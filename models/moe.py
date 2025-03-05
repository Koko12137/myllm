from functools import partial

import torch
import torch.nn as nn

from models.configuration import MyLLMConfigForMoE, MyLLMConfigForCoE
from models.ffn import MyLLMFFN
from models.activation import MyLLMSwiGLU
from utils.hooks import check_nan


class MyLLMMoERouter(nn.Module):
    
    def __init__(self, config: MyLLMConfigForMoE, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk_experts = config.topk_experts
        self.expert_gate = nn.Linear(self.hidden_size, self.num_experts)
        
        # If debug is enabled, register the hook
        if self.debug:
            self.expert_gate.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="expert_gate"))
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]: 
        """Forward pass of the MoE router

        Args:
            x (torch.Tensor): 
                The input tensor with shape (bsz, seq_len, hidden_size)

        Returns:
            tuple[torch.Tensor]: 
                The output logits with shape (bsz, seq_len, topk_experts), last dimension is the expert index
                The output ids with shape (bsz, seq_len, hidden_size)
        """
        # Compute the expert gate
        gate = self.expert_gate(x)
        # Get top k experts
        logits, ids = torch.topk(gate, self.topk_experts, dim=-1)
        # Convert logit to probability
        mask = torch.full_like(gate, -float('inf'))
        mask.scatter_(-1, ids, logits)
        logits = torch.softmax(mask, dim=-1)
        return logits, ids
    

class MyLLMFFNMoE(nn.Module):
    
    def __init__(self, config: MyLLMConfigForMoE, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_share_experts = config.num_share_experts
        # Check num_share_experts smaller than num_experts
        if self.num_share_experts >= self.num_experts:
            raise ValueError("`num_experts` should more than `num_share_experts`")
        
        # Split the intermediate size into each experts
        self.expert_size = self.intermediate_size // self.num_experts
        config.intermediate_size = self.expert_size
        
        # Share experts
        if self.num_share_experts > 0:
            self.share_up = nn.Linear(self.hidden_size, self.num_share_experts * self.expert_size)
            self.share_down = nn.Linear(self.num_share_experts * self.expert_size, self.hidden_size)
            self.up_act = MyLLMSwiGLU(self.expert_size)
        
        # Initialize unique experts and routers
        n = self.num_experts - self.num_share_experts
        config.topk_experts = n     # Modify the topk
        self.experts = nn.ModuleDict({
            f"{layer_idx}-CoE-{idx}-Expert": MyLLMFFN(config, layer_idx, debug) for idx in range(n)
        })
        self.router = MyLLMMoERouter(config, layer_idx, debug)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass of the MoE FFN

        Args:
            x ('torch.Tensor'): 
                The input tensor with shape (bsz, seq_len, hidden_size)

        Returns:
            'torch.Tensor': 
                The output tensor with shape (bsz, seq_len, hidden_size) 
        """
        x_shape = x.size()
        
        # Compute the expert gate
        # Shape: (bsz, seq_len, num_experts), (bsz, seq_len, topk_experts)
        logits, ids = self.router(x)                
        logits = logits.view(-1, logits.size(-1))   # Shape: (bsz * seq_len, num_experts)
        ids = ids.view(-1, ids.size(-1))            # Shape: (bsz * seq_len, topk_experts)
        
        # Create a zero tensor with the same shape as x for the expert output
        x = x.view(-1, x.size(-1))      # Shape: (bsz * seq_len, hidden_size)
        zeros = torch.zeros_like(x)     # Shape: (bsz * seq_len, hidden_size)
        
        # Compute the expert output
        for i, expert in enumerate(self.experts):
            # Create a expert mask
            expert_mask = (ids == i).any(dim=-1)
            # Mask the input tensor
            x_masked = x[expert_mask]
            
            if x_masked.size(0) == 0:
                # Skip the expert if the mask is empty
                continue
            
            # Compute the expert output
            expert_out = expert(x_masked)
            # Get expert score
            scores = logits[expert_mask][:, i]
            # Scale the expert output
            expert_out = expert_out * scores.unsqueeze(-1)
            # Update the expert output
            zeros[expert_mask] += expert_out
            
        # Reshape the expert output
        expert_out = zeros.view(*x_shape)
        
        # Compute share experts
        if self.num_share_experts > 0:
            share_out = self.share_up(x)
            share_out = self.up_act(share_out)
            share_out = self.share_down(share_out)
        
            return expert_out + share_out
        return expert_out
    
    
class MyLLMFFNCoE(MyLLMFFNMoE):
    """Chain of Experts. Reference: https://github.com/ZihanWang314/CoE """
    
    def __init__(self, config: MyLLMConfigForCoE, layer_idx: int, debug: bool = False) -> None: 
        super().__init__(config, layer_idx, debug)
        self.num_chains = config.num_chains
        self.residual = config.residual
        
        config.topk_experts = self.num_experts - self.num_share_experts
        self.routers = nn.ModuleDict({
            f"{layer_idx}-CoE-{idx}-Router": MyLLMMoERouter(config, layer_idx, debug) for idx in range(self.num_chains)
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass of the MoE FFN

        Args:
            x ('torch.Tensor'): 
                The input tensor with shape (bsz, seq_len, hidden_size)

        Returns:
            'torch.Tensor': 
                The output tensor with shape (bsz, seq_len, hidden_size) 
        """
        # Save size and residual
        x_shape = x.size()
        if self.residual:
            residual = x
        
        # Create a zero tensor with the same shape as x for the expert output
        x = x.view(-1, x.size(-1))      # Shape: (bsz * seq_len, hidden_size)
        
        for j, router in enumerate(self.routers):
            
            # Compute the expert gate
            # Shape: (bsz, seq_len, num_experts), (bsz, seq_len, topk_experts)
            logits, ids = router(x)                
            logits = logits.view(-1, logits.size(-1))   # Shape: (bsz * seq_len, num_experts)
            ids = ids.view(-1, ids.size(-1))            # Shape: (bsz * seq_len, topk_experts)
            
            zeros = torch.zeros_like(x)     # Shape: (bsz * seq_len, hidden_size)
            
            # Compute the expert output
            for i, expert in enumerate(self.experts):
                # Create a expert mask
                expert_mask = (ids == i).any(dim=-1)
                # Mask the input tensor
                x_masked = x[expert_mask]
                
                if x_masked.size(0) == 0:
                    # Skip the expert if the mask is empty
                    continue
                
                # Compute the expert output
                expert_out = expert(x_masked)
                # Get expert score
                scores = logits[expert_mask][:, i]
                # Scale the expert output
                expert_out = expert_out * scores.unsqueeze(-1)
                # Update the expert output
                zeros[expert_mask] += expert_out
                
            # Reshape the expert output
            expert_out = zeros.view(*x_shape)
            
            # Compute share experts
            if self.num_share_experts > 0:
                share_out = self.share_up(x).view(-1, self.num_share_experts, self.expert_size)
                share_out = self.up_act(share_out)
                share_out = self.share_down(share_out.view(-1, self.num_share_experts * self.expert_size))
            
                expert_out += share_out
                
            # Apply residual
            if self.residual:
                x = residual + expert_out
            else:
                x = expert_out
            
        return x
            