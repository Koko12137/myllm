import random
from functools import partial

import torch
import torch.nn as nn

from models.configuration import MyLLMConfig
from models.ffn import MyLLMFFN
from models.activation import MyLLMSwiGLU
from models.norm import MyLLMRMSNorm
from utils.hooks import check_nan


class MyLLMMoERouter(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk_experts = config.topk_experts
        self.expert_temp = config.expert_temperature
        self.expert_sample = config.expert_sample
        self.alpha = config.gate_random_alpha
        
        self.expert_gate = nn.Linear(self.hidden_size, self.num_experts - config.num_share_experts)
        # Referring to Deepseek MoE Balance
        self.gate_bias = nn.Parameter(torch.zeros(self.num_experts - config.num_share_experts))
        
        # If debug is enabled, register the hook
        if self.debug:
            self.expert_gate.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="expert_gate"))
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]: 
        """Forward pass of the MoE router

        Args:
            x (torch.Tensor): 
                The input tensor with shape (bsz, seq_len, hidden_size)

        Returns:
            torch.Tensor: 
                The output logits with shape (bsz, seq_len, topk_experts), last dimension is the expert index
                
            torch.Tensor:
                The output ids with shape (bsz, seq_len, topk_experts), last dimension is the expert index
        """
        # Compute the expert gate
        gate: torch.Tensor = self.expert_gate(x)
        
        # Apply temperature
        gate = gate / self.expert_temp
        
        # Compute bias gate
        bias_gate = gate + self.gate_bias
        # Apply noise
        if self.alpha > 0:
            bias_gate += torch.rand(bias_gate.size()) * self.alpha
        
        if not self.expert_sample:
            # Get top k experts
            _, ids = torch.topk(bias_gate, self.topk_experts, dim=-1)
        else:
            # Convert gate values to probabilities
            gate_probs = torch.softmax(bias_gate, dim=-1)
            # Sample experts based on gate probabilities
            ids = torch.multinomial(gate_probs, self.topk_experts, replacement=False)
            
        # Get the logits with out bias
        logits = gate.gather(-1, ids)
        
        # Convert logit to probability
        mask = torch.full_like(gate, -float('inf'))
        mask.scatter_(-1, ids, logits)
        logits = torch.softmax(mask, dim=-1)
        
        return logits, ids
    

class MyLLMFFNMoE(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // (config.topk_experts + config.num_share_experts)
        self.num_experts = config.num_experts
        self.num_share_experts = config.num_share_experts
        self.balance_penalty = config.balance_penalty
        
        # Check num_share_experts smaller than num_experts
        if self.num_share_experts >= self.num_experts:
            raise ValueError("`num_experts` should more than `num_share_experts`")
        
        # Share experts
        if self.num_share_experts > 0:
            self.share_up = nn.Linear(self.hidden_size, self.num_share_experts * self.intermediate_size)
            self.share_down = nn.Linear(self.intermediate_size, self.hidden_size)
            self.up_act = MyLLMSwiGLU(self.intermediate_size)
        
        self.experts = nn.ModuleDict({
            f"{layer_idx}-MoE-{idx}-Expert": MyLLMFFN(
                self.hidden_size, self.intermediate_size, self.hidden_size, layer_idx, debug
            ) for idx in range(self.num_experts - self.num_share_experts)
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
        
        # Penalty for imbalance
        penalty = torch.zeros_like(self.router.gate_bias)
        
        # Compute the expert output
        for i, expert in enumerate(self.experts):
            # Create a expert mask
            expert_mask = (ids == i).any(dim=-1)
            # Mask the input tensor
            x_masked = x[expert_mask]
            
            if x_masked.size(0) == 0:
                # BUG: this will cause dead lock while the other gpu process is waiting for all_gather if we do nothing but continue
                # Reference: https://github.com/deepspeedai/DeepSpeed/issues/5066#issuecomment-1989459339
                if self.training:
                    # Dummy inputs, Shape: (1, hidden_size)
                    dummy_inputs = torch.zeros((1, zeros.shape[-1]), device=x.device, dtype=x.dtype)
                    # Dummy output, this must be assured that the dummy output is zero
                    expert_out = self.experts[expert](dummy_inputs)
                    # Update the expert output
                    zeros[0] += expert_out[0]
                else:
                    continue
            else:
                # Compute the expert output
                expert_out = self.experts[expert](x_masked)
                # Get expert score
                scores = logits[expert_mask][:, i]
                # Scale the expert output
                expert_out = expert_out * scores.unsqueeze(-1)
                # Update the expert output
                zeros[expert_mask] += expert_out
            
        # Update penalty
        penalty = (penalty - penalty.mean(dim=-1)) / penalty.var(dim=-1)
        self.router.gate_bias = nn.Parameter(self.router.gate_bias - penalty * self.balance_penalty)
        
        # Compute share experts
        if self.num_share_experts > 0:
            share_out: torch.Tensor = self.share_up(x)          # Shape: (bsz * seq_len, num_share_experts * intermediate_size)
            share_out = share_out.view(share_out.size(0), self.num_share_experts, self.intermediate_size)
            share_out = self.up_act(share_out)
            share_out = self.share_down(share_out)              # Shape: (bsz * seq_len, num_share_experts, hidden_size)
            share_out = share_out.sum(dim=1)
        
            # Update the expert output
            zeros += share_out
            
        # Reshape the expert output
        expert_out = zeros.view(*x_shape)
        
        return expert_out
    
    
class MyLLMFFNCoE(MyLLMFFNMoE):
    """Chain of Experts. Reference: https://github.com/ZihanWang314/CoE """
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None: 
        super().__init__(config, layer_idx, debug)
        self.num_chains = config.num_chains
        self.residual = config.residual
        
        # Remove the router
        del self.router
        
        self.routers = nn.ModuleDict({
            f"{layer_idx}-CoE-{idx}-Router": MyLLMMoERouter(config, layer_idx, debug) for idx in range(self.num_chains)
        })
        
        # Add norm layers avoiding gradient explosion
        if self.residual:
            self.norms = nn.ModuleDict({
                f"{layer_idx}-CoE-{idx}-Router": MyLLMRMSNorm(config.hidden_size) for idx in range(self.num_chains)
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
        
        # Create a zero tensor with the same shape as x for the expert output
        x = x.view(-1, x.size(-1))      # Shape: (bsz * seq_len, hidden_size)
        
        for j, (name, router) in enumerate(self.routers.items()):
            
            if self.residual:
                residual = x
            
            # Compute the expert gate
            # Shape: (bsz, seq_len, num_experts), (bsz, seq_len, topk_experts)
            logits, ids = router(x)
            logits = logits.view(-1, logits.size(-1))   # Shape: (bsz * seq_len, num_experts)
            ids = ids.view(-1, ids.size(-1))            # Shape: (bsz * seq_len, topk_experts)
            
            zeros = torch.zeros_like(x)     # Shape: (bsz * seq_len, hidden_size)
            
            # Penalty for imbalance
            penalty = torch.zeros_like(router.gate_bias)
            
            # Compute the expert output
            for i, expert in enumerate(self.experts):
                # Create a expert mask
                expert_mask = (ids == i).any(dim=-1)
                # Mask the input tensor
                x_masked = x[expert_mask]
                
                if x_masked.size(0) == 0:
                    # BUG: this will cause dead lock while the other gpu process is waiting for all_gather if we do nothing but continue
                    # Reference: https://github.com/deepspeedai/DeepSpeed/issues/5066#issuecomment-1989459339
                    if self.training:
                        # Dummy inputs, Shape: (1, hidden_size)
                        dummy_inputs = torch.zeros((1, zeros.shape[-1]), device=x.device, dtype=x.dtype)
                        # Dummy output, this must be assured that the dummy output is zero
                        expert_out = self.experts[expert](dummy_inputs)
                        # Update the expert output
                        zeros[0] += expert_out[0]
                    else:
                        continue
                else:
                    # Compute the expert output
                    expert_out = self.experts[expert](x_masked)
                    # Get expert score
                    scores = logits[expert_mask][:, i]
                    # Scale the expert output
                    expert_out = expert_out * scores.unsqueeze(-1)
                    # Update the expert output
                    zeros[expert_mask] += expert_out
            
            # Update penalty
            penalty = (penalty - penalty.mean(dim=-1)) / penalty.var(dim=-1)
            router.gate_bias = nn.Parameter(router.gate_bias - penalty * self.balance_penalty)
            
            # Compute share experts
            if self.num_share_experts > 0:
                share_out: torch.Tensor = self.share_up(x)          # Shape: (bsz * seq_len, num_share_experts * intermediate_size)
                share_out = share_out.view(share_out.size(0), self.num_share_experts, self.intermediate_size)
                share_out = self.up_act(share_out)
                share_out = self.share_down(share_out)              # Shape: (bsz * seq_len, num_share_experts, hidden_size)
                share_out = share_out.sum(dim=1)
            
                # Update the expert output
                zeros += share_out
                
            # Apply residual
            if self.residual:
                x = self.norms[name](residual + zeros)
            else:
                x = zeros
                
        # Convert to source shape
        return x.view(*x_shape)
            