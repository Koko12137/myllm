import torch
import torch.nn as nn


class MyLLMRotaryEmbedding(nn.Module):
    
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0) -> None: 
        super().__init__()
        self.theta = theta
        self.hidden_size = dim
        self.max_seq_len = max_seq_len
        
        # Initialize the rotary position embedding
        inv_freq = 1 / (self.theta ** (torch.arange(0, self.hidden_size, 2) / self.hidden_size))
        # Register buffer
        self.register_buffer("inv_freq", inv_freq)
    
    @torch.no_grad()
    def compute_pos_emb(self, ids: torch.Tensor) -> torch.Tensor:
        # Record the bsz, seq_len, hidden_size
        bsz, seq_len = ids.shape
        # Compute the sinusoidal position encoding, output shape: (bsz, seq_len, hidden_size // 2)
        sinusoid_inp = torch.einsum("bi,j->bij", ids.float(), self.inv_freq)
        # Convert to polar coordinates, output shape: (bsz, 1, hidden_size // 2, 1)
        pos_emb = torch.polar(torch.ones_like(sinusoid_inp), sinusoid_inp)
        return pos_emb.view(bsz, 1, seq_len, -1)
    
    @torch.no_grad()
    def apply_pos_emb(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        # Record the bsz, n, seq_len, hidden_size
        _, n, seq_len, hidden_size = x.shape
        # View the input tensor as (bsz, n, seq_len, hidden_size // 2, 2) and cast to complex numbers
        x = torch.view_as_complex(x.view(-1, n, seq_len, hidden_size // 2, 2))
        # Apply the rotation
        x = x * pos_emb
        # Convert back to real numbers and view as (bsz, seq_len, hidden_size)
        x = torch.view_as_real(x).view(-1, n, seq_len, hidden_size)
        return x
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, ids: torch.Tensor = None, pos_emb: torch.Tensor = None) -> torch.Tensor:
        if pos_emb is None and ids is not None:
            pos_emb = self.compute_pos_emb(ids)
            
        # Compute as complex numbers
        x = self.apply_pos_emb(x, pos_emb)
        return x