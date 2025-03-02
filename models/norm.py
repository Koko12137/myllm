import torch
import torch.nn as nn


class MyLLMRMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None: 
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Record dtype of the input tensor
        input_type = x.dtype
        # Cast the input tensor to float32
        x = x.float()
        # Compute the root mean square
        x_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize the input tensor
        x = x * x_rms
        # Scale the tensor
        x = x * self.gamma
        return x.to(input_type)