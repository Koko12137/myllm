import torch
import torch.nn as nn

    
class MyLLMSwiGLU(nn.Module):
    
    def __init__(self, dim: int) -> None: 
        super().__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Linear(self.dim, self.dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.sigmoid(x) * self.w(x)
        return x