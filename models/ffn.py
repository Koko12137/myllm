from functools import partial

import torch
import torch.nn as nn

from models.configuration import MyLLMConfig
from models.activation import MyLLMSwiGLU
from utils.hooks import check_nan


class MyLLMFFN(nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.layer_idx = layer_idx
        self.debug = debug
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.up = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.down = nn.Linear(self.hidden_channels, self.out_channels, bias=False)
        self.up_act = MyLLMSwiGLU(self.hidden_channels)
        
        # If debug is enabled, register the hook
        if self.debug:
            self.up.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="up"))
            self.down.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="down"))
            self.up_act.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="up_act"))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # Up projection and activation
        out = self.up_act(self.up(x))
        # Down projection
        out = self.down(out)
        return out