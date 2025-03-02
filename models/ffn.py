from functools import partial

import torch
import torch.nn as nn

from models.configuration import MyLLMConfig
from models.activation import MyLLMSwiGLU
from utils.hooks import check_nan


class MyLLMFFN(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_act = MyLLMSwiGLU(self.intermediate_size)
        
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