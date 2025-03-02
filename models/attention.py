import warnings
from typing import Callable
from functools import partial

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv

from models.configuration import MyLLMConfig
from utils.hooks import check_nan


def eager_attn(
    module: nn.Module, 
    q_states: torch.Tensor, 
    k_states: torch.Tensor, 
    v_states: torch.Tensor, 
    attention_mask: torch.Tensor = None,  
    dropout: float = 0.0, 
    scaling: float = None, 
    is_causal: bool = False, 
    **kwargs
) -> tuple[torch.Tensor]: 
    r"""Eager attention implementation. 
    
    Args:
        module ('nn.Module'): 
            The module to apply the attention.
        q_states ('torch.Tensor'): 
            The query states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        k_states ('torch.Tensor'): 
            The key states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        v_states ('torch.Tensor'): 
            The value states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        attention_mask ('torch.Tensor', *optional*):  
            The attention mask with shape (bsz, 1, seq_len, seq_len) 
        dropout ('float', *optional*, defaults to 0.0): 
            The dropout probability. Default to 0.0. 
        scaling ('float', *optional*): 
            The scaling factor for the attention weights. If not provided, it will be calculated as `q_states.size(-1) ** -0.5`. 
        is_causal ('bool', *optional*, defaults to False): 
            Whether the attention is causal or not. 
            
    Returns: 
        `torch.Tensor`: 
            The output tensor after the attention mechanism. 
        
        `torch.Tensor`:
            The attention weights.
    """
    if scaling is None:
        scaling = q_states.size(-1) ** -0.5
        
    # Repeat the key and value states to match the number of attention heads
    # Output shape: (bsz, num_attention_heads, seq_len, attention_head_dim)
    k_states = repeat_kv(k_states, module.num_key_value_groups)
    v_states = repeat_kv(v_states, module.num_key_value_groups)
        
    # Compute the attention weights
    # Output shape: (bsz, num_attention_heads, seq_len, seq_len)
    attn_weights = torch.einsum("bhid,bhjd->bhij", q_states, k_states) * scaling
    
    # Add the causal mask
    attn_weights = attn_weights + attention_mask
    
    # Apply the softmax, the output of softmax in padding rows are not zero
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = torch.dropout(attn_weights, p=dropout, train=module.training)
    
    # Apply attention weights to the value states
    # Output shape: (bsz, num_attention_heads, seq_len, attention_head_dim)
    attn_values = torch.einsum("bhis,bhsj->bhij", attn_weights, v_states)
    attn_values = attn_values.transpose(1, 2).contiguous()
    return attn_values, attn_weights


ATTENTION: dict[str, Callable] = {
    'eager': eager_attn,
    'sdpa': sdpa_attention_forward,
}


class MyLLMGroupAttention(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = getattr(config, 'attention_head_dim', self.hidden_size // self.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        # If debug is enabled, register the hook
        if self.debug:
            self.q_proj.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="q_proj"))
            self.k_proj.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="k_proj"))
            self.v_proj.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="v_proj"))
            self.o_proj.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="o_proj"))

    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_embeddings: torch.Tensor, 
        # Output arguments
        output_attentions: bool = False, 
        # Cache arguments
        past_key_value: Cache = None, 
        cache_position: torch.LongTensor = None, 
        **kwargs
    ) -> tuple[torch.Tensor | None]: 
        r"""Attention is all you need. This method take the following steps:
        1. Compute the query, key, and value states
        2. Apply the position embeddings
        3. Repeat the key and value states to match the number of attention heads
        4. View the key and value states as (bsz, num_attention_heads, seq_len, attention_head_dim)
        
        Args: 
            hidden_states (`torch.Tensor`): 
                The hidden states with shape (bsz, seq_len, hidden_size)
            attention_mask (`torch.Tensor`):
                The attention mask with shape (bsz, 1, seq_len, seq_len)
            position_embeddings (`torch.Tensor`): 
                The position embeddings with shape (bsz, seq_len, hidden_size)
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether to output attentions or not. 
            past_key_value (`Cache`, *optional*): 
                The past key value cache
            cache_position (`torch.LongTensor`, *optional*): 
                The cache position tensor. Shape of (seq_len,)
        
        Returns:
            `torch.Tensor`: 
                The output tensor after the attention mechanism.
                
            `torch.Tensor` or `None`:
                The attention weights if `output_attentions` is enabled.
        """
        # Input shape: (bsz, seq_len, hidden_size)
        input_shape = hidden_states.shape[:-1]
        # Hidden shape: (bsz, seq_len, n, attention_head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # Compute the query, key, and value
        # Output shape: (bsz, seq_len, num_attention_heads, attention_head_dim)
        q_states: torch.Tensor = self.q_proj(hidden_states).view(hidden_shape)
        # Output shape: (bsz, seq_len, num_key_value_heads, attention_head_dim)
        k_states: torch.Tensor = self.k_proj(hidden_states).view(hidden_shape)
        # Output shape: (bsz, seq_len, num_key_value_heads, attention_head_dim)
        v_states: torch.Tensor = self.v_proj(hidden_states).view(hidden_shape)
        
        # Transpose the query, key and value states to (bsz, n, seq_len, attention_head_dim)
        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)
        
        # Apply the position embeddings
        q_states = q_states * position_embeddings
        k_states = k_states * position_embeddings
        
        # If Cache is provided, update the key and value states
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            k_states, v_states = past_key_value.update(k_states, v_states, cache_kwargs)
            
        attn_impl = ATTENTION[self.config._attn_implementation]
        # Check if the output attentions is enabled
        if self.config._attn_implementation != 'eager':
            if self.config._attn_implementation == 'sdpa' and output_attentions:
                warnings.warn("The output_attentions argument is not supported for the sdpa implementation. Falling back to eager implementation.")
                attn_impl = eager_attn
        
        # Check if the attention is causal, if attention_mask is None, it is designated as causal
        is_causal = True if attention_mask is None else False
        
        # Apply the attention mechanism
        attn_out, attn_weights = attn_impl(
            self, 
            q_states, 
            k_states, 
            v_states, 
            attention_mask, 
            self.dropout if self.training else 0.0, 
            self.scaling, 
            is_causal, 
        )
        
        # Transpose the attention output to (bsz, seq_len, n * attention_head_dim)
        attn_out = attn_out.reshape(*input_shape, -1)
        # Apply the output linear layer
        attn_out = self.o_proj(attn_out)
        return attn_out, attn_weights