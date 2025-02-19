import warnings
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from transformers import Cache, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast, 
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv

from models.configuration import MyLLMConfig


def check_nan(module: nn.Module, grad_input: nn.Module, grad_output: nn.Module, layer_idx: int, name: str) -> None:
    if grad_output[0] is not None:
        output_nan = torch.isnan(grad_output[0]).sum().item()
        assert output_nan == 0, f"NaN detected in layer {layer_idx}, name: {name}"
    if grad_input[0] is not None:
        input_nan = torch.isnan(grad_input[0]).sum().item()
        assert input_nan == 0, f"NaN detected in layer {layer_idx}, name: {name}"
    

def eager_attn(
    module: nn.Module, 
    q_states: torch.Tensor, 
    k_states: torch.Tensor, 
    v_states: torch.Tensor, 
    attention_mask: torch.Tensor, 
    dropout: float = 0.0, 
    scaling: float = None, 
    **kwargs
) -> tuple[torch.Tensor]: 
    r"""Eager attention implementation. 
    
    Args:
        module (nn.Module): 
            The module to apply the attention.
        q_states (torch.Tensor): 
            The query states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        k_states (torch.Tensor): 
            The key states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        v_states (torch.Tensor): 
            The value states with shape (bsz, num_attention_heads, seq_len, attention_head_dim)
        attention_mask (torch.Tensor): 
            The attention mask with shape (bsz,1, seq_len, seq_len) 
        dropout (float): 
            The dropout probability. Default to 0.0. 
        scaling (float): 
            The scaling factor for the attention weights. 
    """
    if scaling is None:
        scaling = q_states.size(-1) ** -0.5
        
    # Reapeat the key and value states to match the number of attention heads
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
    
    
class MyLLMSwiGLU(nn.Module):
    
    def __init__(self, dim: int) -> None: 
        super().__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Linear(self.dim, self.dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.sigmoid(x) * self.w(x)
        return x
    

class MyLLMForward(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None: 
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.debug = debug
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_act = nn.SiLU()
        
        # If debug is enabled, register the hook
        if self.debug:
            self.up.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="up"))
            self.gate.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="gate"))
            self.down.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="down"))
            self.up_act.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="up_act"))
            # self.down_act.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="down_act"))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.down(self.up_act(self.up(x) * self.gate(x)))
        return out


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
            hidden_states (torch.Tensor): 
                The hidden states with shape (bsz, seq_len, hidden_size)
            attention_mask (torch.Tensor):
                The attention mask with shape (bsz, 1, seq_len, seq_len)
            position_embeddings (torch.Tensor): 
                The position embeddings with shape (bsz, seq_len, hidden_size)
            past_key_value (Cache): 
                The past key value cache
            cache_position (torch.LongTensor): 
                The cache position tensor. 
        
        Returns:
            tuple[torch.Tensor | None]: 
                The output tensor and the attention weights.
        """
        # Input shape: (bsz, seq_len, hidden_size)
        input_shape = hidden_states.shape[:-1]
        # Hidden shape: (bsz, seq_len, n, attention_head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, seq_len, n, dim = hidden_shape
        
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
            # TODO: Learn how to implement this
            raise NotImplementedError("Cache is not implemented yet.")
        
        # Get attention implementation
        attn_impl = ATTENTION[self.config._attn_implementation]
        # Check if the output attentions is enabled
        if self.config._attn_implementation != 'eager':
            if self.config._attn_implementation == 'sdpa' and kwargs.get('output_attentions', False):
                warnings.warn("The output_attentions argument is not supported for the sdpa implementation. Falling back to eager implementation.")
                attn_impl = eager_attn
        
        is_causal = True if attention_mask is None else False
        
        # Apply the attention
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


class MyLLMDecoderLayer(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.debug = debug
        
        self.attn = MyLLMGroupAttention(config, layer_idx=layer_idx, debug=debug)
        self.ffn = MyLLMForward(config, layer_idx, debug)
        self.input_norm = MyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_norm = MyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # If debug is enabled, register the hook
        if self.debug:
            self.input_norm.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="input_norm"))
            self.output_norm.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="output_norm"))
        
    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_embeddings: torch.Tensor, 
        output_attentions: bool = False, 
        past_key_value: Cache = None, 
        cache_position: torch.LongTensor = None, 
    ) -> tuple[torch.Tensor]:
        # Record the hidden states
        residual = hidden_states
        
        # Normalize the input
        hidden_states = self.input_norm(hidden_states)
        
        # Self-Attention
        hidden_states, attn_weights = self.attn(
            hidden_states, 
            attention_mask=attention_mask, 
            position_embeddings=position_embeddings, 
            output_attentions=output_attentions, 
            past_key_value=past_key_value, 
            cache_position=cache_position, 
        )
        # Set hidden states to residual
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        # Normalize the output
        hidden_states = self.output_norm(hidden_states)
        # Feed Forward Network
        hidden_states = self.ffn(hidden_states)
        # Add the residual
        hidden_states = hidden_states + residual
        
        outputs = (hidden_states,) 
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
    
    
class MyLLMPreTrainedModel(PreTrainedModel):
    
    config_class = MyLLMConfig
    base_model_prefix = "myllm" 
    supports_gradient_checkpointing = False     # TODO: Implement gradient checkpointing
    _no_split_modules = ["MyLLMDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # Initialize the padding index to zero vector
                module.weight.data[module.padding_idx].zero_()
        
    
class MyLLMModel(MyLLMPreTrainedModel):
    
    def __init__(self, config: MyLLMConfig, debug: bool = False) -> None:
        super().__init__(config)
        self.config = config
        self.debug = debug
        
        # Initialize embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) 
        
        # Initialize the model components
        self.layers = nn.ModuleList([
            MyLLMDecoderLayer(config, idx, debug) for idx in range(config.num_hidden_layers)
        ])
        
        # Initialize other necessary components
        self.norm = MyLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MyLLMRotaryEmbedding(config.attention_head_dim, config.max_position_embeddings, config.rope_theta)
        self.gradient_checkpointing = None  # TODO: Implement gradient checkpointing
        
        # If debug is enabled, register the hook
        if self.debug:
            self.norm.register_full_backward_hook(partial(check_nan, layer_idx=-1, name="norm"))
            self.embed_tokens.register_full_backward_hook(partial(check_nan, layer_idx=-1, name="embed_tokens"))
        
        # Post initialization
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        assert isinstance(value, nn.Embedding), f"Value should be of type nn.Embedding, got {type(value)}"
        self.embed_tokens = value
        
    def forward(
        self,
        attention_mask: torch.Tensor, 
        position_ids: torch.LongTensor, 
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        # Output arguments
        use_cache: bool = False,                    # TODO: Not implemented yet
        output_attentions: bool = False, 
        output_hidden_states: bool = False, 
        return_dict: bool = False, 
        # Cache arguments
        past_key_values: Cache = None,              # TODO: Not implemented yet
        cache_position: torch.LongTensor = None,    # TODO: Not implemented yet
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        # Get the embeddings
        if inputs_embeds is None:
            hidden_states: torch.Tensor = self.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds should be provided.")
        
        # Container for all the hidden states and attentions
        all_hidden_states = ()
        all_attentions = () if output_attentions else None
        
        # Record the shape of the hidden states
        bsz, seq_len, _ = hidden_states.shape
        
        # Create cache if not provided and cache is enabled
        if use_cache:
            raise NotImplementedError("Cache is not implemented yet.")
        
        # Compute the position embeddings
        # If position cache is provided, update the position embeddings
        if cache_position is not None:
            # TODO: Learn how to implement this
            raise NotImplementedError("Cache is not implemented yet.")
        
        # Create a ones-like states with shape of (1, 1, seq_len, attention_head_dim)
        position_embeddings = torch.ones(1, 1, seq_len, self.config.attention_head_dim, device=hidden_states.device)
        
        # Apply the position embeddings
        # Output shape: (bsz, seq_len, num_attention_heads, attention_head_dim)
        position_embeddings = self.rotary_emb(position_embeddings, position_ids).to(hidden_states.dtype)
        
        causal_mask = self._update_causal_mask(
            attention_mask, 
            input_tensor=hidden_states, 
            cache_position=cache_position, 
            past_key_values=past_key_values, 
            output_attentions=output_attentions, 
        )
        
        # Forward pass through the model layers
        for idx, layer in enumerate(self.layers):
            # Check if the gradient checkpointing is enabled
            if self.gradient_checkpointing:
                raise NotImplementedError("Gradient checkpointing is not implemented yet.")
            else:
                layer_outputs = layer(
                    hidden_states, 
                    attention_mask=causal_mask, 
                    position_embeddings=position_embeddings, 
                    output_attentions=output_attentions, 
                    past_key_value=past_key_values, 
                )
            
            # Update the hidden states
            hidden_states = layer_outputs[0]
            
            # Update the attentions
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        # Normalize the hidden states
        hidden_states = self.norm(hidden_states)
        
        # Add the hidden states to the hidden states list after the last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Create the output
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, 
            past_key_values=past_key_values, 
            hidden_states=all_hidden_states, 
            attentions=all_attentions,
        )
        
        if return_dict:
            return output
        else:
            return output.to_tuple()
        
    def _prepare_4d_causal_mask(
        self, 
        attention_mask: torch.Tensor, 
        dtype: torch.dtype, 
        device: torch.device, 
    ) -> torch.Tensor:
        """Create a 4D causal mask for the attention mechanism. 
        HF Implementation Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
        I'm not familiar about all the details of all methods of HF Implementation, so I just implemented the method that I need.
        
        Args:
            attention_mask (torch.Tensor): 
                The attention mask with shape (bsz, seq_len)
            device (torch.device): 
                The device of the attention mask needed to be converted to.
                
        Returns:
            torch.Tensor: 
                The 4D causal mask with shape (bsz, 1, seq_len, seq_len)
        """
        # Record the shape of the attention mask
        bsz, seq_len = attention_mask.shape
        
        dtype_min = torch.finfo(dtype).min
        
        # Create a triangular mask with min values in the upper triangular part
        # Output shape: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), dtype_min, device=device), diagonal=1)
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len).to(dtype)
        # Convert the dims specified by the input attention_mask to -inf
        attention_mask = attention_mask.view(bsz, 1, 1, seq_len)
        # Reapeat the attention mask
        attention_mask = attention_mask.repeat(1, 1, seq_len, 1)
        # Convert the causal mask to min where the value of attention mask is -1
        causal_mask = causal_mask.masked_fill(attention_mask == 0, dtype_min)
        # Convert the causal mask to 0 if all of the values are min in a row
        attention_mask = attention_mask.transpose(-1, -2)
        return causal_mask * attention_mask
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ) -> torch.Tensor:
        """Update the causal mask for the attention mechanism. 
        
        Args:
            attention_mask (torch.Tensor): 
                The attention mask with shape (bsz, seq_len)
            input_tensor (torch.Tensor): 
                The input tensor with shape (bsz, seq_len, hidden_size)
            cache_position (torch.Tensor): 
                The cache position tensor with shape (bsz, seq_len)
            past_key_values (Cache): 
                The past key values cache
            output_attentions (bool): 
                Whether to output attentions or not
        """
        dtype = input_tensor.dtype
        device = input_tensor.device
        
        # Create the causal mask
        causal_mask = self._prepare_4d_causal_mask(attention_mask, dtype, device)
        return causal_mask
    
    
class MyLLMForCausalLM(MyLLMPreTrainedModel, GenerationMixin):
    """MyLLMForCausalLM is a causal language model that predicts the next token in a sequence. 
    
    MyLLMForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. 
    From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
    - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
    - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
    """
    
    def __init__(self, config: MyLLMConfig, debug: bool = False) -> None:
        super().__init__(config)
        self.config = config
        self.debug = debug
        self.vocab_size = config.vocab_size
        
        # Initalize the causal model
        self.model = MyLLMModel(config, self.debug)
        # Initialize the output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # If debug is enabled, register the hook
        if self.debug:
            self.lm_head.register_full_backward_hook(partial(check_nan, layer_idx=-2, name="lm_head"))
        
        # Post initialization
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        assert isinstance(value, nn.Embedding), f"Value should be of type nn.Embedding, got {type(value)}"
        self.model.set_input_embeddings(value)
        
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Linear) -> None:
        assert isinstance(value, nn.Linear), f"Value should be of type nn.Linear, got {type(value)}"
        self.lm_head = value
        
    def get_decoder(self) -> MyLLMModel:
        return self.model

    def set_decoder(self, value: MyLLMModel) -> None:
        assert isinstance(value, MyLLMModel), f"Value should be of type MyLLMModel, got {type(value)}"
        self.model = value
        
    def forward(
        self,
        attention_mask: torch.Tensor, 
        position_ids: torch.LongTensor, 
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        # Target arguments
        labels: torch.LongTensor = None, 
        # Output arguments
        use_cache: bool = None,                     # TODO: Not implemented yet
        output_attentions: bool = None, 
        output_hidden_states: bool = None, 
        return_dict: bool = None, 
        num_logits_to_keep: int = 0, 
        # Cache arguments
        past_key_values: Cache = None,              # TODO: Not implemented yet
        cache_position: torch.LongTensor = None,    # TODO: Not implemented yet
    ) -> tuple | CausalLMOutputWithPast:
        # Check if output_hidden_states is enabled
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        # Check if output_attentions is enabled
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        # Check if return_dict is enabled
        return_dict = return_dict if return_dict else self.config.use_return_dict
        
        # Input shape: (bsz, seq_len), Output hidden states shape: (bsz, seq_len, hidden_size)
        outputs: BaseModelOutputWithPast | tuple = self.model(
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict, 
            past_key_values=past_key_values, 
            cache_position=cache_position, 
        )
        
        # Get the output hidden states
        hidden_states: torch.Tensor = outputs[0]
        
        # Forward pass through the lm head
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])    # Output shape: (bsz, seq_len, vocab_size)
        logits = logits.view(-1, self.vocab_size)                           # Output shape: (bsz * seq_len, vocab_size)
        
        # Mask the logits and labels, we do not update the padding tokens
        logits = logits[attention_mask.view(-1)]
        labels = labels.view(-1)[attention_mask.view(-1)]
        
        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
            
        # Create the output
        outputs = CausalLMOutputWithPast(
            loss=loss, 
            logits=logits, 
            past_key_values=outputs.past_key_values, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions, 
        )
        
        if return_dict:
            return outputs
        else:
            return outputs.to_tuple()
        
    def prepare_inputs_for_generation(
        self,
        attention_mask: torch.Tensor, 
        position_ids: torch.LongTensor, 
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        # Target arguments
        labels: torch.LongTensor = None, 
        # Output arguments
        use_cache: bool = None,                     # TODO: Not implemented yet
        output_attentions: bool = None, 
        output_hidden_states: bool = None, 
        return_dict: bool = None, 
        # Cache arguments
        past_key_values: Cache = None,              # TODO: Not implemented yet
        cache_position: torch.LongTensor = None,    # TODO: Not implemented yet
    ) -> dict:
        raise NotImplementedError("Prepare inputs for generation is not implemented yet.")
        