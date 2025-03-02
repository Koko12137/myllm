import warnings
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast, 
)

from models.configuration import MyLLMConfig, MyLLMConfigForMoE
from models.rope import MyLLMRotaryEmbedding
from models.ffn import MyLLMFFN
from models.moe import MyLLMFFNMoE
from models.norm import MyLLMRMSNorm
from models.attention import MyLLMGroupAttention
from utils.hooks import check_nan


class MyLLMDecoderLayer(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int, debug: bool = False) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.debug = debug
        
        self.attn = MyLLMGroupAttention(config, layer_idx=layer_idx, debug=debug)
        self.input_norm = MyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_norm = MyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Check type of Feed Forward Network
        if config.use_moe and config.moe_type == "ffn":
            self.ffn = MyLLMFFNMoE(config, layer_idx, debug)
        else:
            self.ffn = MyLLMFFN(config, layer_idx, debug)
        
        # If debug is enabled, register the hook
        if self.debug:
            self.input_norm.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="input_norm"))
            self.output_norm.register_full_backward_hook(partial(check_nan, layer_idx=layer_idx, name="output_norm"))
        
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
        **kwargs: dict, 
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
    supports_gradient_checkpointing = True
    _no_split_modules = ["MyLLMDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    _supports_cache_class = True
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
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        
        # Initialize embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) 
        
        # Initialize the model components
        self.layers = nn.ModuleList([
            MyLLMDecoderLayer(config, idx, debug) for idx in range(config.num_hidden_layers)
        ])
        
        # Initialize other necessary components
        self.norm = MyLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MyLLMRotaryEmbedding(
            config.attention_head_dim, config.max_position_embeddings, config.rope_theta
        )
        
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
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        position_ids: torch.LongTensor = None, 
        attention_mask: torch.Tensor = None, 
        # Output arguments
        use_cache: bool = False,                    
        output_attentions: bool = False, 
        output_hidden_states: bool = False, 
        return_dict: bool = False, 
        # Cache arguments
        past_key_values: Cache = None,              
        cache_position: torch.LongTensor = None,    
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if self.gradient_checkpointing and use_cache and self.training:
            raise ValueError("Gradient checkpointing is not compatible with use_cache=True during training.")
        
        # Get the embeddings
        if inputs_embeds is None:
            hidden_states: torch.Tensor = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Record the shape of the hidden states
        seq_len = hidden_states.size(1)
        
        # Container for all the hidden states and attentions
        all_hidden_states = ()
        all_attentions = () if output_attentions else None
        
        # Create cache if not provided and cache is enabled
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            
        # Update the cache position
        if cache_position is None:
            past_kv_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_kv_length, past_kv_length + hidden_states.size(1), device=hidden_states.device
            )
            
        if position_ids is None: 
            position_ids = cache_position.unsqueeze(0)
            
        # Update the causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, 
            input_tensor=hidden_states, 
            cache_position=cache_position, 
            past_key_values=past_key_values, 
            output_attentions=output_attentions, 
        )
        
        # Create a ones-like states with shape of (1, 1, seq_len, attention_head_dim)
        position_embeddings = torch.ones(
            1, 1, seq_len, self.config.attention_head_dim, device=hidden_states.device
        )
        
        # Apply the position embeddings
        # Output shape: (bsz, seq_len, num_attention_heads, attention_head_dim)
        position_embeddings = self.rotary_emb(position_embeddings, position_ids).to(hidden_states.dtype)
        
        # Forward pass through the model layers
        for layer in self.layers:
            # Check if the gradient checkpointing is enabled
            if self.gradient_checkpointing:
                layer_outputs = checkpoint(
                    layer.__call__,     # If the `forward` method is passed, hooks will not be called
                    hidden_states, 
                    causal_mask, 
                    position_embeddings, 
                    output_attentions, 
                    past_key_values, 
                )
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
        if self.config._attn_implementation == "flash_attention_2":
            raise NotImplementedError("Flash Attention 2 is not implemented yet.")
        
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)   # Static Cache and JIT Not Supported Yet
        
        # If non-padding inputs are provided, we can ignore the causal mask for the sdpa implementation
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if not using_static_cache: 
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask, 
                    inputs_embeds=input_tensor, 
                    past_key_values_length=past_key_values_length, 
                    is_training=self.training, 
                ):
                    return None
        
        dtype = input_tensor.dtype
        device = input_tensor.device
        seq_len = input_tensor.size(1)
        
        if using_static_cache:
            # Static cache is not supported yet
            raise NotImplementedError("Static cache is not supported yet.")
        
        # Create the causal mask
        causal_mask = self._prepare_4d_causal_mask_with_cache_position(
            seq_len, 
            key_len=past_key_values_length + seq_len, 
            batch_size=input_tensor.size(0), 
            dtype=dtype, 
            device=device, 
            cache_position=cache_position, 
            attention_mask=attention_mask, 
        )
        return causal_mask
        
    @staticmethod
    def _prepare_4d_causal_mask_with_cache_position(
        query_len: int, 
        key_len: int, 
        batch_size: int, 
        dtype: torch.dtype, 
        device: torch.device, 
        cache_position: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
    ) -> torch.Tensor:
        """Create a 4D causal mask for the attention mechanism. 
        
        Args:
            query_len (`int`):
                The length of the query tensor.
            key_len (`int`): 
                The length of the key tensor. 
            batch_size (`int`): 
                The batch size of the input tensor.
            dtype (`torch.dtype`):
                The data type of the input tensor.
            device (`torch.device`):
                The device of the input tensor. 
            cache_position (`torch.Tensor`): 
                The cache position tensor with shape (seq_len,)
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask with shape (bsz, seq_len)
                
        Returns:
            `torch.Tensor`: 
                The 4D causal mask with shape (bsz, 1, query_len, key_len)
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        
        dtype_min = torch.finfo(dtype).min
        
        # Create a triangular mask with min values in the upper triangular part
        # Output shape: (1, 1, query_len, key_len)
        causal_mask = torch.full((query_len, key_len), dtype_min, dtype=dtype, device=device)
        if query_len == key_len:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        elif query_len > 1:
            # If a new long query is added, and query length is greater than key length, the torch.triu will not work properly
            causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask.view(1, 1, query_len, key_len).to(dtype)
        causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
        
        if attention_mask is not None:
            # Convert the dims specified by the input attention_mask to -inf
            attention_mask = attention_mask[:, None, None, :]
            # Repeat the attention mask
            attention_mask = attention_mask.repeat(1, 1, query_len, 1)
            # Convert the causal mask to min where the value of attention mask is -1
            causal_mask = causal_mask.masked_fill(attention_mask == 0, dtype_min)
            
            if query_len == key_len:
                # Convert the causal mask to 0 if all of the values are min in a row
                attention_mask = attention_mask.transpose(-1, -2)
                causal_mask = causal_mask * attention_mask
            
        return causal_mask
    
    
class MyLLMForCausalLM(MyLLMPreTrainedModel, GenerationMixin):
    """MyLLMForCausalLM is a causal language model that predicts the next token in a sequence. """
    
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
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        position_ids: torch.LongTensor = None, 
        attention_mask: torch.Tensor = None, 
        # Training arguments
        labels: torch.LongTensor = None, 
        # Output arguments
        use_cache: bool = None,                     
        output_attentions: bool = None, 
        output_hidden_states: bool = None, 
        return_dict: bool = None, 
        num_logits_to_keep: int = 0, 
        # Cache arguments
        past_key_values: Cache = None,              
        cache_position: torch.LongTensor = None,    
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
        # logits = logits[attention_mask.view(-1)]
        # labels = labels.view(-1)[attention_mask.view(-1)]
        
        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))
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
        