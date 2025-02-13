import torch
import torch.nn as nn

from transformers import Cache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast, 
)

from models.configuration import MyLLMConfig


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
    
    def __init__(self, dim: int, eps: float = 1e-5) -> None: 
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the root mean square
        x_rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize the input tensor
        x = x / x_rms
        # Scale the tensor
        x = x * self.gamma
        return x
    
    
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
    
    def __init__(self, config: MyLLMConfig) -> None: 
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.swiglu_up = MyLLMSwiGLU(config.intermediate_size)
        self.swiglu_down = MyLLMSwiGLU(config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up projection
        x = self.up(x)
        x = self.swiglu_up(x)
        
        # Down projection
        x = self.down(x)
        x = self.swiglu_down(x)
        return x


class MyLLMGroupAttention(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.q = nn.Linear(config.hidden_size, config.attention_head_dim * config.num_attention_heads)
        self.k = nn.Linear(config.hidden_size, config.attention_head_dim * config.num_kv_heads)
        self.v = nn.Linear(config.hidden_size, config.attention_head_dim * config.num_kv_heads)
        self.o = nn.Linear(config.attention_head_dim * config.num_attention_heads, config.hidden_size)
        
    def apply_attn(
        self, 
        q_states: torch.Tensor, 
        k_states: torch.Tensor, 
        v_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
    ) -> tuple[torch.Tensor]:
        # Record the shape of the query states
        bsz, n, seq_len, dim = k_states.shape
        
        # repeat the key and value states
        repeats = self.config.num_attention_heads // self.config.num_kv_heads
        k_states = k_states.view(bsz, n, 1, seq_len, dim)
        v_states = v_states.view(bsz, n, 1, seq_len, dim)
        k_states = k_states.repeat(1, 1, repeats, 1, 1) # Output shape: (bsz, n, repeats, seq_len, dim)
        v_states = v_states.repeat(1, 1, repeats, 1, 1) # Output shape: (bsz, n, repeats, seq_len, dim)
        # View the key and value states as (bsz, num_attention_heads, seq_len, attention_head_dim)
        k_states = k_states.view(bsz, self.config.num_attention_heads, seq_len, dim).contiguous()
        v_states = v_states.view(bsz, self.config.num_attention_heads, seq_len, dim).contiguous()
        
        # Compute the attention weights
        # Output shape: (bsz, num_attention_heads, seq_len, seq_len)
        attn_weights = torch.einsum("bhid,bhjd->bhij", q_states, k_states) / self.config.attention_head_dim
        
        # Create a triangular mask with -inf values in the upper triangular part
        # Output shape: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -torch.inf, device=q_states.device), diagonal=1
        ).view(1, 1, seq_len, seq_len)
        # Convert the dims specified by the input attention_mask to -inf
        attention_mask = attention_mask.view(bsz, 1, 1, seq_len).to(torch.float64)
        # Reapeat the attention mask
        attention_mask = attention_mask.repeat(1, 1, seq_len, 1) - 1
        # Convert the attention_mask to -inf where value is 0
        attention_mask[attention_mask == -1] = -torch.inf
        # Add the causal mask
        attn_weights = attn_weights + causal_mask + attention_mask
        
        # Safe softmax
        attn_weights = attn_weights.exp()
        attn_weights_max = attn_weights.max(dim=-1, keepdim=True).values
        attn_weights = attn_weights - attn_weights_max
        # Mask the attention weights
        attn_weights = attn_weights.masked_fill(attention_mask == -torch.inf, 0.0)
        attn_weights_sum = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights / (attn_weights_sum + 1e-6)
        attn_weights = attn_weights.type_as(v_states)
        
        # Apply attention weights to the value states
        # Output shape: (bsz, num_attention_heads, seq_len, attention_head_dim)
        attn_values = torch.einsum("bhis,bhsj->bhij", attn_weights, v_states)
        
        return attn_values, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_embeddings: torch.Tensor, 
        output_attentions: bool = False, 
        past_key_value: Cache = None, 
    ) -> tuple[torch.Tensor]:
        # Record the shape of the hidden states
        bsz, seq_len, _ = hidden_states.shape # Input shape: (bsz, seq_len, hidden_size)
        
        # Compute the query, key, and value
        # Output shape: (bsz, seq_len, num_attention_heads, attention_head_dim)
        q_states: torch.Tensor = self.q(hidden_states).view(bsz, seq_len, self.config.num_attention_heads, -1)
        # Output shape: (bsz, seq_len, num_kv_heads, attention_head_dim)
        k_states: torch.Tensor = self.k(hidden_states).view(bsz, seq_len, self.config.num_kv_heads, -1)
        # Output shape: (bsz, seq_len, num_kv_heads, attention_head_dim)
        v_states: torch.Tensor = self.v(hidden_states).view(bsz, seq_len, self.config.num_kv_heads, -1)
        
        # Transpose the query, key and value states to (bsz, num_attention_heads, seq_len, attention_head_dim)
        q_states = q_states.transpose(1, 2).contiguous()
        k_states = k_states.transpose(1, 2).contiguous()
        v_states = v_states.transpose(1, 2).contiguous()
        
        # Apply the position embeddings
        q_states = q_states * position_embeddings
        k_states = k_states * position_embeddings
        
        # If Cache is provided, update the key and value states
        if past_key_value is not None:
            # TODO: Learn how to implement this
            raise NotImplementedError("Cache is not implemented yet.")
        
        # Apply the attention
        attn_out, attn_weights = self.apply_attn(
            q_states, 
            k_states, 
            v_states, 
            attention_mask, 
        )
        
        # Transpose the attention output to (bsz, seq_len, num_attention_heads * attention_head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        # Apply the output linear layer
        attn_out = self.o(attn_out)
        
        out = (attn_out,)
        if output_attentions:
            out += (attn_weights,)
        return out
        

class MyLLMDecoderLayer(nn.Module):
    
    def __init__(self, config: MyLLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attn = MyLLMGroupAttention(config, layer_idx=layer_idx)
        self.ffn = MyLLMForward(config)
        self.input_rms_norm = MyLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_rms_norm = MyLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_embeddings: torch.Tensor, 
        output_attentions: bool = False, 
        past_key_value: Cache = None, 
    ) -> tuple[torch.Tensor]:
        # Record the hidden states
        residual = hidden_states
        
        # Normalize the input
        hidden_states = self.input_rms_norm(hidden_states)
        
        # Self-Attention
        out = self.attn(
            hidden_states, 
            attention_mask=attention_mask, 
            position_embeddings=position_embeddings, 
            output_attentions=output_attentions, 
            past_key_value=past_key_value, 
        )
        if output_attentions:
            hidden_states, attn_weights = out
        else:
            hidden_states = out[0]
            attn_weights = None
        
        # Add the residual
        hidden_states = hidden_states + residual
        
        # Record the hidden states
        residual = hidden_states
        
        # Normalize the output
        hidden_states = self.output_rms_norm(hidden_states)
        
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
    _supports_flash_attn_2 = False
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)
        
    
class MyLLMModel(MyLLMPreTrainedModel):
    
    def __init__(self, config: MyLLMConfig) -> None:
        super().__init__(config)
        self.config = config
        
        # Initialize embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_idx) 
        
        # Initialize the model components
        self.layers = nn.ModuleList([
            MyLLMDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ])
        
        # Initialize other necessary components
        self.norm = MyLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MyLLMRotaryEmbedding(config.attention_head_dim, config.max_position_embeddings, config.rope_theta)
        self.gradient_checkpointing = None  # TODO: Implement gradient checkpointing
        
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
        # Get the embeddings
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds should be provided.")
        
        # Container for all the hidden states and attentions
        all_hidden_states = ()
        all_attentions = () if output_attentions else None
        
        # Record the shape of the hidden states
        bsz, seq_len, _ = hidden_states.shape
        
        # Compute the position embeddings
        # If position cache is provided, update the position embeddings
        if cache_position:
            # TODO: Learn how to implement this
            raise NotImplementedError("Cache is not implemented yet.")
        else:
            # Create a ones-like states with shape of (1, 1, seq_len, attention_head_dim)
            ones = torch.ones(1, 1, seq_len, self.config.attention_head_dim, device=hidden_states.device)
            
            # Apply the position embeddings
            # Output shape: (bsz, seq_len, num_attention_heads, attention_head_dim)
            ones = self.rotary_emb(ones, position_ids)
        
        # Forward pass through the model layers
        for idx, layer in enumerate(self.layers):
            # Check if the gradient checkpointing is enabled
            if self.gradient_checkpointing:
                raise NotImplementedError("Gradient checkpointing is not implemented yet.")
            else:
                layer_outputs = layer(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    position_embeddings=ones, 
                    output_attentions=output_attentions, 
                    past_key_value=past_key_values, 
                )
            
            # Update the hidden states
            hidden_states = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Update the attentions
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        # Normalize the hidden states
        hidden_states = self.norm(hidden_states)
        
        # Add the hidden states to the hidden states list after the last layer
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
    
    
class MyLLMForCausalLM(MyLLMPreTrainedModel):
    
    def __init__(self, config: MyLLMConfig) -> None:
        super().__init__(config)
        self.config = config
        
        # Initalize the causal model
        self.model = MyLLMModel(config)
        # Initialize the output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
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
        logits = self.lm_head(hidden_states)    # Output shape: (bsz, seq_len, vocab_size)
        
        # Compute loss if labels are provided
        if labels is not None:
            # num_labels shape: (bsz, seq_len)
            num_labels = labels.shape[-1]
            # keep_logits shape: (bsz * seq_len, vocab_size)
            keep_logits = logits[:, -num_labels:].view(-1, logits.size(-1))
            
            loss = self.loss_fn(keep_logits, labels.view(-1))
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
        