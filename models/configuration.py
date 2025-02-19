from transformers import Qwen2Config


class MyLLMConfig(Qwen2Config):
    r"""LMConfig is the configuration class for MyLLM with Group Query Attention."""
    model_type = "myllm"

    def __init__(
        self, 
        vocab_size: int = 20000, 
        hidden_size: int = 1024, 
        intermediate_size: int = 4096, 
        num_hidden_layers: int = 16, 
        num_attention_heads: int = 16, 
        num_key_value_heads: int = 8, 
        attention_head_dim: int = 128, 
        max_position_embeddings: int = 512,
        rms_norm_eps: float = 0.000001, 
        rope_theta: float = 10000.0, 
        padding_idx: int = 0, 
        dropout: float = 0.1, 
        attn_implementation: str = "sdpa", 
        **kwargs,
    ) -> None:
        """Initialize MyLLMConfig.
        
        Args: 
            vocab_size (`int`, *optional*, defaults to 20000): 
                The size of the vocabulary.
            hidden_size (`int`, *optional*, defaults to 1024): 
                The hidden size of the model.
            intermediate_size (`int`, *optional*, defaults to 4096): 
                The intermediate size of the model. 
            num_hidden_layers (`int`, *optional*, defaults to 16):
                The number of hidden layers in the model.
            num_attention_heads (`int`, *optional*, defaults to 16): 
                The number of attention heads in the model. 
            num_key_value_heads (`int`, *optional*, defaults to 8):
                The number of key value heads in the model.
            attention_head_dim (`int`, *optional*, defaults to 128):
                The dimension of the attention heads.
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum position embeddings.
            rms_norm_eps (`float`, *optional*, defaults to 0.000001):
                The epsilon value for RMSNorm.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The theta value for ROPE.
            padding_idx (`int`, *optional*, defaults to 0):
                The padding index.
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout rate.
            attn_implementation (`str`, *optional*, defaults to "sdpa"):
                The implementation of attention. 
        """
        super().__init__(
            vocab_size=vocab_size, 
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            num_hidden_layers=num_hidden_layers, 
            num_attention_heads=num_attention_heads, 
            num_key_value_heads=num_key_value_heads, 
            attention_head_dim=attention_head_dim,
            max_position_embeddings=max_position_embeddings, 
            rms_norm_eps=rms_norm_eps, 
            use_cache=False, 
            rope_theta=rope_theta, 
            attn_implementation=attn_implementation, 
        )
        self.padding_idx = padding_idx 
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
