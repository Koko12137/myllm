from transformers import PretrainedConfig


class MyLLMConfig(PretrainedConfig):
    r"""The configuration class for MyLLM with Group Query Attention."""
    model_type = "myllm"

    def __init__(
        self, 
        vocab_size: int = 20000, 
        rms_norm_eps: float = 0.000001, 
        dropout: float = 0.1, 
        hidden_size: int = 1024, 
        intermediate_size: int = 2048, 
        num_hidden_layers: int = 16, 
        num_attention_heads: int = 16, 
        num_key_value_heads: int = 8, 
        attention_head_dim: int = 128, 
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0, 
        attn_implementation: str = "sdpa", 
        **kwargs,
    ) -> None:
        """Initialize MyLLMConfig.
        
        Args: 
            vocab_size (`int`, *optional*, defaults to 20000): 
                The size of the vocabulary.
            rms_norm_eps (`float`, *optional*, defaults to 0.000001):
                The epsilon value for RMSNorm.
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout rate.
            hidden_size (`int`, *optional*, defaults to 1024): 
                The hidden size of the model.
            intermediate_size (`int`, *optional*, defaults to 2048): 
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
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The theta value for ROPE. 
            attn_implementation (`str`, *optional*, defaults to "sdpa"):
                The implementation of attention. 
        """
        super().__init__(
            attn_implementation=attn_implementation, 
        )
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        # MLP Arguments
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Attention Arguments
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_dim = attention_head_dim
        # Positional Embeddings Arguments
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        # Other Arguments
        self.use_cache = kwargs.get('use_cache', False)
        self.use_moe = kwargs.get('use_moe', False)
        
        # Check if the model is using Mixture of Experts
        if self.use_moe:
            raise ValueError("Mixture of Experts is not supported in this configuration. Please use MyLLMConfigForMoE instead.")


class MyLLMConfigForMoE(MyLLMConfig):
    r"""The configuration class for MyLLM with Mixture of Experts."""
    model_type = "myllm_moe"

    def __init__(
        self, 
        num_experts: int = 16, 
        topk_experts: int = 4, 
        **kwargs,
    ) -> None:
        """Initialize MyLLMConfigForMoE.
        
        Args: 
            num_experts (`int`, *optional*, defaults to 16): 
                The number of experts in the model.
            topk_experts (`int`, *optional*, defaults to 4):
                The number of top k experts to select.
        """
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.topk_experts = topk_experts
        self.use_moe = True
        self.moe_type = kwargs.get('moe_type', 'ffn')
        