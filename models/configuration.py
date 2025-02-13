from transformers import PretrainedConfig


class MyLLMConfig(PretrainedConfig):
    r"""LMConfig is the configuration class for MyLLM with Group Query Attention."""
    model_type = "myllm"

    def __init__(
        self,
        vocab_size: int = 20000, 
        padding_idx: int = 0, 
        hidden_size: int = 768, 
        intermediate_size: int = 3072, 
        output_size: int = 768, 
        num_hidden_layers: int = 24, 
        num_attention_heads: int = 16, 
        attention_head_dim: int = 128, 
        num_kv_heads: int = 8, 
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0, 
        max_position_embeddings: int = 512,
        dropout: float = 0.1, 
        **kwargs,
    ) -> None:
        r"""
        Initializes the configuration class for MyLLM with Group Query Attention.
        
        Args:
            vocab_size (`int`, *optional*, defaults to 200000): 
                The size of the vocabulary. 
            padding_idx (`int`, *optional*, defaults to 0): 
                The padding index. 
            hidden_size (`int`, *optional*, defaults to 768):   
                The size of the hidden layer.
            intermediate_size (`int`, *optional*, defaults to 3072): 
                The size of intermediate layer in MLP. 
            output_size (`int`, *optional*, defaults to 768):
                The size of the output layer. 
            num_hidden_layers (`int`, *optional*, defaults to 24):
                The number of hidden layers.
            num_attention_heads (`int`, *optional*, defaults to 16):
                The number of attention heads. 
            attention_head_dim (`int`, *optional*, defaults to 128):
                The dimension of the attention head. 
            num_kv_heads (`int`, *optional*, defaults to 8):
                The number of key-value heads. 
            rms_norm_eps (`float`, *optional*, defaults to 1e-5):
                The epsilon value for RMS normalization.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The theta value for ROPE.
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum length of the position embeddings. 
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout rate. 
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx 
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size 
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_kv_heads = num_kv_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
