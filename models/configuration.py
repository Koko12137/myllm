from enum import Enum

from transformers import PretrainedConfig


class ValidMoE(Enum):
    FFN = "ffn"
    FFN_SHARE = 'ffn_share'
    
    
class MyLLMConfig(PretrainedConfig):
    r"""The configuration class for MyLLM with Group Query Attention."""
    model_type = "myllm"

    def __init__(
        self, 
        vocab_size: int = 151665, 
        ignore_index: int = 151643, 
        num_hidden_layers: int = 16, 
        # MLP Arguments
        hidden_size: int = 768, 
        intermediate_size: int = 3840, 
        rms_norm_eps: float = 0.000001, 
        dropout: float = 0.1, 
        # Attention Arguments
        num_attention_heads: int = 16, 
        num_key_value_heads: int = 8, 
        attention_head_dim: int = 128, 
        attn_implementation: str = "sdpa", 
        # Positional Embeddings Arguments
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0, 
        # Generation Arguments
        use_cache: bool = True, 
        max_length: int = 512, 
        # Mixture of Experts
        use_moe: bool = True, 
        moe_type: str = "ffn", 
        num_experts: int = 32, 
        topk_experts: int = 2, 
        num_share_experts: int = 2, 
        moe_dropout: float = 0.1, 
        expert_temperature: float = 1.0, 
        expert_sample: bool = False, 
        gate_random_alpha: float = 0.0, 
        balance_penalty: float = 0.0001, 
        # Chain of Experts
        use_coe: bool = True, 
        num_chains: int = 2, 
        residual: bool = True, 
        **kwargs,
    ) -> None:
        r"""Initialize MyLLMConfig.
        
        Args: 
            vocab_size (`int`, *optional*, defaults to 151665): 
                The size of the vocabulary.
            ignore_index (`int`, *optional*, defaults to 151643): 
                The ignore index for loss calculation.
            num_hidden_layers (`int`, *optional*, defaults to 16):
                The number of hidden layers in the model.
                
            hidden_size (`int`, *optional*, defaults to 768): 
                The hidden size of the model.
            intermediate_size (`int`, *optional*, defaults to 3072): 
                The intermediate size of the model. 
            rms_norm_eps (`float`, *optional*, defaults to 0.000001):
                The epsilon value for RMSNorm.
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout rate.
                
            num_attention_heads (`int`, *optional*, defaults to 16): 
                The number of attention heads in the model. 
            num_key_value_heads (`int`, *optional*, defaults to 8):
                The number of key value heads in the model.
            attention_head_dim (`int`, *optional*, defaults to 128):
                The dimension of the attention heads.
            attn_implementation (`str`, *optional*, defaults to "sdpa"):
                The implementation of attention. 
                
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum position embeddings.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The theta value for ROPE. 
                
            use_cache (`bool`, *optional*, defaults to True):
                Whether to use cache for generation. 
            
            use_moe (`bool`, *optional*, defaults to True): 
                Whether to use Mixture of Experts. 
            moe_type (`str`, *optional*, defaults to "ffn"):
                The type of Mixture of Experts. 
            num_experts (`int`, *optional*, defaults to 32): 
                The number of experts in the model.
            topk_experts (`int`, *optional*, defaults to 2):
                The number of top k experts to select. 
            num_share_experts (`int`, *optional*, defaults to 1):
                The number of share experts, if 0 is set, then all experts are unique. 
            moe_dropout (`float`, *optional*, defaults to 0.1): 
                The dropout rate for Mixture of Experts. 
            expert_temperature (`float`, *optional*, defaults to 1.0): 
                The temperature for the scaling the router probability. 
            expert_sample (`bool`, *optional*, defaults to False): 
                Whether to sample the expert. If False, greedy selection is used.
            gate_random_alpha (`float`, *optional*, defaults to 0.0)
                Router of MoE will apply random noise for gate logits before top-k, and this is using for 
                scaling the random noise.
            balance_penalty
                
            use_coe (`bool`, *optional*, defaults to True): 
                Whether to use Chain of Experts. 
            num_chains (`int`, *optional*, defaults to 2): 
                The number of chains for CoE. 
            residual (`bool`, *optional*, defaults to True): 
                Whether to use residual connection in CoE. 
        """
        super().__init__(
            attn_implementation=attn_implementation, **kwargs
        )
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ignore_index = ignore_index
        
        # MLP Arguments
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        
        # Attention Arguments
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_dim = attention_head_dim
        
        # Positional Embeddings Arguments
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        # Generation Arguments
        self.use_cache = use_cache
        self.max_length = max_length
        
        # Mixture of Experts
        self.use_moe = use_moe
        self.moe_type = ValidMoE(moe_type).value
        if use_moe: 
            self.model_type = f"{self.model_type}_{moe_type}_moe"
        self.num_experts = num_experts
        self.topk_experts = topk_experts
        self.num_share_experts = num_share_experts
        self.moe_dropout = moe_dropout
        self.expert_temperature = expert_temperature
        self.expert_sample = expert_sample
        self.gate_random_alpha = gate_random_alpha
        self.balance_penalty = balance_penalty
            
        # Chain of Experts
        self.use_coe = use_coe
        if use_coe:
            self.model_type = f"{self.model_type}_coe"
        self.num_chains = num_chains
        self.residual = residual
