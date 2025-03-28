import torch.nn as nn


def model_inspect(model: nn.Module) -> None:
    # Print model structure
    print(model)
    
    # Get the total parameters
    p_counts = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"Trainable parameters: {p_counts / 1e6:.2f}M")
    
    # Get the total parameters of head and embeddings
    p_head_counts = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
    p_emb_counts = sum(p.numel() for p in model.model.embed_tokens.parameters() if p.requires_grad)
    print(f"Head Trainable parameters: {p_head_counts / 1e6:.2f}M")
    print(f"Embedding Trainable parameters: {p_emb_counts / 1e6:.2f}M")
    
    # Get the total parameters of core
    p_core_counts = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Core Trainable parameters: {p_core_counts / 1e6:.2f}M")
    print(f"Core Trainable parameters without embeddings and head: {(p_core_counts - p_emb_counts - p_head_counts) / 1e6:.2f}M")
    
    # Check if the model is moe
    # if model.config.use_moe:
    #     # Get the active parameters
    #     pass    # TODO: extract the experts 