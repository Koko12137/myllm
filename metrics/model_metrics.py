import torch
import torch.nn as nn


def perplexity(y_true: torch.LongTensor, y_pred: torch.FloatTensor, ignore_index: int = 151643) -> float:
    """Calculate the perplexity of the model.
    Reference: https://en.wikipedia.org/wiki/Perplexity

    Args:
        y_true (`torch.LongTensor`): 
            The true labels.
        y_pred (`torch.FloatTensor`): 
            The predicted logits.

    Returns:
        `float`: 
            The perplexity of the model.
    """
    y_true = torch.tensor(y_true.reshape(-1))
    y_pred = torch.tensor(y_pred.reshape(-1, y_pred.shape[-1]))
    # Get the number of tokens
    n_tokens = y_true.shape[0]
    
    # Get the cross entropy
    cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')(y_pred, y_true)
    
    # Calculate the perplexity
    perplexity = torch.exp(cross_entropy / n_tokens)
    return perplexity.item()
