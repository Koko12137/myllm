import torch
import torch.nn as nn


def check_nan(module: nn.Module, grad_input: nn.Module, grad_output: nn.Module, layer_idx: int, name: str) -> None:
    if grad_output[0] is not None:
        output_nan = torch.isnan(grad_output[0]).sum().item()
        assert output_nan == 0, f"NaN detected in layer {layer_idx}, name: {name}"
    if grad_input[0] is not None:
        input_nan = torch.isnan(grad_input[0]).sum().item()
        assert input_nan == 0, f"NaN detected in layer {layer_idx}, name: {name}"
        