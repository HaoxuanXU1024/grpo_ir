"""
LoRA (Low-Rank Adaptation) implementation for AdaIR model fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer implementation."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Get dimensions from original layer
        if isinstance(original_layer, nn.Linear):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif isinstance(original_layer, nn.Conv2d):
            in_features = original_layer.in_channels
            out_features = original_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type: {type(original_layer)}")
        
        # LoRA decomposition: W = W0 + BA, where B is (out_features, rank), A is (rank, in_features)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        original_out = self.original_layer(x)
        
        # LoRA forward pass
        if isinstance(self.original_layer, nn.Linear):
            lora_out = F.linear(x, (self.lora_B @ self.lora_A) * self.scaling)
        elif isinstance(self.original_layer, nn.Conv2d):
            # For Conv2d, reshape LoRA weights to match conv kernel format
            kernel = (self.lora_B @ self.lora_A).view(
                self.original_layer.out_channels,
                self.original_layer.in_channels,
                1, 1
            ) * self.scaling
            lora_out = F.conv2d(
                x, kernel,
                bias=None,
                stride=self.original_layer.stride,
                padding=self.original_layer.padding,
                dilation=self.original_layer.dilation,
                groups=self.original_layer.groups
            )
        
        return original_out + self.dropout(lora_out)


def apply_lora_to_adair(
    model: nn.Module,
    targets: Union[str, List[str]],
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0
) -> int:
    """
    Apply LoRA adapters to specified layers in AdaIR model.
    
    Args:
        model: The AdaIR model to apply LoRA to
        targets: Target layer names (e.g., 'attn', 'cross_attn', 'mlp')
        rank: LoRA rank (low-rank decomposition parameter)
        alpha: LoRA alpha (scaling parameter)
        dropout: Dropout rate for LoRA layers
    
    Returns:
        Number of layers that LoRA was applied to
    """
    if isinstance(targets, str):
        targets = [targets]
    
    applied_count = 0
    
    def apply_lora_recursive(module: nn.Module, module_name: str = ""):
        nonlocal applied_count
        
        for name, child in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Check if this layer should have LoRA applied
            should_apply = False
            for target in targets:
                if target in name.lower() or target in full_name.lower():
                    should_apply = True
                    break
            
            if should_apply and isinstance(child, (nn.Linear, nn.Conv2d)):
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                applied_count += 1
                print(f"[LoRA] Applied to {full_name} ({type(child).__name__})")
            else:
                # Recursively apply to children
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    
    print(f"[LoRA] Total applied to {applied_count} layers")
    return applied_count


def enable_lora_training(model: nn.Module):
    """Enable training for LoRA parameters only."""
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_parameters(model: nn.Module):
    """Get all LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model: nn.Module):
    """Merge LoRA weights into original weights (for inference)."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            # Merge LoRA weights into original layer
            with torch.no_grad():
                if isinstance(module.original_layer, nn.Linear):
                    delta_weight = (module.lora_B @ module.lora_A) * module.scaling
                    module.original_layer.weight.add_(delta_weight)
                elif isinstance(module.original_layer, nn.Conv2d):
                    delta_weight = (module.lora_B @ module.lora_A).view(
                        module.original_layer.out_channels,
                        module.original_layer.in_channels,
                        1, 1
                    ) * module.scaling
                    module.original_layer.weight.add_(delta_weight)
