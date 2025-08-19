import torch
import torch.nn as nn


class LoRAConv1x1(nn.Module):
    """LoRA adapter for Conv2d with kernel_size=1.

    y = base_conv(x) + scale * up(down(x))
    where down: in_c -> r, up: r -> out_c, both Conv2d(k=1)
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d), "LoRAConv1x1 expects nn.Conv2d"
        assert tuple(base_conv.kernel_size) == (1, 1), "LoRAConv1x1 only supports 1x1 conv"

        self.base = base_conv
        in_c = base_conv.in_channels
        out_c = base_conv.out_channels
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / float(self.rank)

        self.down = nn.Conv2d(in_c, self.rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(self.rank, out_c, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base conv by default; caller can override
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = self.up(self.dropout(self.down(x))) * self.scaling
        return y + z


def _maybe_wrap_attr(module: nn.Module, attr_name: str, rank: int, alpha: float, dropout: float) -> int:
    if not hasattr(module, attr_name):
        return 0
    layer = getattr(module, attr_name)
    if isinstance(layer, nn.Conv2d) and tuple(layer.kernel_size) == (1, 1):
        wrapped = LoRAConv1x1(layer, rank=rank, alpha=alpha, dropout=dropout)
        setattr(module, attr_name, wrapped)
        return 1
    return 0


def apply_lora_to_adair(net: nn.Module, targets: str = "attn,cross_attn", rank: int = 4, alpha: float = 4.0, dropout: float = 0.0) -> int:
    """Inject LoRA into AdaIR attention 1x1 convs.

    - targets: comma-separated in {"attn", "cross_attn", "ffn"}
    - returns: number of layers wrapped
    """
    target_set = set([t.strip() for t in targets.split(',') if t.strip()])
    wrapped = 0

    for m in net.modules():
        cls = m.__class__.__name__
        if "Attention" == cls and ("attn" in target_set):
            # net.model.Attention: qkv, project_out are 1x1 convs
            wrapped += _maybe_wrap_attr(m, 'qkv', rank, alpha, dropout)
            wrapped += _maybe_wrap_attr(m, 'project_out', rank, alpha, dropout)
        elif "Chanel_Cross_Attention" == cls and ("cross_attn" in target_set):
            wrapped += _maybe_wrap_attr(m, 'q', rank, alpha, dropout)
            wrapped += _maybe_wrap_attr(m, 'kv', rank, alpha, dropout)
            wrapped += _maybe_wrap_attr(m, 'project_out', rank, alpha, dropout)
        elif "FeedForward" == cls and ("ffn" in target_set):
            wrapped += _maybe_wrap_attr(m, 'project_in', rank, alpha, dropout)
            wrapped += _maybe_wrap_attr(m, 'project_out', rank, alpha, dropout)

    return wrapped


