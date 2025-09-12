# src/models/lora.py
from typing import Iterable, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in wrapper for nn.Linear with LoRA (A@B) added to the frozen base weight.
    deltaW = B @ A, where A: [r, in], B: [out, r], scale = alpha / r
    """

    def __init__(
        self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = (self.alpha / self.r) if self.r > 0 else 0.0
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # match device & dtype of the frozen base weight
        dev = self.base.weight.device
        dt = self.base.weight.dtype

        if self.r > 0:
            # A: [r, in], B: [out, r]
            self.A = nn.Parameter(
                torch.zeros(self.r, self.base.in_features, device=dev, dtype=dt)
            )
            self.B = nn.Parameter(
                torch.zeros(self.base.out_features, self.r, device=dev, dtype=dt)
            )
            # init per LoRA paper: A ~ N(0, 0.02), B zero
            nn.init.normal_(self.A, std=0.02)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = F.linear(x, self.base.weight, self.base.bias)
        if self.r > 0:
            # (B @ A) has shape [out, in]
            delta = torch.matmul(self.B, self.A) * self.scale
            out = out + F.linear(self.drop(x), delta, None)
        return out

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        return [] if self.r == 0 else [self.A, self.B]


def _resolve_parent(root: nn.Module, path: str):
    """
    Given 'encoder.layer.0.attn.q_proj', return (parent_module, leaf_attr_name).
    """
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_keys: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> List[nn.Parameter]:
    """
    Replace nn.Linear layers whose qualified name contains any of `target_keys`
    with LoRALinear(base=that_linear, r, alpha, dropout). Returns new trainable
    LoRA parameters.
    """
    lora_params: List[nn.Parameter] = []

    # Collect candidate modules first to avoid changing the module tree while iterating.
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(k in name for k in target_keys):
            to_replace.append((name, mod))

    for fqname, lin in to_replace:
        parent, leaf = _resolve_parent(model, fqname)
        wrapped = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, wrapped)
        lora_params.extend(list(wrapped.lora_parameters))

    return lora_params
