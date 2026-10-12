"""Loop classifier heads on top of a pair of image embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn

from .swin import EMBED_DIM


class LoopHead(nn.Module):
    """MLP that scores whether two embedded images overlap (form a loop).

    Two variants were trained:

    * ``mlp2``: 1536 -> 512 -> 1, dropout 0.2, sigmoid output, BCE loss
    * ``mlp3``: 1536 -> 512 -> 256 -> 1, dropout 0.3, logit output, BCE-with-logits loss

    ``forward`` always returns a loop probability in [0, 1].
    """

    VARIANTS = ("mlp2", "mlp3")

    def __init__(self, variant: str = "mlp3", embed_dim: int = EMBED_DIM):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        d = 2 * embed_dim
        if variant == "mlp2":
            self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1))
        else:
            self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def logits(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ea, eb], dim=1)).squeeze(1)

    def forward(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(ea, eb))


def load_head(path: str, variant: str) -> LoopHead:
    """Load weights saved by the original training scripts (``classifier.*`` or ``fc.*`` keys)."""
    head = LoopHead(variant)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.split(".", 1)[1] if k.split(".", 1)[0] in ("classifier", "fc", "net") else k: v for k, v in sd.items()}
    head.net.load_state_dict(sd)
    return head.eval()
