"""Reader for UCAMGEN synthetic underwater camera datasets.

A dataset folder holds ``IMAGES/IMAGE00001.png ...`` (a bottom-looking camera
moving over a large "world" image) and ``OVERLAP.csv``, an N x N matrix of
pairwise image overlap in percent. Following the original training data, a
pair of images is a *loop* when they overlap at all (overlap > 0 %).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class UcamgenDataset:
    root: Path
    overlap: np.ndarray            # (N, N) int, percent overlap

    @classmethod
    def load(cls, root) -> UcamgenDataset:
        root = Path(root)
        overlap = np.loadtxt(root / "OVERLAP.csv", delimiter=",", dtype=np.int16)
        if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
            raise ValueError(f"OVERLAP.csv must be square, got {overlap.shape}")
        return cls(root, overlap)

    def __len__(self) -> int:
        return self.overlap.shape[0]

    def image_path(self, i: int) -> Path:
        return self.root / "IMAGES" / f"IMAGE{i + 1:05d}.png"

    def load_image(self, i: int) -> np.ndarray:
        """RGB uint8 image (H, W, 3)."""
        from PIL import Image

        return np.asarray(Image.open(self.image_path(i)).convert("RGB"))

    def pairs(self):
        """All unordered pairs i < j: ``(i, j, overlap_percent, is_loop)``."""
        i, j = np.triu_indices(len(self), k=1)
        ov = self.overlap[i, j]
        return i, j, ov, ov > 0
