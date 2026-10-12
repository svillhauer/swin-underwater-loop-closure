"""Uniform interface over loop detectors: embed each image once, then score pairs.

Both detectors are Siamese at heart, so per-image features can be computed once
and reused for every pair. That makes scoring all N(N-1)/2 pairs of a dataset,
or every candidate in a SLAM loop, cheap.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


@dataclass
class LoopDetector:
    name: str
    featurize: Callable[[list[np.ndarray]], torch.Tensor]       # RGB uint8 images -> (N, ...) features
    classify: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (B, ...) x2 -> (B,) loop probability
    device: torch.device

    @torch.inference_mode()
    def score_pairs(self, feats: torch.Tensor, i: np.ndarray, j: np.ndarray, batch: int = 16384) -> np.ndarray:
        out = []
        for k in range(0, len(i), batch):
            a = feats[torch.from_numpy(np.asarray(i[k:k + batch]))].to(self.device)
            b = feats[torch.from_numpy(np.asarray(j[k:k + batch]))].to(self.device)
            out.append(self.classify(a, b).float().cpu())
        return torch.cat(out).numpy() if out else np.empty(0, np.float32)


def scnn_detector(h5_path: str, device: torch.device, name: str = "SCNN") -> LoopDetector:
    from skimage import img_as_float
    from skimage.transform import resize

    from .scnn import INPUT_SIZE, load_keras_h5

    model = load_keras_h5(h5_path).to(device)

    @torch.inference_mode()
    def featurize(images):
        x = np.stack([resize(img_as_float(im), (INPUT_SIZE, INPUT_SIZE)) for im in images]).astype(np.float32)
        return model.encode(torch.from_numpy(x).permute(0, 3, 1, 2).to(device)).cpu()

    return LoopDetector(name, featurize, model.classify, device)


def swin_detector(head_path: str, variant: str, device: torch.device, name: str = "Swin-T",
                  embedder=None) -> LoopDetector:
    from .heads import load_head
    from .swin import SwinEmbedder

    embedder = embedder or SwinEmbedder(device=str(device))
    head = load_head(head_path, variant).to(device)
    return LoopDetector(name, lambda images: torch.from_numpy(embedder.embed(images)), head, device)
