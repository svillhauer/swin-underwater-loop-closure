"""Frozen Swin-T image embedder (ImageNet-pretrained, from Hugging Face)."""
from __future__ import annotations

import numpy as np
import torch

MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
EMBED_DIM = 768


class SwinEmbedder:
    """Maps RGB images to 768-d global descriptors (Swin pooler output).

    The backbone is frozen; only the loop classifier on top is trained.
    Preprocessing: resize to 224 x 224, ImageNet mean/std normalisation.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        from transformers import AutoImageProcessor, SwinModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        proc = AutoImageProcessor.from_pretrained(model_name)
        self.mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.model = SwinModel.from_pretrained(model_name).to(self.device).eval()

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        from PIL import Image

        x = np.stack([np.asarray(Image.fromarray(im).resize((224, 224), Image.BILINEAR)) for im in images])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def embed(self, images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        out = []
        for k in range(0, len(images), batch_size):
            x = self.preprocess(images[k:k + batch_size]).to(self.device)
            out.append(self.model(pixel_values=x).pooler_output.float().cpu())
        return torch.cat(out).numpy()
