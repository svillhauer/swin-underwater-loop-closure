"""PyTorch port of the Siamese CNN loop detector of Burguera et al. (2022).

Architecture (as trained in the original Keras code):
    shared encoder: 3 x [Conv 3x3 stride 2 'same' -> LeakyReLU(0.2) -> BatchNorm]
                    64x64x3 -> 32x32x128 -> 16x16x128 -> 8x8x16
    head:           concat(enc(a), enc(b)) -> flatten (channels-last order)
                    -> BatchNorm -> Dense 128 sigmoid -> Dense 16 sigmoid -> Dense 1 sigmoid

``load_keras_h5`` reads the original ``.h5`` weights directly with h5py, so
neither TensorFlow nor Keras is needed at inference time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_BN_EPS = 1e-3
INPUT_SIZE = 64


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(cout, eps=KERAS_BN_EPS)

    def forward(self, x):
        # TensorFlow 'same' padding for k=3, s=2 on even inputs pads 0 before, 1 after.
        x = self.conv(F.pad(x, (0, 1, 0, 1)))
        return self.bn(F.leaky_relu(x, 0.2))


class SiameseCNN(nn.Module):
    def __init__(self, filters=(128, 128, 16), dense=(128, 16)):
        super().__init__()
        chans = (3, *filters)
        self.encoder = nn.Sequential(*[_ConvBlock(a, b) for a, b in zip(chans[:-1], chans[1:], strict=True)])
        n_flat = 2 * filters[-1] * (INPUT_SIZE // 8) ** 2
        self.bn = nn.BatchNorm1d(n_flat, eps=KERAS_BN_EPS)
        sizes = (n_flat, *dense, 1)
        self.dense = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 64, 64) in [0, 1] -> (B, 8, 8, 16) channels-last feature map."""
        return self.encoder(x).permute(0, 2, 3, 1)

    def classify(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """Loop probability from two encoded images."""
        x = torch.cat([fa, fb], dim=-1).flatten(1)
        x = self.bn(x)
        for layer in self.dense:
            x = torch.sigmoid(layer(x))
        return x.squeeze(1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(a), self.encode(b))


def load_keras_h5(path: str) -> SiameseCNN:
    """Build a SiameseCNN and copy the weights from an original SNNLOOP ``.h5`` file."""
    import h5py

    model = SiameseCNN()
    with h5py.File(path, "r") as f:
        w = f["model_weights"]

        def bn(dst, grp):
            dst.weight.data = torch.from_numpy(grp["gamma"][()])
            dst.bias.data = torch.from_numpy(grp["beta"][()])
            dst.running_mean.data = torch.from_numpy(grp["moving_mean"][()])
            dst.running_var.data = torch.from_numpy(grp["moving_variance"][()])

        for i, block in enumerate(model.encoder):
            sfx = "" if i == 0 else f"_{i}"
            conv = w[f"encoder_model/conv2d{sfx}"]
            block.conv.weight.data = torch.from_numpy(np.transpose(conv["kernel"][()], (3, 2, 0, 1)).copy())
            block.conv.bias.data = torch.from_numpy(conv["bias"][()])
            bn(block.bn, w[f"encoder_model/batch_normalization{sfx}"])
        bn(model.bn, w["batch_normalization/batch_normalization"])
        for i, layer in enumerate(model.dense):
            name = "dense" if i == 0 else f"dense_{i}"
            g = w[f"{name}/{name}"]
            layer.weight.data = torch.from_numpy(g["kernel"][()].T.copy())
            layer.bias.data = torch.from_numpy(g["bias"][()])
    return model.eval()
