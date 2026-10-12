import h5py
import numpy as np
import torch
import torch.nn.functional as F

from loopclosure.scnn import SiameseCNN, load_keras_h5


def write_fake_keras_h5(path, rng):
    """Random weights laid out exactly like the original SNNLOOP .h5 files."""
    with h5py.File(path, "w") as f:
        w = f.create_group("model_weights")
        cin = 3
        for i, cout in enumerate((128, 128, 16)):
            sfx = "" if i == 0 else f"_{i}"
            g = w.create_group(f"encoder_model/conv2d{sfx}")
            g["kernel"] = rng.normal(size=(3, 3, cin, cout)).astype(np.float32) * 0.1
            g["bias"] = rng.normal(size=cout).astype(np.float32) * 0.1
            b = w.create_group(f"encoder_model/batch_normalization{sfx}")
            for k, v in (("gamma", 1), ("beta", 0), ("moving_mean", 0), ("moving_variance", 1)):
                b[k] = (v + 0.1 * rng.normal(size=cout)).astype(np.float32) if k != "moving_variance" else \
                    (1 + 0.1 * rng.random(cout)).astype(np.float32)
            cin = cout
        b = w.create_group("batch_normalization/batch_normalization")
        for k in ("gamma", "beta", "moving_mean"):
            b[k] = rng.normal(size=2048).astype(np.float32)
        b["moving_variance"] = (1 + rng.random(2048)).astype(np.float32)
        for name, (a, o) in {"dense": (2048, 128), "dense_1": (128, 16), "dense_2": (16, 1)}.items():
            g = w.create_group(f"{name}/{name}")
            g["kernel"] = rng.normal(size=(a, o)).astype(np.float32) * 0.05
            g["bias"] = rng.normal(size=o).astype(np.float32) * 0.05


def keras_same_conv_reference(x_nhwc, kernel_hwio, bias):
    """Direct numpy implementation of TF Conv2D(stride=2, padding='same') on even inputs."""
    x = np.pad(x_nhwc, ((0, 0), (0, 1), (0, 1), (0, 0)))
    n, h, w, _ = x_nhwc.shape
    out = np.zeros((n, h // 2, w // 2, kernel_hwio.shape[3]), np.float64)
    for r in range(h // 2):
        for c in range(w // 2):
            patch = x[:, 2 * r:2 * r + 3, 2 * c:2 * c + 3, :]
            out[:, r, c] = np.tensordot(patch, kernel_hwio, axes=([1, 2, 3], [0, 1, 2]))
    return out + bias


def test_conv_matches_tensorflow_same_padding(tmp_path):
    rng = np.random.default_rng(0)
    write_fake_keras_h5(tmp_path / "m.h5", rng)
    model = load_keras_h5(str(tmp_path / "m.h5"))
    x = rng.random((2, 64, 64, 3)).astype(np.float32)
    with h5py.File(tmp_path / "m.h5") as f:
        k, b = f["model_weights/encoder_model/conv2d/kernel"][()], f["model_weights/encoder_model/conv2d/bias"][()]
    ref = keras_same_conv_reference(x, k, b)
    conv = model.encoder[0].conv
    got = conv(F.pad(torch.from_numpy(x).permute(0, 3, 1, 2), (0, 1, 0, 1))).permute(0, 2, 3, 1)
    np.testing.assert_allclose(got.detach().numpy(), ref, atol=1e-4)


def test_shapes_and_probability_output(tmp_path):
    write_fake_keras_h5(tmp_path / "m.h5", np.random.default_rng(1))
    model = load_keras_h5(str(tmp_path / "m.h5"))
    a, b = torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64)
    assert model.encode(a).shape == (4, 8, 8, 16)
    p = model(a, b)
    assert p.shape == (4,) and torch.all((p > 0) & (p < 1))


def test_model_is_in_eval_mode_after_loading(tmp_path):
    write_fake_keras_h5(tmp_path / "m.h5", np.random.default_rng(2))
    assert not load_keras_h5(str(tmp_path / "m.h5")).training
    assert SiameseCNN().training
