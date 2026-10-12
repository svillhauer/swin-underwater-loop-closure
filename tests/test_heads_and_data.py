import numpy as np
import torch

from loopclosure.detectors import LoopDetector
from loopclosure.heads import LoopHead, load_head
from loopclosure.ucamgen import UcamgenDataset


def test_heads_output_probabilities():
    for variant in LoopHead.VARIANTS:
        head = LoopHead(variant).eval()
        p = head(torch.randn(5, 768), torch.randn(5, 768))
        assert p.shape == (5,) and torch.all((p >= 0) & (p <= 1))


def test_load_head_accepts_original_key_prefixes(tmp_path):
    src = LoopHead("mlp2")
    for prefix in ("classifier", "fc"):
        torch.save({f"{prefix}.{k}": v for k, v in src.net.state_dict().items()}, tmp_path / f"{prefix}.pth")
        loaded = load_head(str(tmp_path / f"{prefix}.pth"), "mlp2")
        a, b = torch.randn(3, 768), torch.randn(3, 768)
        torch.testing.assert_close(loaded(a, b), src.eval()(a, b))


def test_ucamgen_pairs_follow_the_overlap_matrix(tmp_path):
    ov = np.array([[100, 40, 0, 0], [40, 100, 5, 0], [0, 5, 100, 0], [0, 0, 0, 100]])
    np.savetxt(tmp_path / "OVERLAP.csv", ov, delimiter=",", fmt="%d")
    ds = UcamgenDataset.load(tmp_path)
    i, j, o, y = ds.pairs()
    assert len(ds) == 4 and len(i) == 6 and np.all(i < j)
    assert {(a, b) for a, b, loop in zip(i, j, y, strict=True) if loop} == {(0, 1), (1, 2)}
    assert ds.image_path(0).name == "IMAGE00001.png"


def test_detector_scores_pairs_in_batches():
    feats = torch.arange(10.0).view(10, 1)
    det = LoopDetector("sum", lambda ims: feats, lambda a, b: (a + b).squeeze(1), torch.device("cpu"))
    i, j = np.array([0, 1, 2, 9]), np.array([1, 2, 3, 0])
    np.testing.assert_array_equal(det.score_pairs(feats, i, j, batch=3), [1, 3, 5, 9])
