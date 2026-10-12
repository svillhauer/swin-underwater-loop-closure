import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from loopclosure.gslam import GSLAM_DIR, cached_matcher_class, import_gslam, prepare_dataset  # noqa: E402

pytestmark = pytest.mark.skipif(not (GSLAM_DIR / "datasimulator.py").exists(),
                                reason="GSLAM submodule not checked out")


def make_ucamgen(root, n=6):
    ov = np.eye(n, dtype=int) * 100
    for k in range(n - 1):
        ov[k, k + 1] = ov[k + 1, k] = 60
    np.savetxt(root / "OVERLAP.csv", ov, delimiter=",", fmt="%d")
    for axis, val in zip("XYOH", (5.0, -60.0, 0.1, 0.0), strict=True):
        rel = np.where(ov > 0, val, 123.0)            # non-loop entries must be zeroed by the converter
        np.fill_diagonal(rel, 0)
        np.savetxt(root / f"IMGRPOS{axis}.csv", rel, delimiter=",")
    odom = np.zeros((4, n))
    odom[:, 0] = [600, 600, np.pi / 2, 300]
    odom[1, 1:] = -60
    np.savetxt(root / "ODOM.csv", odom, delimiter=",")


def test_prepared_dataset_loads_in_upstream_datasimulator(tmp_path):
    make_ucamgen(tmp_path)
    cache = prepare_dataset(tmp_path, tmp_path / "cache")
    import_gslam()
    from datasimulator import DataSimulator

    sim = DataSimulator(str(cache), loadImages=False, minOverlap=1)
    detected, motion, gt = sim.match_images(0, 1, addNoise=False)
    assert gt and detected and np.allclose(motion, [5, -60, 0.1])
    assert not sim.match_images(0, 3, addNoise=False)[2]
    assert sim.relPosX[0, 3] == 0                       # non-loop relative pose zeroed
    assert sim.endStep == 5


def test_cached_matcher_survives_featureless_images():
    import_gslam()
    matcher = cached_matcher_class()()
    blank = np.zeros((240, 320, 3), np.uint8)
    matcher.define_images(blank, blank)
    assert matcher.estimate() is False
