"""Adapter for the GSLAM front-end of Burguera et al. (2022), used unmodified.

GSLAM (https://github.com/aburguera/GSLAM) is vendored as a git submodule in
``third_party/GSLAM``. This module

* puts it on ``sys.path``,
* shims two library API changes instead of patching the upstream code:
  ``cv2.xfeatures2d.SIFT_create`` (now ``cv2.SIFT_create``, OpenCV >= 4.4) and
  ``np.Inf`` (removed in NumPy 2.0),
* converts a UCAMGEN dataset into the pickle files GSLAM's ``DataSimulator``
  reads, in a cache folder, without touching the dataset itself, and
* provides an ``ImageMatcher`` subclass that caches SIFT features per image and
  tolerates images with too few keypoints (otherwise an exception in upstream).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

GSLAM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "GSLAM"


def import_gslam():
    if not (GSLAM_DIR / "graphoptimizer.py").exists():
        raise RuntimeError(f"GSLAM not found in {GSLAM_DIR}. Run: git submodule update --init")
    if str(GSLAM_DIR) not in sys.path:
        sys.path.insert(0, str(GSLAM_DIR))
    import cv2

    if not hasattr(cv2, "xfeatures2d"):
        cv2.xfeatures2d = SimpleNamespace(SIFT_create=cv2.SIFT_create)
    if not hasattr(np, "Inf"):
        np.Inf = np.inf


def prepare_dataset(ucamgen_dir, cache_dir) -> Path:
    """Write GSLAM's pre-processed pickles for a UCAMGEN dataset (same format as GSLAM's
    ``preprocess_dataset.py``: sparse overlap and relative-pose matrices, non-loops zeroed)."""
    from scipy.sparse import lil_matrix

    src, out = Path(ucamgen_dir), Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ODOM.pkl").exists():
        return out

    def dump(name, obj):
        with open(out / name, "wb") as f:
            pickle.dump(obj, f)

    overlap = np.loadtxt(src / "OVERLAP.csv", delimiter=",")
    no_loop = np.where(overlap <= 0)
    dump("NOLOOPIDX.pkl", no_loop)
    dump("OVERLAP.pkl", lil_matrix(overlap))
    for axis in "XYOH":
        rel = np.loadtxt(src / f"IMGRPOS{axis}.csv", delimiter=",")
        rel[no_loop] = 0
        dump(f"IMGRPOS{axis}.pkl", lil_matrix(rel))
    dump("ODOM.pkl", np.loadtxt(src / "ODOM.csv", delimiter=","))
    return out


def cached_matcher_class():
    """ImageMatcher that reuses SIFT features of images it has already seen."""
    import cv2
    from imagematcher import ImageMatcher

    class CachedImageMatcher(ImageMatcher):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cache = {}

        def _get_sift_(self, image):
            key = image.__array_interface__["data"][0]   # same array object -> same features
            if key not in self._cache:
                kp, des = cv2.SIFT_create().detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
                self._cache[key] = (kp, des if des is not None else np.empty((0, 128), np.float32))
            return self._cache[key]

        def _get_matches_(self, d_ref, d_cur):
            if len(d_ref) < 2 or len(d_cur) < 2:
                return np.empty((0, 2), int)
            good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(d_ref, d_cur, k=2) if len(p) == 2)
                    if m.distance < self.matchThreshold * n.distance]
            return np.array([[m.queryIdx, m.trainIdx] for m in good]).reshape(-1, 2)

        def estimate(self):
            # Upstream assumes at least a few SIFT matches; with none (featureless seabed) it
            # crashes instead of failing. Treat < 3 matches as a failed verification.
            self.kpRef, self.dRef = self._get_sift_(self.iRef)
            self.kpCur, self.dCur = self._get_sift_(self.iCur)
            if len(self._get_matches_(self.dRef, self.dCur)) < 3:
                self.theMotion, self.hasFailed = np.zeros((3, 1)), True
                return False
            return super().estimate()

        def define_images(self, i_ref, i_cur):
            # Keep the caller's uint8 arrays (no copy) so the feature cache can key on them.
            self._reset_()
            self.iRef, self.iCur = i_ref, i_cur

    return CachedImageMatcher
