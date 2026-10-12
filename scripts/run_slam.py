#!/usr/bin/env python3
"""Run underwater visual graph SLAM with a chosen loop detector as stage 1.

Pipeline (Burguera et al., 2022), executed with the original GSLAM code:
  1. learned loop detector  - fast pre-selection of candidate loops (SCNN or Swin)
  2. image-based filter     - SIFT + RANSAC must confirm the two images overlap
  3. pose-based filter      - GraphOptimizer rejects loops inconsistent with the graph
  then pose-graph optimisation.

Odometry and loop motions come from GSLAM's DataSimulator (ground truth + noise),
exactly as in the original evaluation, so the detector is the only variable.
Detectors: ``none`` (odometry only), ``oracle`` (ground-truth overlap), ``scnn``, ``swin``.

Usage:
    python scripts/run_slam.py --data data/test_low_feature --detector scnn --scnn weights/scnn_sea.h5
    python scripts/run_slam.py --data data/test_low_feature --detector swin \
        --swin-head weights/swin_mlp3.pth --variant mlp3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from loopclosure.detectors import scnn_detector, swin_detector
from loopclosure.gslam import cached_matcher_class, import_gslam, prepare_dataset
from loopclosure.ucamgen import UcamgenDataset

NOISE_LEVELS = [(0.625, 3.14 / (180 * 4)), (2.5, 3.14 / 180), (5, 2 * 3.14 / 180)]  # GSLAM defaults


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="UCAMGEN dataset folder")
    p.add_argument("--detector", choices=["none", "oracle", "scnn", "swin"], required=True)
    p.add_argument("--scnn", help="SCNN .h5 weights")
    p.add_argument("--swin-head", help="Swin loop-head weights")
    p.add_argument("--variant", default="mlp3", choices=["mlp2", "mlp3"])
    p.add_argument("--threshold", type=float, default=0.5, help="Detector decision threshold")
    p.add_argument("--no-ransac", action="store_true", help="Skip stage 2")
    p.add_argument("--no-filter", action="store_true", help="Skip stage 3")
    p.add_argument("--loop-margin", type=int, default=5, help="Ignore the most recent N frames")
    p.add_argument("--loop-separation", type=int, default=5, help="Check every N-th older frame")
    p.add_argument("--noise-level", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default="results/slam")
    p.add_argument("--name", default=None, help="Run name (defaults to the detector)")
    return p.parse_args()


def main():
    args = parse_args()
    import_gslam()
    from datasimulator import DataSimulator
    from graphoptimizer import GraphOptimizer
    from transform2d import compose_trajectory
    from util import compute_absolute_trajectory_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = UcamgenDataset.load(args.data)
    cache = prepare_dataset(args.data, Path(args.out) / "cache" / Path(args.data).name)
    m_sigma, a_sigma = NOISE_LEVELS[args.noise_level]
    sim = DataSimulator(str(cache), loadImages=False, minOverlap=1, motionSigma=m_sigma, angleSigma=a_sigma)
    if args.max_steps:
        sim.set_start_end(endStep=args.max_steps)
    n = sim.endStep + 1
    cov = sim.odoCovariance

    images = [ds.load_image(k) for k in range(n)]
    det = None
    if args.detector == "scnn":
        det = scnn_detector(args.scnn, device, "SCNN")
    elif args.detector == "swin":
        det = swin_detector(args.swin_head, args.variant, device, f"Swin-T + {args.variant.upper()}")
    feats = det.featurize(images) if det else None

    matcher = cached_matcher_class()()
    optimizer = GraphOptimizer(initialID=0, minLoops=5, doFilter=not args.no_filter)
    stats = []                       # (from, to, gt, stage1, stage2, stage3)
    t = {"detector": 0.0, "ransac": 0.0, "graph": 0.0}
    all_ids = [0]
    pre = 0
    while sim.update():
        cur, _ = sim.get_image()
        _, odo, _ = sim.match_images(pre, cur, addNoise=True, simulateFailures=False)
        optimizer.add_odometry(cur, odo.reshape(3, 1), cov)
        cands = all_ids[:-args.loop_margin:args.loop_separation]
        if cands:
            gt = ds.overlap[cands, cur] >= 1
            tic = time.time()
            if det is None:
                stage1 = gt.copy() if args.detector == "oracle" else np.zeros(len(cands), bool)
            else:
                stage1 = det.score_pairs(feats, np.full(len(cands), cur), np.asarray(cands)) >= args.threshold
            t["detector"] += time.time() - tic
            for c, g, s1 in zip(cands, gt, stage1, strict=True):
                # Simulated loop motion, as in GSLAM's evaluation (MOTION_ESTIMATOR=0)
                _, motion, _ = sim.match_images(c, cur, addNoise=True, simulateFailures=True)
                s2 = False
                if s1:
                    if args.no_ransac:
                        s2 = True
                    else:
                        tic = time.time()
                        matcher.define_images(images[c], images[cur])
                        # RANSAC draws from NumPy's global RNG; give it a per-pair seed and restore the
                        # simulator's state so every detector sees identical odometry and loop noise.
                        rng_state = np.random.get_state()
                        np.random.seed(c * 100003 + cur)
                        try:
                            s2 = bool(matcher.estimate())
                        except (ValueError, IndexError, np.linalg.LinAlgError):
                            s2 = False
                        np.random.set_state(rng_state)
                        t["ransac"] += time.time() - tic
                    if s2:
                        optimizer.add_loop(c, cur, motion.reshape(3, 1), cov)
                stats.append([c, cur, int(g), int(s1), int(s2), 0])
            tic = time.time()
            for sel in optimizer.validate():
                for row in reversed(stats):
                    if row[0] == sel[0] and row[1] == sel[1]:
                        row[5] = 1
                        break
            optimizer.optimize()
            t["graph"] += time.time() - tic
        all_ids.append(cur)
        pre = cur
        if cur % 100 == 0:
            print(f"  step {cur}/{n - 1}  loops accepted so far: {sum(r[5] for r in stats)}", flush=True)

    stats = np.array(stats, dtype=int).reshape(-1, 6)
    traj = np.array([np.array(v.pose) for v in optimizer.theVertices]).T
    gt_traj = compose_trajectory(sim.gtOdom[0:3, 1:])
    errors, ate_mean, ate_std = compute_absolute_trajectory_error(traj, gt_traj)

    def stage(col):
        g, p = stats[:, 2] == 1, stats[:, col] == 1
        return {"tp": int((g & p).sum()), "fp": int((~g & p).sum()), "fn": int((g & ~p).sum()),
                "tn": int((~g & ~p).sum())}

    name = args.name or (args.detector if det is None else det.name)
    result = {
        "name": name, "dataset": str(args.data), "steps": int(n), "candidates_checked": int(len(stats)),
        "true_loops_available": int(stats[:, 2].sum()) if len(stats) else 0,
        "stage1_detector": stage(3) if len(stats) else {}, "stage2_ransac": stage(4) if len(stats) else {},
        "stage3_filter": stage(5) if len(stats) else {},
        "ate_mean": float(ate_mean), "ate_std": float(ate_std), "seconds": t,
        "config": {k: v for k, v in vars(args).items() if k != "out"},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace("__", "_")
    (out / f"{slug}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / f"{slug}.npz", trajectory=traj, ground_truth=gt_traj, ate=np.asarray(errors),
                        loops=stats)
    print(json.dumps({k: result[k] for k in ("name", "stage1_detector", "stage2_ransac", "stage3_filter",
                                             "ate_mean", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
