#!/usr/bin/env python3
"""Score loop detectors on every image pair of a held-out UCAMGEN world.

Each detector sees the same pairs: all N(N-1)/2 unordered pairs of images in
the test dataset, labelled as loops when their ground-truth overlap is > 0 %.
Images are embedded once per detector and pairs are scored in batches.

Usage:
    python scripts/benchmark.py --test data/test_low_feature \
        --scnn weights/scnn_sea.h5 \
        --swin-head "Swin-T + MLP-2=weights/swin_mlp2_sea.pth:mlp2" \
        --swin-head "Swin-T + MLP-3=weights/swin_mlp3.pth:mlp3"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from loopclosure import metrics
from loopclosure.detectors import scnn_detector, swin_detector
from loopclosure.swin import SwinEmbedder
from loopclosure.ucamgen import UcamgenDataset

OVERLAP_BINS = [(1, 10), (10, 25), (25, 50), (50, 101)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, help="UCAMGEN dataset folder (held-out world)")
    p.add_argument("--scnn", help="Original SNNLOOP .h5 weights (Siamese CNN baseline)")
    p.add_argument("--swin-head", action="append", default=[], metavar="LABEL=PATH:VARIANT",
                   help="Swin loop head to evaluate; repeatable")
    p.add_argument("--out", default="results/benchmark.json")
    p.add_argument("--save-scores", default="results/scores.npz", help="Per-pair scores for plotting")
    p.add_argument("--batch", type=int, default=16384)
    return p.parse_args()


def summarise(y, ov, s) -> dict:
    res = {
        "at_0.5": metrics.at_threshold(y, s, 0.5),
        "best_f1": metrics.best_f1(y, s),
        "roc_auc": metrics.roc_auc(y, s),
        "average_precision": metrics.average_precision(y, s),
        "recall_by_overlap": {},
    }
    pred = s >= 0.5
    for lo, hi in OVERLAP_BINS:
        m = (ov >= lo) & (ov < hi)
        res["recall_by_overlap"][f"{lo}-{min(hi, 100)}%"] = {"n": int(m.sum()), "recall": float(pred[m].mean())}
    return res


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = UcamgenDataset.load(args.test)
    i, j, ov, y = ds.pairs()
    print(f"{args.test}: {len(ds)} images, {len(y):,} pairs, {y.sum():,} loops ({y.mean():.2%})")
    images = [ds.load_image(k) for k in range(len(ds))]

    results = {"dataset": str(args.test), "n_images": len(ds), "n_pairs": int(len(y)),
               "n_loops": int(y.sum()), "loop_rate": float(y.mean()), "models": {}}
    scores = {"overlap": ov, "label": y}

    detectors = []
    val_thresholds = {}
    if args.scnn:
        detectors.append(scnn_detector(args.scnn, device, "SCNN (Burguera et al.)"))
    embedder = SwinEmbedder(device=str(device)) if args.swin_head else None
    for spec in args.swin_head:
        label, rest = spec.split("=", 1)
        path, variant = rest.rsplit(":", 1)
        detectors.append(swin_detector(path, variant, device, label, embedder=embedder))
        meta = Path(path).with_suffix(".json")
        if meta.exists():   # threshold selected on a validation world by train_head.py
            val_thresholds[label] = json.loads(meta.read_text())["threshold"]

    for det in detectors:
        tic = time.time()
        feats = det.featurize(images)
        s = det.score_pairs(feats, i, j, args.batch)
        results["models"][det.name] = summarise(y, ov, s) | {"seconds": time.time() - tic}
        if det.name in val_thresholds:
            results["models"][det.name]["at_val_threshold"] = metrics.at_threshold(y, s, val_thresholds[det.name])
        scores[det.name] = s

    for name, r in results["models"].items():
        a = r["at_0.5"]
        print(f"  {name:28s} P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1']:.3f} "
              f"FP={a['fp']:,}  ROC-AUC={r['roc_auc']:.3f}  AP={r['average_precision']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    np.savez_compressed(args.save_scores, **scores)
    print(f"Wrote {args.out} and {args.save_scores}")


if __name__ == "__main__":
    main()
