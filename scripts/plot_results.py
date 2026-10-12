#!/usr/bin/env python3
"""README figures from results/scores.npz (benchmark.py) and results/slam/*.npz (run_slam.py).

Usage:
    python scripts/plot_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopclosure import metrics

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots 1-4 (validated palette), fixed per model
COLORS = {
    "Swin-T + MLP-3": "#2a78d6",
    "SCNN (Burguera et al.)": "#eb6834",
    "SCNN": "#eb6834",
    "Swin-T + MLP-2": "#1baf7a",
    "Swin-T + MLP-3, retrained": "#eda100",
}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": SECONDARY, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def style(ax):
    ax.grid(color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_pr(panels, out):
    """panels: [(scores.npz path, title)] -> side-by-side precision-recall curves."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5), dpi=200)
    for ax, (path, title) in zip(np.atleast_1d(axes), panels, strict=True):
        s = np.load(path)
        y = s["label"].astype(bool)
        for name in [k for k in s.files if k not in ("label", "overlap")]:
            p, r, _ = metrics.precision_recall_curve(y, s[name])
            ap = metrics.average_precision(y, s[name])
            ax.plot(r, p, color=COLORS.get(name, MUTED), linewidth=2, label=f"{name}  (AP {ap:.2f})", zorder=3)
        ax.axhline(y.mean(), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(0.99, y.mean() - 0.01, f"chance = loop rate ({y.mean():.3f})", ha="right", va="top", fontsize=9,
                color=SECONDARY)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall (share of true loops found)")
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=8)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Precision")
    fig.suptitle("Pair-level loop detection on unseen worlds", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png")
    plt.close(fig)


def plot_trajectories(rows, out):
    """rows: [(slam results dir, row title)] -> grid of SLAM estimates vs. ground truth."""
    runs = [("odometry_only", "Odometry only"), ("scnn", "SCNN"), ("swin-t_mlp-3", "Swin-T + MLP-3"),
            ("oracle_ground-truth_loops", "Oracle loops")]
    fig, axes = plt.subplots(len(rows), len(runs), figsize=(4.0 * len(runs), 4.3 * len(rows)), dpi=170)
    axes = np.atleast_2d(axes)
    for r, (slam_dir, row_title) in enumerate(rows):
        for c, (slug, title) in enumerate(runs):
            ax = axes[r, c]
            d = np.load(Path(slam_dir) / f"{slug}.npz")
            meta = json.loads((Path(slam_dir) / f"{slug}.json").read_text())
            gt, full, loops = d["ground_truth"], d["trajectory"], d["loops"]
            ax.plot(gt[0], gt[1], color=MUTED, linewidth=1.5, linestyle=(0, (3, 2)), label="ground truth")
            acc = loops[loops[:, 5] == 1]
            for a, b in acc[:, :2]:
                ax.plot(full[0, [a, b]], full[1, [a, b]], color="#1baf7a", linewidth=0.6, alpha=0.6, zorder=2)
            ax.plot(full[0, 1:], full[1, 1:], color=COLORS.get(meta["name"], INK), linewidth=1.8,
                    label="SLAM estimate", zorder=3)
            head = f"{row_title}\n" if c == 0 else "\n"
            ax.set_title(f"{head}{title}: ATE {meta['ate_mean']:.0f}  ·  {len(acc)} loops", fontsize=10.5,
                         loc="left", fontweight="bold")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#1baf7a", linewidth=1.2))
    fig.legend(handles, ["ground truth", "SLAM estimate", "accepted loop closure"], frameon=False, fontsize=10,
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "slam_trajectories.png")
    plt.close(fig)


def _loop_example(ds):
    """A loop revisited much later (>= 40% overlap) and a frame from elsewhere (0% overlap)."""
    i, j = np.where(np.triu(ds.overlap, k=len(ds) // 8) >= 40)
    k = int(np.argmax(j - i))
    a, b = int(i[k]), int(j[k])
    far = int(np.flatnonzero(ds.overlap[a] == 0)[len(ds) // 3])
    return a, b, far


def plot_worlds(train_dir, test_dirs, out):
    """Example frames from the training world and, per test world, a far-apart loop vs. a non-loop."""
    from PIL import Image

    from loopclosure.ucamgen import UcamgenDataset

    rows = [("Training world: sea surface", train_dir, [1, 300, 700], None)]
    for root, title in test_dirs:
        ds = UcamgenDataset.load(root)
        a, b, far = _loop_example(ds)
        rows.append((title, root, [a + 1, b + 1, far + 1],
                     [f"frame {a + 1}", f"frame {b + 1}: same place ({ds.overlap[a, b]}% overlap)",
                      f"frame {far + 1}: elsewhere (0% overlap)"]))
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.9 * len(rows)), dpi=150)
    for r, (title, root, frames, caps) in enumerate(rows):
        for c, k in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(Image.open(Path(root) / "IMAGES" / f"IMAGE{k:05d}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if caps:
                ax.set_xlabel(caps[c], fontsize=9.5, color=SECONDARY)
        axes[r, 0].set_title(title, loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "worlds.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/media")
    p.add_argument("--train", default="data/train_sea")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worlds = [("data/test_sand", "results/scores_sand.npz", "results/slam_sand", "Unseen world A: sandy shallows"),
              ("data/test_rocky", "results/scores_rocky.npz", "results/slam_rocky", "Unseen world B: rocky seabed")]
    plot_pr([(scores, title) for _, scores, _, title in worlds], out)
    plot_trajectories([(slam, title) for _, _, slam, title in worlds], out)
    plot_worlds(args.train, [(data, title) for data, _, _, title in worlds], out)
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
