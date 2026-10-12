#!/usr/bin/env python3
"""Train a loop-classifier head on frozen Swin-T embeddings.

Images are embedded once; the head is then trained on embedding pairs, which
takes seconds per epoch. Validation uses *separate worlds* (``--val``), so the
score reflects generalisation to unseen terrain rather than to new pairs of
already-seen images.

Usage:
    python scripts/train_head.py --train data/train_sea --val data/val_high_feature \
        --variant mlp3 --out weights/swin_mlp3_new.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loopclosure import metrics
from loopclosure.heads import LoopHead
from loopclosure.swin import SwinEmbedder
from loopclosure.ucamgen import UcamgenDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True, help="UCAMGEN dataset folders to train on")
    p.add_argument("--val", nargs="+", required=True, help="Held-out worlds for validation")
    p.add_argument("--variant", default="mlp3", choices=LoopHead.VARIANTS)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--neg-per-pos", type=float, default=4.0,
                   help="Negatives sampled per positive each epoch (loops are only ~3%% of pairs)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def embed_world(embedder, path):
    ds = UcamgenDataset.load(path)
    emb = torch.from_numpy(embedder.embed([ds.load_image(k) for k in range(len(ds))]))
    i, j, _, y = ds.pairs()
    return emb, i, j, y


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SwinEmbedder(device=str(device))
    train = [embed_world(embedder, p) for p in args.train]
    val = [embed_world(embedder, p) for p in args.val]
    print("train loops:", sum(int(w[3].sum()) for w in train), " val pairs:", sum(len(w[3]) for w in val))

    head = LoopHead(args.variant).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history, best, threshold = [], -1.0, 0.5

    for epoch in range(args.epochs):
        head.train()
        tot, n = 0.0, 0
        for emb, i, j, y in train:
            pos = np.flatnonzero(y)
            neg = rng.choice(np.flatnonzero(~y), size=int(len(pos) * args.neg_per_pos), replace=False)
            idx = rng.permutation(np.r_[pos, neg])
            for k in range(0, len(idx), args.batch):
                b = idx[k:k + args.batch]
                # Pairs are unordered: randomly swap sides so the head learns a symmetric function.
                swap = rng.random(len(b)) < 0.5
                a_idx, b_idx = np.where(swap, j[b], i[b]), np.where(swap, i[b], j[b])
                logits = head.logits(emb[a_idx].to(device), emb[b_idx].to(device))
                loss = loss_fn(logits, torch.from_numpy(y[b]).float().to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
                n += len(b)

        head.eval()
        with torch.inference_mode():
            ys, ss = [], []
            for emb, i, j, y in val:
                for k in range(0, len(i), 65536):
                    ss.append(head(emb[i[k:k + 65536]].to(device), emb[j[k:k + 65536]].to(device)).cpu().numpy())
                ys.append(y)
            y_val, s_val = np.concatenate(ys), np.concatenate(ss)
        ap = metrics.average_precision(y_val, s_val)
        row = {"epoch": epoch, "train_loss": tot / n, "val_roc_auc": metrics.roc_auc(y_val, s_val),
               "val_average_precision": ap}
        history.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if ap > best:
            best = ap
            # Decision threshold chosen on the validation worlds (max F1), never on test data.
            threshold = metrics.best_f1(y_val, s_val)["threshold"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.net.state_dict(), args.out)

    meta = {"variant": args.variant, "threshold": threshold, "best_val_average_precision": best,
            "config": vars(args), "history": history}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Best validation AP {best:.4f}, validation-selected threshold {threshold:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
