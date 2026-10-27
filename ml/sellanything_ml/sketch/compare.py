"""Compare sketch classifiers on drawings none of them trained on.

    python -m sellanything_ml.sketch.compare checkpoints/sketch_cnn_345.pt checkpoints/sketch_cnn_345_15k.pt

Every model here trained on drawings taken from the first 15,000 of each class, so drawings
15,000-15,499 (the --start/--end default) are unseen by all of them: the same fair test for each.
Prints overall top-1/top-3, the classes named with --show (default: cat), and how many classes got
better or worse between the first and the last model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import download_first_n, load_categories
from .infer import SketchClassifier


@torch.no_grad()
def per_class_hits(clf: SketchClassifier, x: np.ndarray, y: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """For every drawing: was the right class first, and was it in the top 3."""
    top1, top3 = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for i in range(0, len(y), batch):
        xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).view(-1, 1, 28, 28).to(clf.device)
        best3 = clf.model(xb).topk(3, dim=1).indices.cpu().numpy()
        target = y[i:i + batch, None]
        top1[i:i + batch] = best3[:, 0] == target[:, 0]
        top3[i:i + batch] = (best3 == target).any(axis=1)
    return top1, top3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--start", type=int, default=15000)
    parser.add_argument("--end", type=int, default=15500)
    parser.add_argument("--show", nargs="*", default=["cat"])
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    categories = load_categories()
    x = np.concatenate([download_first_n(c, args.end)[args.start:args.end] for c in categories])
    y = np.repeat(np.arange(len(categories)), args.end - args.start)
    print(f"{len(y):,} drawings ({args.start:,}-{args.end - 1:,} of each of {len(categories)} classes)")

    per_class = {}
    for path in args.checkpoints:
        clf = SketchClassifier(path, args.device)
        assert clf.categories == categories, f"{path} has a different class list"
        top1, top3 = per_class_hits(clf, x, y)
        per_class[path] = top1.reshape(len(categories), -1).mean(axis=1)
        shown = {c: {"top1": round(float(top1[y == categories.index(c)].mean()), 4),
                     "top3": round(float(top3[y == categories.index(c)].mean()), 4)} for c in args.show}
        print(json.dumps({"checkpoint": str(path), "top1": round(float(top1.mean()), 4),
                          "top3": round(float(top3.mean()), 4), **shown}))
    if len(args.checkpoints) > 1:
        diff = per_class[args.checkpoints[-1]] - per_class[args.checkpoints[0]]
        order = np.argsort(diff)
        print(f"classes better: {(diff > 0).sum()}, worse: {(diff < 0).sum()}, same: {(diff == 0).sum()}")
        print("most worse:", [(categories[i], round(float(diff[i]), 3)) for i in order[:5]])
        print("most better:", [(categories[i], round(float(diff[i]), 3)) for i in order[::-1][:5]])


if __name__ == "__main__":
    main()
