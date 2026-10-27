"""Does the painting bring back more of the photo's fine lines with more hair strokes? (owner, 2026-10-19: show a tabby
with all 500 hairs next to the photo, to decide whether coverage should go up.) The same eight digital layers as the
repaint (tools/repaint.py), with the three hair layers (fur, dark hairs, light hairs) multiplied by --times, painted on
the CPU for a few tabbies already painted at x1 on the Mac. A test, not training data (runs/hair_count_test/).

Measured on the cat pixels: L1, and per line map (tools/repaint_tidy.line_maps: texture, thin dark lines, thin light
lines) the recall sum(min(photo, painting)) / sum(photo), i.e. how much of the photo's fine-line mass the painting
has at the same place.

    PYTHONPATH=.:tools python tools/hair_count_test.py --names pixabay_cat_004798 --times 2 --out runs/hair_count_test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import repaint as R
import repaint_tidy as T

HAIR_KINDS = {"fur", "dark hairs", "light hairs"}


def line_recall(painting_u8: np.ndarray, target_u8: np.ndarray, tri: np.ndarray) -> dict[str, float]:
    ph, pa = T.line_maps(target_u8, tri), T.line_maps(painting_u8, tri)
    return {n: round(float(np.minimum(ph[k], pa[k]).sum() / ph[k].sum()), 3) for k, n in ((5, "texture"), (6, "dark"), (7, "light"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--times", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    R.LAYERS[:] = [(nm, ln, w, n * a.times if nm in HAIR_KINDS else n, e, k) for nm, ln, w, n, e, k in R.LAYERS]
    rows = []
    for name in a.names:
        f = next((R.ML / "data/processed/repaint/mac").glob(f"shard_*/afhq__{name}.npz"))
        old = R.load(f)
        t0 = time.time()
        record, final = R.paint_one(old["target"], old["tri"], "cpu")
        painting = (final.clip(0, 1) * 255).round().astype(np.uint8)
        cat_px = old["tri"] == 1
        l1 = lambda p: round(float(np.abs(p.astype(np.float32) / 255 - old["target"].astype(np.float32) / 255)[cat_px].mean()), 4)  # noqa: E731
        item = {k: old[k] for k in ("target", "tri", "points", "box")}
        R.save(a.out / f"{name}_x{a.times}.npz", record, final, item, {"times": a.times, "test": "hair_count_test"})
        row = {"name": name, "times": a.times, "strokes": int(sum(len(s.xy) for _, s in record)), "seconds": round(time.time() - t0),
               "l1": l1(painting), "recall": line_recall(painting, old["target"], old["tri"]),
               "x1": {"strokes": 1848, "l1": l1(old["painting"]), "recall": line_recall(old["painting"], old["target"], old["tri"])}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / f"summary_x{a.times}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
