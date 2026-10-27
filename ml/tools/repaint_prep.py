"""Inputs for the digital repaint (owner, 2026-10-18), cut once so every machine paints exactly the same thing.

Every cat offered to 1_整體打底 or 5_毛流 by tools/review_export.py (the whole-face and fur models' cats; the owner's
manual review only removes cats later, so painting does not wait for it) is cut to the head's box
(review_export.face_crop's square), resized to the painter's 256 px, and stored with:

  target  the photo, uint8 (256, 256, 3); white where the subject mask says background
  tri     0 background (the subject mask says not cat: real silhouette, painted as white paper),
          1 cat (the mask says cat and the pixel is inside review_export.head_shape),
          2 uncertain (the mask says cat but outside the head shape, e.g. the chest under the chin or the back behind
            the ears, or beyond the photo): not learned (section 15.6; the chin is uncertain, owner 2026-10-18)
  points  the nine face points on the 256 canvas (CAT order), so any region (eyes, ears, nose and mouth, fur) and any
          zoomed part canvas can be computed later from the strokes alone
  box     the crop in the photo's pixels, and source / id, so the photo (and its appearance attributes) can be found

in shards of --shard cats (data/processed/repaint/inputs/shard_NNN.npz).

    PYTHONPATH=.:tools python tools/repaint_prep.py
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import review_export as RX

ML = Path(__file__).resolve().parents[1]
OUT = ML / "data" / "processed" / "repaint" / "inputs"
SIDE = 256


def prepare(item: tuple) -> dict:
    source, cid, photo_path, mask_path, p = item
    photo = Image.open(photo_path).convert("RGB")
    w, h = photo.size
    apple = np.asarray(Image.open(mask_path).convert("L").resize(photo.size, Image.BILINEAR), np.float32) / 255
    head = RX.head_shape(p, (h, w))
    ys, xs = np.nonzero(head)
    lo, hi = np.array([xs.min(), ys.min()], float), np.array([xs.max(), ys.max()], float) + 1
    box = RX.square((lo + hi) / 2, 1.08 * float((hi - lo).max()))
    side = box[2] - box[0]

    def cut(arr: np.ndarray, fill: int) -> np.ndarray:
        img = Image.fromarray(arr)
        sheet = Image.new(img.mode, (side, side), fill if img.mode == "L" else (fill,) * 3)
        sheet.paste(img, (-box[0], -box[1]))
        return np.asarray(sheet.resize((SIDE, SIDE), Image.LANCZOS if img.mode == "RGB" else Image.BILINEAR))

    rgb = np.asarray(photo)
    in_photo = cut(np.full((h, w), 255, np.uint8), 0) > 127
    cat_mask = cut((apple * 255).astype(np.uint8), 0) > 127
    in_head = cut((head * 255).astype(np.uint8), 0) > 127
    target = cut(rgb, 255).copy()
    background = in_photo & ~cat_mask
    target[background] = 255
    tri = np.full((SIDE, SIDE), 2, np.uint8)
    tri[background] = 0
    tri[cat_mask & in_head & in_photo] = 1
    pts = (p - np.array(box[:2], float)) * (SIDE / side)
    return {"source": source, "id": cid, "target": target, "tri": tri, "points": pts.astype(np.float32),
            "box": np.array(box, np.int32)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N cats only (a test)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    offered = {(r["source"], r["id"]) for r in csv.DictReader(open(RX.OUT / "manifest.csv"))
               if r["not_whole"] == "" or r["not_fur"] == ""}
    items = [c for c in RX.cats() if (c[0], c[1]) in offered]
    if a.limit:
        items = items[:a.limit]
    with Pool(a.workers) as pool:
        done = pool.map(prepare, items, chunksize=8)
    index = []
    for k in range(0, len(done), a.shard):
        part = done[k:k + a.shard]
        name = f"shard_{k // a.shard:03d}.npz"
        np.savez_compressed(a.out / name, source=np.array([d["source"] for d in part]), id=np.array([d["id"] for d in part]),
                            target=np.stack([d["target"] for d in part]), tri=np.stack([d["tri"] for d in part]),
                            points=np.stack([d["points"] for d in part]), box=np.stack([d["box"] for d in part]))
        index.append({"shard": name, "cats": len(part)})
    (a.out / "index.json").write_text(json.dumps({"cats": len(done), "shards": index}, indent=1))
    print(json.dumps({"cats": len(done), "shards": len(index)}))


if __name__ == "__main__":
    main()
