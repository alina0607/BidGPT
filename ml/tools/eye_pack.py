"""Everything the eye-material fit needs for each cat, packed so it can run where the photos are not (owner, 2026-10-24:
repaint every cat's eyes in the digital style on Kaggle overnight). One tools/eye_data_l8.Cat per cat, built here from
the photo, its mask, the CAT/AFHQ repaint file and the L8 base, and stored as: the photo crop on the eye canvas
(target, uint8) and its subject mask (uint8), L8's painting on the crop (base, uint8), the photo's eye points on the
crop, the eye distance, the crop and face boxes and the plan's two eye outlines on the crop. eye_data_l8 --pack reads
these instead of the photos; both the Mac and Kaggle fit from the pack, so the material is the same wherever a cat is
painted. The cats are eye_data_l8's: every AFHQ cat with an L8 base whose eyes the owner's review keeps.

    PYTHONPATH=.:tools python tools/eye_pack.py --part 0/4 --out data/kaggle/eye_digital/pack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eye_data_l8 as D

CHUNK = 350                                                     # cats per file


def u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(a, np.float32), 0, 1) * 255).round().astype(np.uint8)


def pack(cat: D.Cat) -> dict[str, np.ndarray]:
    return {"target": u8(cat.target), "mask": u8(cat.photo.mask.numpy()), "base": u8(cat.base),
            "eyes": cat.eyes.astype(np.float32), "d": np.float32(cat.d), "ebox": np.asarray(cat.ebox, np.float64),
            "fbox": np.asarray(cat.fbox, np.float64), "outline_0": cat.outlines[0].astype(np.float32),
            "outline_1": cat.outlines[1].astype(np.float32)}


def names() -> list[str]:
    keep = D.eyes_ok()
    return [x for x in sorted(f.name[:-4] for f in D.BASES.glob("afhq__*.npz") if not f.name.endswith(".tmp.npz"))
            if x in keep]


def main() -> None:
    """Pack the i-th of k interleaved parts of the cats into files of CHUNK cats."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chunks", type=int, default=0, help="stop after this many files (memory grows per cat)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    i, k = map(int, a.part.split("/"))
    mine = names()[i::k]
    rel = D.photo_paths()
    written = 0
    for c in range(0, len(mine), CHUNK):
        path = a.out / f"pack_{i}of{k}_{c // CHUNK:02d}.npz"
        if path.exists():
            continue
        rows = [pack(D.Cat(n, rel)) for n in mine[c:c + CHUNK]]
        np.savez_compressed(path.with_suffix(".tmp.npz"), names=np.array(mine[c:c + CHUNK]),
                            **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
        path.with_suffix(".tmp.npz").rename(path)
        print(json.dumps({"file": path.name, "cats": len(rows)}), flush=True)
        written += 1
        if a.max_chunks and written >= a.max_chunks:
            print(json.dumps({"stopped": "max_chunks"}), flush=True)
            return
    print(json.dumps({"done": len(mine)}), flush=True)


if __name__ == "__main__":
    main()
