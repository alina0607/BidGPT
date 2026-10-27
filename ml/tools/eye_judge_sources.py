"""Are the eyes of the CAT-dataset repaints as clear as AFHQ's? (owner, 2026-10-21: would more CAT cats make L draw
better eyes?) The eye judge (tools/eye_judge.py) on the repainted training paintings (all eight layers) and on their
photos, for the CAT cats repainted so far and the same number of AFHQ cats; also the face size (eye distance in the
256-pixel canvas). CPU.

    PYTHONPATH=.:tools python tools/eye_judge_sources.py --n 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eye_judge as EJ

ML = Path(__file__).resolve().parents[1]
RP = ML / "data" / "processed" / "repaint"


def load(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paint, photo, d = [], [], []
    for f in files:
        z = np.load(f)
        paint.append(z["painting"] / 255)
        photo.append(z["target"] / 255)
        d.append(float(np.linalg.norm(z["points"][0] - z["points"][1])))
    return np.stack(paint).astype(np.float32), np.stack(photo).astype(np.float32), np.array(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    cat = sorted(f for f in (RP / "mac_cat").glob("shard_*/*.npz") if not f.name.endswith(".tmp.npz"))[:a.n]
    afhq = sorted(f for f in (RP / "kaggle_t4_run1/out").glob("shard_00[0-2]/*.npz") if not f.name.endswith(".tmp.npz"))
    rng = np.random.default_rng(0)
    afhq = [afhq[i] for i in sorted(rng.choice(len(afhq), len(cat), replace=False))]
    out = {}
    for name, files in (("cat", cat), ("afhq", afhq)):
        paint, photo, d = load(files)
        out[name] = {"n": len(files), "painting": EJ.eye_score(paint), "photo": EJ.eye_score(photo),
                     "eye_distance_px": [round(float(d.mean()), 1), round(float(np.median(d)), 1)]}
        print(json.dumps({name: out[name]}), flush=True)
    (ML / "runs" / "brush_l_sources").mkdir(parents=True, exist_ok=True)
    (ML / "runs" / "brush_l_sources" / "eye_judge.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
