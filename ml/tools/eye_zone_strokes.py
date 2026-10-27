"""How many strokes does a run paint in the eye region? Counted per stage on the paintings a run generated (the tokens
tools/brushgpt_fid_diversity.py saved) and on the real validation paintings, with the eye region of aligned faces:
the aligned eye points (the data file's align_mid ± align_dist / 2), radius brushgpt.EYE_ZONE eye distances, the
region eye weighting uses. Eye weighting on a stroke's x and y tokens makes a model paint there more often (runs I and
J about twice as often as the real paintings, 2026-10-18); this counts it.

    PYTHONPATH=. python tools/eye_zone_strokes.py --data data/processed/brushgpt/tokens_runs_HI.npz \
        --runs I_aligned_points_eyes J_cat_aligned_points_eyes --out runs/brushgpt/eye_zone_strokes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import brushgpt as G
from sellanything_ml.sketch import brushtokens as T


def per_stage(rows: np.ndarray, eyes: np.ndarray, dist: float) -> np.ndarray:
    """(n, stages) strokes whose centre lies within EYE_ZONE * dist of either eye."""
    out = []
    for row in rows:
        stages = T.decode(G.strokes_only(row).tolist())
        out.append([int((np.linalg.norm(st["xy"][:, None, :] - eyes[None], axis=2).min(1) < G.EYE_ZONE * dist).sum())
                    for st in stages])
    return np.array(out)


def summary(counts: np.ndarray) -> dict:
    total = counts.sum(1)
    return {"paintings": len(counts), "per_stage": counts.mean(0).round(1).tolist(), "total_mean": round(float(total.mean()), 1),
            "total_se": round(float(total.std(ddof=1) / np.sqrt(len(total))), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="an aligned token file (it holds align_mid, align_dist)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.data)
    mid, dist = z["align_mid"], float(z["align_dist"])
    eyes = np.array([[mid[0] - dist / 2, mid[1]], [mid[0] + dist / 2, mid[1]]])
    report = {"data": str(a.data), "eyes": eyes.round(2).tolist(), "radius_px": round(G.EYE_ZONE * dist, 2),
              "real_val": summary(per_stage(z["tokens_overlap"][z["split"] == "val"], eyes, dist))}
    for run in a.runs:
        f = G.RUN_DIR / run / "fid_diversity_t1.0.npz"
        report[run] = summary(per_stage(np.load(f)["tokens"], eyes, dist)) if f.exists() else "no saved samples"
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
