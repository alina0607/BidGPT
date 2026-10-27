"""Which parts of the face does a Brush GPT run predict worst? Teacher-forced loss of the validation paintings, split
by where each stroke sits on the face.

Every stroke goes to one region by its centre and the painting's nine face points, in this order of priority:
    eyes         within EYE_ZONE (0.35) eye distances of either eye (the eye region of the eye-weighted loss)
    nose_mouth   within 0.35 eye distances of the nose tip or the mouth; the nose tip is estimated as
                 eye midpoint + 0.7 (mouth - eye midpoint), as in oilprocess.Face
    ears         inside either ear triangle (outer base, tip, inner base)
    other        everything else: forehead, cheeks, the outline of the fur and the background
All five tokens of a stroke belong to its region. For each region: the share of strokes, the mean loss per token,
the loss of each field, and the loss in each stage. Nothing is sampled; the model only scores the real paintings.

    nice -n 19 env BRUSHGPT_DEVICE=cpu PYTHONPATH=. python tools/region_losses.py --run I_aligned_points_eyes \
        --data data/processed/brushgpt/tokens_runs_HI.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import brushgpt as G
from sellanything_ml.sketch import brushtokens as T
from sellanything_ml.sketch import catbrush as B

REGIONS = ("eyes", "nose_mouth", "ears", "other")
NOSE_MOUTH_ZONE = 0.35


def inside_triangle(px: np.ndarray, py: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """(n, s) bool: points (px, py) of each row inside that row's triangle tri (n, 3, 2)."""
    def side(a, b):
        return (b[:, None, 0] - a[:, None, 0]) * (py - a[:, None, 1]) - (b[:, None, 1] - a[:, None, 1]) * (px - a[:, None, 0])
    s1, s2, s3 = side(tri[:, 0], tri[:, 1]), side(tri[:, 1], tri[:, 2]), side(tri[:, 2], tri[:, 0])
    return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))


def stroke_regions(rows: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position of each stroke's first token (s,) and the region index of each stroke (n, s)."""
    _, _, slot = G.layout_of(rows)
    xs = np.flatnonzero(slot == 1 + G.FIELDS.index("x"))
    cx = (rows[:, xs] - T.OFFSET["x"] + 0.5) * B.SIDE / T.NXY
    cy = (rows[:, xs + 1] - T.OFFSET["y"] + 0.5) * B.SIDE / T.NXY
    p = points.astype(float)
    d = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)[:, None]
    mid = (p[:, 0] + p[:, 1]) / 2
    nose = mid + 0.7 * (p[:, 2] - mid)

    def near(c, r):
        return np.hypot(cx - c[:, None, 0], cy - c[:, None, 1]) < r * d

    eyes = near(p[:, 0], G.EYE_ZONE) | near(p[:, 1], G.EYE_ZONE)
    nose_mouth = near(nose, NOSE_MOUTH_ZONE) | near(p[:, 2], NOSE_MOUTH_ZONE)
    ears = inside_triangle(cx, cy, p[:, [3, 4, 5]]) | inside_triangle(cx, cy, p[:, [6, 7, 8]])
    region = np.full(cx.shape, REGIONS.index("other"))
    region[ears] = REGIONS.index("ears")
    region[nose_mouth] = REGIONS.index("nose_mouth")
    region[eyes] = REGIONS.index("eyes")
    return xs, region


def token_losses(model, rows: np.ndarray, dev: str, batch: int = 8) -> np.ndarray:
    """(n, len - 1) teacher-forced cross-entropy of every predicted token, with the run's field restriction."""
    import torch
    import torch.nn.functional as F
    mask = G.allowed(dev, model.brush_cfg.points) if model.brush_cfg.heads else None
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            t = torch.from_numpy(rows[i:i + batch]).to(dev)
            logits = G.field_logits(model(t[:, :-1])[0].float(), 1, mask)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1), reduction="none")
            out.append(ce.view(len(t), -1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dev = G.device()
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", dev)
    cfg = model.brush_cfg
    rows, ids = G.load_rows(a.data, a.split, 0, cfg.order, points=cfg.points)
    points = G.load_points(a.data, a.split)
    t0 = time.time()
    ce = token_losses(model, rows, dev)
    xs, region = stroke_regions(rows, points)
    stage_of_stroke = np.repeat(np.arange(len(G.STROKES)), G.STROKES)

    out = {"run": a.run, "data": str(a.data), "split": a.split, "cats": len(rows), "device": dev,
           "zones": {"eyes": G.EYE_ZONE, "nose_mouth": NOSE_MOUTH_ZONE}, "regions": {}}
    for r, name in enumerate(REGIONS):
        sel = region == r                                     # (n, s) strokes of this region
        fields = {f: float(ce[:, xs + k - 1][sel].mean()) for k, f in enumerate(G.FIELDS)}   # ce index = position - 1
        tok = np.stack([ce[:, xs + k - 1] for k in range(T.TOKENS_PER_STROKE)], -1)          # (n, s, 5)
        stages = {}
        for k in range(len(G.STROKES)):
            m = sel & (stage_of_stroke == k)[None]
            stages[str(k + 1)] = {"strokes": int(m.sum()), "loss": float(tok[m].mean()) if m.any() else None}
        out["regions"][name] = {"strokes": int(sel.sum()), "share": float(sel.mean()), "loss": float(tok[sel].mean()),
                                "fields": fields, "stages": stages}
    all_tok = np.stack([ce[:, xs + k - 1] for k in range(T.TOKENS_PER_STROKE)], -1)
    out["all_strokes_loss"] = float(all_tok.mean())
    out["seconds"] = round(time.time() - t0, 1)
    path = G.RUN_DIR / a.run / f"region_losses_{a.split}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v["loss"], 4), round(v["share"], 3)) for k, v in out["regions"].items()}),
          "all", round(out["all_strokes_loss"], 4), f"{out['seconds']} s ->", path)


if __name__ == "__main__":
    main()
