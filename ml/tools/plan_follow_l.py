"""Do L5's strokes follow its own sketch plan? (owner, 2026-10-20: prove the guesses about L5's weak eyes.)

Three kinds of painting, each read against the plan written at its own start (the first n validation cats):
  real        the training painting (L5's five binned layers) with the cat's plan: what following the plan looks like;
  free        L5 samples from nothing: the model writes the plan, then paints;
  real plan   L5 is handed the real cat's plan (its tokens up to PLAN_END are written in) and paints the rest.
If the free paintings miss their own eyes but the real-plan ones hit them, the plans are at fault; if both miss, the
strokes do not follow the plan. Measures per painting, for the two eye outlines of its plan:
  judge       the eye judge's confidence (tools/eye_judge.py), as in section 17.15;
  judge_off   distance from each eye the judge finds to the centre of the plan's outline for that eye, in eye distances
              (the plan's two outline centres);
  contrast    colour difference (RGB, 0-1, Euclidean) between the inside of the outline and a ring around it 0.15 eye
              distances wide: an eye differs from the fur around it;
  strokes     number of strokes (all five layers) centred inside the two outlines;
and, for the plans themselves (real vs free): eye distance, outline area / eye distance squared, and how far the eyes'
midpoint is from the head outline's centre (eye distances).

    PYTHONPATH=.:tools python tools/plan_follow_l.py --run L5 --n 128 --temperatures 1.0 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.ndimage import binary_dilation

import eye_judge as EJ
from sellanything_ml.sketch import brushgpt_l as G
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
BL = ML / "data" / "processed" / "brush_l"
YY, XX = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
CENTRES = np.c_[XX.ravel(), YY.ravel()]


def plan_stats(p: dict) -> dict | None:
    if len(p["eyes"]) != 2:
        return None
    c = np.array([e.mean(0) for e in p["eyes"]])
    d = float(np.linalg.norm(c[0] - c[1]))
    if d < 4:
        return None
    area = [abs(0.5 * np.sum(e[:, 0] * np.roll(e[:, 1], 1) - np.roll(e[:, 0], 1) * e[:, 1])) for e in p["eyes"]]
    head = np.concatenate(p["head"]) if p["head"] else None
    return {"centres": c, "d": d, "area": float(np.mean(area)) / d ** 2,
            "eyes_from_head_centre": float(np.linalg.norm(c.mean(0) - head.mean(0)) / d) if head is not None else np.nan}


def eye_contrast(img: np.ndarray, p: dict, d: float) -> tuple[float, np.ndarray]:
    """Mean over the plan's two eyes of the colour difference inside vs a ring around the outline; and the union of
    the two insides (SIDE, SIDE)."""
    ring_w = max(2, int(round(0.15 * d)))
    contrast, inside_any = [], np.zeros((B.SIDE, B.SIDE), bool)
    for e in p["eyes"]:
        m = Polygon(e).contains_points(CENTRES).reshape(B.SIDE, B.SIDE)
        inside_any |= m
        ring = binary_dilation(m, iterations=ring_w) & ~m
        contrast.append(float(np.linalg.norm(img[m].mean(0) - img[ring].mean(0))) if m.sum() >= 3 and ring.sum() >= 3 else np.nan)
    return float(np.nanmean(contrast)) if np.isfinite(contrast).any() else np.nan, inside_any


def measure(img: np.ndarray, seq: np.ndarray, judge_pts: np.ndarray, judge_conf: np.ndarray) -> dict | None:
    p = L.decode_plan(seq)
    st = plan_stats(p)
    if st is None:
        return None
    contrast, inside_any = eye_contrast(img, p, st["d"])
    xy = np.concatenate([L.stroke_values(r, k)["xy"] for k, r in enumerate(L.decode_layers(seq)[:5]) if len(r)])
    ix, iy = np.clip(xy[:, 0].astype(int), 0, B.SIDE - 1), np.clip(xy[:, 1].astype(int), 0, B.SIDE - 1)
    strokes = int(inside_any[iy, ix].sum())
    off = np.linalg.norm(judge_pts[:2] - st["centres"], axis=1).mean() / st["d"]
    return {"judge": float(judge_conf[:2].mean()), "judge_off": float(off), "contrast": contrast,
            "strokes": strokes, "plan_d": st["d"], "plan_area": st["area"], "plan_eyes_from_head": st["eyes_from_head_centre"]}


def summary(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        out[k] = [round(float(v.mean()), 4), round(float(v.std() / np.sqrt(len(v))), 4)]
    return out


def score(kind: str, imgs: np.ndarray, seqs: list[np.ndarray], photo: np.ndarray | None = None) -> tuple[dict, list]:
    pts, conf = EJ.judge(imgs.astype(np.float32))
    rows, skipped = [], 0
    for i, s in enumerate(seqs):
        r = measure(imgs[i], s, pts[i], conf[i])
        if r is None:
            skipped += 1
            continue
        if photo is not None:                                       # the photo, read against the same plan
            p = L.decode_plan(s)
            r["photo_contrast"] = eye_contrast(photo[i], p, plan_stats(p)["d"])[0]
        rows.append(r)
    s_ = summary(rows)
    s_["no_two_eyes"] = skipped
    print(json.dumps({kind: s_}), flush=True)
    return s_, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="L5")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.8])
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    z = np.load(ML / "data/kaggle/brush_l5/data/data_L5.npz")
    val = [str(n) for n, s in zip(z["names"], z["split"]) if s == "val"][:a.n]
    real = [np.load(BL / "cats" / f"{n}.npz") for n in val]
    real_seq = [r["tokens"] for r in real]
    real_img = np.stack([r["painting"] / 255 for r in real])
    photo = np.stack([np.load(next((ML / "data/processed/repaint").glob(f"**/shard_*/{n}.npz")))["target"] / 255 for n in val])
    res = {"cats": len(val)}
    res["real"], rows_real = score("real", real_img, real_seq, photo)
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", G.device())
    prefix = [s[:int(np.flatnonzero(s == L.SP["PLAN_END"])[0]) + 1] for s in real_seq]
    out_dir = G.RUN_DIR / a.run / "plan_follow"
    out_dir.mkdir(exist_ok=True)
    keep = {"real": rows_real}
    for t in a.temperatures:
        for kind, pre in (("free", None), ("real_plan", prefix)):
            seqs = G.sample(model, len(val), t, a.seed, max_len=3072, prefix=pre)
            imgs = np.stack([L.render(L.decode_layers(s)[:5]) for s in seqs])
            key = f"{kind}_t{t}"
            res[key], keep[key] = score(key, imgs, seqs)
            np.savez_compressed(out_dir / f"{key}_s{a.seed}.npz",
                                tokens=np.array([np.pad(s, (0, 3072 - len(s)), constant_values=-1) for s in seqs]),
                                lengths=np.array([len(s) for s in seqs]), paintings=(imgs * 255).round().astype(np.uint8))
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    (out_dir / "rows.json").write_text(json.dumps(keep))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
