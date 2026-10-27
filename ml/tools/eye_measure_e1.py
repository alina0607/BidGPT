"""Measures of the eye model's paintings that the eye judge does not see (owner, 2026-10-22), on the images
tools/eye_eval_e1.py saved (CPU only, nothing is sampled again, every cat is counted). Each eye is the outline in L8's
plan on the 256 px face canvas; its pupil is the plan's pupil ellipse nearest to it.

  speckle   mean size of the colour detail inside the iris (outline minus pupil, eroded 2 px so the rim does not
            count): |image - its 5 x 5 box blur| per pixel (RGB 0-1, Euclidean), averaged over the iris
  lr_diff   distance between the two irises' mean colours (RGB 0-1, Euclidean)
  dark      share of the eye (whole outline) with luminance under 0.15
  pupil     the plan's pupil area over the eye outline's area (what L8 asked for)
  profile   mean luminance in five rings from the plan pupil's centre outwards (distance over the eye's radius,
            sqrt(outline area / pi): 0-0.2 ... 0.8-1.0) inside the outline: a dark centre fading lighter outwards
            is what an eye looks like (owner, 2026-10-22)
  contrast  iris luminance minus the centre's (outline minus iris, under 0.6 eye radii: the pupil)
  outward   share of eyes whose first four rings get lighter ring by ring

Conditions: real (L8 alone, + E1 at 0.8 and 1.0, + the eye material = the training target) and free (L8 sampled at
1.0 / 0.8 / 0.6, alone and + E1). Means with standard errors; for dark, also its correlation with pupil and the
share of black-hole eyes (dark over 0.6).

    PYTHONPATH=.:tools python tools/eye_measure_e1.py --run E1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import eyegpt_l8 as EG

SIDE = 256
DARK, HOLE = 0.15, 0.6


Masks = list[tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]
RINGS = 5


def eye_masks(plan: dict[str, object]) -> Masks | None:
    """(eye mask, iris mask, pupil share, ring index (-1 outside the rings), centre mask) for each of the plan's two
    eyes; None without two outlines."""
    if len(plan["eyes"]) != 2:
        return None
    ys, xs = np.mgrid[0:SIDE, 0:SIDE] + 0.5
    out = []
    for xy in plan["eyes"]:
        im = Image.new("L", (SIDE, SIDE), 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in (xy + 0.5)], fill=1)
        eye = np.asarray(im, bool)
        pupil = np.zeros_like(eye)
        share = 0.0
        cx, cy = xy.mean(0) + 0.5
        if plan["pupils"]:
            c = xy.mean(0) + 0.5
            p = min(plan["pupils"], key=lambda q: np.linalg.norm(q["centre"] + 0.5 - c))
            (cx, cy), (rx, ry), a = p["centre"] + 0.5, np.maximum(p["radii"], 0.5), p["angle"]
            u = (xs - cx) * np.cos(a) + (ys - cy) * np.sin(a)
            v = -(xs - cx) * np.sin(a) + (ys - cy) * np.cos(a)
            pupil = (u / rx) ** 2 + (v / ry) ** 2 <= 1
            share = float(np.pi * rx * ry / max(eye.sum(), 1))
        iris = ndimage.binary_erosion(eye, iterations=2) & ~pupil
        rr = np.hypot(xs - cx, ys - cy) / np.sqrt(max(eye.sum(), 1) / np.pi)
        ring = np.where(eye & (rr < 1), np.minimum((rr * RINGS).astype(int), RINGS - 1), -1)
        out.append((eye, iris, share, ring, eye & ~iris & (rr < 0.6)))
    return out


def measure(img: np.ndarray, masks: Masks | None) -> dict[str, object] | None:
    if masks is None:
        return None
    x = img.astype(np.float32) / 255
    detail = np.linalg.norm(x - ndimage.uniform_filter(x, size=(5, 5, 1)), axis=-1)
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    eyes = []
    for eye, iris, share, ring, centre in masks:
        ok = iris.sum() >= 10
        prof = [float(lum[ring == k].mean()) if (ring == k).sum() > 3 else np.nan for k in range(RINGS)]
        eyes.append({"speckle": float(detail[iris].mean()) if ok else np.nan,
                     "colour": x[iris].mean(0) if ok else np.full(3, np.nan),
                     "dark": float((lum[eye] < DARK).mean()), "pupil": share, "profile": prof,
                     "contrast": float(lum[iris].mean() - lum[centre].mean()) if ok and centre.sum() > 3 else np.nan})
    return {k: [e[k] for e in eyes] for k in ("speckle", "dark", "pupil", "profile", "contrast")} | \
        {"lr_diff": float(np.linalg.norm(eyes[0]["colour"] - eyes[1]["colour"]))}


def mean_se(v: list[float] | np.ndarray) -> list[float | int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(v.mean()), 4), round(float(v.std(ddof=1) / np.sqrt(len(v))), 4), len(v)]


def summary(rows: list[dict[str, object] | None]) -> dict[str, object]:
    rows = [r for r in rows if r is not None]
    sp = [s for r in rows for s in r["speckle"]]
    dk = np.array([d for r in rows for d in r["dark"]])
    pu = np.array([p for r in rows for p in r["pupil"]])
    hole = dk > HOLE
    return {"speckle": mean_se(sp), "lr_diff": mean_se([r["lr_diff"] for r in rows]), "dark": mean_se(dk),
            "dark_vs_pupil_r": round(float(np.corrcoef(dk, pu)[0, 1]), 3), "holes": [int(hole.sum()), len(dk)],
            "pupil_of_holes": round(float(pu[hole].mean()), 3) if hole.any() else None,
            "pupil_of_others": round(float(pu[~hole].mean()), 3),
            "profile": [round(float(v), 3) for v in np.nanmean([p for r in rows for p in r["profile"]], 0)],
            "contrast": mean_se([c for r in rows for c in r["contrast"]]),
            "outward": round(float(np.mean([bool(np.all(np.diff(p[:4]) > 0)) for r in rows for p in r["profile"]])), 3)}


def main() -> None:
    """Measure every saved painting of the run's eval and write <run>/eval/measures.json."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="E1")
    a = ap.parse_args()
    ev = EG.RUN_DIR / a.run / "eval"
    res: dict[str, dict] = {}
    r = np.load(ev / "real_images.npz")
    masks = [eye_masks(L.decode_plan(np.load(EG.BASES / f"{n}.npz")["tokens"])) for n in r["names"]]
    res["real"] = {k: summary([measure(img, m) for img, m in zip(r[k], masks)])
                   for k in ("alone", "with_t0.8", "with_t1.0", "material")}
    f = np.load(ev / "free_images.npz")
    res["free"] = {}
    for t in ("1.0", "0.8", "0.6"):
        masks = [eye_masks(L.decode_plan(s)) for s in f[f"tokens_t{t}"]]
        for k in ("alone", "with"):
            res["free"][f"{k}_l8t{t}"] = summary([measure(img, m) for img, m in zip(f[f"{k}_t{t}"], masks)])
    (ev / "measures.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
