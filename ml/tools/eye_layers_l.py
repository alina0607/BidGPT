"""The eye model's data on Brush GPT L (owner, 2026-10-21): tools/eye_layers.py's four eye layers (outline, iris, pupil,
highlight) redone on the new base and outline, on the first 24 validation cats, to see the pictures and the measures
before making eye data for every cat.

What changes from tools/eye_layers.py (zh notebook section 14.5):
  base     the L8 training painting (data/processed/brush_l/cats8: the digital repaint's eight layers, binned, as L8
           learns them), re-rendered as vector strokes in the zoomed eye crop (positions, sizes and soft edges scaled),
           instead of the old painter's four stages;
  outline  the eye outline in the cat's sketch plan (SAM on a zoomed crop, 16 points, tools/sketch_plan.py), smoothed
           to a closed curve (Fourier series of the radius, 4 terms), instead of tracing rays from the eye point.
To see which change does what, every cat is painted four ways (base old/L8 x outline rays/plan), with the same four
layers ("layers" of tools/eye_layers.py) fitted to the same photo crop, and measured the same way: L1 over the crop,
eye-disc L1 (within 0.35 eye distances), rim strength (eye_crop_styles.rim_strength, independent of any outline), and
overflow (L1 1.5-5 px outside the plan outline, where colour spilling out of the eye shows). CPU, 2 threads (as 14.5).

    PYTHONPATH=.:tools python tools/eye_layers_l.py --n 24 --out runs/eye_layers_l/24cats
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

import eye_crop_styles as E
import eye_layers as EL
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

CATS8 = E.ML / "data" / "processed" / "brush_l" / "cats8"
REPAINT = E.ML / "data" / "processed" / "repaint"
HARM = 4


def repaint_file(name: str) -> Path:
    return next(p for root in ("kaggle_t4_run1/out", "kaggle_t4_run2/out", "mac")
                for p in (REPAINT / root).glob(f"shard_*/{name}.npz"))


def to_crop(xy_face: np.ndarray, fbox: np.ndarray, ebox: tuple) -> np.ndarray:
    """Face-canvas pixels (continuous) -> eye-crop pixels."""
    k_face, k_eye = (fbox[2] - fbox[0]) / B.SIDE, B.SIDE / (ebox[2] - ebox[0])
    return ((fbox[:2] + xy_face * k_face) - np.array(ebox[:2])) * k_eye


def l8_on_crop(tokens: np.ndarray, fbox: np.ndarray, ebox: tuple, pad: bool = True) -> np.ndarray:
    """The L8 painting of a sequence rendered into the eye crop: every binned stroke's centre, size and soft edge scaled
    by the zoom, each layer with its own brush (blending layers smudge), in order.

    pad (2026-10-21): painted on a canvas widened by the largest stroke window on every side, then cut to the crop.
    A stroke's centre is a sigmoid of [0, SIDE) (catbrush.Strokes), so on the bare crop a stroke centred off the crop
    is clamped onto its edge instead of falling off it; pad=False keeps that old behaviour (zh notebook 17.23-17.24
    were measured with it). Strokes whose window misses the crop are skipped (they paint nothing on it)."""
    scale = (fbox[2] - fbox[0]) / B.SIDE * B.SIDE / (ebox[2] - ebox[0])
    layers = L.decode_layers(tokens)
    brushes = []
    for k in range(len(layers)):
        br = L.brush(k)
        br = {**br, "edge_px": br["edge_px"] * scale}
        with E.brush_settings(br):
            base = (float(br["length"] * scale), float(br["width"] * scale))
            brushes.append((br, base, B.window_half(base) + 1))
    side0 = B.SIDE
    p = max(h for _, _, h in brushes) if pad else 0
    side = side0 + 2 * p
    ys, xs = torch.meshgrid(torch.arange(side) + 0.5, torch.arange(side) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(side, side, 3)
    values = []                                             # in crop pixels, before SIDE is widened below
    for k, rows in enumerate(layers):
        v = L.stroke_values(rows, k) if len(rows) else None
        if v is not None:
            v["xy"] = to_crop(v["xy"], fbox, ebox) + p
            v["length_width"] = v["length_width"] * scale
            v = {key: np.asarray(val, np.float32) for key, val in v.items()}
            if pad:
                half = brushes[k][2]
                on = np.all((v["xy"] > p - half) & (v["xy"] < p + side0 + half), 1)
                v = {key: val[on] for key, val in v.items()} if on.any() else None
        values.append(v)
    try:
        B.SIDE = side                                       # catbrush maps stroke centres through SIDE
        with torch.no_grad():
            for (br, base, _), v in zip(brushes, values):
                if v is None:
                    continue
                with E.brush_settings(br):
                    s = B.Strokes.from_stage(v, base)
                    canvas = E.blend_composite(canvas, s) if br["blend"] else B.composite(canvas, s, grid)
    finally:
        B.SIDE = side0
    return canvas[p:p + side0, p:p + side0].numpy()


def plan_outlines(tokens: np.ndarray, fbox: np.ndarray, ebox: tuple) -> list[np.ndarray]:
    """The plan's two eye outlines in the eye crop, each smoothed to a closed curve: the radius around the outline's
    centre as a Fourier series (HARM terms) fitted to its 16 points; (OUTLINE_POINTS, 4): x, y, tangent, strength."""
    out = []
    for e in L.decode_plan(tokens)["eyes"]:
        p = to_crop(e + 0.5, fbox, ebox)                                  # plan points are whole pixels (centres)
        c = p.mean(0)
        th, r = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), np.hypot(*(p - c).T)
        des = lambda a: np.column_stack([np.ones_like(a)] + [f(h * a) for h in range(1, HARM + 1) for f in (np.cos, np.sin)])  # noqa: E731
        coef = np.linalg.lstsq(des(th), r, rcond=None)[0]
        dense = np.linspace(0, 2 * np.pi, EL.OUTLINE_POINTS, endpoint=False)
        rad = des(dense) @ coef
        x, y = c[0] + rad * np.cos(dense), c[1] + rad * np.sin(dense)
        tangent = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)) % np.pi
        out.append(np.c_[x, y, tangent, np.ones_like(x)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(E.SPLIT)
    ids = [str(i) for i in z["ids"][z["split"] == "val"]]
    rows = []
    for rel in ids:
        if len(rows) == a.n:
            break
        name = Path(rel).stem
        if not (CATS8 / f"afhq__{name}.npz").exists() or not (EL.PAINT_DIR / f"{name}.npz").exists():
            continue
        rp = np.load(repaint_file(f"afhq__{name}"))
        fbox = rp["box"].astype(float)
        pts = fbox[:2] + rp["points"] * (fbox[2] - fbox[0]) / B.SIDE          # face points in the photo's pixels
        ebox = E.eye_box(pts)
        photo = B.load(str(E.AFHQ / rel), str(E.MASKS / f"{name}_mask1.png"), ebox)
        target = photo.target.numpy()
        eyes = (pts[:2] - np.array(ebox[:2])) * (B.SIDE / (ebox[2] - ebox[0]))
        d = float(np.linalg.norm(eyes[0] - eyes[1]))
        tokens = np.load(CATS8 / f"afhq__{name}.npz")["tokens"]
        bases = {"old": EL.face_on_crop(name, ebox), "l8": l8_on_crop(tokens, fbox, ebox)}
        outlines = {"rays": E.eye_contours(target, eyes, d), "plan": plan_outlines(tokens, fbox, ebox)}
        inside = EL.interior(outlines["plan"])                             # the measuring region: the plan outline
        dist_out = distance_transform_edt(~inside)
        band = (dist_out > 1.5) & (dist_out <= 5)
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        disc = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < 0.35 * d
        row = {"id": rel, "eye_dist_px": round(d, 1), "photo_rim": round(E.rim_strength(target, eyes, d), 4),
               "plan_inside_px": int(inside.sum())}
        images = {"photo": target}

        def measure(img: np.ndarray) -> dict:
            err = np.abs(img - target)
            return {"l1": round(float(err.mean()), 4), "eye_l1": round(float(err[disc].mean()), 4),
                    "overflow_l1": round(float(err[band].mean()), 4), "rim": round(E.rim_strength(img, eyes, d), 4)}

        for b, base in bases.items():
            images[f"{b}_face"] = base
            row[f"{b}_face"] = measure(base)
            for o, cont in outlines.items():
                t0 = time.time()
                specs = EL.variants(target, eyes, d, cont)[0]["layers"]
                final = E.paint(photo.target, photo.mask, specs, start=base)[-1]
                images[f"{b}_{o}"] = final
                row[f"{b}_{o}"] = {**measure(final), "seconds": round(time.time() - t0, 1)}
        np.savez_compressed(a.out / f"{name}.npz", eyes=eyes, plan_0=outlines["plan"][0], plan_1=outlines["plan"][1],
                            rays_0=outlines["rays"][0], rays_1=outlines["rays"][1], **images)
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = ["old_face", "old_rays", "old_plan", "l8_face", "l8_rays", "l8_plan"]
    summary = {k: {m: round(float(np.mean([r[k][m] for r in rows])), 4) for m in ("l1", "eye_l1", "overflow_l1", "rim")}
               for k in keys}
    summary["photo_rim"] = round(float(np.mean([r["photo_rim"] for r in rows])), 4)
    (a.out / "summary.json").write_text(json.dumps({"cats": rows, "summary": summary}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
