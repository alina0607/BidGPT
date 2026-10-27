"""Why the eye material was painted again (owner, 2026-10-26: show, with pictures and numbers, what was wrong with the
earlier materials and what the new one fixes). Three materials of the same cats, each against the photo's eye crop
(tools/eye_pack.py's files):

  layers   data/processed/brush_l/eye_l8           E1-E5's material: 4 layers, rim = "outline", L8 base
  digital  data/processed/brush_l/eye_l8_digital   Kaggle 10-24: digital painter, rim fitted freely, L8 base
  new      data/processed/brush_l/eye_l8_rim_hybrid  Mac 10-25/26: hybrid base + rim along the photo's rim line

Measured per cat (both eyes together) on the finished painting, around the photo's rim line (eye_layers.rim_on_photo,
the line the new rim follows; on some slit pupils it is pulled in towards the pupil):
  rim_green   share of the rim band (within 2.5 px of the photo's rim line) that is green: G above both R and B by
              more than 0.04 (the photo's own share is reported as "photo")
  rim_light   mean luminance of the rim band minus the photo's (the photo's rim is dark; above 0 = a lighter rim)
  fur_l1      mean |painting - photo| on the fur ring 4-20 px outside the photo's rim line
  eye_l1      mean |painting - photo| inside the photo's rim line

How far the rim strokes stick out is not measured: the rim layer's strokes are moved by the fit, many end up inside
the iris and are covered by later layers, so neither their ends nor the layer alone tell what is seen. The zoomed
figure shows it instead.

    PYTHONPATH=.:tools python tools/eye_material_compare.py --out runs/eye_layers/material_compare --figure 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

import eye_layers as EL

ML = Path(__file__).resolve().parents[1]
B = ML / "data" / "processed" / "brush_l"
PACK = ML / "data" / "kaggle" / "eye_digital" / "pack"
MATERIALS = {"layers": B / "eye_l8", "digital": B / "eye_l8_digital", "new": B / "eye_l8_rim_hybrid"}
LABELS = {"layers": "原本 layers（E1–E5）", "digital": "digital（Kaggle 10-24）", "new": "新：混合底圖 + 新眼眶"}
GREEN, BAND, RING = 0.04, 2.5, (4.0, 20.0)


def lum(c: np.ndarray) -> np.ndarray:
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114


def photo_rim(p: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """The photo's rim lines, the distance of every pixel to them and the inside of the eyes."""
    target = p["target"].astype(np.float32) / 255
    rims = EL.rim_on_photo(target, [p["outline_0"].astype(np.float64), p["outline_1"].astype(np.float64)], float(p["d"]))
    line = np.ones((256, 256), bool)
    for r in rims:
        xs, ys = np.clip(r[:, 0].astype(int), 0, 255), np.clip(r[:, 1].astype(int), 0, 255)
        line[ys, xs] = False
    return rims, distance_transform_edt(line), EL.interior(rims).reshape(256, 256)


def measure(painted: np.ndarray, target: np.ndarray, dist: np.ndarray, inside: np.ndarray) -> dict:
    """The four measures of one painting (or of the photo itself) in 0-1."""
    band, ring = dist <= BAND, (~inside) & (dist >= RING[0]) & (dist <= RING[1])
    green = painted[..., 1] - np.maximum(painted[..., 0], painted[..., 2]) > GREEN
    return {"rim_green": float(green[band].mean()),
            "rim_light": float(lum(painted)[band].mean() - lum(target)[band].mean()),
            "fur_l1": float(np.abs(painted - target)[ring].mean()),
            "eye_l1": float(np.abs(painted - target)[inside].mean())}


def pack_index() -> dict[str, tuple[Path, int]]:
    out = {}
    for f in sorted(PACK.glob("pack_*.npz")):
        if not f.name.endswith(".tmp.npz"):
            out.update({str(n): (f, j) for j, n in enumerate(np.load(f)["names"])})
    return out


def tile(img: np.ndarray, eye: np.ndarray, d: float, rim: np.ndarray | None, side: int) -> Image.Image:
    """One eye of an eye-canvas image, zoomed; the photo's rim line drawn when given."""
    r = 0.42 * d
    x0, y0 = int(eye[0] - r), int(eye[1] - r)
    im = Image.fromarray(img).crop((x0, y0, int(eye[0] + r), int(eye[1] + r))).resize((side, side), Image.LANCZOS)
    if rim is not None:
        s = side / (2 * r)
        pts = [((x - x0) * s, (y - y0) * s) for x, y in rim[:, :2]]
        ImageDraw.Draw(im).line(pts + pts[:1], fill=(255, 255, 0), width=1)
    return im


def figure(names: list[str], rows: dict, idx: dict, path: Path, side: int = 260) -> None:
    """Each cat's left eye zoomed: photo, photo + its rim line, the three materials; each tile's measures below."""
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 15)
    small = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 13)
    cols = ["照片", "照片 + 照片的眼眶線（黃）"] + [LABELS[m] for m in MATERIALS]
    W, H = side + 6, side + 46
    img = Image.new("RGB", (len(cols) * W + 6, 28 + len(names) * H), "white")
    d = ImageDraw.Draw(img)
    for c, t in enumerate(cols):
        d.text((6 + c * W, 5), t, fill=(0, 0, 0), font=font)
    for i, n in enumerate(names):
        f, j = idx[n]
        p = {k: v[j] for k, v in np.load(f).items() if k != "names"}
        rims, _, _ = photo_rim(p)
        k = 0
        eye, dd = p["eyes"][k], float(p["d"])
        y = 28 + i * H
        img.paste(tile(p["target"], eye, dd, None, side), (6, y))
        img.paste(tile(p["target"], eye, dd, rims[k], side), (6 + W, y))
        d.text((6, y + side + 4), n.replace("afhq__", ""), fill=(90, 90, 90), font=small)
        for c, m in enumerate(MATERIALS):
            z = np.load(MATERIALS[m] / f"{n}.npz")
            img.paste(tile(z["painted"], eye, dd, None, side), (6 + (c + 2) * W, y))
            q = rows[n][m]
            d.text((6 + (c + 2) * W, y + side + 4), f"眼眶帶：綠 {q['rim_green']:.0%}  比照片亮 {q['rim_light']:+.2f}",
                   fill=(0, 0, 0), font=small)
            d.text((6 + (c + 2) * W, y + side + 22), f"毛色差 {q['fur_l1']:.3f}  眼睛差 {q['eye_l1']:.3f}",
                   fill=(0, 0, 0), font=small)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=int, default=4, help="cats in the zoomed figure (evenly spaced, not picked)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pack_index()
    names = sorted(n for n in idx if all((MATERIALS[m] / f"{n}.npz").exists() for m in MATERIALS))
    rows = {}
    for f in sorted({f for f, _ in idx.values()}):
        z = np.load(f)
        packed = {k: z[k] for k in z.files}
        for j, n in enumerate(map(str, packed["names"])):
            if n not in names:
                continue
            p = {k: v[j] for k, v in packed.items() if k != "names"}
            rims, dist, inside = photo_rim(p)
            target = p["target"].astype(np.float32) / 255
            rows[n] = {m: measure(np.load(path / f"{n}.npz")["painted"].astype(np.float32) / 255, target, dist, inside)
                       for m, path in MATERIALS.items()}
            rows[n]["photo"] = measure(target, target, dist, inside)
    mean = {m: {k: round(float(np.mean([r[m][k] for r in rows.values()])), 4) for k in rows[names[0]][m]}
            for m in [*MATERIALS, "photo"]}
    res = {"cats": len(rows), "mean": mean}
    (a.out / "compare.json").write_text(json.dumps({**res, "cats_rows": rows}))
    print(json.dumps(res, indent=1))
    if a.figure:
        step = len(names) // a.figure
        figure([names[i * step + step // 2] for i in range(a.figure)], rows, idx, a.out / "zoom.png")


if __name__ == "__main__":
    main()
