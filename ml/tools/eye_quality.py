"""Which cats' eyes can teach a clear black pupil (owner, 2026-10-24: remove from training the cats whose eyes are closed
or squinting, too dark to see, cut by the subject mask (a white gap in the eye), or whose plan eye outline is not on the
photo's eye; keep dilated black pupils and slit pupils). Measured on the photo crop of every packed cat
(tools/eye_pack.py) for each eye, inside L8's plan outline (where the eye material is fitted), CPU only:

  open    the outline's height over its width (its two principal extents); L8's plan outlines never get flat (1st
          percentile 0.57), so this does not find closed eyes and is only reported
  differ  distance between the mean colour inside the outline and in a ring of fur around it (RGB 0-1): an open eye
          (iris colour, or a dilated black pupil) differs from the fur, a closed or squinting eye is fur inside too
  bright  the 95th percentile luminance inside the outline (reported only, see RULES)
  cut     share of the outline outside the subject mask (mask under 0.5): there the photo was turned to white paper
  offset  distance from the outline's centre to the photo's eye point, over the eye distance

A cat is removed when either eye fails a rule (both eyes are in its sequence). Writes quality.json (every cat: measures
and reasons) and, per rule, a figure of the removed cats nearest the threshold and the kept cats just past it (the
outline drawn in green), so the threshold can be checked by eye.

    PYTHONPATH=.:tools python tools/eye_quality.py --out runs/eye_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ML = Path(__file__).resolve().parents[1]
PACK = ML / "data" / "kaggle" / "eye_digital" / "pack"
DATA_L8 = ML / "data" / "processed" / "brush_l" / "data_L8.npz"
LUM = np.array([0.299, 0.587, 0.114], np.float32)
# Thresholds set by looking at the cats ranked along each measure (2026-10-24): differ <= 0.019 (the 60 lowest) are
# nearly all closed or unseen eyes, at 0.026 about half, from 0.032 mostly open ones; bright does not separate unseen
# eyes from dark irises and dilated pupils (the 70 darkest are mostly good black pupils) and the unseen ones are
# already low in differ (inside and fur both dark), so it is only reported; cut: 2-3% is a sliver at the photo's
# edge, from 26% the eye is half white paper; offset: from 0.20 six of the eight cats have an outline off the eye or
# misshapen, between 0.15 and 0.20 most outlines are on the eye.
RULES = {                                     # name: (measure, removed when below / above, threshold)
    "closed": ("differ", "below", 0.020),     # closed, squinting or too dark to see
    "cut": ("cut", "above", 0.05),
    "misplaced": ("offset", "above", 0.20),
}


def eye_measures(target: np.ndarray, mask: np.ndarray, outline: np.ndarray, eye: np.ndarray, d: float) -> dict[str, float]:
    xy = outline[:, :2]
    x0, y0 = np.clip(np.floor(xy.min(0)).astype(int), 0, 255)
    x1, y1 = np.clip(np.ceil(xy.max(0)).astype(int) + 1, 1, 256)
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    m = Polygon(xy).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    c = xy - xy.mean(0)
    ext = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(c.T)), 1e-9))           # minor, major
    res = {"open": float(ext[0] / ext[1]), "offset": float(np.linalg.norm(xy.mean(0) - eye) / d)}
    if m.sum() < 20:
        return {**res, "bright": 0.0, "cut": 1.0, "differ": 0.0}
    img = target.astype(np.float32) / 255
    lum = img[y0:y1, x0:x1] @ LUM
    full = np.zeros((256, 256), bool)
    full[y0:y1, x0:x1] = m
    ring = ndimage.binary_dilation(full, iterations=max(2, int(0.08 * d))) & ~ndimage.binary_dilation(full, iterations=2)
    ring &= mask >= 128
    differ = float(np.linalg.norm(img[full].mean(0) - img[ring].mean(0))) if ring.sum() > 20 else 0.0
    return {**res, "bright": float(np.percentile(lum[m], 95)), "cut": float((mask[y0:y1, x0:x1][m] < 128).mean()),
            "differ": differ}


def reasons(q: dict[str, float]) -> list[str]:
    out = []
    for name, (key, way, t) in RULES.items():
        if (q[key] < t) if way == "below" else (q[key] > t):
            out.append(name)
    return out


def worse(eyes: list[dict[str, float]]) -> dict[str, float]:
    """The cat's measure from its worse eye for each rule."""
    return {"open": min(e["open"] for e in eyes), "differ": min(e["differ"] for e in eyes), "bright": min(e["bright"] for e in eyes),
            "cut": max(e["cut"] for e in eyes), "offset": max(e["offset"] for e in eyes)}


def figure(rule: str, rows: dict[str, dict], cache: dict[str, tuple[dict, int]], path: Path, n: int) -> None:
    key, way, t = RULES[rule]
    removed = sorted((x for x in rows if rule in rows[x]["reasons"]), key=lambda x: rows[x][key], reverse=way == "below")
    kept = sorted((x for x in rows if rule not in rows[x]["reasons"]), key=lambda x: rows[x][key], reverse=way == "above")
    shown = [(x, "剔除") for x in removed[:n]] + [(x, "保留") for x in kept[:n]]
    S, font = 140, ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 15)
    img = Image.new("RGB", (n * (S + 4) + 4, 2 * (2 * S + 50) + 10), "white")
    d = ImageDraw.Draw(img)
    for i, (x, tag) in enumerate(shown):
        z, j = cache[x]
        col, row = i % n, i // n
        ox, oy = 4 + col * (S + 4), 4 + row * (2 * S + 50)
        r = 0.42 * float(z["d"][j])
        for k in (0, 1):
            ex, ey = z["eyes"][j][k]
            box = (ex - r, ey - r)
            crop = Image.fromarray(z["target"][j]).crop((int(box[0]), int(box[1]), int(ex + r), int(ey + r))).resize((S, S))
            dc = ImageDraw.Draw(crop)
            pts = [((px - int(box[0])) * S / (2 * r), (py - int(box[1])) * S / (2 * r)) for px, py in z[f"outline_{k}"][j][:, :2]]
            dc.line(pts + pts[:1], fill=(0, 255, 0), width=1)
            img.paste(crop, (ox, oy + k * S))
        d.text((ox, oy + 2 * S + 4), f"{tag} {key} {rows[x][key]:.2f}", fill=(200, 0, 0) if tag == "剔除" else (0, 0, 0), font=font)
        d.text((ox, oy + 2 * S + 24), rows[x]["split"], fill=(0, 0, 0), font=font)
    img.save(path)


def main() -> None:
    """Measure every packed cat, write quality.json, the counts and the figures."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10, help="cats per row of a figure")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z8 = np.load(DATA_L8)
    split_of = dict(zip(map(str, z8["names"]), map(str, z8["split"])))
    rows, cache = {}, {}
    for f in sorted(PACK.glob("pack_*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = dict(np.load(f))
        for j, name in enumerate(z["names"]):
            eyes = [eye_measures(z["target"][j], z["mask"][j], z[f"outline_{k}"][j], z["eyes"][j][k], float(z["d"][j]))
                    for k in (0, 1)]
            q = worse(eyes)
            rows[str(name)] = {**{k: round(v, 4) for k, v in q.items()}, "reasons": reasons(q), "split": split_of[str(name)]}
            cache[str(name)] = (z, j)
    (a.out / "quality.json").write_text(json.dumps({"rules": RULES, "cats": rows}))
    count = {"cats": len(rows), "removed": {}, "by_split": {}}
    for s in ("train", "val", "test"):
        names = [x for x in rows if rows[x]["split"] == s]
        count["by_split"][s] = {"cats": len(names), "removed": sum(bool(rows[x]["reasons"]) for x in names)}
    for rule in RULES:
        count["removed"][rule] = sum(rule in r["reasons"] for r in rows.values())
    count["removed"]["any"] = sum(bool(r["reasons"]) for r in rows.values())
    (a.out / "counts.json").write_text(json.dumps(count, indent=1))
    print(json.dumps(count))
    for rule in RULES:
        figure(rule, rows, cache, a.out / f"rule_{rule}.png", a.n)


if __name__ == "__main__":
    main()
