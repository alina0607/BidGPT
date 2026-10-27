"""Two flaws of the new eye material (owner, 2026-10-26: how to handle them), counted on every painted cat:

  off      the photo's rim line left the eye: eye_layers.rim_on_photo looks for the rim only RIM_REACH eye distances
           either side of the plan's outline, so where the plan's outline is farther than that from the photo's rim
           the strongest dark line within reach is fur, a lid fold or the pupil. Per eye, the share of the outline's
           points whose best slide is at the edge of the search (the rim may be farther) or whose line there is weak
           (dark_lines under WEAK, about what fur gives).
  leftover L8's base still showing: the hybrid base keeps L8 inside the plan's outline widened by 2 px and softened,
           but the eye layers may only paint inside the outline itself (and the rim within reach of its line), so
           where L8's colour is far from the photo's it can stay in the painting. Per cat, the pixels in L8's part of
           the base (weight over 0.1) that the painting left as they were (|painting - base| < SAME) while they are
           far from the photo (|base - photo| > FAR); split into inside the plan's outline and the widened band.

    PYTHONPATH=.:tools python tools/eye_material_flaws.py --out runs/eye_layers/material_flaws
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import map_coordinates

import eye_crop_styles as E
import eye_layers as EL
import eye_material_compare as C

WEAK, SAME, FAR = 0.03, 0.03, 0.15


def off_share(target: np.ndarray, outline: np.ndarray, d: float) -> float:
    """rim_on_photo's search before smoothing: the share of points at the search's edge or on a weak line."""
    lines = E.dark_lines(E.luminance(target))
    k = max(2, int(round(EL.RIM_REACH * d)))
    offs = np.arange(-k, k + 1, dtype=float)
    x, y = outline[:, 0], outline[:, 1]
    r = np.maximum(np.hypot(x - x.mean(), y - y.mean()), 1e-6)
    ux, uy = (x - x.mean()) / r, (y - y.mean()) / r
    sx, sy = x[:, None] + offs[None] * ux[:, None], y[:, None] + offs[None] * uy[:, None]
    vals = map_coordinates(lines, [sy.ravel() - 0.5, sx.ravel() - 0.5], order=1, mode="nearest").reshape(sx.shape)
    j = vals.argmax(1)
    return float(((j == 0) | (j == len(offs) - 1) | (vals.max(1) < WEAK)).mean())


def leftover(p: dict, z) -> tuple[int, int, np.ndarray]:
    """L8 pixels left in the painting far from the photo: inside the outlines, in the widened band, and where."""
    target = p["target"].astype(np.float32) / 255
    base, painted = z["base"].astype(np.float32) / 255, z["painted"].astype(np.float32) / 255
    inside = EL.interior([p["outline_0"].astype(np.float64), p["outline_1"].astype(np.float64)])
    w = ndimage.gaussian_filter(ndimage.binary_dilation(inside, iterations=2).astype(np.float32), 1.5)
    left = (w > 0.1) & (np.abs(painted - base).mean(-1) < SAME) & (np.abs(base - target).mean(-1) > FAR)
    return int((left & inside).sum()), int((left & ~inside).sum()), left


def tile(img: np.ndarray, eye: np.ndarray, d: float, side: int, line=None, mark=None) -> Image.Image:
    r = 0.42 * d
    x0, y0 = int(eye[0] - r), int(eye[1] - r)
    a = img.copy()
    if mark is not None:
        a[mark] = (255, 0, 255)
    im = Image.fromarray(a).crop((x0, y0, int(eye[0] + r), int(eye[1] + r))).resize((side, side), Image.LANCZOS)
    s = side / (2 * r)
    for pts, colour in line or []:
        q = [((x - x0) * s, (y - y0) * s) for x, y in pts[:, :2]]
        ImageDraw.Draw(im).line(q + q[:1], fill=colour, width=2)
    return im


def figure(picks: list[tuple[str, int, str]], idx: dict, path: Path, side: int = 220) -> None:
    """Photo + the plan's outline (green) and the photo's rim line (yellow); the new painting; L8 left over (magenta)."""
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 14)
    cols = ["照片 + plan 眼眶（綠）+ 照片眼眶線（黃）", "新教材", "新教材，留下的 L8 底色（洋紅）"]
    W, H = side + 6, side + 24
    img = Image.new("RGB", (len(cols) * W + 6, 26 + len(picks) * H), "white")
    dr = ImageDraw.Draw(img)
    for c, t in enumerate(cols):
        dr.text((6 + c * W, 5), t, fill=(0, 0, 0), font=font)
    for r, (n, k, tag) in enumerate(picks):
        f, j = idx[n]
        p = {key: v[j] for key, v in np.load(f).items() if key != "names"}
        z = np.load(C.MATERIALS["new"] / f"{n}.npz")
        outline = p[f"outline_{k}"].astype(np.float64)
        rim = EL.rim_on_photo(p["target"].astype(np.float32) / 255, [outline], float(p["d"]))[0]
        _, _, left = leftover(p, z)
        eye, d, y = outline[:, :2].mean(0), float(p["d"]), 26 + r * H
        img.paste(tile(p["target"], eye, d, side, [(outline, (0, 255, 0)), (rim, (255, 255, 0))]), (6, y))
        img.paste(tile(z["painted"], eye, d, side), (6 + W, y))
        img.paste(tile(z["painted"], eye, d, side, mark=left), (6 + 2 * W, y))
        dr.text((6, y + side + 3), f"{tag}  {n.replace('afhq__', '')} 第 {k} 眼", fill=(0, 0, 0), font=font)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = C.pack_index()
    exclude = {line.split("\t")[0] for line in open(C.B / "eye_exclude_2026-10-24.tsv")}
    rows = {}
    for f in sorted({f for f, _ in idx.values()}):
        z = np.load(f)
        packed = {k: z[k] for k in z.files}
        for j, n in enumerate(map(str, packed["names"])):
            p = {k: v[j] for k, v in packed.items() if k != "names"}
            target = p["target"].astype(np.float32) / 255
            row = {"off": [round(off_share(target, p[f"outline_{k}"].astype(np.float64), float(p["d"])), 3)
                           for k in (0, 1)], "excluded": n in exclude}
            new = C.MATERIALS["new"] / f"{n}.npz"
            if new.exists():
                li, lb, _ = leftover(p, np.load(new))
                row["left_inside"], row["left_band"] = li, lb
            rows[n] = row
    off = np.array([max(r["off"]) for r in rows.values()])
    kept = [r for r in rows.values() if not r["excluded"]]
    painted = [r for r in kept if "left_inside" in r]
    left = np.array([r["left_inside"] + r["left_band"] for r in painted])
    res = {"cats": len(rows), "kept": len(kept),
           "off_worse_eye_percentiles": {q: round(float(np.percentile(off, q)), 3) for q in (50, 75, 90, 95, 99)},
           "off_kept_cats_over": {t: sum(max(r["off"]) > t for r in kept) for t in (0.2, 0.3, 0.4, 0.5)},
           "off_excluded_over_0.3": sum(max(r["off"]) > 0.3 for r in rows.values() if r["excluded"]),
           "painted_kept": len(painted),
           "left_px_percentiles": {q: int(np.percentile(left, q)) for q in (50, 75, 90, 95, 99)},
           "left_share_in_band": round(sum(r["left_band"] for r in painted) / max(1, int(left.sum())), 3),
           "left_kept_cats_over": {t: int((left > t).sum()) for t in (20, 50, 100, 200)}}
    (a.out / "flaws.json").write_text(json.dumps({**res, "rows": rows}))
    print(json.dumps(res, indent=1))
    pn = sorted((n for n in rows if "left_inside" in rows[n] and not rows[n]["excluded"]))

    def spread(names: list[str], key, bands: list[tuple[float, float, str]], per: int) -> list[tuple[str, int, str]]:
        out = []
        for lo, hi, tag in bands:
            g = [n for n in names if lo < key(n) <= hi]
            step = max(1, len(g) // per)
            out += [(n, int(np.argmax(rows[n]["off"])), f"{tag}（{key(n):.2f}）" if key(n) < 2 else f"{tag}（{key(n):.0f} px）")
                    for n in g[step // 2::step][:per]]
        return out

    figure(spread(pn, lambda n: max(rows[n]["off"]), [(0.5, 1.0, "跑出眼睛 >0.5"), (0.3, 0.5, "0.3–0.5"), (0.15, 0.3, "0.15–0.3")], 3),
           idx, a.out / "off_examples.png")
    figure(spread(pn, lambda n: rows[n]["left_inside"] + rows[n]["left_band"], [(200, 1e9, "L8 留下 >200"), (50, 200, "50–200"), (10, 50, "10–50")], 3),
           idx, a.out / "left_examples.png")


if __name__ == "__main__":
    main()
