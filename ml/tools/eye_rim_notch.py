"""How often the photo's rim line is pulled into the eye, and whether it shows in the new material (owner, 2026-10-26:
the yellow line of tools/eye_material_compare.py dips in at slit pupils and highlights; count it on every painted cat,
check whether the finished painting shows it, then decide whether to change anything).

eye_layers.rim_on_photo slides every point of the plan's eye outline along its ray from the outline's centre, at most
RIM_REACH eye distances, to the strongest thin dark line, then averages the slides over RIM_WINDOW neighbours. A dent
is where a stretch of the line slid inwards more than the line around it:

  dent   per point, (the slide averaged over WIDE points, circular) - (its own slide), in px; above 0 = inwards
  depth  an eye's largest dent; a notch when depth > NOTCH_PX

Whether the painting shows it: the same rim finder run on the painting itself (same plan outline). Its dent at the
photo notch's points (the largest within +-SPAN points) is the painting's depth there. The new material (rim strokes
along the photo's line) is compared with the digital material of 10-24 (rim fitted from the plan outline, never along
the photo's line): when both paintings dent there, it is the painted pupil or iris pulling the finder, not the rim;
"shown" is new's depth minus digital's depth > NOTCH_PX / 2.

    PYTHONPATH=.:tools python tools/eye_rim_notch.py --out runs/eye_layers/rim_notch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eye_layers as EL
import eye_material_compare as C

WIDE, SPAN, NOTCH_PX = 91, 10, 3.0


def slides(rim: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """How far each point of the found line lies outside the outline along its ray (px; negative = inside)."""
    c = outline[:, :2].mean(0)
    return np.hypot(*(rim[:, :2] - c).T) - np.hypot(*(outline[:, :2] - c).T)


def dents(s: np.ndarray) -> np.ndarray:
    pad = WIDE // 2
    wide = np.convolve(np.r_[s[-pad:], s, s[:pad]], np.ones(WIDE) / WIDE, mode="valid")
    return wide - s


def around(dent: np.ndarray, i: int) -> float:
    """The largest dent within SPAN points of point i (circular)."""
    return float(dent[np.arange(i - SPAN, i + SPAN + 1) % len(dent)].max())


def eye_rows(p: dict, painted: dict[str, np.ndarray]) -> list[dict]:
    """Both eyes of one cat: the photo's depth and where, and each painting's depth there."""
    outlines = [p["outline_0"].astype(np.float64), p["outline_1"].astype(np.float64)]
    d = float(p["d"])
    photo = EL.rim_on_photo(p["target"].astype(np.float32) / 255, outlines, d)
    found = {m: EL.rim_on_photo(img, outlines, d) for m, img in painted.items()}
    rows = []
    for k in (0, 1):
        dent = dents(slides(photo[k], outlines[k]))
        i = int(dent.argmax())
        row = {"eye": k, "depth": round(float(dent[i]), 2), "at": i}
        for m in painted:
            row[m] = round(around(dents(slides(found[m][k], outlines[k])), i), 2)
        rows.append(row)
    return rows


def figure(picks: list[tuple[str, int, str]], rows: dict, idx: dict, path: Path, side: int = 230) -> None:
    """Each picked eye zoomed: photo + its found line (yellow, the notch's points red), digital, new, new + line."""
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 15)
    small = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 13)
    cols = ["照片", "照片 + 照片眼眶線（缺口紅色）", "digital（10-24）", "新教材", "新教材 + 照片眼眶線"]
    W, H = side + 6, side + 44
    img = Image.new("RGB", (len(cols) * W + 6, 28 + len(picks) * H), "white")
    dr = ImageDraw.Draw(img)
    for c, t in enumerate(cols):
        dr.text((6 + c * W, 5), t, fill=(0, 0, 0), font=font)
    for r, (n, k, tag) in enumerate(picks):
        f, j = idx[n]
        p = {key: v[j] for key, v in np.load(f).items() if key != "names"}
        outline = p[f"outline_{k}"].astype(np.float64)
        rim = EL.rim_on_photo(p["target"].astype(np.float32) / 255, [outline], float(p["d"]))[0]
        e = rows[n][k]
        notch = np.arange(e["at"] - SPAN, e["at"] + SPAN + 1) % len(rim)
        eye, d = outline[:, :2].mean(0), float(p["d"])
        y = 28 + r * H

        def put(arr, c, line):
            t = C.tile(arr, eye, d, rim if line else None, side)
            if line:
                rr = 0.42 * d
                s, x0, y0 = side / (2 * rr), int(eye[0] - rr), int(eye[1] - rr)
                ImageDraw.Draw(t).line([((x - x0) * s, (yy - y0) * s) for x, yy in rim[notch, :2]], fill=(255, 0, 0), width=3)
            img.paste(t, (6 + c * W, y))

        put(p["target"], 0, False)
        put(p["target"], 1, True)
        put(np.load(C.MATERIALS["digital"] / f"{n}.npz")["painted"], 2, False)
        new = np.load(C.MATERIALS["new"] / f"{n}.npz")["painted"]
        put(new, 3, False)
        put(new, 4, True)
        dr.text((6, y + side + 4), f"{tag}  {n.replace('afhq__', '')} 第 {k} 眼", fill=(0, 0, 0), font=small)
        dr.text((6, y + side + 22), f"照片缺口 {e['depth']:.1f} px · 畫上：digital {e['digital']:.1f} px、新 {e['new']:.1f} px",
                fill=(0, 0, 0), font=small)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-group", type=int, default=3, help="eyes per group in the figure (evenly spaced, not picked)")
    ap.add_argument("--worst", type=int, default=8, help="eyes in worst.png: the largest new-minus-digital dents")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = C.pack_index()
    names = sorted(n for n in idx if all((C.MATERIALS[m] / f"{n}.npz").exists() for m in ("digital", "new")))
    rows = {}
    for f in sorted({f for f, _ in idx.values()}):
        z = np.load(f)
        packed = {k: z[k] for k in z.files}
        for j, n in enumerate(map(str, packed["names"])):
            if n in names:
                p = {k: v[j] for k, v in packed.items() if k != "names"}
                painted = {m: np.load(C.MATERIALS[m] / f"{n}.npz")["painted"].astype(np.float32) / 255
                           for m in ("digital", "new")}
                rows[n] = eye_rows(p, painted)
    eyes = [(n, e) for n in rows for e in rows[n]]
    notch = [(n, e) for n, e in eyes if e["depth"] > NOTCH_PX]
    shown = [(n, e) for n, e in notch if e["new"] - e["digital"] > NOTCH_PX / 2]
    depth = np.array([e["depth"] for _, e in eyes])
    res = {"cats": len(rows), "eyes": len(eyes),
           "depth_px_percentiles": {q: round(float(np.percentile(depth, q)), 2) for q in (50, 75, 90, 95, 99)},
           "notch_eyes": len(notch), "notch_cats": len({n for n, _ in notch}),
           "shown_eyes": len(shown), "shown_cats": len({n for n, _ in shown})}
    (a.out / "notch.json").write_text(json.dumps({**res, "rows": rows}))
    print(json.dumps(res, indent=1))
    groups = {"畫上看得到": shown, "有缺口、畫上看不到": [x for x in notch if x not in shown],
              "沒有缺口": [x for x in eyes if x[1]["depth"] <= NOTCH_PX]}
    picks = []
    for tag, g in groups.items():
        g = sorted(g, key=lambda x: (x[0], x[1]["eye"]))
        step = max(1, len(g) // a.per_group)
        picks += [(n, e["eye"], tag) for n, e in g[step // 2::step][:a.per_group]]
    figure(picks, rows, idx, a.out / "examples.png")
    worst = sorted(notch, key=lambda x: x[1]["digital"] - x[1]["new"])[:a.worst]
    figure([(n, e["eye"], f"最嚴重第 {r + 1}") for r, (n, e) in enumerate(worst)], rows, idx, a.out / "worst.png")


if __name__ == "__main__":
    main()
