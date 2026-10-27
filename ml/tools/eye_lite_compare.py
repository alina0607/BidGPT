"""How close the shorter digital eye paintings come to digital (owner, 2026-10-24: "the final result should look like
runs/eye_layers/digital6/steps_000031.png"), on what tools/eye_layers.py saved (CPU only). Each eye is the traced outline's
inside nearest to that eye point; its centre is the eye point.

  to_digital  mean |painting - digital| in the eye (RGB 0-1, per pixel and channel)
  contrast    iris luminance (0.45-0.9 eye radii) minus the centre's (under 0.3 eye radii; radius sqrt(area / pi))
  outward     share of eyes whose five rings (0-1 eye radii) get lighter ring by ring for the first four
  speckle     mean |image - its 5 x 5 box blur| inside the eye eroded 2 px (as tools/eye_measure_e1.py)

The photo is measured too, as the reference. Figures: the first n cats in order (not picked), each eye zoomed.

    PYTHONPATH=.:tools python tools/eye_lite_compare.py --run runs/eye_layers/lite24 --n 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

STYLES = ("photo", "digital", "lite_nofibre", "lite_410", "lite_290", "lite_290_to_digital", "lite_200",
          "lite_200_to_digital", "layers", "face")
STROKES = {"photo": "", "face": "0", "digital": "862", "lite_nofibre": "562", "lite_410": "410", "lite_290": "290",
           "lite_290_to_digital": "290", "lite_200": "200", "lite_200_to_digital": "200", "layers": "192"}
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def eye_regions(z: np.lib.npyio.NpzFile) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(eye mask, distance over eye radius, eroded mask) per eye."""
    yy, xx = np.mgrid[:256, :256] + 0.5
    eyes, inside = z["eyes"], z["inside"]
    near = np.argmin([np.hypot(xx - x, yy - y) for x, y in eyes], 0)
    out = []
    for k, (x, y) in enumerate(eyes):
        m = inside & (near == k)
        rr = np.hypot(xx - x, yy - y) / np.sqrt(max(m.sum(), 1) / np.pi)
        out.append((m, rr, ndimage.binary_erosion(m, iterations=2)))
    return out


def measure(img: np.ndarray, regs: list, digital: np.ndarray) -> dict[str, float]:
    x = np.clip(np.asarray(img, np.float32), 0, 1)
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    detail = np.linalg.norm(x - ndimage.uniform_filter(x, size=(5, 5, 1)), axis=-1)
    res = {"to_digital": [], "contrast": [], "outward": [], "speckle": []}
    for m, rr, er in regs:
        if m.sum() < 30:
            continue
        res["to_digital"].append(float(np.abs(x - digital)[m].mean()))
        centre, ring = m & (rr < 0.3), m & (rr >= 0.45) & (rr < 0.9)
        if centre.sum() > 3 and ring.sum() > 3:
            res["contrast"].append(float(lum[ring].mean() - lum[centre].mean()))
        prof = [lum[m & (rr >= a) & (rr < a + 0.2)].mean() if (m & (rr >= a) & (rr < a + 0.2)).sum() > 3 else np.nan
                for a in np.arange(0, 1, 0.2)]
        res["outward"].append(float(np.all(np.diff(prof[:4]) > 0)))
        res["speckle"].append(float(detail[er].mean()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    files = sorted(f for f in a.run.glob("*.npz"))
    acc = {s: {k: [] for k in ("to_digital", "contrast", "outward", "speckle")} for s in STYLES}
    for f in files:
        z = np.load(f)
        regs = eye_regions(z)
        for s in STYLES:
            if s in z.files:
                for k, v in measure(z[s], regs, z["digital"]).items():
                    acc[s][k] += v
    lines = ["| | strokes | to digital | pupil contrast | lighter outwards | speckle |", "|---|---|---|---|---|---|"]
    table = {}
    for s in STYLES:
        if not acc[s]["contrast"]:
            continue
        m = {k: float(np.mean(v)) for k, v in acc[s].items()}
        table[s] = m
        lines.append(f"| {s} | {STROKES[s]} | {m['to_digital']:.4f} | {m['contrast']:.3f} | {m['outward']:.0%} | {m['speckle']:.3f} |")
    print(f"{len(files)} cats, {len(acc['digital']['contrast'])} eyes")
    print("\n".join(lines))
    (a.run / "lite_table.md").write_text(f"{len(files)} cats\n\n" + "\n".join(lines) + "\n")
    (a.run / "lite_measures.json").write_text(json.dumps(table, indent=1))
    S, font = 200, ImageFont.truetype(FONT, 15)
    shown = [s for s in STYLES if s in table]
    out = Image.new("RGB", (len(shown) * (S + 6) + 6, a.n * (S + 6) + 40), "white")
    d = ImageDraw.Draw(out)
    for j, s in enumerate(shown):
        d.text((6 + j * (S + 6), 8), f"{s} {STROKES[s]}", fill=(0, 0, 0), font=font)
    for i, f in enumerate(files[:a.n]):
        z = np.load(f)
        x, y = z["eyes"][0]
        r = 0.45 * float(np.linalg.norm(z["eyes"][0] - z["eyes"][1]))
        box = (int(x - r), int(y - r), int(x + r), int(y + r))
        for j, s in enumerate(shown):
            im = Image.fromarray((np.clip(z[s], 0, 1) * 255).astype(np.uint8)).crop(box).resize((S, S), Image.LANCZOS)
            out.paste(im, (6 + j * (S + 6), 34 + i * (S + 6)))
    out.save(a.run / "lite_eyes.png")
    print(a.run / "lite_eyes.png")


if __name__ == "__main__":
    main()
