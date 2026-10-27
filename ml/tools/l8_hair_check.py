"""L8's hair layers against its training data (owner, 2026-10-27: in the game the fine hairs look sudden and do not
follow the fur, unlike the training paintings). The first N validation cats (not picked), three ways each: the
training painting (the digital repaint), L8 from the cat's real plan (data/processed/brush_l/l8_gen), and L8's own
free samples (runs/brushgpt/L8/samples_t1.0_s0.npz). zh notebook 17.29.

  stats    runs/brushgpt/L8/hair_stats.json: per hair stroke, its colour against the paint under it (3 x 3 px of the
           painting before that layer), the share more than 0.2 off, darker or lighter than it (0.03 luminance), on
           white paper (luminance over 0.92) and its length
  grain    runs/brushgpt/L8/hair_flow.json: per hair stroke, width, length, opacity, and its angle against the grain
           of the block-in under it (layers 1-5; structure tensor of the luminance, sigma 3 px; the grain is along
           the stripes, at right angles to the strongest change); also only where the grain is clear (coherence
           over 0.5). Random angles give 45 degrees.
  figures  notebooks/08_brush_gpt_細毛_訓練vsL8_zh.png (4 cats, middle of the face: before the hairs and finished),
           08_brush_gpt_毛層放大_訓練vsL8_zh.png (2 cats at about the game's zoom: before and after layer ⑥ (細毛)) and
           08_brush_gpt_L8溫度_zh.png (L8's free samples at temperature 1.0, 0.8, 0.6, from the eye judge's runs)

    PYTHONPATH=.:tools python tools/l8_hair_check.py stats grain figures --n 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from sellanything_ml.sketch import brushtokens_l as L

ML = Path(__file__).resolve().parents[1]
B = ML / "data" / "processed" / "brush_l"
RUN = ML / "runs" / "brushgpt" / "L8"
NB = ML / "notebooks"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
HAIRS = (5, 6, 7)


def lum(c: np.ndarray) -> np.ndarray:
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114


def val_cats(n: int) -> tuple[np.lib.npyio.NpzFile, list[int]]:
    z = np.load(B / "data_L8.npz")
    return z, [i for i, s in enumerate(z["split"]) if s == "val"][:n]


def sequences(n: int, free: bool = True) -> dict[str, list[np.ndarray]]:
    z, val = val_cats(n)
    out = {"training (val cats)": [z["tokens"][z["offsets"][i]:z["offsets"][i + 1]] for i in val],
           "L8, real plan (same cats)": [np.load(B / "l8_gen" / f"{z['names'][i]}.npz")["tokens"] for i in val]}
    if free:
        fs = np.load(RUN / "samples_t1.0_s0.npz")
        out["L8, free (own plan)"] = [fs["tokens"][i][:fs["lengths"][i]] for i in range(min(n, len(fs["lengths"])))]
    return out


def stats(n: int) -> dict:
    out = {}
    for name, seqs in sequences(n).items():
        r = {k: {"diff": [], "darker": [], "lighter": [], "white": [], "len": [], "far": []} for k in HAIRS}
        for t in seqs:
            layers = L.decode_layers(t)
            for k in HAIRS:
                if k >= len(layers) or not len(layers[k]):
                    continue
                before = L.render(layers[:k])
                v = L.stroke_values(layers[k], k)
                xy = np.clip(v["xy"].astype(int), 1, 254)
                under = np.stack([before[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(0) for x, y in xy])
                d = np.abs(v["colour"] - under).mean(1)
                r[k]["diff"] += d.tolist()
                r[k]["far"] += (d > 0.2).tolist()
                r[k]["darker"] += (lum(v["colour"]) < lum(under) - 0.03).tolist()
                r[k]["lighter"] += (lum(v["colour"]) > lum(under) + 0.03).tolist()
                r[k]["white"] += (lum(under) > 0.92).tolist()
                r[k]["len"] += v["length_width"][:, 0].tolist()
        out[name] = {k: {m: round(float(np.mean(x)), 3) for m, x in r[k].items()} | {"n": len(r[k]["diff"])} for k in r}
    (RUN / "hair_stats.json").write_text(json.dumps(out, indent=1))
    return out


def grain(img: np.ndarray, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the direction along the grain (at right angles to the strongest change), 0..pi, and its coherence."""
    y = lum(img)
    gx, gy = ndimage.sobel(y, 1), ndimage.sobel(y, 0)
    jxx, jyy, jxy = (ndimage.gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
    return np.mod(theta + np.pi / 2, np.pi), coh


def flow(n: int) -> dict:
    out = {}
    for name, seqs in sequences(n, free=False).items():
        r = {k: {"width": [], "length": [], "alpha": [], "off_grain": [], "off_grain_strong": []} for k in HAIRS}
        for t in seqs:
            lay = L.decode_layers(t)
            if len(lay) < 8:
                continue
            along, coh = grain(L.render(lay[:5]))
            for k in HAIRS:
                v = L.stroke_values(lay[k], k)
                x, y = np.clip(v["xy"].astype(int), 0, 255).T
                d = np.abs(np.mod(v["angle"] - along[y, x] + np.pi / 2, np.pi) - np.pi / 2)
                r[k]["width"] += v["length_width"][:, 1].tolist()
                r[k]["length"] += v["length_width"][:, 0].tolist()
                r[k]["alpha"] += v["alpha"].tolist()
                r[k]["off_grain"] += np.degrees(d).tolist()
                r[k]["off_grain_strong"] += np.degrees(d[coh[y, x] > 0.5]).tolist()
        out[name.split(" (")[0]] = {k: {m: round(float(np.mean(x)), 3) for m, x in r[k].items()} for k in r}
    (RUN / "hair_flow.json").write_text(json.dumps(out, indent=1))
    return out


def tile(a: np.ndarray, crop: tuple, side: int) -> Image.Image:
    return Image.fromarray((a.clip(0, 1) * 255).round().astype(np.uint8)).crop(crop).resize((side, side), Image.LANCZOS)


def grid_figure(rows: list[list[np.ndarray]], cols: list[str], crop: tuple, side: int, path: Path) -> None:
    font = ImageFont.truetype(FONT, 17)
    img = Image.new("RGB", (len(cols) * (side + 8) + 8, 34 + len(rows) * (side + 8)), "white")
    d = ImageDraw.Draw(img)
    for c, t in enumerate(cols):
        d.text((8 + c * (side + 8), 8), t, fill=(0, 0, 0), font=font)
    for r, row in enumerate(rows):
        for c, a in enumerate(row):
            img.paste(tile(a, crop, side), (8 + c * (side + 8), 34 + r * (side + 8)))
    img.save(path)


def figures() -> None:
    z, val = val_cats(4)
    pairs = [(z["tokens"][z["offsets"][i]:z["offsets"][i + 1]], np.load(B / "l8_gen" / f"{z['names'][i]}.npz")["tokens"])
             for i in val]
    rows = []
    for train, l8 in pairs:
        row = []
        for t in (train, l8):
            lay = L.decode_layers(t)
            row += [L.render(lay[:5]), L.render(lay)]
        rows.append(row)
    grid_figure(rows, ["訓練資料：細毛之前", "訓練資料：畫完", "L8（同一隻的真實 plan）：細毛之前", "L8：畫完"],
                (48, 48, 208, 208), 300, NB / "08_brush_gpt_細毛_訓練vsL8_zh.png")
    rows = []
    for train, l8 in pairs[:2]:
        row = []
        for t in (train, l8):
            lay = L.decode_layers(t)
            row += [L.render(lay[:5]), L.render(lay[:6])]
        rows.append(row)
    grid_figure(rows, ["訓練資料：毛之前", "訓練資料：+ ⑥ 細毛", "L8：毛之前", "L8：+ ⑥ 細毛"],
                (40, 70, 168, 198), 384, NB / "08_brush_gpt_毛層放大_訓練vsL8_zh.png")
    font, side, crop = ImageFont.truetype(FONT, 17), 240, (48, 48, 208, 208)
    img = Image.new("RGB", (4 * (side + 8) + 90, 10 + 3 * (side + 8)), "white")
    d = ImageDraw.Draw(img)
    for r, t in enumerate(("1.0", "0.8", "0.6")):
        p = np.load(RUN / f"eye_judge_samples_t{t}_s3.npz")["paintings"]
        d.text((8, 10 + r * (side + 8) + side // 2), f"溫度 {t}", fill=(0, 0, 0), font=font)
        for c in range(4):
            a = p[c].astype(np.float32) / 255 if p.dtype == np.uint8 else p[c]
            img.paste(tile(a, crop, side), (90 + c * (side + 8), 10 + r * (side + 8)))
    img.save(NB / "08_brush_gpt_L8溫度_zh.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", nargs="+", choices=("stats", "grain", "figures"))
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if "stats" in a.what:
        print(json.dumps(stats(a.n), indent=1), flush=True)
    if "grain" in a.what:
        print(json.dumps(flow(a.n), indent=1), flush=True)
    if "figures" in a.what:
        figures()
        print("figures written", flush=True)


if __name__ == "__main__":
    main()
