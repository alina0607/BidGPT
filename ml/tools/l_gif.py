"""A training cat of Brush GPT L painted from its token sequence, stroke by stroke, as a GIF (owner, 2026-10-19): what
the model is taught to do. First the sketch plan appears outline by outline (faint lines; the plan tokens), then the
five layers' strokes in their order; the photo is on the right. Blending strokes are applied a few at a time (in
training data they were fitted acting together; replaying them in small groups differs by a few thousandths).

The sketch lines are drawn on the paper the strokes are painted over, for display only: the training paintings start
on white.

    PYTHONPATH=.:tools python tools/l_gif.py --cats afhq__flickr_cat_000018 --out runs/brush_l_gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import eye_crop_styles as E
import sketch_plan as SP
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
CATS = ML / "data" / "processed" / "brush_l" / "cats"
PLANS = ML / "data" / "processed" / "sketch_plan" / "afhq"
LAYER_ZH = ["① 大片打底", "② 中型色塊", "③ 暈染", "④ 小色塊", "⑤ 暈染"]
PER_FRAME = [2, 4, 10, 8, 15]                 # strokes added per frame in each layer
SCALE = 1.5
FONT = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 22)


def plan_parts(plan: np.lib.npyio.NpzFile) -> list[tuple[str, list]]:
    """The plan's outlines in the order the tokens write them, for drawing one part at a time."""
    pupils = []
    for p in plan["pupils"]:
        pupils.append(None if np.isnan(p).any() else {"centre": p[:2], "radii": p[2:4], "angle": float(p[4])})
    nose = plan["nose"] if len(plan["nose"]) else None
    return [("頭與耳朵的輪廓", [list(plan["head"]), [], [], None, np.zeros((0, 2))]),
            ("眼眶", [[], list(plan["eyes"]), [], None, np.zeros((0, 2))]),
            ("瞳孔", [[], [], pupils, None, np.zeros((0, 2))]),
            ("鼻子", [[], [], [], nose, np.zeros((0, 2))]),
            ("嘴", [[], [], [], None, plan["mouth"]])]


def sketch_image(parts: list, upto: int) -> np.ndarray:
    head, eyes, pupils, nose, mouth = [], [], [], None, np.zeros((0, 2))
    for _, (h, e, p, n, m) in parts[:upto]:
        head += h
        eyes += e
        pupils += p
        nose = n if n is not None else nose
        mouth = m if len(m) else mouth
    img = SP.render_sketch(head, eyes, pupils, nose, mouth, grey=170, scale=1)
    return np.asarray(img.convert("RGB"), np.float32) / 255


def frame(painting: np.ndarray, photo: np.ndarray, caption: str) -> Image.Image:
    a = Image.fromarray((painting.clip(0, 1) * 255).astype(np.uint8)).resize((int(B.SIDE * SCALE),) * 2, Image.LANCZOS)
    b = Image.fromarray(photo).resize((int(B.SIDE * SCALE),) * 2, Image.LANCZOS)
    out = Image.new("RGB", (int(B.SIDE * SCALE) * 2 + 12, int(B.SIDE * SCALE) + 44), "white")
    out.paste(a, (0, 44))
    out.paste(b, (int(B.SIDE * SCALE) + 12, 44))
    d = ImageDraw.Draw(out)
    d.text((8, 10), caption, fill=(0, 0, 0), font=FONT)
    d.text((int(B.SIDE * SCALE) + 20, 10), "照片（目標，不確定區照原樣）", fill=(90, 90, 90), font=FONT)
    return out


def make(name: str, out: Path) -> Path:
    seq = np.load(CATS / f"{name}.npz")["tokens"]
    plan = np.load(next(PLANS.glob(f"shard_*/{name}.npz")), allow_pickle=True)
    photo = np.load(next((ML / "data/processed/repaint").glob(f"**/shard_*/{name}.npz")))["target"]
    parts = plan_parts(plan)
    frames, durations = [frame(np.ones((B.SIDE, B.SIDE, 3)), photo, "空白畫布")], [700]
    for k in range(1, len(parts) + 1):
        frames.append(frame(sketch_image(parts, k), photo, f"構圖計畫：{parts[k - 1][0]}"))
        durations.append(600)
    keys = [("構圖計畫", sketch_image(parts, len(parts)))]
    canvas = torch.from_numpy(sketch_image(parts, len(parts)).copy())
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    layers = L.decode_layers(seq)
    total, done = sum(len(r) for r in layers), 0
    with torch.no_grad():
        for k, rows in enumerate(layers):
            br = L.brush(k)
            for i in range(0, len(rows), PER_FRAME[k]):
                chunk = rows[i:i + PER_FRAME[k]]
                with E.brush_settings(br):
                    s = B.Strokes.from_stage(L.stroke_values(chunk, k), (br["length"], br["width"]))
                    canvas = E.blend_composite(canvas, s) if br["blend"] else B.composite(canvas, s, grid)
                done += len(chunk)
                frames.append(frame(canvas.numpy(), photo, f"{LAYER_ZH[k]}　第 {done} / {total} 筆"))
                durations.append(150 if k else 250)
            keys.append((f"{LAYER_ZH[k]}後（{done} 筆）", canvas.numpy().copy()))
    frames.append(frame(canvas.numpy(), photo, f"完成：{total} 筆，序列 {len(seq):,} 個 token"))
    durations.append(3000)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save(out / f"{name}_final.png")
    keys.append(("照片", photo.astype(np.float32) / 255))
    w = B.SIDE
    strip = Image.new("RGB", (w * len(keys) + 6 * (len(keys) - 1), w + 34), "white")
    d = ImageDraw.Draw(strip)
    small = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 15)
    for j, (cap, img) in enumerate(keys):
        strip.paste(Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)), (j * (w + 6), 34))
        d.text((j * (w + 6) + 4, 8), cap, fill=(0, 0, 0), font=small)
    strip.save(out / f"{name}_steps.png")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cats", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ML / "runs" / "brush_l_gif")
    a = ap.parse_args()
    for name in a.cats:
        print(make(name, a.out))


if __name__ == "__main__":
    main()
