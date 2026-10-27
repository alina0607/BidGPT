"""Figure for tools/eye_data_l8.py --compare: per cat, the photo's eye crop with the plan's eye outlines (green), L8's
base on the crop, the eye data (four layers ordered around the eyes and snapped to the token grid), the upper-bound
painter (no order, no snapping: zh notebook 17.24), and the whole face with the eye data's strokes painted straight
onto it (eye_data_l8.eyes_on_face). All of it is training material fitted to the photo, not anything a model generated. The first cats of the run, not picked.

    PYTHONPATH=.:tools python tools/eye_data_l8_figure.py --run runs/eye_data_l8/trial16 --n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eye_data_l8 as D
from sellanything_ml.sketch import brushtokens_l as L

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
COLS = ["照片（綠＝計畫的眼眶）", "L8 底圖", "教材（排序＋分格）", "上限畫家（17.24）", "整張臉＋教材筆觸"]
S = 200


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.run / "progress_0of1.jsonl")][:a.n]
    big, small = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    out = Image.new("RGB", (len(COLS) * (S + 8) + 8, len(rows) * (S + 26) + 70), "white")
    d = ImageDraw.Draw(out)
    d.text((8, 8), f"眼睛模型的教材：對照片擬合，不是模型生成（前 {len(rows)} 隻，沒有挑）", fill=(0, 0, 0), font=big)
    for j, cap in enumerate(COLS):
        d.text((8 + j * (S + 8), 40), cap, fill=(0, 0, 0), font=small)
    for i, r in enumerate(rows):
        z, c = np.load(a.run / f"{r['name']}.npz"), np.load(a.run / f"{r['name']}.compare.npz")
        seq = np.load(D.BASES / f"{r['name']}.npz")["tokens"]
        face = L.render(L.decode_layers(seq))
        whole = D.eyes_on_face(face, [z[f"bins_{k}"] for k in D.LAYER_NAMES], tuple(z["ebox"]), z["fbox"])
        imgs = [c["photo"], z["base"], z["painted"], c["upper_bound"], (whole * 255).round().astype(np.uint8)]
        y0 = 64 + i * (S + 26)
        for j, img in enumerate(imgs):
            im = Image.fromarray(img).resize((S, S), Image.LANCZOS)
            if j == 0:
                dr = ImageDraw.Draw(im)
                for e in (0, 1):
                    p = z[f"plan_{e}"] * (S / 256)
                    dr.line([tuple(q) for q in p] + [tuple(p[0])], fill=(0, 200, 60), width=1)
            out.paste(im, (8 + j * (S + 8), y0))
        txt = (f"{r['name'].split('__')[1]}　眼睛 L1：底圖 {r['base']['eye_l1']:.3f} → 資料 {r['painted']['eye_l1']:.3f}"
               f"（上限 {r['upper_bound']['eye_l1']:.3f}）　眼框強度 {r['base']['rim']:.3f} → {r['painted']['rim']:.3f}"
               f"（上限 {r['upper_bound']['rim']:.3f}，照片 {r['photo_rim']:.3f}）")
        d.text((8, y0 + S + 3), txt, fill=(60, 60, 60), font=small)
    path = a.run / "compare_zh.png"
    out.save(path)
    print(path, out.size)


if __name__ == "__main__":
    main()
