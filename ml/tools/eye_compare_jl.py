"""Figure: J's and L5's generated cats side by side, whole and zoomed on the eyes (owner, 2026-10-20). The first 8
samples of each, not picked: J's from its FID evaluation (temperature 1, seed 11), L5's from tools/eye_judge_l.py
(temperature 1, seed 3). The eye crop is centred where the eye judge (tools/eye_judge.py) puts the two eyes, 2.4 eye
distances wide; the judge's eye confidence is written under each crop.

    PYTHONPATH=.:tools python tools/eye_compare_jl.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eye_judge as EJ
from sellanything_ml.sketch import brushgpt as G

ML = Path(__file__).resolve().parents[1]
RUNS = ML / "runs" / "brushgpt"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
N, S, CROP = 8, 256, 200


def crop_eyes(img: np.ndarray, pts: np.ndarray) -> Image.Image:
    a, b = pts[0], pts[1]
    c, d = (a + b) / 2, max(float(np.linalg.norm(a - b)), 20.0)
    h = 1.2 * d
    im = Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))
    return im.crop((int(c[0] - h), int(c[1] - h * 0.6), int(c[0] + h), int(c[1] + h * 0.6))).resize((CROP, int(CROP * 0.6)), Image.LANCZOS)


def main() -> None:
    j = np.stack([G.render(r).clip(0, 1) for r in np.load(RUNS / "J_cat_aligned_points_eyes/fid_diversity_t1.0.npz")["tokens"][:N]])
    l5 = np.load(RUNS / "L5/eye_judge_samples_t1.0_s3.npz")["paintings"][:N] / 255
    big, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    W = N * (S + 6)
    out = Image.new("RGB", (W, 2 * (40 + S + 6 + int(CROP * 0.6) + 30) + 50), "white")
    d = ImageDraw.Draw(out)
    d.text((6, 10), "J 與 L5 生成的新貓：整張臉與眼睛放大（各取前 8 張，沒有挑；溫度 1.0；每張下方是眼睛評審的信心）", fill=(0, 0, 0), font=big)
    y = 50
    for name, imgs in (("J（完整模型 + CAT，第 7,300 步）", j), ("L5（數位版五層，第 2,100 步）", l5)):
        pts, conf = EJ.judge(imgs.astype(np.float32))
        d.text((6, y + 8), name, fill=(0, 0, 0), font=big)
        y += 40
        for i in range(N):
            out.paste(Image.fromarray((imgs[i].clip(0, 1) * 255).astype(np.uint8)).resize((S, S)), (i * (S + 6), y))
            out.paste(crop_eyes(imgs[i], pts[i, :2]), (i * (S + 6) + (S - CROP) // 2, y + S + 6))
            d.text((i * (S + 6) + 70, y + S + 6 + int(CROP * 0.6) + 4), f"眼睛 {conf[i, :2].mean():.2f}", fill=(0, 0, 0), font=small)
        y += S + 6 + int(CROP * 0.6) + 30
    path = RUNS / "L5" / "eyes_J_vs_L5_zh.png"
    out.save(path)
    print(path, out.size)


if __name__ == "__main__":
    main()
