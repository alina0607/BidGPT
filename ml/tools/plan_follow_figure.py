"""Figure for tools/plan_follow_l.py: the first 8 validation cats, each as its L5 training painting and as L5's painting
when handed the same cat's plan (temperature 1.0, seed 3), with the plan's eye outlines (green) and pupils (red)
drawn on top, and the eyes zoomed. Not picked.

    PYTHONPATH=.:tools python tools/plan_follow_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sketch_plan as SPL
from sellanything_ml.sketch import brushtokens_l as L

ML = Path(__file__).resolve().parents[1]
RUN = ML / "runs" / "brushgpt" / "L5"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
N, S = 8, 256


def with_plan(img: np.ndarray, seq: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """img: (SIDE, SIDE, 3) uint8."""
    p = L.decode_plan(seq)
    pu = [{"centre": q["centre"], "radii": q["radii"], "angle": q["angle"]} for q in p["pupils"]]
    sk = SPL.render_sketch([], p["eyes"], pu, None, np.zeros((0, 2)), under=img.copy(),
                           scale=1, colours=True)
    c = np.concatenate(p["eyes"]).mean(0)
    d = np.linalg.norm(p["eyes"][0].mean(0) - p["eyes"][1].mean(0))
    h = 1.2 * d
    crop = Image.fromarray(img).crop(
        (int(c[0] - h), int(c[1] - 0.6 * h), int(c[0] + h), int(c[1] + 0.6 * h))).resize((200, 120), Image.LANCZOS)
    return sk, crop


def main() -> None:
    z = np.load(ML / "data/kaggle/brush_l5/data/data_L5.npz")
    val = [str(n) for n, s in zip(z["names"], z["split"]) if s == "val"][:N]
    g = np.load(RUN / "plan_follow" / "real_plan_t1.0_s3.npz")
    big, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    out = Image.new("RGB", (N * (S + 6), 2 * (40 + S + 6 + 120 + 8) + 50), "white")
    d = ImageDraw.Draw(out)
    d.text((6, 10), "同一份構圖計畫：上面是訓練畫，下面是把這份計畫交給 L5 後它畫的（綠線＝計畫的眼眶，紅線＝瞳孔；前 8 隻驗證貓，沒有挑）",
           fill=(0, 0, 0), font=big)
    y = 50
    for label, imgs, seqs in (("訓練畫（L5 的前五層）", [np.load(ML / f"data/processed/brush_l/cats/{n}.npz")["painting"] for n in val],
                               [np.load(ML / f"data/processed/brush_l/cats/{n}.npz")["tokens"] for n in val]),
                              ("L5 照真貓的計畫畫的（溫度 1.0）", list(g["paintings"][:N]),
                               [g["tokens"][i][:g["lengths"][i]] for i in range(N)])):
        d.text((6, y + 8), label, fill=(0, 0, 0), font=big)
        y += 40
        for i in range(N):
            sk, crop = with_plan(np.asarray(imgs[i], np.uint8), seqs[i])
            out.paste(sk, (i * (S + 6), y))
            out.paste(crop, (i * (S + 6) + 28, y + S + 6))
        y += S + 6 + 120 + 8
    path = RUN / "plan_follow" / "real_plan_vs_training_zh.png"
    out.save(path)
    print(path, out.size)


if __name__ == "__main__":
    main()
