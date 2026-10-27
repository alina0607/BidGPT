"""Figure for tools/eye_layers_l.py: the first 8 cats as the photo crop with both outlines (green: the plan's, orange:
traced ray by ray), the old result (old base, ray outline: zh notebook section 14.5), the L8 base alone, and the four
layers on the L8 base with each outline. Not picked.

    PYTHONPATH=.:tools python tools/eye_layers_l_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ML = Path(__file__).resolve().parents[1]
RUN = ML / "runs" / "eye_layers_l" / "24cats"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
COLS = [("photo", "照片（綠＝計畫，橘＝射線）"), ("old_rays", "舊底圖＋射線外框"), ("l8_face", "L8 底圖（未加四層）"),
        ("l8_rays", "L8＋四層（射線外框）"), ("l8_plan", "L8＋四層（計畫的眼眶）")]
N, S = 8, 200


def main() -> None:
    files = sorted(RUN.glob("*.npz"))[:N]
    big, small = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    out = Image.new("RGB", (len(COLS) * (S + 8) + 8, N * (S + 6) + 70), "white")
    d = ImageDraw.Draw(out)
    d.text((8, 8), "眼睛模型的資料（前 8 隻驗證貓，沒有挑）", fill=(0, 0, 0), font=big)
    for j, (_, cap) in enumerate(COLS):
        d.text((8 + j * (S + 8), 40), cap, fill=(0, 0, 0), font=small)
    for i, f in enumerate(files):
        z = np.load(f)
        for j, (k, _) in enumerate(COLS):
            im = Image.fromarray((z[k].clip(0, 1) * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
            if k == "photo":
                dr = ImageDraw.Draw(im)
                sc = S / 256
                for key, col in (("rays", (255, 140, 0)), ("plan", (0, 200, 60))):
                    for e in (0, 1):
                        c = z[f"{key}_{e}"][:, :2] * sc
                        dr.line([tuple(p) for p in c] + [tuple(c[0])], fill=col, width=1)
            out.paste(im, (8 + j * (S + 8), 64 + i * (S + 6)))
    path = RUN / "compare_zh.png"
    out.save(path)
    print(path, out.size)


if __name__ == "__main__":
    main()
