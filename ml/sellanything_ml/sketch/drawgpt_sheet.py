"""Side-by-side drawings for looking, not scoring: real drawings and each model setting, same classes
and seeds (a drawing GPT row uses the same seed in every setting, so differences come from the model).

    python -m sellanything_ml.sketch.drawgpt_sheet --setting v1:0.6:1 --setting v2:0.6:1 --setting v2:0.6:2 \
        --classes dog tiger cat elephant bicycle house --seeds 0 1 2 --out runs/drawgpt_v2/sheet.png

Each setting is version:temperature:guidance. The drawings go through the game's own path
(decode, tidy, simplify), so this is what a player would see.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import drawgpt, pipeline
from .raster import simplify
from .strokes import StrokeSet, load_cached

CELL, PAD, LABEL = 128, 6, 18


def _draw(img: ImageDraw.ImageDraw, strokes, ox: int, oy: int) -> None:
    for xs, ys in strokes:
        pts = [(ox + x * CELL / 256, oy + y * CELL / 256) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            img.line(pts, fill=0, width=2)
        elif pts:
            img.point(pts, fill=0)


def sheet(settings: list[str], classes: list[str], seeds: list[int], out: Path) -> Path:
    cfg1 = drawgpt.version("v1")
    cats = drawgpt.categories(cfg1)
    centers = np.load(drawgpt._paths(cfg1)["dict"])
    sp = pipeline.splits(drawgpt._pcfg(cfg1))
    rows = ["real"] + settings
    models = {}
    for s in settings:
        ver = s.split(":")[0]
        if ver not in models:
            models[ver] = drawgpt.load_model(drawgpt.version(ver), torch.device("cpu"))
    w = len(classes) * len(seeds)
    canvas = Image.new("L", (PAD + w * (CELL + PAD) + 120, PAD + len(rows) * (CELL + PAD) + LABEL), 255)
    g = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        oy = LABEL + PAD + r * (CELL + PAD)
        g.text((PAD + w * (CELL + PAD), oy + CELL // 2), row, fill=0)
        for ci, name in enumerate(classes):
            c = cats.index(name)
            if r == 0:
                g.text((PAD + ci * len(seeds) * (CELL + PAD), 2), name, fill=0)
            for si, seed in enumerate(seeds):
                ox = PAD + (ci * len(seeds) + si) * (CELL + PAD)
                g.rectangle([ox, oy, ox + CELL, oy + CELL], outline=200)
                if row == "real":
                    base = c * cfg1.per_class
                    val = sp["val"][(sp["val"] >= base) & (sp["val"] < base + cfg1.per_class)]
                    k = int(np.random.default_rng(seed).choice(val)) - base
                    strokes = cast(StrokeSet, load_cached(name, cfg1.per_class)).strokes(k)
                else:
                    ver, t, gd = row.split(":")
                    cfg = drawgpt.version(ver)
                    (seq,) = drawgpt.draw(models[ver], c, 1, float(t), seed=seed * 1000 + c,
                                          guidance=float(gd), no_class=drawgpt.no_class_token(cfg))
                    strokes = drawgpt.tidy(drawgpt.decode(seq, centers), bool(len(seq) and seq[-1] == drawgpt.END))
                _draw(g, simplify(strokes), ox, oy)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", action="append", required=True, help="version:temperature:guidance")
    ap.add_argument("--classes", nargs="+", default=["dog", "tiger", "cat", "elephant", "bicycle", "house"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    print(sheet(a.setting, a.classes, a.seeds, a.out))


if __name__ == "__main__":
    main()
