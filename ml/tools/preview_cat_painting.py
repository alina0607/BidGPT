"""Render already-optimised cat strokes into a deterministic progress contact sheet.

This never runs gradient descent or claims a GPU: it only replays saved stages on CPU.

    cd ml
    python tools/preview_cat_painting.py data/processed/afhq_brush_face ../docs/images/cat_painting_process.png
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sellanything_ml.sketch import catbrush

ML = Path(__file__).resolve().parents[1]
AFHQ = ML / "data" / "raw" / "afhq"
MASKS = ML / "data" / "processed" / "afhq_masks"
LABELS = ("photo", "24 strokes", "88 strokes", "248 strokes", "608 strokes")
SIDE = catbrush.SIDE


def _source_for(stem: str) -> Path:
    matches = list(AFHQ.glob(f"*/cat/{stem}.jpg")) + list(AFHQ.glob(f"*/cat/{stem}.png"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one AFHQ source for {stem}, found {len(matches)}")
    return matches[0]


def render(npz_path: Path) -> list[np.ndarray]:
    saved = np.load(npz_path)
    source = _source_for(npz_path.stem)
    mask = MASKS / f"{npz_path.stem}_mask1.png"
    photo = catbrush.load(str(source), str(mask), tuple(int(x) for x in saved["box"]))
    yy, xx = torch.meshgrid(torch.arange(SIDE) + 0.5, torch.arange(SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xx, yy], -1)
    canvas = torch.ones((SIDE, SIDE, 3), dtype=torch.float32)
    frames = [photo.target.numpy()]
    for k, (length, width, _) in enumerate(catbrush.STAGES[:4]):
        stage = {key: saved[f"s{k}_{key}"] for key in ("xy", "length_width", "angle", "colour", "alpha")}
        with torch.no_grad():
            canvas = catbrush.composite(canvas, catbrush.Strokes.from_stage(stage, (length, width)), grid)
        frames.append(canvas.numpy())
    return frames


def main() -> None:
    source_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    candidates = sorted(path for path in source_dir.glob("*.npz") if path.stem != "log")
    chosen = random.Random(20261010).sample(candidates, min(count, len(candidates)))
    label_h, name_w = 36, 104
    sheet = Image.new("RGB", (name_w + SIDE * len(LABELS), label_h + SIDE * len(chosen)), "#f7efe3")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for col, label in enumerate(LABELS):
        draw.text((name_w + col * SIDE + 8, 12), label, fill="#633a2d", font=font)
    for row, path in enumerate(chosen):
        y = label_h + row * SIDE
        draw.text((8, y + SIDE // 2), path.stem, fill="#633a2d", font=font)
        for col, frame in enumerate(render(path)):
            tile = Image.fromarray((frame.clip(0, 1) * 255).astype(np.uint8))
            sheet.paste(tile, (name_w + col * SIDE, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"{output}: {', '.join(path.stem for path in chosen)}")


if __name__ == "__main__":
    main()
