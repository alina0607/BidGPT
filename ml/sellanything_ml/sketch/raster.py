"""Turn strokes into the 28×28 bitmaps the classifier was trained on.

Quick, Draw!'s pipeline, reproduced so a player's drawing looks like the training data:

1. simplify (dataset "simplified" format): align to the top-left, scale so the longest side is
   255, resample at 1-pixel spacing, then Ramer–Douglas–Peucker with ε = 2.0;
2. rasterize (dataset "numpy_bitmap" format): center in a 256×256 box, draw 16-pixel round
   strokes with 16 pixels of padding, anti-aliased down to 28×28.

Google rendered step 2 with cairo; here we draw 8× larger with PIL and average down, which
approximates cairo's anti-aliasing. raster.py's tests compare against the real bitmaps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

Stroke = tuple[list[float], list[float]]    # ([x0, x1, ...], [y0, y1, ...])
StrokeIn = tuple[Sequence[float], Sequence[float]]   # a stroke read in: integer or float coordinates


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker: drop points closer than epsilon to the simplified line."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    norm = np.hypot(*seg)
    if norm == 0:
        dists = np.hypot(*(points - start).T)
    else:
        dists = np.abs(seg[0] * (points[:, 1] - start[1]) - seg[1] * (points[:, 0] - start[0])) / norm
    i = int(np.argmax(dists))
    if dists[i] > epsilon:
        left = _rdp(points[:i + 1], epsilon)
        right = _rdp(points[i:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _resample(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    if len(points) < 2:
        return points
    seg = np.hypot(*np.diff(points, axis=0).T)
    dist = np.concatenate([[0], np.cumsum(seg)])
    if dist[-1] == 0:
        return points[:1]
    samples = np.arange(0, dist[-1], spacing)
    if samples[-1] != dist[-1]:
        samples = np.append(samples, dist[-1])
    return np.stack([np.interp(samples, dist, points[:, 0]), np.interp(samples, dist, points[:, 1])], 1)


def simplify(strokes: Sequence[StrokeIn], epsilon: float = 2.0) -> list[Stroke]:
    """Raw strokes in any coordinates → Quick, Draw! 'simplified' strokes (0..255, integers)."""
    arrays = [np.stack([np.asarray(xs, float), np.asarray(ys, float)], 1) for xs, ys in strokes if len(xs)]
    if not arrays:
        return []
    allp = np.vstack(arrays)
    lo = allp.min(axis=0)
    extent = (allp.max(axis=0) - lo).max()
    scale = 255.0 / extent if extent > 0 else 1.0
    out = []
    for a in arrays:
        p = _rdp(_resample((a - lo) * scale), epsilon)
        if len(p) == 1:                      # a dot: the dataset stores it as two equal points
            p = np.vstack([p, p])
        p = np.round(p).astype(int)
        out.append((p[:, 0].tolist(), p[:, 1].tolist()))
    return out


def rasterize(strokes: Sequence[StrokeIn], side: int = 28, line_diameter: float = 16.0,
              padding: float = 16.0, supersample: int = 8) -> np.ndarray:
    """Simplified strokes (0..255) → (side, side) uint8 bitmap, ink = 255, background = 0."""
    big = side * supersample
    image = Image.new("L", (big, big), 0)
    if not strokes:
        return np.zeros((side, side), np.uint8)
    draw = ImageDraw.Draw(image)
    total_padding = padding * 2 + line_diameter
    scale = side / (256.0 + total_padding) * supersample
    allp = np.vstack([np.stack([xs, ys], 1) for xs, ys in strokes]).astype(float)
    offset = (256.0 - allp.max(axis=0)) / 2.0                 # center the drawing's bounding box
    width = line_diameter * scale
    radius = width / 2.0
    for xs, ys in strokes:
        pts = [((x + offset[0] + total_padding / 2) * scale, (y + offset[1] + total_padding / 2) * scale)
               for x, y in zip(xs, ys)]
        if len(pts) > 1:
            draw.line(pts, fill=255, width=max(1, int(round(width))), joint="curve")
        for x, y in (pts[0], pts[-1]) if len(pts) > 1 else pts:   # round caps
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image.reduce(supersample), dtype=np.uint8)


def strokes_to_bitmap(strokes: list[Stroke]) -> np.ndarray:
    """Player strokes (game pixel coordinates) → classifier input."""
    return rasterize(simplify(strokes))
