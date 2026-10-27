"""Render stroke points into images on the GPU, at any resolution, with optional augmentation.

The 28x28 bitmaps are fixed; drawing the strokes ourselves lets the CNN see them at higher
resolution, and lets augmentation move the points before drawing (a rotated line stays sharp,
instead of a blurred resampled bitmap). The same function renders the player's strokes in the game,
so training and play see identical images.

Geometry follows Quick, Draw!: a drawing sits in a 256x256 box with 16 px of padding and is drawn
with 16 px round lines; the image is that box scaled to `side` pixels.
"""

from __future__ import annotations

import math

import torch

BOX = 256.0
PAD = 16.0
LINE = 16.0


def fit_to_box(pts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """(B, L, 2) float points -> centred in the 256 box with padding, keeping the aspect ratio."""
    B, L, _ = pts.shape
    valid = torch.arange(L, device=pts.device)[None, :] < lengths[:, None]           # (B, L)
    big = torch.finfo(pts.dtype).max
    lo = torch.where(valid[..., None], pts, torch.full_like(pts, big)).amin(1)       # (B, 2)
    hi = torch.where(valid[..., None], pts, torch.full_like(pts, -big)).amax(1)
    size = (hi - lo).amax(1).clamp(min=1.0)                                          # (B,)
    scale = (BOX - 2 * PAD) / size
    centre = (lo + hi) / 2
    return (pts - centre[:, None]) * scale[:, None, None] + BOX / 2


def augment_points(pts: torch.Tensor, gen: torch.Generator, max_degrees: float = 12,
                   scale_range: tuple[float, float] = (0.9, 1.1), max_shift: float = 0.08) -> torch.Tensor:
    """Random rotation, per-axis scale (Sketch-RNN's 0.9-1.1) and shift, applied to the points.
    Random numbers come from `gen` (CPU), so a resumed run draws the same ones."""
    B = pts.shape[0]
    angle = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(max_degrees)
    sx = torch.empty(B).uniform_(*scale_range, generator=gen)
    sy = torch.empty(B).uniform_(*scale_range, generator=gen)
    shift = (torch.rand(B, 2, generator=gen) * 2 - 1) * max_shift * BOX
    cos, sin = torch.cos(angle), torch.sin(angle)
    m = torch.stack([torch.stack([cos * sx, -sin * sy], 1), torch.stack([sin * sx, cos * sy], 1)], 1)
    m, shift = m.to(pts.device), shift.to(pts.device)
    c = pts - BOX / 2
    return torch.einsum("bij,blj->bli", m, c) + BOX / 2 + shift[:, None]


def render(points: torch.Tensor, lengths: torch.Tensor, side: int = 64,
           gen: torch.Generator | None = None, chunk: int = 48) -> torch.Tensor:
    """(B, L, 3) points (x, y, end of stroke; any numeric dtype) -> (B, 1, side, side) float in [0, 1].

    Each pixel's ink is set by its distance to the nearest line segment (anti-aliased over one
    pixel). Segments join consecutive points of the same stroke; a single-point stroke is a dot.
    """
    B, L, _ = points.shape
    dev = points.device
    xy = fit_to_box(points[..., :2].float(), lengths)
    if gen is not None:
        xy = augment_points(xy, gen)
    xy = xy * (side / BOX)                                   # pixel units
    radius = LINE / 2 * side / BOX
    idx = torch.arange(L, device=dev)
    valid = idx[None, :] < lengths[:, None]
    ends = points[..., 2] > 0
    # segment i goes from point i to point i+1 unless point i ends its stroke; a lone point (end of
    # stroke whose previous point also ended one) becomes a zero-length segment, i.e. a dot.
    nxt = torch.cat([xy[:, 1:], xy[:, -1:]], 1)
    joins = valid & ~ends & torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], 1)
    prev_end = torch.cat([torch.ones_like(ends[:, :1]), ends[:, :-1]], 1)
    dots = valid & ends & prev_end
    a = xy
    b = torch.where(joins[..., None], nxt, xy)
    use = joins | dots                                       # (B, L)

    coords = (torch.arange(side, device=dev, dtype=torch.float32) + 0.5)
    py, px = torch.meshgrid(coords, coords, indexing="ij")
    p = torch.stack([px, py], -1).view(1, 1, side * side, 2)          # (1, 1, P, 2)
    best = torch.full((B, side * side), float("inf"), device=dev)
    for s in range(0, L, chunk):
        sa, sb, su = a[:, s:s + chunk, None], b[:, s:s + chunk, None], use[:, s:s + chunk]
        ab = sb - sa                                                      # (B, C, 1, 2)
        t = ((p - sa) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)
        t = t.clamp(0, 1)
        d2 = ((sa + t[..., None] * ab - p) ** 2).sum(-1)                  # (B, C, P)
        d2 = torch.where(su[..., None], d2, torch.full_like(d2, float("inf")))
        best = torch.minimum(best, d2.amin(1))
    ink = (radius + 0.5 - best.sqrt()).clamp(0, 1)
    return ink.view(B, 1, side, side)
