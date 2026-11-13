"""A cat face painted in the order of a real oil painting, with the fur strokes laid along the fur's growth.

catbrush.py fits every stage to the colour photo, big brush to small, and starts each stroke along the image's
local structure (which on a tabby follows the stripes, across the fur), then lets the angle drift wherever the L1
loss pulls it. This painter keeps catbrush's stroke model (centre, length, width, angle, colour, opacity, painted
one over another) and its direct optimisation, but changes what each stage is fitted to and what it may do,
following the studio steps the owner wrote down on 2026-10-16:

    1  sketch        charcoal lines: the silhouette and the big structure, no fur
    2  fix lines     the charcoal dusted off (redrawn at DUST of its opacity), key lines restated in thin umber
    3  underpaint    one umber hue, light to dark: the cat's light and shadow as a sculpture
    4  block-in      local colour from a blurred photo with highlights held down, strokes along the fur
    5  features      eyes, nose and mouth from the sharp photo, only near the face points
    6a dark fur      tapered strokes along the fur that may only darken the canvas
    6b mid fur       tapered strokes along the fur, the sharp photo
    6c light fur     dry-brush tapered strokes along the fur that may only lighten the canvas
    7a whiskers      long hair-thin strokes from the whisker pads and above the eyes
    7b ear hairs     short hair-thin strokes inside the ears
    8  glaze         a few large, nearly transparent, soft-edged strokes over the whole cat

"Along the fur" is a field built from the nine face points: fur leaves the nose (up the bridge and over the
forehead, out across the cheeks and drooping below the eye line, down the chin), runs along the eye's axis around
the eyes and from base to tip in the ears; where the photo's own fine hair texture is coherent and roughly agrees,
the field bends toward it. Fur strokes may turn at most `Step.dev` away from the field at their centre.

Each finished stage is snapped to token bins like brushtokens (1 px positions, 32 angles over [0, π), 6 × 6 sizes
within 0.5–1.5 × the stage's brush, 12 levels a colour channel, 4 opacity levels within the stage's own range), so
the painting is exactly what its tokens say. A decoder needs nothing but the strokes: the field and the targets are
only used to fit them. 600 strokes are 3,000 tokens, about the 3,040 of catbrush's four stages.

    python -m sellanything_ml.sketch.oilprocess <photo.png> <mask.png> <points.npy> <out.png>
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import cast
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import catbrush as B

SIDE = B.SIDE
ITERS = B.ITERS
DUST = 0.35                                  # the charcoal's opacity after it is dusted off
CHARCOAL = (0.20, 0.19, 0.18)
UMBER_LINE = (0.33, 0.20, 0.11)
UMBER_DARK = (0.24, 0.15, 0.09)              # burnt umber, thick
UMBER_WASH = (0.86, 0.78, 0.66)              # burnt umber thinned with turpentine
NXY, NANGLE, LEVELS_SIZE, LEVELS_RGB, LEVELS_ALPHA = 256, 32, 6, 12, 4


@dataclass(frozen=True)
class Step:
    name: str
    base: tuple[float, float]                # brush (length, width) in px; strokes are 0.5–1.5 × this
    n: int
    shape: str                               # "flat" (catbrush's), "taper" (pointed both ends), "glaze" (soft edge)
    alpha: tuple[float, float]
    colour: str                              # "charcoal" | "umber" (fixed) | "value" (umber ramp) | "free"
    target: str                              # see Guides.target
    region: str                              # see Guides.regions; outside it the target is "leave the canvas"
    angle: str = "free"                      # "free" (starts along the target's structure) | "flow" (± dev)
    dev: float = 0.0                         # radians
    dry: bool = False                        # broken bristle coverage
    weight: float = 0.0                      # extra loss weight inside the region (1 + weight there)
    protect: float = 0.0                     # extra loss weight on the eyes, where the target is the canvas itself


STEPS = [
    Step("1 構圖打型", (48, 2.5), 28, "taper", (0.35, 0.8), "charcoal", "lines_coarse", "all"),
    Step("2 拍粉固定", (20, 2.0), 24, "taper", (0.5, 0.9), "umber", "lines_fine", "keys"),
    Step("3 單色打底", (64, 28), 40, "flat", (0.5, 1.0), "value", "value", "all"),
    Step("4 固有色鋪底", (40, 16), 80, "flat", (0.6, 1.0), "free", "blocks", "cat", "flow", math.radians(30)),
    Step("5 五官", (12, 5), 64, "flat", (0.5, 1.0), "free", "photo", "features"),
    Step("6a 底毛（暗）", (22, 5), 96, "taper", (0.4, 0.9), "free", "darker", "fur", "flow", math.radians(15)),
    Step("6b 中層毛", (16, 4), 120, "taper", (0.5, 1.0), "free", "photo", "fur", "flow", math.radians(15)),
    Step("6c 表層亮毛", (14, 3), 96, "taper", (0.4, 0.9), "free", "lighter", "fur", "flow", math.radians(15), True),
    Step("7a 鬍鬚", (56, 1.5), 16, "taper", (0.6, 1.0), "free", "photo", "whiskers", "flow", math.radians(40)),
    Step("7b 耳內細毛", (12, 1.2), 24, "taper", (0.5, 1.0), "free", "photo", "ears", "flow", math.radians(25)),
    Step("8 罩染", (96, 40), 12, "glaze", (0.05, 0.3), "free", "photo", "cat"),
]

# The third painter (2026-10-17): the same process with the eyes painted the way the studio steps say — iris, then
# pupil and lids, then a catchlight at the very end — after the pilot showed STEPS losing the eyes (notebook 08 §10.4):
#   1. the block-in takes the sharp photo inside the eyes (no blur, no held-down highlights), so the umber
#      underpainting is painted over in the eyes' own colour;
#   2. each eye gets its own strokes, large to small (5a iris 14 × 8, 5b pupil and lids 8 × 3, 20 + 20 an eye), with
#      the eye region's error weighted 5 × (Singh & Zheng 2021; AttentionPainter 2025), nose and mouth apart;
#   3. fur, whiskers and glaze avoid the eyes: their region leaves the eyes out and changing the eyes costs 11 ×
#      (Collomosse & Hall 2002; Wang et al. 2013) — a penalty, not a hard clip, so the strokes stay plain tokens;
#   4. four small strokes that may only lighten the eye cores come last, after the glaze (the catchlight).
# 604 strokes: the fur gives up 40 of its strokes to the eyes.
STEPS_EYES = [
    *STEPS[:3],
    Step("4 固有色鋪底", (40, 16), 80, "flat", (0.6, 1.0), "free", "blocks_eyes", "cat", "flow", math.radians(30)),
    Step("5a 虹膜", (14, 8), 40, "flat", (0.6, 1.0), "free", "photo", "eyes", weight=4),
    Step("5b 瞳孔與眼瞼", (8, 3), 40, "flat", (0.6, 1.0), "free", "photo", "eyes", weight=4),
    Step("5c 鼻與嘴", (12, 5), 24, "flat", (0.5, 1.0), "free", "photo", "nose_mouth"),
    Step("6a 底毛（暗）", (22, 5), 80, "taper", (0.4, 0.9), "free", "darker", "fur_eyes", "flow", math.radians(15), protect=10),
    Step("6b 中層毛", (16, 4), 96, "taper", (0.5, 1.0), "free", "photo", "fur_eyes", "flow", math.radians(15), protect=10),
    Step("6c 表層亮毛", (14, 3), 96, "taper", (0.4, 0.9), "free", "lighter", "fur_eyes", "flow", math.radians(15), True,
         protect=10),
    Step("7a 鬍鬚", (56, 1.5), 16, "taper", (0.6, 1.0), "free", "photo", "whiskers_eyes", "flow", math.radians(40),
         protect=10),
    STEPS[9],
    Step("8 罩染", (96, 40), 12, "glaze", (0.05, 0.3), "free", "photo", "cat", protect=4),
    Step("9 高光", (4, 3), 4, "flat", (0.6, 1.0), "free", "lighter", "catchlight"),
]
PAINTERS = {"oil": STEPS, "oil_eyes": STEPS_EYES}


def fur_steps(steps: list[Step]) -> list[int]:
    return [k for k, s in enumerate(steps) if s.name.startswith("6")]


FUR_STEPS = fur_steps(STEPS)
DENSE_ABOVE = 1 / 3


# ---------------------------------------------------------------- image helpers

def _gauss(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W) or (H, W, C) image, edges reflected."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = img[None, None] if img.dim() == 2 else img.permute(2, 0, 1)[:, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x[0, 0] if img.dim() == 2 else x[:, 0].permute(1, 2, 0)


def _sobel(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8
    x = F.pad(g[None, None], (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x, kx)[0, 0], F.conv2d(x, kx.transpose(2, 3))[0, 0]


def _lum(img: torch.Tensor) -> torch.Tensor:
    return img @ torch.tensor([0.299, 0.587, 0.114])


def texture(img: torch.Tensor, window: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Fine-scale orientation along the hair texture (perpendicular to the gradient) and its coherence in [0, 1]."""
    gx, gy = _sobel(_lum(img))
    jxx, jyy, jxy = (_gauss(t, window) for t in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy) + math.pi / 2
    coh = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return theta, coh


def _doubled(theta: torch.Tensor) -> torch.Tensor:
    """Orientation as (cos 2θ, sin 2θ): a stroke at θ and at θ + π is the same stroke."""
    return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta)], -1)


def _angle(doubled: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.atan2(doubled[..., 1], doubled[..., 0])


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a − b| between orientations, in [0, π/2]."""
    d = np.mod(np.asarray(a) - np.asarray(b), math.pi)
    return np.minimum(d, math.pi - d)


# ---------------------------------------------------------------- face geometry and the fur field

class Face:
    """The nine face points on the canvas: eyes 0–1, mouth 2, left ear 3 (outer base), 4 (tip), 5 (inner base),
    right ear 6 (inner base), 7 (tip), 8 (outer base)."""

    def __init__(self, points: np.ndarray):
        p = torch.as_tensor(points, dtype=torch.float32)
        self.p = p
        self.eye_l, self.eye_r, self.mouth = p[0], p[1], p[2]
        mid = (p[0] + p[1]) / 2
        self.d = float(torch.linalg.norm(p[1] - p[0]))
        self.nose = mid + 0.7 * (p[2] - mid)
        self.right = (p[1] - p[0]) / self.d
        self.up = (mid - p[2]) / torch.linalg.norm(mid - p[2])
        self.ears = [(p[3], p[4], p[5]), (p[8], p[7], p[6])]   # (outer base, tip, inner base)

    @staticmethod
    def from_photo(points: np.ndarray, box: tuple[int, int, int, int]) -> Face:
        """Points in photo pixels, moved onto the canvas that `catbrush.load(..., box)` cuts from that photo."""
        left, top, right, _ = box
        return Face((np.asarray(points, float) - [left, top]) * SIDE / (right - left))


def _grid() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(SIDE) + 0.5, torch.arange(SIDE) + 0.5, indexing="ij")
    return torch.stack([xs, ys], -1)


def _disc(grid: torch.Tensor, c: torch.Tensor, r: float, soft: float = 2.0) -> torch.Tensor:
    return torch.sigmoid((r - torch.linalg.norm(grid - c, dim=-1)) / soft)


def _triangle(grid: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, soft: float = 2.0,
              shrink: float = 0.0) -> torch.Tensor:
    centroid = (a + b + c) / 3
    a, b, c = (centroid + (1 - shrink) * (v - centroid) for v in (a, b, c))
    dist = []
    for p, q, other in ((a, b, c), (b, c, a), (c, a, b)):
        e = q - p
        n = torch.stack([-e[1], e[0]]) / torch.linalg.norm(e)
        if torch.dot(other - p, n) < 0:
            n = -n
        dist.append((grid - p) @ n)
    return torch.sigmoid(torch.stack(dist).min(0).values / soft)


def template_field(face: Face, grid: torch.Tensor) -> torch.Tensor:
    """The fur's direction from anatomy alone, as doubled-angle unit vectors (H, W, 2)."""
    rel = grid - face.nose
    v = rel / (torch.linalg.norm(rel, dim=-1, keepdim=True) + 1e-6)
    below = torch.sigmoid(-(rel @ face.up) / (0.3 * face.d))
    lateral = ((rel @ face.right).abs() / face.d).clamp(0, 1)
    v = v - (0.45 * below * lateral)[..., None] * face.up
    radial = _doubled(torch.atan2(v[..., 1], v[..., 0]))
    eye_dir = _doubled(torch.atan2(face.right[1], face.right[0])).expand_as(radial)
    w_eye = cast(torch.Tensor, sum(torch.exp(-(torch.linalg.norm(grid - e, dim=-1) / (0.45 * face.d)) ** 2)
                                   for e in (face.eye_l, face.eye_r)))
    parts, weights = [radial, eye_dir], [torch.ones_like(w_eye), 2.0 * w_eye]
    for outer, tip, inner in face.ears:
        root = (outer + inner) / 2
        parts.append(_doubled(torch.atan2(tip[1] - root[1], tip[0] - root[0])).expand_as(radial))
        weights.append(6.0 * _triangle(grid, outer, tip, inner, soft=4.0))
    out = cast(torch.Tensor, sum(w[..., None] * p for w, p in zip(weights, parts)))
    return out / (torch.linalg.norm(out, dim=-1, keepdim=True) + 1e-6)


def fur_field(face: Face, target: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(field, template): the template bent toward the photo's fine hair texture where that texture is coherent and
    within about 45° of it, then smoothed; both as doubled-angle unit vectors (H, W, 2)."""
    tmpl = template_field(face, grid)
    theta, coh = texture(target)
    img = _doubled(theta)
    agree = (tmpl * img).sum(-1).clamp(min=0)                # cos 2Δ: 1 when parallel, 0 at 45° and beyond
    w = (0.6 * coh * agree ** 2)[..., None]
    f = _gauss((1 - w) * tmpl + w * img, 2.0)
    return f / (torch.linalg.norm(f, dim=-1, keepdim=True) + 1e-6), tmpl


# ---------------------------------------------------------------- what each step is fitted to

class Guides:
    """Regions, targets and the fur field of one photo."""

    def __init__(self, photo: B.Photo, face: Face):
        self.photo, self.face = photo, face
        grid = _grid()
        t, m, d = photo.target, photo.mask, face.d
        self.field, self.template = fur_field(face, t, grid)
        lum = _lum(t)
        edge = _sobel(_gauss(m, 1.5))
        band = torch.sqrt(edge[0] ** 2 + edge[1] ** 2)
        self.band = (band / band.max()).clamp(0, 1)
        eyes = torch.maximum(_disc(grid, face.eye_l, 0.34 * d), _disc(grid, face.eye_r, 0.34 * d))
        eye_core = torch.maximum(_disc(grid, face.eye_l, 0.24 * d), _disc(grid, face.eye_r, 0.24 * d))
        nose, mouth = _disc(grid, face.nose, 0.25 * d), _disc(grid, face.mouth, 0.25 * d)
        tips = [_disc(grid, tip, 0.3 * d) for _, tip, _ in face.ears]
        keys = torch.stack([_disc(grid, face.eye_l, 0.45 * d), _disc(grid, face.eye_r, 0.45 * d),
                            _disc(grid, face.nose, 0.35 * d), _disc(grid, face.mouth, 0.3 * d), *tips,
                            self.band]).max(0).values
        rel = grid - face.nose
        side, down = (rel @ face.right).abs() / d, -(rel @ face.up) / d
        pads = torch.sigmoid((side - 0.2) / 0.05) * torch.sigmoid((1.6 - side) / 0.05) \
            * torch.sigmoid((down + 0.3) / 0.05) * torch.sigmoid((0.6 - down) / 0.05)
        brows = torch.maximum(*(_disc(grid, e + 0.5 * d * face.up, 0.3 * d) for e in (face.eye_l, face.eye_r)))
        near_cat = (_gauss(m, 3.0) > 0.2).float()
        ears = torch.maximum(*(_triangle(grid, o, tip, i, shrink=0.15) for o, tip, i in face.ears))
        whiskers = torch.maximum(pads, brows) * near_cat
        self.eyes = eyes * near_cat
        self.regions = {
            "all": torch.ones_like(m), "cat": m, "keys": keys,
            "features": torch.stack([eyes, nose, mouth]).max(0).values * near_cat,
            "fur": m * (1 - eye_core),
            "whiskers": whiskers,
            "ears": ears * m,
            "eyes": self.eyes, "nose_mouth": torch.maximum(nose, mouth) * near_cat,
            "catchlight": torch.maximum(_disc(grid, face.eye_l, 0.2 * d), _disc(grid, face.eye_r, 0.2 * d)) * near_cat,
            "fur_eyes": m * (1 - eyes), "whiskers_eyes": whiskers * (1 - eyes),
        }
        self.lines = {"lines_coarse": self._lines(lum, 2.5, m, 0.93, 0.99),
                      "lines_fine": self._lines(lum, 1.0, keys * m, 0.88, 0.98)}
        lo, hi = (torch.quantile(lum[m > 0.5], q) for q in (0.05, 0.95))
        value = _gauss(((lum - lo) / (hi - lo)).clamp(0, 1), 2.0)[..., None]
        ramp = torch.tensor(UMBER_DARK) + (torch.tensor(UMBER_WASH) - torch.tensor(UMBER_DARK)) * value
        self.value = m[..., None] * ramp + (1 - m[..., None])
        self.value_level = value[..., 0]
        mm = m[..., None]
        blur = _gauss(t * mm, 4.0) / (_gauss(m, 4.0)[..., None] + 1e-4)
        self.blocks = mm * (0.08 + 0.78 * blur.clamp(0, 1)) + (1 - mm)
        self.blocks_eyes = self.eyes[..., None] * t + (1 - self.eyes[..., None]) * self.blocks

    def _lines(self, lum, sigma, where, q_lo, q_hi) -> torch.Tensor:
        gx, gy = _sobel(_gauss(lum, sigma))
        g = torch.sqrt(gx ** 2 + gy ** 2)
        lo, hi = (torch.quantile(g[where > 0.5], q) for q in (q_lo, q_hi))
        return torch.maximum(((g - lo) / (hi - lo)).clamp(0, 1) * where, self.band)

    def target(self, step: Step, start: torch.Tensor) -> torch.Tensor:
        """What the step's strokes are fitted to, over the canvas they start from; outside the step's region the
        target is the canvas itself, so strokes there only cost."""
        t = self.photo.target
        if step.target in self.lines:
            s = self.lines[step.target][..., None]
            ink = torch.tensor(CHARCOAL if step.colour == "charcoal" else UMBER_LINE)
            raw = start * (1 - s) + ink * s
        else:
            raw = {"value": self.value, "blocks": self.blocks, "blocks_eyes": self.blocks_eyes, "photo": t,
                   "darker": torch.minimum(t, start), "lighter": torch.maximum(t, start)}[step.target]
        r = self.regions[step.region][..., None]
        return r * raw + (1 - r) * start

    def weight(self, step: Step) -> torch.Tensor:
        """The per-pixel loss weight of a step: 1, plus `weight` in its region, plus `protect` on the eyes."""
        return 1 + step.weight * self.regions[step.region] + step.protect * self.eyes

    def start_angles(self, step: Step, target: torch.Tensor) -> torch.Tensor:
        if step.angle == "flow":
            return _angle(self.field)
        return B.orientation(target)


# ---------------------------------------------------------------- brushes and compositing

def taper(brush: B.Brush, px: torch.Tensor, py: torch.Tensor, dry: bool = False):
    """A hair stroke: full width in the middle, pointed at both ends, with bristle streaks (broken when dry)."""
    dx, dy = px - brush.x0, py - brush.y0
    u, v = dx * brush.cos + dy * brush.sin, -dx * brush.sin + dy * brush.cos
    t = (u / brush.half_length).clamp(-1, 1)
    hw = brush.half_width * (1 - t * t).clamp(min=0) ** 0.7
    cover = torch.sigmoid((hw - v.abs()) / B.EDGE_PX) * torch.sigmoid((brush.half_length - u.abs()) / B.EDGE_PX)
    phase = v / brush.half_width
    if dry:
        cover = cover * (0.35 + 0.65 * (0.5 + 0.5 * torch.sin(phase * 6 + brush.x0 * 1.7)))
    return cover, 1 + 0.06 * torch.sin(phase * 9 + brush.x0)


GLAZE_EDGE_PX = 6.0


def glaze(brush: B.Brush, px: torch.Tensor, py: torch.Tensor, dry: bool = False):
    """catbrush's rounded rectangle with a 6 px soft edge and no bristles: a thin film of transparent paint."""
    dx, dy = px - brush.x0, py - brush.y0
    u, v = dx * brush.cos + dy * brush.sin, -dx * brush.sin + dy * brush.cos
    rho = 0.5 * brush.half_width
    qx, qy = u.abs() - (brush.half_length - rho).clamp(min=0), v.abs() - (brush.half_width - rho)
    outside = (qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-8).sqrt()
    sdf = outside + torch.maximum(qx, qy).clamp(max=0) - rho
    cover = torch.sigmoid(-sdf / GLAZE_EDGE_PX)
    return cover, torch.ones_like(cover)


def flat(brush: B.Brush, px: torch.Tensor, py: torch.Tensor, dry: bool = False):
    return B.rounded_rect(brush, px, py)


SHAPES: dict[str, Callable] = {"flat": flat, "taper": taper, "glaze": glaze}


def _composite_dense(canvas, p: B.StrokeParams, grid, shape) -> torch.Tensor:
    cover, shade = shape(B.Brush.of(p, 2), grid[None, ..., 0], grid[None, ..., 1])
    a = cover * p.alpha[:, None, None]
    keep = torch.flip(torch.cumprod(torch.flip(1 - a, [0]), 0), [0])
    after = torch.cat([keep[1:], torch.ones_like(keep[:1])])
    paint = (a * after)[..., None] * (p.colour[:, None, None, :] * shade[..., None]).clamp(0, 1)
    return paint.sum(0) + canvas * keep[0][..., None]


def _composite_windowed(canvas, p: B.StrokeParams, half: int, shape) -> torch.Tensor:
    """catbrush.composite_windowed with the brush's shape as a parameter."""
    h, w = canvas.shape[:2]
    n, dev = len(p.xy), p.xy.device
    off = torch.arange(-half, half + 1, device=dev)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ix = p.xy[:, 0].detach().floor().long()[:, None] + ox.reshape(1, -1)
    iy = p.xy[:, 1].detach().floor().long()[:, None] + oy.reshape(1, -1)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    cover, shade = shape(B.Brush.of(p, 1), ix + 0.5, iy + 0.5)
    a = (cover * p.alpha[:, None] * valid).clamp(max=1 - 1e-6)
    col = (p.colour[:, None, :] * shade[..., None]).clamp(0, 1)
    pix = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).reshape(-1)
    order = torch.argsort(pix * n + torch.arange(n, device=dev).repeat_interleave(a.shape[1]))
    px_s, a_s, col_s = pix[order], a.reshape(-1)[order], col.reshape(-1, 3)[order]
    first = torch.ones_like(px_s, dtype=torch.bool)
    first[1:] = px_s[1:] != px_s[:-1]
    seg = torch.cumsum(first.long(), 0) - 1
    starts = torch.nonzero(first).squeeze(1)
    rank = torch.arange(len(px_s), device=dev) - starts[seg]
    log_keep = torch.zeros(len(starts), int(rank.max()) + 1, device=dev).index_put((seg, rank), torch.log1p(-a_s))
    later = torch.flip(torch.cumsum(torch.flip(log_keep, [1]), 1), [1]) - log_keep
    paint = torch.zeros(h * w, 3, device=dev).index_add(0, px_s, (a_s * torch.exp(later[seg, rank]))[:, None] * col_s)
    keep = torch.zeros(h * w, device=dev).index_add(0, px_s[starts], log_keep.sum(1))
    return (canvas.reshape(-1, 3) * torch.exp(keep)[:, None] + paint).reshape(h, w, 3)


def composite(canvas: torch.Tensor, p: B.StrokeParams, step: Step, grid: torch.Tensor) -> torch.Tensor:
    shape = SHAPES[step.shape]
    if step.dry:
        def shape(brush, px, py, _s=shape):                   # noqa: E306
            return _s(brush, px, py, dry=True)
    margin = 4 * B.EDGE_PX + 2
    half = int(math.ceil(math.hypot(0.75 * step.base[0], 0.75 * step.base[1]) + margin))
    if step.shape == "glaze" or 2 * half + 1 > DENSE_ABOVE * SIDE:
        return _composite_dense(canvas, p, grid, shape)
    return _composite_windowed(canvas, p, half, shape)


# ---------------------------------------------------------------- one step's strokes

def _logit(p) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


class StepStrokes:
    """Unconstrained parameters of one step's strokes."""

    def __init__(self, step: Step, xy, angle0, colour0, field: torch.Tensor):
        n = len(xy)
        self.step, self.field = step, field
        self.xy = _logit(torch.as_tensor(xy) / SIDE)
        self.size = torch.zeros(n, 2)
        self.alpha = torch.full((n,), 1.5)
        self.angle = torch.zeros(n) if step.angle == "flow" else torch.as_tensor(angle0, dtype=torch.float32)
        self.colour: torch.Tensor | None
        if step.colour == "free":
            self.colour = _logit(colour0)
        elif step.colour == "value":
            self.colour = _logit(colour0)                   # (n,) position on the umber ramp
        else:
            self.colour = None

    def tensors(self) -> list[torch.Tensor]:
        return [t for t in (self.xy, self.size, self.angle, self.colour, self.alpha) if t is not None]

    def flow_at(self, xy: torch.Tensor) -> torch.Tensor:
        g = (xy.detach() / SIDE * 2 - 1).view(1, 1, -1, 2)
        f = F.grid_sample(self.field.permute(2, 0, 1)[None], g, align_corners=False)[0, :, 0].T
        return _angle(f)

    def params(self) -> B.StrokeParams:
        s = self.step
        xy = torch.sigmoid(self.xy) * SIDE
        lw = (0.5 + torch.sigmoid(self.size)) * torch.tensor(s.base)
        angle = self.flow_at(xy) + s.dev * torch.tanh(self.angle) if s.angle == "flow" else self.angle
        n = len(xy)
        if s.colour == "free":
            colour = torch.sigmoid(cast(torch.Tensor, self.colour))
        elif s.colour == "value":
            level = torch.sigmoid(cast(torch.Tensor, self.colour))[:, None]
            colour = torch.tensor(UMBER_DARK) + (torch.tensor(UMBER_WASH) - torch.tensor(UMBER_DARK)) * level
        else:
            colour = torch.tensor(CHARCOAL if s.colour == "charcoal" else UMBER_LINE).expand(n, 3)
        a_lo, a_hi = s.alpha
        return B.StrokeParams(xy, lw, angle, colour, a_lo + (a_hi - a_lo) * torch.sigmoid(self.alpha))


def _bin(x, lo, hi, n):
    return np.clip(((np.asarray(x) - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _centre(i, lo, hi, n):
    return lo + (np.asarray(i) + 0.5) * (hi - lo) / n


def snap(st: B.Stage, step: Step) -> B.Stage:
    """The step's strokes moved to the centres of their token bins."""
    base = np.array(step.base, float)
    return {"xy": _centre(_bin(st["xy"], 0, SIDE, NXY), 0, SIDE, NXY).astype(np.float32),
            "length_width": (_centre(_bin(st["length_width"] / base, 0.5, 1.5, LEVELS_SIZE), 0.5, 1.5, LEVELS_SIZE)
                             * base).astype(np.float32),
            "angle": _centre(_bin(np.mod(st["angle"], math.pi), 0, math.pi, NANGLE), 0, math.pi,
                             NANGLE).astype(np.float32),
            "colour": _centre(_bin(st["colour"], 0, 1, LEVELS_RGB), 0, 1, LEVELS_RGB).astype(np.float32),
            "alpha": _centre(_bin(st["alpha"], *step.alpha, LEVELS_ALPHA), *step.alpha, LEVELS_ALPHA)
            .astype(np.float32)}


def stage_params(st: B.Stage) -> B.StrokeParams:
    t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in st.items()}
    return B.StrokeParams(t["xy"], t["length_width"], t["angle"], t["colour"], t["alpha"])


def dusted(st: B.Stage) -> B.Stage:
    return {**st, "alpha": st["alpha"] * DUST}


def render(stages: list[B.Stage], steps: list[Step] = STEPS) -> list[np.ndarray]:
    """The canvas after each step, from the strokes alone (the sketch is dusted before step 2)."""
    grid = _grid()
    canvas, frames = torch.ones(SIDE, SIDE, 3), []
    with torch.no_grad():
        for k, (st, step) in enumerate(zip(stages, steps, strict=True)):
            if k == 1:
                canvas = composite(torch.ones(SIDE, SIDE, 3), stage_params(dusted(stages[0])), steps[0], grid)
            canvas = composite(canvas, stage_params(st), step, grid)
            frames.append(canvas.numpy())
    return frames


# ---------------------------------------------------------------- painting

def new_strokes(step: Step, guides: Guides, start: torch.Tensor, target: torch.Tensor,
                rng: np.random.Generator) -> StepStrokes:
    """Strokes placed with probability ∝ how far the canvas is from the target, starting in the target's colour."""
    region = guides.regions[step.region].flatten().numpy()
    err = (target - start).abs().sum(-1).flatten().numpy() + 1e-6 * region + 1e-9   # a region with no error left
    #                                                                                  is still where strokes go
    idx = rng.choice(err.size, step.n, replace=False, p=err / err.sum())
    ys, xs = idx // SIDE, idx % SIDE
    angle0 = guides.start_angles(step, target)[ys, xs]
    colour0 = guides.value_level[ys, xs] if step.colour == "value" else target[ys, xs]
    return StepStrokes(step, np.c_[xs, ys] + 0.5, angle0, colour0, guides.field)


def paint(photo: B.Photo, face: Face, seed: int = 0, iters: int = ITERS, log: Callable[[str], None] = print,
          steps: list[Step] = STEPS) -> tuple[list[B.Stage], list[np.ndarray], Guides]:
    """The steps' snapped strokes, the canvas after each step, and the guides they were fitted with."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    guides, grid = Guides(photo, face), _grid()
    canvas = torch.ones(SIDE, SIDE, 3)
    stages: list[B.Stage] = []
    frames: list[np.ndarray] = []
    for k, step in enumerate(steps):
        if k == 1:
            canvas = composite(torch.ones(SIDE, SIDE, 3), stage_params(dusted(stages[0])), steps[0], grid)
        start = canvas.detach()
        target = guides.target(step, start)
        w = guides.weight(step)[..., None]
        s = new_strokes(step, guides, start, target, rng)
        for t in s.tensors():
            t.requires_grad_(True)
        opt = torch.optim.Adam(s.tensors(), lr=0.05)
        for _ in range(iters):
            loss = (w * (composite(start, s.params(), step, grid) - target).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = s.params()
            st = snap({"xy": p.xy.numpy(), "length_width": p.lw.numpy(), "angle": p.angle.numpy(),
                       "colour": p.colour.numpy(), "alpha": p.alpha.numpy()}, step)
            canvas = composite(start, stage_params(st), step, grid)
        stages.append(st)
        frames.append(canvas.numpy())
        log(f"  {step.name}: {step.n} strokes, L1 to photo {(canvas - photo.target).abs().mean().item():.4f}")
    return stages, frames, guides


if __name__ == "__main__":
    from PIL import Image
    photo_path, mask_path, points_path, out = sys.argv[1:5]
    pts = np.load(points_path)
    photo = B.load(photo_path, mask_path)
    w = Image.open(photo_path).size[0]
    stages, frames, _ = paint(photo, Face(pts * SIDE / w))
    Image.fromarray((np.concatenate([photo.target.numpy()] + frames, 1).clip(0, 1) * 255).astype(np.uint8)).save(out)
