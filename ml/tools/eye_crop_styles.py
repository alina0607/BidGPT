"""Which way of painting gives clear eyes? Paint only the eyes (an eye crop) of a few cats in several styles and
measure each against the photo. Painting an eye crop is far cheaper than a whole face, and the eyes are twice as big
on the canvas, so a 1–2 px eye rim can be painted at all.

The eye crop is a square around the midpoint of the two eye points, two eye distances wide, resized to the 256-px
canvas: the eyes lie 128 px apart (on a face canvas about 82). Styles:

  old         catbrush's four stages, unchanged (608 strokes);
  old_liner   the same, then thin dark strokes for the eye rims (10 x 1.5 px, placed and weighted where the photo
              has thin dark lines near the eyes) and a few small bright strokes for the highlights;
  flat        flat colour: the four stages painted on the photo reduced to 12 colours, with opaque strokes, then the
              same rim and highlight strokes against the photo;
  old_edge    the four stages, then 120 thin strokes fitted to the photo's thin dark lines inside the eye zone (an
              extra loss term next to L1) and opaque highlight strokes on the brightest spots;
  old_edge5   the same with the thin-dark-line term five times as strong, also on the fourth stage, and 160 thin
              strokes (it games the measure: bright green and purple streaks next to dark ones);
  old_trace   the four stages, then strokes traced straight from the photo: short dark ones along its thin dark
              lines near the eyes, small bright ones on its brightest spots near the eyes (not fitted);
  old_contour the four stages, then the eye outlines traced like a painter would: along 72 rays from each eye
              point, the first strong thin dark line from the inside out, smoothed into a closed outline, painted
              with short dark strokes along it; then the bright highlight strokes of old_trace;
  rim_fit     old_edge without its highlight strokes: the four stages and the 120 fitted thin rim strokes only;
  rim_trace   the four stages, then the same 120 fitted thin rim strokes, but starting on the traced outline of
              old_contour (placed on its pixels, along its tangent) instead of anywhere on the thin-dark-line map;
              their position, angle, size and colour are then fitted as in old_edge (the colour is not fixed);
  rim_trace_hl  rim_trace, then old_edge's 8 fitted highlight strokes last (rim layer, then highlight layer);
  old_edge_dark  the four stages, then 160 thin strokes with the term five times as strong but each stroke's colour
              fixed to the photo's colour where it starts (on a dark line), so it can only draw dark lines.

Measures against the photo: L1 over the crop, L1 in the eye discs (0.35 eye distances), L1 in the zone the rim strokes
are fitted in (0.45 eye distances: catches rim strokes in a wrong colour), and the thin-dark-line
strength around each eye (the strongest line, darker than the mean of its two sides 2 px away, along 36 rays from
0.08 to 0.40 eye distances): the photo's own value is the target. Painted on the CPU (one GPU job at a time), not
snapped to the token grid (a painter comparison only).

    PYTHONPATH=. python tools/eye_crop_styles.py --n 24 --out runs/eye_crop_styles
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates, uniform_filter

from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
AFHQ, MASKS = ML / "data" / "raw" / "afhq", ML / "data" / "processed" / "afhq_masks"
LABELS = ML / "data" / "processed" / "afhq_cats_mae_pad1.5.npz"
SPLIT = ML / "data" / "processed" / "brushgpt" / "tokens_runs_DEFG.npz"
SOURCE = 512
WIDTH = 2.0                 # the crop's side, in eye distances
LINER = (10, 1.5, 80)       # rim strokes: length, width, count
HIGHLIGHT = (4, 3, 8)       # highlight strokes
TRACE = (6, 2, 150)         # traced rim strokes: length, width, at most this many
CONTOUR_RAYS = 72           # old_contour: rays per eye
CONTOUR_FLOOR = 0.02        # a ray keeps its point when the line there is at least this much darker than its sides
CONTOUR_STROKE = (5, 2)     # length, width of an outline stroke
RIM_WEIGHT, LIGHT_WEIGHT = 20.0, 20.0
EDGE_WEIGHT = 1.0           # old_edge: the thin-dark-line term next to L1, averaged over the eye zone
COLOURS = 12                # the flat style's palette


def eye_box(points: np.ndarray) -> tuple[int, int, int, int]:
    eyes = points[:2]
    side = min(WIDTH * float(np.linalg.norm(eyes[0] - eyes[1])), SOURCE)
    left, top = np.clip(eyes.mean(0) - side / 2, 0, SOURCE - side)
    return round(left), round(top), round(left + side), round(top + side)


def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=img.dtype)


def dark_lines(lum: np.ndarray, s: int = 2) -> np.ndarray:
    """Per pixel, how much darker it is than the mean of its two sides s px away, in the best of four directions."""
    p = np.pad(lum, s, mode="edge")
    h, w = lum.shape
    out = np.zeros_like(lum)
    for dy, dx in ((0, s), (s, 0), (s, s), (s, -s)):
        a = p[s + dy:s + dy + h, s + dx:s + dx + w]
        b = p[s - dy:s - dy + h, s - dx:s - dx + w]
        out = np.maximum(out, (a + b) / 2 - lum)
    return out.clip(0)


def rim_strength(img: np.ndarray, eyes: np.ndarray, d: float, s: float = 2.0) -> float:
    """The thin-dark-line strength around the eyes (see the module docstring)."""
    lum = luminance(img.astype(np.float64))
    out = []
    for e in eyes:
        best = []
        for th in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            r = np.arange(0.08 * d, 0.40 * d, 0.5)
            at = lambda rr: (e[1] + rr * np.sin(th) - 0.5, e[0] + rr * np.cos(th) - 0.5)   # noqa: E731
            c, a, b = (map_coordinates(lum, at(r + k), order=1) for k in (0, -s, s))
            best.append(float(np.max((a + b) / 2 - c)))
        out.append(np.mean(best))
    return float(np.mean(out))


def posterise(target: np.ndarray, mask: np.ndarray, k: int, seed: int = 0, iters: int = 15) -> np.ndarray:
    """The cat's pixels reduced to k colours (k-means in numpy; sklearn's crashes next to torch on macOS)."""
    on = mask > 0.5
    px = target[on].astype(np.float64)
    centres = px[np.random.default_rng(seed).choice(len(px), k, replace=False)]
    for _ in range(iters):
        label = ((px[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([px[label == j].mean(0) if (label == j).any() else centres[j] for j in range(k)])
    out = target.copy()
    out[on] = centres[label].astype(target.dtype)
    return out


def dark_lines_t(lum: torch.Tensor, s: int = 2) -> torch.Tensor:
    """dark_lines in torch, so a painting's thin dark lines can be fitted to the photo's by gradient descent."""
    p = torch.nn.functional.pad(lum[None, None], (s, s, s, s), mode="replicate")[0, 0]
    h, w = lum.shape
    out = torch.zeros_like(lum)
    for dy, dx in ((0, s), (s, 0), (s, s), (s, -s)):
        a = p[s + dy:s + dy + h, s + dx:s + dx + w]
        b = p[s - dy:s - dy + h, s - dx:s - dx + w]
        out = torch.maximum(out, (a + b) / 2 - lum)
    return out.clamp(min=0)


@contextmanager
def brush_settings(spec: dict):
    """A stage's own brush: edge softness in px ("edge_px", catbrush's 0.5 by default; watercolour washes use more),
    opacity range ("alpha_range", catbrush's 0.3-1.0; coloured pencil goes lower) and bristle streaks ("bristle",
    catbrush's 0.06; 0 for a smooth digital brush). The window margin grows with
    the edge so a soft stroke's faint rim is not cut off."""
    saved = B.EDGE_PX, B.ALPHA_RANGE, B.WINDOW_MARGIN, B.BRISTLE
    B.EDGE_PX = spec.get("edge_px", saved[0])
    B.ALPHA_RANGE = spec.get("alpha_range", saved[1])
    B.WINDOW_MARGIN = max(saved[2], int(np.ceil(8 * B.EDGE_PX)))
    B.BRISTLE = spec.get("bristle", saved[3])
    try:
        yield
    finally:
        B.EDGE_PX, B.ALPHA_RANGE, B.WINDOW_MARGIN, B.BRISTLE = saved


def paint(target: torch.Tensor, mask: torch.Tensor, specs: list[dict], seed: int = 0,
          start: np.ndarray | None = None, record: list | None = None, dev: str = "cpu") -> list[np.ndarray]:
    """catbrush.paint generalised: each stage has its own brush, count, placement map, loss weights and opacity.
    With `start`, painting continues on that canvas (e.g. the old painter's finished painting). With `record`, each
    fitted stage's (spec, strokes) is appended to it, for replaying the painting one stroke at a time. `dev` is the
    torch device the fitting runs on (CPU by default; the stroke placement stays on the CPU's random generator)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    target, mask = target.to(dev), mask.to(dev)
    ys, xs = torch.meshgrid(torch.arange(B.SIDE, device=dev) + 0.5, torch.arange(B.SIDE, device=dev) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    angles = B.orientation(target.cpu()).to(dev)          # catbrush builds its filter on the CPU
    canvas = torch.ones_like(target) if start is None else torch.from_numpy(np.asarray(start, np.float32)).to(dev)
    frames = []
    for spec in specs:
        if "xy" in spec:                         # traced strokes: placed from the photo, not fitted
            fixed = B.Strokes(xy=B._logit(torch.as_tensor(spec["xy"], dtype=torch.float32) / B.SIDE),
                              size=torch.zeros(len(spec["xy"]), 2), angle=torch.as_tensor(spec["angle"], dtype=torch.float32),
                              colour=B._logit(torch.as_tensor(spec["colour"], dtype=torch.float32).clamp(0.02, 0.98)),
                              alpha=torch.full((len(spec["xy"]),), 4.0), base=(spec["length"], spec["width"]))
            with torch.no_grad():
                canvas = B.composite(canvas, fixed, grid)
            frames.append(canvas.cpu().numpy().clip(0, 1))
            continue
        with brush_settings(spec):
            canvas, strokes = _fit_stage(spec, target, mask, canvas, grid, angles, rng)
        if record is not None:                   # the fitted strokes, in order, to replay one at a time
            record.append((spec, strokes))
        frames.append(canvas.cpu().numpy().clip(0, 1))
    return frames


BLEND_SAMPLES = 9           # points along a blending stroke that its smudge averages


def blend_composite(canvas: torch.Tensor, strokes: B.Strokes) -> torch.Tensor:
    """Blending strokes (a painter's clean, dry brush dragged through wet paint): they carry no paint of their own.
    Under a stroke, each pixel moves towards the mean of the canvas along the stroke's direction, over the stroke's
    length (BLEND_SAMPLES points), by the stroke's coverage times its opacity: out = (1 - a) c + a smudge for one
    stroke. The strokes of a stage act together on the same canvas; where several overlap, their pulls are averaged
    (weights a, divided by max(1, sum a)). Along a radius this draws out the fibres of an iris; along the rim it
    smooths the colour into a gradient. Differentiable in position, size, angle and opacity."""
    p = strokes.params()
    h, w = canvas.shape[:2]
    half = B.window_half(strokes.base)
    dev = canvas.device
    off = torch.arange(-half, half + 1, device=dev)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ix = p.xy[:, 0].detach().floor().long()[:, None] + ox.reshape(1, -1)
    iy = p.xy[:, 1].detach().floor().long()[:, None] + oy.reshape(1, -1)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    cover, _ = B.rounded_rect(B.Brush.of(p, 1), ix + 0.5, iy + 0.5)
    a = cover * p.alpha[:, None] * valid                                          # (n, window)
    t = torch.linspace(-0.5, 0.5, BLEND_SAMPLES, device=dev)[None, :] * p.lw[:, :1]          # (n, K) along the stroke
    sx = (ix + 0.5)[..., None] + (t * p.angle.cos()[:, None])[:, None, :]       # (n, window, K)
    sy = (iy + 0.5)[..., None] + (t * p.angle.sin()[:, None])[:, None, :]
    gridn = torch.stack([sx / w * 2 - 1, sy / h * 2 - 1], -1).reshape(1, -1, BLEND_SAMPLES, 2)
    img = canvas.permute(2, 0, 1)[None]
    smudge = torch.nn.functional.grid_sample(img, gridn, mode="bilinear", padding_mode="border", align_corners=False)
    smudge = smudge[0].mean(-1).T                                                 # (n * window, 3)
    pix = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).reshape(-1)
    flat = canvas.reshape(-1, 3)
    pull = a.reshape(-1, 1) * (smudge - flat[pix])
    num = torch.zeros(h * w, 3, device=dev).index_add(0, pix, pull)
    den = torch.zeros(h * w, device=dev).index_add(0, pix, a.reshape(-1))
    return (flat + num / den.clamp(min=1)[:, None]).reshape(h, w, 3)


def _fit_stage(spec: dict, target: torch.Tensor, mask: torch.Tensor, canvas: torch.Tensor, grid: torch.Tensor,
               angles: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, B.Strokes]:
    dev = canvas.device
    on = lambda t: t.to(dev) if isinstance(t, torch.Tensor) else t   # noqa: E731
    goal = on(spec.get("target", target))
    weight = on(spec.get("weight", torch.ones(B.SIDE, B.SIDE)))[..., None]
    place = spec.get("place")
    if place is None:
        place = ((goal - canvas).abs().sum(-1) * mask).cpu().numpy()
    p = place.flatten() + 1e-9
    idx = rng.choice(p.size, spec["n"], replace=False, p=p / p.sum())
    if spec.get("order") is not None:            # a fixed stroke order (e.g. around each eye), set before fitting
        idx = idx[spec["order"](idx % B.SIDE + 0.5, idx // B.SIDE + 0.5)]
    py, px = idx // B.SIDE, idx % B.SIDE
    xy = torch.tensor(np.c_[px, py] + 0.5, dtype=torch.float32, device=dev)
    start_angle = on(spec.get("angles", angles))   # an angle map of its own, e.g. the traced outline's tangents
    ti = torch.as_tensor(idx, device=dev)
    s = B.Strokes(xy=B._logit(xy / B.SIDE), size=torch.zeros(spec["n"], 2, device=dev),
                  angle=start_angle.flatten()[ti].clone(),
                  colour=B._logit(goal[torch.as_tensor(py, device=dev), torch.as_tensor(px, device=dev)].clamp(0.02, 0.98)),
                  alpha=torch.full((spec["n"],), spec.get("alpha", 1.5), device=dev), base=(spec["length"], spec["width"]))
    blend = spec.get("blend", False)             # blending strokes: no paint, only a smudge (blend_composite)
    draw = (lambda c, st: blend_composite(c, st)) if blend else (lambda c, st: B.composite(c, st, grid))
    params = ([s.xy, s.size] + ([] if spec.get("angle_fixed") else [s.angle])
              + ([] if spec.get("colour_fixed") or blend else [s.colour])
              + ([] if spec.get("alpha_fixed") else [s.alpha]))
    for t in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=0.05)
    edge = spec.get("edge")                  # (zone, coefficient): fit the thin dark lines inside the zone too
    protect = spec.get("protect")
    if edge is not None:
        edge = (on(edge[0]), edge[1])
    if protect is not None:
        protect = (on(protect[0]), protect[1])
    if edge is not None:
        lum_w = torch.tensor([0.299, 0.587, 0.114], device=dev)
        goal_lines = dark_lines_t(goal @ lum_w)
    for _ in range(B.ITERS):
        painted = draw(canvas, s)
        loss = ((painted - goal).abs() * weight).mean()
        if edge is not None:
            zone, coef = edge
            loss = loss + coef * ((dark_lines_t(painted @ lum_w) - goal_lines).abs() * zone).sum() / zone.sum()
        if protect is not None:              # (outside, coefficient): changing these pixels costs more
            outside, pcoef = protect
            loss = loss + pcoef * ((painted - canvas).abs().mean(-1) * outside).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if spec.get("order_by_contrast") and not blend:
        # lay the strokes from the lowest to the highest contrast with the canvas under them (owner, 2026-10-18:
        # the finer and the sharper a mark, the later), then fit a while longer in that order
        with torch.no_grad():
            pr = s.params()
            ix = pr.xy[:, 0].long().clamp(0, B.SIDE - 1)
            iy = pr.xy[:, 1].long().clamp(0, B.SIDE - 1)
            order = torch.argsort((pr.colour - canvas[iy, ix]).abs().mean(-1))
        for t in s.tensors():                     # every stroke parameter, fitted or fixed, follows the new order
            t.data = t.data[order]
        opt = torch.optim.Adam(params, lr=0.02)
        for _ in range(B.ITERS // 3):
            painted = draw(canvas, s)
            loss = ((painted - goal).abs() * weight).mean()
            if protect is not None:
                outside, pcoef = protect
                loss = loss + pcoef * ((painted - canvas).abs().mean(-1) * outside).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return draw(canvas, s), s


def peaks(score: np.ndarray, n: int, spacing: float, floor: float) -> np.ndarray:
    """Up to n pixels (x, y) of highest score above floor, at least `spacing` px apart (greedy)."""
    order = np.argsort(score.ravel())[::-1]
    chosen = []
    for i in order:
        if score.flat[i] < floor or len(chosen) == n:
            break
        y, x = divmod(int(i), score.shape[1])
        if all((x - cx) ** 2 + (y - cy) ** 2 >= spacing ** 2 for cx, cy in chosen):
            chosen.append((x, y))
    return np.array(chosen, float).reshape(-1, 2)


def traced(target: np.ndarray, rim: np.ndarray, light: np.ndarray) -> list[dict]:
    """Strokes placed straight from the photo: short dark strokes along its thin dark lines near the eyes (each in the
    photo's colour there, along the line's direction), then small bright strokes on its brightest spots."""
    angles = B.orientation(torch.from_numpy(target)).numpy()
    line = peaks(rim, TRACE[2], spacing=2.5, floor=0.25)
    dots = peaks(light, HIGHLIGHT[2], spacing=4.0, floor=0.35)
    pick = lambda pts: (pts + 0.5, angles[pts[:, 1].astype(int), pts[:, 0].astype(int)],   # noqa: E731
                        target[pts[:, 1].astype(int), pts[:, 0].astype(int)])
    out = []
    for pts, (ln, w, _) in ((line, TRACE), (dots, HIGHLIGHT)):
        if len(pts):
            xy, ang, col = pick(pts)
            out.append({"length": ln, "width": w, "xy": xy, "angle": ang, "colour": col})
    return out


def eye_contours(target: np.ndarray, eyes: np.ndarray, d: float) -> list[np.ndarray]:
    """Each eye's outline as the painter would find it: along 72 rays from the eye point, the first strong thin dark
    line from the inside out (at least 60% of the strongest on that ray, between 0.10 and 0.45 eye distances), the
    radii smoothed around the eye (circular median of 7), rays without a clear line dropped. (n, 4) per eye:
    x, y, the tangent's angle, and the line's strength."""
    lum = luminance(target.astype(np.float64))
    out = []
    for e in eyes:
        th = np.linspace(0, 2 * np.pi, CONTOUR_RAYS, endpoint=False)
        r = np.arange(0.10 * d, 0.45 * d, 0.5)
        radius, strength = np.zeros(len(th)), np.zeros(len(th))
        for k, t in enumerate(th):
            at = lambda rr: (e[1] + rr * np.sin(t) - 0.5, e[0] + rr * np.cos(t) - 0.5)   # noqa: E731
            c, a, b = (map_coordinates(lum, at(r + q), order=1) for q in (0, -2, 2))
            line = (a + b) / 2 - c
            first = int(np.argmax(line >= 0.6 * line.max()))
            radius[k], strength[k] = r[first], line[first]
        pad = np.r_[radius[-3:], radius, radius[:3]]
        radius = np.array([np.median(pad[k:k + 7]) for k in range(len(th))])
        x, y = e[0] + radius * np.cos(th), e[1] + radius * np.sin(th)
        tangent = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)) % np.pi
        keep = strength >= CONTOUR_FLOOR
        out.append(np.c_[x, y, tangent, strength][keep])
    return out


def contour_place(contours: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Where and at what angle fitted rim strokes start when they follow the traced outlines: a placement map that is
    the line's strength on the outline's pixels and their 4-neighbours, and the outline's tangent there."""
    place, angle = np.zeros((B.SIDE, B.SIDE)), np.zeros((B.SIDE, B.SIDE))
    for x, y, tangent, strength in (np.concatenate(contours) if contours else np.zeros((0, 4))):
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            xi, yi = int(np.clip(x + dx, 0, B.SIDE - 1)), int(np.clip(y + dy, 0, B.SIDE - 1))
            if strength > place[yi, xi]:
                place[yi, xi], angle[yi, xi] = strength, tangent
    return place / max(place.max(), 1e-6), angle


def contour_strokes(target: np.ndarray, contours: list[np.ndarray]) -> dict | None:
    """Short dark strokes along the traced outlines, each in the darkest photo colour within 2 px of the line."""
    pts = np.concatenate(contours) if contours else np.zeros((0, 4))
    if not len(pts):
        return None
    xi = np.clip(pts[:, 0].astype(int), 0, B.SIDE - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, B.SIDE - 1)
    lum = luminance(target)
    colour = []
    for x, y in zip(xi, yi, strict=True):
        patch = target[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
        colour.append(patch[np.argmin(luminance(patch))])
    del lum
    return {"length": CONTOUR_STROKE[0], "width": CONTOUR_STROKE[1], "xy": pts[:, :2], "angle": pts[:, 2],
            "colour": np.array(colour)}


def styles(target: np.ndarray, mask: np.ndarray, eyes: np.ndarray, d: float) -> dict[str, list[dict]]:
    lum = luminance(target)
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    near = lambda r: np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < r * d   # noqa: E731
    rim = dark_lines(lum) * near(0.45)
    rim /= max(np.percentile(rim[rim > 0], 99), 1e-6) if (rim > 0).any() else 1.0
    light = (lum - uniform_filter(lum, 9)).clip(0) * near(0.25)
    light /= max(np.percentile(light[light > 0], 99), 1e-6) if (light > 0).any() else 1.0
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))   # noqa: E731
    old = [{"length": ln, "width": w, "n": n} for ln, w, n in B.STAGES[:4]]
    finish = [{"length": LINER[0], "width": LINER[1], "n": LINER[2], "place": rim.clip(0, 1),
               "weight": t(1 + RIM_WEIGHT * rim.clip(0, 1))},
              {"length": HIGHLIGHT[0], "width": HIGHLIGHT[1], "n": HIGHLIGHT[2], "place": light.clip(0, 1),
               "weight": t(1 + LIGHT_WEIGHT * light.clip(0, 1))}]
    flat_target = t(posterise(target, mask, COLOURS))
    flat = [{**s, "target": flat_target, "alpha": 4.0, "alpha_fixed": True} for s in old]
    zone = t(near(0.45))
    edge = [{"length": LINER[0], "width": LINER[1], "n": 120, "place": rim.clip(0, 1), "edge": (zone, EDGE_WEIGHT)},
            {"length": HIGHLIGHT[0], "width": HIGHLIGHT[1], "n": HIGHLIGHT[2], "place": light.clip(0, 1) ** 4,
             "weight": t(1 + LIGHT_WEIGHT * light.clip(0, 1)), "alpha": 4.0}]
    strong = [*old[:3], {**old[3], "edge": (zone, 5 * EDGE_WEIGHT)},
              {**edge[0], "n": 160, "edge": (zone, 5 * EDGE_WEIGHT)}, edge[1]]
    dark = [*old, {**edge[0], "n": 160, "edge": (zone, 5 * EDGE_WEIGHT), "colour_fixed": True}, edge[1]]
    outline, outline_angle = contour_place(eye_contours(target, eyes, d))
    angle_map = B.orientation(torch.from_numpy(target)).numpy()
    angle_map[outline > 0] = outline_angle[outline > 0]
    traced_rim = {**edge[0], "place": outline + 0.05 * rim.clip(0, 1), "angles": t(angle_map)}
    return {"old": old, "rim_fit": old + edge[:1], "rim_trace": old + [traced_rim], "rim_trace_hl": old + [traced_rim, edge[1]],
            "old_liner": old + finish, "flat": flat + finish, "old_edge": old + edge, "old_edge5": strong,
            "old_edge_dark": dark, "old_trace": old + traced(target, rim.clip(0, 1), light.clip(0, 1)),
            "old_contour": old + [x for x in (contour_strokes(target, eye_contours(target, eyes, d)),) if x]
            + [x for x in traced(target, np.zeros_like(rim), light.clip(0, 1)) if x]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24, help="validation cats")
    ap.add_argument("--out", type=Path, default=ML / "runs" / "eye_crop_styles")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--styles", default="old,old_liner,flat,old_edge")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    labels = np.load(LABELS)
    index = {str(i): k for k, i in enumerate(labels["ids"])}
    z = np.load(SPLIT)
    ids = [str(i) for i in z["ids"][z["split"] == "val"][:a.n]]
    rows = []
    for rel in ids:
        name = Path(rel).stem
        pts = labels["points"][index[rel]]
        box = eye_box(pts)
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box)
        scale = B.SIDE / (box[2] - box[0])
        eyes = (pts[:2] - np.array(box[:2])) * scale
        d = float(np.linalg.norm(eyes[0] - eyes[1]))
        target, mask = photo.target.numpy(), photo.mask.numpy()
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        near_eye = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0)
        disc = near_eye < 0.35 * d
        ring = near_eye < 0.45 * d                   # the zone the rim strokes are fitted in
        row = {"id": rel, "box": list(box), "eye_dist_px": round(d, 1), "photo_rim": round(rim_strength(target, eyes, d), 4)}
        images = {"photo": target}
        table = styles(target, mask, eyes, d)
        base = table["old"]
        t0 = time.time()
        old_final = paint(photo.target, photo.mask, base)[-1]       # every other style continues from it
        old_seconds = time.time() - t0
        for style, specs in table.items():
            if style not in a.styles.split(","):
                continue
            t0 = time.time()
            if style == "old":
                final = old_final
            elif specs[:len(base)] == base:
                final = paint(photo.target, photo.mask, specs[len(base):], start=old_final)[-1]
            else:
                final = paint(photo.target, photo.mask, specs)[-1]
            images[style] = final
            row[style] = {"l1": round(float(np.abs(final - target).mean()), 4),
                          "eye_l1": round(float(np.abs(final - target)[disc].mean()), 4),
                          "rim_zone_l1": round(float(np.abs(final - target)[ring].mean()), 4),
                          "rim": round(rim_strength(final, eyes, d), 4), "strokes": sum(s["n"] if "n" in s else len(s["xy"]) for s in specs),
                          "seconds": round(time.time() - t0 + (old_seconds if specs[:len(base)] == base else 0), 1)}
        contours = eye_contours(target, eyes, d)
        light = (luminance(target) - uniform_filter(luminance(target), 9)).clip(0)
        yy2, xx2 = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        light *= np.min([np.hypot(xx2 - x, yy2 - y) for x, y in eyes], 0) < 0.25 * d
        light /= max(np.percentile(light[light > 0], 99), 1e-6) if (light > 0).any() else 1.0
        np.savez_compressed(a.out / f"{name}.npz", eyes=eyes, contour_0=contours[0], contour_1=contours[1],
                            highlights=peaks(light.clip(0, 1), HIGHLIGHT[2], spacing=4.0, floor=0.35), **images)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    for style in a.styles.split(","):
        summary[style] = {k: round(float(np.mean([r[style][k] for r in rows])), 4)
                          for k in ("l1", "eye_l1", "rim_zone_l1", "rim", "seconds")}
    summary["photo_rim"] = round(float(np.mean([r["photo_rim"] for r in rows])), 4)
    (a.out / "summary.json").write_text(json.dumps({"cats": rows, "summary": summary}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
