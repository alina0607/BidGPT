"""The eye model's painter (owner, 2026-10-18: a second, trained model paints the eye detail on a zoomed eye canvas, on
top of the face model's painting; CLAUDE.md). This builds what that model would learn from: for each cat, the face
painting (the old painter's four stages, snapped to the token grid exactly as the face model learns them) rendered
into the eye crop of tools/eye_crop_styles.py, then four eye layers fitted on top of it, in the owner's order:

  1 outline   120 thin strokes (10 x 1.5 px) starting on the traced eye outline, along its tangent, fitted to the
              photo with the thin-dark-line term (eye_crop_styles' rim_trace);
  2 iris      40 strokes (12 x 6 px) placed inside the outline, fitted to the photo there;
  3 pupil     24 strokes (8 x 3 px) placed on the darkest pixels inside the outline;
  4 highlight 8 strokes (4 x 3 px, opaque) on the brightest spots inside the outline, last.

Layers 2-4 must keep the colour inside the outline: changing any pixel outside the outline's interior (the outline
itself included) costs PROTECT times the L1 of the change, a loss penalty rather than a clip, so every stroke stays an
ordinary stroke. The outline layer may also paint up to OUTLINE_REACH px outside the outline and nowhere else: the
fur around the eye is the face model's, not the eye model's. Variants, all from the same face painting:

  face        the face painting alone (what the eye model is given);
  layers      the four layers in the owner's order (outline first);
  fill_first  iris, pupil, outline, highlight (outline after the colour, as a check of the order);
  rim_hl      outline and highlight only (no iris or pupil layer);
  pencil      coloured pencil: finer outline (160 x 8 x 1 px), 400 fine iris strokes (9 x 1.2 px) at low opacity
              (0.12-0.6) with a fixed radial direction (the iris fibres), 60 pupil strokes, crisp highlight dots;
  watercolour 30 soft-edged washes (22 x 16 px, edge 3 px, opacity 0.12-0.5) for the iris gradient, 160 radial glaze
              strokes, soft pupil strokes, highlight dots (outline first as always);
  oil         realistic oil (blended): 40 soft washes (edge 2 px) for the gradient, 240 fine radial strokes for the
              fibres, a crisp pupil, crisp highlight dots: the blending of watercolour with the detail of pencil.

  smooth      blended, with the outline traced as one smooth closed curve (smooth_outlines) instead of ray by ray;
  fine        smooth outline, and every layer after the base as fine as the pupil's edge (owner, 2026-10-18): outline
              240 x 6 x 1 px, a big base (30 x 12 x 8), iris 400 x 6 x 1.5, blending 150 x 6 x 2, pupil 120 x 4 x 1.5,
              highlight 12 x 3 x 2; the colour layers keep RIM_KEEP px off the outline.
  digital     a digital painter's nine steps (owner, 2026-10-18): dark lid line (200 x 6 x 1.6), soft base (20 x 22 x 16,
              edge 3 px), soft blocks of light and shade (50 x 12 x 8, edge 2), blending (100), fine radial fibres
              (300 x 8 x 1.2, opacity 0.1-0.5), blending (80), pupil (100 x 4 x 1.5), crisp highlight (12 x 3 x 2).
  digital_smooth  digital without bristle streaks (catbrush.BRISTLE 0 for every layer): a smooth digital brush.
  lite_*      digital with fewer strokes per step (owner, 2026-10-24: the eye material should look like digital,
              runs/eye_layers/digital6/steps_000031.png, but the eye model writes about 8 tokens a stroke, so 862
              strokes are too long to train on the Mac); LITE gives each step's count, 0 drops the step;
  <lite>_to_digital  the same strokes fitted to the digital painting of the cat instead of the photo, so the fewer
              strokes copy digital's look rather than finding their own (needs digital earlier in --styles).
  Measures and the saved outline use the smooth outline for every variant; it comes from SAM (sam_outlines, the
  default since 2026-10-18) or from the thin dark lines (smooth_outlines, --tracer lines).
  blended    the four layers with a blending layer after the iris colour: 80 strokes (10 x 4 px) that carry no paint
              and smudge the canvas along their direction (eye_crop_styles.blend_composite), starting radial, kept RIM_KEEP px
              off the outline so the rim stays dark;
              the highlight comes after and stays crisp.

A stage's edge softness and opacity range are its own (eye_crop_styles.brush_settings): the eye model is a separate
model and need not share the face model's oil brush.

Measures as eye_crop_styles (L1, eye-disc L1, rim strength), plus overflow: the L1 in the band 1.5-5 px outside the
traced outline, where colour spilling out of the eye would show. Diagnostic for the painter; CPU only. Results depend on
the number of CPU threads (summation order changes the floats, and 150 fitting steps amplify it: one cat's rim 0.0736
with 2 threads, 0.0784 with 4); runs are reproducible at the same --threads, and all comparisons here use 2.

    PYTHONPATH=. python tools/eye_layers.py --n 4 --out runs/eye_layers/pilot4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter

import eye_crop_styles as E
from sellanything_ml.sketch import brushtokens as T
from sellanything_ml.sketch import catbrush as B

PAINT_DIR = E.ML / "data" / "processed" / "afhq_brush_face"
IRIS, PUPIL = (12, 6, 40), (8, 3, 24)
PROTECT = 10.0
HARMONICS = 2                # smooth outline: Fourier terms of the radius around the eye (2: ellipse-like)
REFINE, SNAP = 3, 0.06       # smooth outline: refits, each ray snapped to the strongest line this near the curve
OUTLINE_POINTS = 360
SAM_MODEL = "facebook/sam-vit-base"
SAM_HARMONICS = 4            # SAM's mask is reliable: the outline may follow it more closely than the line tracer
# Strokes per step of the shorter digital paintings: lid line, soft base, light and shade, blending, radial fibres,
# blending, pupil, highlight (digital: 200, 20, 50, 100, 300, 80, 100, 12 = 862).
LITE = {"lite_nofibre": (200, 20, 50, 100, 0, 80, 100, 12),    # 562: what the 300 fibres add
        "lite_410": (100, 20, 50, 80, 100, 0, 48, 12),
        "lite_290": (60, 16, 40, 60, 60, 0, 42, 12),           # about 1.5 times layers' 192 strokes
        "lite_200": (40, 12, 30, 40, 40, 0, 30, 8)}            # about layers' length
RIM_KEEP = 2.0               # px inside the outline that blending must not smudge (painters blend the iris, not the rim)
OUTLINE_REACH = 3.0          # px outside the traced outline the outline layer may still paint
CHUNK = 64


def face_on_crop(name: str, ebox: tuple[int, int, int, int]) -> np.ndarray:
    """The face painting of `name`, snapped to the token grid, rendered into the eye crop `ebox` (source pixels)."""
    z = np.load(PAINT_DIR / f"{name}.npz")
    fbox = z["box"].astype(float)
    k_face, k_eye = (fbox[2] - fbox[0]) / B.SIDE, B.SIDE / (ebox[2] - ebox[0])
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(B.SIDE, B.SIDE, 3)
    with torch.no_grad():
        for s in range(4):
            st = {key: z[f"s{s}_{key}"] for key in ("xy", "angle", "length_width", "colour", "alpha")}
            st = T.stroke_params(T.stroke_tokens(st, s), s)                      # what the face model learns
            xy = ((fbox[:2] + st["xy"] * k_face) - np.array(ebox[:2])) * k_eye
            lw = st["length_width"] * k_face * k_eye
            for i in range(0, len(xy), CHUNK):
                j = slice(i, i + CHUNK)
                p = B.StrokeParams(*(torch.as_tensor(v[j], dtype=torch.float32)
                                     for v in (xy, lw, st["angle"], st["colour"], st["alpha"])))
                canvas = B.composite_dense(canvas, p, grid)
    return canvas.numpy()


def smooth_outlines(target: np.ndarray, eyes: np.ndarray, d: float) -> list[np.ndarray]:
    """Each eye's outline as one smooth closed curve, close to the almond a cat's eye is. Along 72 rays from the eye
    point, the thin-dark-line profile (darker than its two sides 2 px away). The radius as a function of the angle is
    a low-order Fourier series (HARMONICS = 2: ellipse-like), fitted by weighted least squares, first to each ray's
    first strong line (as eye_crop_styles.eye_contours), then REFINE times to the strongest line within SNAP eye
    distances of the current curve, rays more than 2.5 median absolute deviations off the curve dropped each time
    (a ray that caught the brow or a stripe). (OUTLINE_POINTS, 4) per eye: x, y, tangent, strength (1).

    Checked by eye on the first 6 validation cats (2026-10-18): 7 of 12 eyes traced well; it fails where the fur
    above the eye is as dark as the rim (cat 000023), on narrow almond eyes (cat 000041, both), and where a dark
    stripe runs next to the eye. Colour-based and random-walker tracing did no better (6 of 12, and worse on others):
    a learned segmentation (e.g. SAM prompted with the eye point) is the likely fix."""
    lum = E.luminance(target.astype(np.float64))
    th = np.linspace(0, 2 * np.pi, E.CONTOUR_RAYS, endpoint=False)
    r = np.arange(0.10 * d, 0.45 * d, 0.5)

    def design(a: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(a)] + [f(h * a) for h in range(1, HARMONICS + 1) for f in (np.cos, np.sin)])

    def fit(radius: np.ndarray, w: np.ndarray) -> np.ndarray:
        for _ in range(3):
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(design(th) * sw[:, None], radius * sw, rcond=None)[0]
            res = np.abs(design(th) @ coef - radius)
            mad = np.median(res[w > 0]) if (w > 0).any() else 0.0
            w = np.where(res > max(2.5 * mad, 0.02 * d), 0.0, w)
        return coef

    out = []
    for e in eyes:
        profile = np.zeros((len(th), len(r)))
        for k, t in enumerate(th):
            at = lambda rr: (e[1] + rr * np.sin(t) - 0.5, e[0] + rr * np.cos(t) - 0.5)   # noqa: E731
            c, a, b = (map_coordinates(lum, at(r + q), order=1) for q in (0, -2, 2))
            profile[k] = (a + b) / 2 - c
        first = np.argmax(profile >= 0.6 * profile.max(1, keepdims=True), 1)
        radius, strength = r[first], profile[np.arange(len(th)), first]
        w = np.where(strength >= E.CONTOUR_FLOOR, strength, 0.0)
        if (w > 0).sum() < 2 * HARMONICS + 6:                         # too few lines: a circle of 0.25 eye distances
            coef = np.r_[0.25 * d, np.zeros(2 * HARMONICS)]
        else:
            coef = fit(radius, w)
            for _ in range(REFINE):                                   # snap each ray to the line nearest the curve
                near = np.abs(r[None, :] - (design(th) @ coef)[:, None]) <= SNAP * d
                masked = np.where(near, profile, -np.inf)
                pick = np.argmax(masked, 1)
                radius, strength = r[pick], profile[np.arange(len(th)), pick]
                coef = fit(radius, np.where(strength >= E.CONTOUR_FLOOR, strength, 0.0))
        dense = np.linspace(0, 2 * np.pi, OUTLINE_POINTS, endpoint=False)
        rad = design(dense) @ coef
        x, y = e[0] + rad * np.cos(dense), e[1] + rad * np.sin(dense)
        tangent = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)) % np.pi
        out.append(np.c_[x, y, tangent, np.ones_like(x)])
    return out


_SAM: dict = {}


def sam_outlines(target: np.ndarray, eyes: np.ndarray, d: float) -> list[np.ndarray]:
    """Each eye's outline from SAM (facebook/sam-vit-base, Apache-2.0; owner's OK to download it, 2026-10-18), prompted
    with three points inside the eye (its point and 0.13 eye distances either way along the line between the eyes) and
    two outside (0.5 eye distances above and below). Of SAM's three masks, the best-scoring one of an eye's size
    (0.08-0.42 eye distances as a disc's radius); then only the part connected to the eye point, opened by a small disc
    (removes thin spurs of fur), and its radius per ray fitted by a Fourier series (SAM_HARMONICS) with rays more than
    2.5 median absolute deviations off dropped. Falls back to smooth_outlines when no mask fits. (OUTLINE_POINTS, 4)."""
    import torch as _t
    from scipy.ndimage import binary_opening
    from scipy.ndimage import label as cc_label
    from transformers import SamModel, SamProcessor
    if not _SAM:
        _SAM["model"] = SamModel.from_pretrained(SAM_MODEL, use_safetensors=True).eval()
        _SAM["proc"] = SamProcessor.from_pretrained(SAM_MODEL)
    model, proc = _SAM["model"], _SAM["proc"]
    img = (target.clip(0, 1) * 255).astype(np.uint8)
    axis = (eyes[1] - eyes[0]) / d
    perp = np.array([-axis[1], axis[0]])
    th = np.linspace(0, 2 * np.pi, E.CONTOUR_RAYS, endpoint=False)
    rr = np.arange(0.0, 0.5 * d, 0.5)
    disc = np.hypot(*np.mgrid[-3:4, -3:4]) <= max(1.0, 0.025 * d)
    fallback = smooth_outlines(target, eyes, d)
    out = []
    for j, e in enumerate(eyes):
        pos = [e, e + 0.13 * d * axis, e - 0.13 * d * axis]
        neg = [e + 0.5 * d * perp, e - 0.5 * d * perp]
        inp = proc(images=img, input_points=[[[list(map(float, q)) for q in pos + neg]]],
                   input_labels=[[[1, 1, 1, 0, 0]]], return_tensors="pt")
        with _t.no_grad():
            res = model(**inp, multimask_output=True)
        masks = proc.image_processor.post_process_masks(res.pred_masks, inp["original_sizes"],
                                                        inp["reshaped_input_sizes"])[0][0].numpy()
        scores = res.iou_scores[0, 0].numpy()
        area = masks.reshape(len(masks), -1).sum(1)
        ok = (area > np.pi * (0.08 * d) ** 2) & (area < np.pi * (0.42 * d) ** 2)
        if not ok.any():
            out.append(fallback[j])
            continue
        m = binary_opening(masks[int(np.argmax(np.where(ok, scores, -1)))], disc)
        lab, _ = cc_label(m)
        cx, cy = int(np.clip(e[0], 0, B.SIDE - 1)), int(np.clip(e[1], 0, B.SIDE - 1))
        if lab[cy, cx] == 0:
            out.append(fallback[j])
            continue
        m = lab == lab[cy, cx]
        radius = np.zeros(len(th))
        for k, t in enumerate(th):
            px = np.clip((e[0] + rr * np.cos(t)).astype(int), 0, B.SIDE - 1)
            py = np.clip((e[1] + rr * np.sin(t)).astype(int), 0, B.SIDE - 1)
            inside_ray = m[py, px]
            radius[k] = rr[int(np.argmin(inside_ray)) - 1] if not inside_ray.all() else rr[-1]
        des = lambda a: np.column_stack([np.ones_like(a)] + [f(h * a) for h in range(1, SAM_HARMONICS + 1)   # noqa: E731
                                                             for f in (np.cos, np.sin)])
        w = (radius > 0.03 * d).astype(float)
        for _ in range(3):
            coef = np.linalg.lstsq(des(th) * w[:, None], radius * w, rcond=None)[0]
            res_ = np.abs(des(th) @ coef - radius)
            w = np.where(res_ > max(2.5 * np.median(res_[w > 0]), 0.015 * d), 0.0, w)
        dense = np.linspace(0, 2 * np.pi, OUTLINE_POINTS, endpoint=False)
        rad = des(dense) @ coef
        x, y = e[0] + rad * np.cos(dense), e[1] + rad * np.sin(dense)
        tangent = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)) % np.pi
        out.append(np.c_[x, y, tangent, np.ones_like(x)])
    return out


FIBRE_KEEP = 5.0                     # digital_rim_long: px inside the outline the radial fibres stay off the rim
RIM_REACH, RIM_WINDOW = 0.07, 21     # rim_on_photo: how far a point may slide (eye distances); smoothing (points)


def rim_on_photo(target: np.ndarray, contours: list[np.ndarray], d: float) -> list[np.ndarray]:
    """Each outline moved onto the photo's eye rim (owner, 2026-10-25: the rim must be a smooth line like the photo's,
    nothing sticking out): every point slides along the ray from the outline's centre, at most RIM_REACH eye distances
    either way, to the strongest thin dark line there (eye_crop_styles.dark_lines, so the dark inside of a pupil does
    not pull it); the slides are then averaged along the outline (RIM_WINDOW points, circular) so the line stays as
    smooth as the outline. Same shape as the input: (n, 4) x, y, tangent, strength."""
    lines = E.dark_lines(E.luminance(target))
    k = max(2, int(round(RIM_REACH * d)))
    offs = np.arange(-k, k + 1, dtype=float)
    out = []
    for c in contours:
        x, y = c[:, 0], c[:, 1]
        cx, cy = x.mean(), y.mean()
        r = np.maximum(np.hypot(x - cx, y - cy), 1e-6)
        ux, uy = (x - cx) / r, (y - cy) / r
        sx, sy = x[:, None] + offs[None] * ux[:, None], y[:, None] + offs[None] * uy[:, None]
        vals = map_coordinates(lines, [sy.ravel() - 0.5, sx.ravel() - 0.5], order=1, mode="nearest").reshape(sx.shape)
        best = offs[np.argmax(vals, 1)]
        pad = RIM_WINDOW // 2
        best = np.convolve(np.r_[best[-pad:], best, best[:pad]], np.ones(RIM_WINDOW) / RIM_WINDOW, mode="valid")
        nx, ny = x + best * ux, y + best * uy
        tangent = np.arctan2(np.roll(ny, -1) - np.roll(ny, 1), np.roll(nx, -1) - np.roll(nx, 1)) % np.pi
        out.append(np.c_[nx, ny, tangent, c[:, 3]])
    return out


def interior(contours: list[np.ndarray]) -> np.ndarray:
    """Pixels inside the traced outlines (each eye's outline points, in angle order, as a polygon)."""
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    pts = np.c_[xx.ravel(), yy.ravel()]
    inside = np.zeros(B.SIDE * B.SIDE, bool)
    for c in contours:
        if len(c) >= 12:
            inside |= Polygon(c[:, :2]).contains_points(pts)
    return inside.reshape(B.SIDE, B.SIDE)


def variants(target: np.ndarray, eyes: np.ndarray, d: float, contours: list[np.ndarray]) -> tuple[dict, dict]:
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))   # noqa: E731
    lum = E.luminance(target)
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    near = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0)
    rim = E.dark_lines(lum) * (near < 0.45 * d)
    rim /= max(np.percentile(rim[rim > 0], 99), 1e-6) if (rim > 0).any() else 1.0
    inside = interior(contours)
    if inside.sum() < 50:                                        # outline not found: a disc of 0.25 eye distances
        inside = near < 0.25 * d
    protect = (t(~inside), PROTECT)
    region = inside | (distance_transform_edt(~inside) <= OUTLINE_REACH)   # the eye and its outline: the eye model's
    keep_out = (t(~region), PROTECT)                                       # whole territory; never the fur around it
    light = (lum - uniform_filter(lum, 9)).clip(0) * inside
    light /= max(np.percentile(light[light > 0], 99), 1e-6) if (light > 0).any() else 1.0
    outline, outline_angle = E.contour_place(contours)
    angle_map = B.orientation(torch.from_numpy(target)).numpy()
    angle_map[outline > 0] = outline_angle[outline > 0]
    zone = t(near < 0.45 * d)
    start = outline if (outline > 0).sum() >= 120 else outline + 0.05 * rim.clip(0, 1) * region
    rim_layer = {"length": E.LINER[0], "width": E.LINER[1], "n": 120, "place": start + 1e-6 * region,
                 "angles": t(angle_map), "edge": (zone, E.EDGE_WEIGHT), "protect": keep_out}
    iris = {"length": IRIS[0], "width": IRIS[1], "n": IRIS[2], "place": inside.astype(float), "protect": protect,
            "weight": t(inside)}
    pupil = {"length": PUPIL[0], "width": PUPIL[1], "n": PUPIL[2], "place": (1 - lum) ** 4 * inside, "protect": protect,
             "weight": t(inside)}
    highlight = {"length": E.HIGHLIGHT[0], "width": E.HIGHLIGHT[1], "n": E.HIGHLIGHT[2], "place": light.clip(0, 1) ** 4 + 1e-6 * inside,
                 "weight": t(1 + E.LIGHT_WEIGHT * light.clip(0, 1)), "alpha": 4.0, "protect": protect}
    table = {"layers": [rim_layer, iris, pupil, highlight], "fill_first": [iris, pupil, rim_layer, highlight],
             "rim_hl": [rim_layer, highlight]}
    # Media with their own brushes (the eye model need not share the face model's oil brush). The iris of a cat is a
    # radial gradient with fibres running out from the pupil: the fine iris strokes keep a radial direction.
    nearest = np.argmin([np.hypot(xx - x, yy - y) for x, y in eyes], 0)
    radial = t(np.mod(np.arctan2(yy - eyes[nearest, 1], xx - eyes[nearest, 0]), np.pi))
    fine_rim = {**rim_layer, "length": 8, "width": 1.0, "n": 160}
    crisp_hl = {**highlight, "length": 3, "width": 2.5}
    inner = {"place": inside.astype(float), "weight": t(inside), "protect": protect}
    pencil = [fine_rim,
              {**inner, "length": 9, "width": 1.2, "n": 400, "angles": radial, "angle_fixed": True, "alpha_range": (0.12, 0.6)},
              {**inner, "length": 6, "width": 1.5, "n": 60, "place": (1 - lum) ** 4 * inside, "alpha_range": (0.2, 1.0)},
              crisp_hl]
    washes = {**inner, "length": 22, "width": 16, "n": 30, "edge_px": 3.0, "alpha_range": (0.12, 0.5)}
    watercolour = [{**rim_layer, "edge_px": 0.8}, washes,
                   {**inner, "length": 10, "width": 2, "n": 160, "edge_px": 1.2, "angles": radial, "angle_fixed": True,
                    "alpha_range": (0.12, 0.5)},
                   {**inner, "length": 10, "width": 5, "n": 24, "edge_px": 1.5, "place": (1 - lum) ** 4 * inside,
                    "alpha_range": (0.3, 0.9)},
                   highlight]
    oil = [fine_rim, {**washes, "edge_px": 2.0, "n": 40},
           {**inner, "length": 8, "width": 1.0, "n": 240, "angles": radial, "angle_fixed": True, "alpha_range": (0.12, 0.6)},
           {**inner, "length": 8, "width": 3, "n": 40, "edge_px": 1.0, "place": (1 - lum) ** 4 * inside, "alpha_range": (0.3, 1.0)},
           crisp_hl]
    # Blending (owner, 2026-10-18: every painter blends the iris until it glows like a small planet): after the iris
    # colour, 80 blending strokes (eye_crop_styles.blend_composite) start along the radius and are fitted, then the
    # pupil and the highlight, which is never blended.
    core = distance_transform_edt(inside) > RIM_KEEP          # the iris without the outline's inner edge
    blending = {**inner, "length": 10, "width": 4, "n": 80, "blend": True, "angles": radial, "place": core.astype(float),
                "protect": (t(~core), PROTECT)}
    table.update({"pencil": pencil, "watercolour": watercolour, "oil": oil,
                  "blended": [rim_layer, iris, blending, pupil, highlight]})
    # Fine (owner, 2026-10-18): only the base may use a big brush; every later layer is as fine as the pupil's edge.
    # The colour layers stay off the outline's inner edge (core), so the rim drawn first is kept.
    keep_core = (t(~core), PROTECT)
    in_core = {"weight": t(core), "protect": keep_core}
    table["fine"] = [
        {**rim_layer, "length": 6, "width": 1.0, "n": 240},
        {**in_core, "length": 12, "width": 8, "n": 30, "place": core.astype(float)},                 # base (big)
        {**in_core, "length": 6, "width": 1.5, "n": 400, "place": core.astype(float), "angles": radial},
        {**blending, "length": 6, "width": 2, "n": 150},
        {**in_core, "length": 4, "width": 1.5, "n": 120, "place": (1 - lum) ** 4 * core},
        {**highlight, "length": 3, "width": 2, "n": 12, "protect": keep_core}]
    # Digital painting (owner, 2026-10-18: a digital painter's nine steps for a cat's eye): a dark lid line, a soft
    # base, soft blocks of light and shade, blending, fine radial fibres, blending again, the pupil, a crisp highlight.
    # Soft edges and low opacity throughout, as a digital brush; every layer still fitted to the photo.
    table["digital"] = [
        {**rim_layer, "length": 6, "width": 1.6, "n": 200},
        {**in_core, "length": 22, "width": 16, "n": 20, "place": core.astype(float), "edge_px": 3.0, "alpha_range": (0.3, 0.9)},
        {**in_core, "length": 12, "width": 8, "n": 50, "place": core.astype(float), "edge_px": 2.0, "alpha_range": (0.2, 0.8)},
        {**blending, "length": 12, "width": 6, "n": 100},
        {**in_core, "length": 8, "width": 1.2, "n": 300, "place": core.astype(float), "angles": radial, "angle_fixed": True,
         "edge_px": 0.8, "alpha_range": (0.1, 0.5)},
        {**blending, "length": 6, "width": 2, "n": 80},
        {**in_core, "length": 4, "width": 1.5, "n": 100, "place": (1 - lum) ** 4 * core, "edge_px": 0.8},
        {**highlight, "length": 3, "width": 2, "n": 12, "protect": keep_core}]
    table["digital_smooth"] = [{**layer, "bristle": 0.0} for layer in table["digital"]]   # no bristle streaks
    # digital_rim (owner, 2026-10-25: the rim was green or purple and stuck out): the lid line starts on the photo's
    # rim (rim_on_photo), its strokes keep that smooth line's tangent (angle_fixed: nothing turns outwards) and the
    # photo's colour where they start, on the dark rim (colour_fixed: never the iris's green); place and size are fitted.
    rim_c = rim_on_photo(target, contours, d)
    rim_place, rim_angle = E.contour_place(rim_c)
    rim_angles = angle_map.copy()
    rim_angles[rim_place > 0] = rim_angle[rim_place > 0]
    lid = {**table["digital"][0], "place": rim_place + 1e-6 * region, "angles": t(rim_angles), "angle_fixed": True,
           "colour_fixed": True}
    table["digital_rim"] = [lid] + table["digital"][1:]
    # digital_rim_long: the same rim with longer strokes (10 px instead of 6, 160 instead of 200) so neighbours overlap
    # into one line instead of dashes
    # its radial fibres also keep FIBRE_KEEP px off the rim (they reached within RIM_KEEP of it and, 8 px long, poked
    # through the rim as dark ticks)
    deep = distance_transform_edt(inside) > FIBRE_KEEP
    fibres = {**table["digital"][4], "place": deep.astype(float), "protect": (t(~deep), PROTECT), "weight": t(deep)}
    table["digital_rim_long"] = [{**lid, "length": 10, "n": 160}] + table["digital"][1:4] + [fibres] + table["digital"][5:]
    for name, ns in LITE.items():                               # the same steps with fewer strokes (0 drops a step)
        table[name] = [{**layer, "n": n} for layer, n in zip(table["digital"], ns) if n]
    return table, {"inside": inside}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24, help="validation cats, in eye_crop_styles' order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--styles", default="layers,fill_first,rim_hl", help="which variants besides face")
    ap.add_argument("--tracer", default="sam", choices=("sam", "lines"), help="the smooth outline: SAM or thin dark lines")
    ap.add_argument("--part", default="0/1", help="i/k: every k-th of the n cats, starting at the i-th (parallel runs)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    labels = np.load(E.LABELS)
    index = {str(i): k for k, i in enumerate(labels["ids"])}
    z = np.load(E.SPLIT)
    ids = [str(i) for i in z["ids"][z["split"] == "val"][:a.n]]
    part, parts = map(int, a.part.split("/"))
    ids = ids[part::parts]
    rows = []
    for rel in ids:
        name = Path(rel).stem
        pts = labels["points"][index[rel]]
        ebox = E.eye_box(pts)
        photo = B.load(str(E.AFHQ / rel), str(E.MASKS / f"{name}_mask1.png"), ebox)
        eyes = (pts[:2] - np.array(ebox[:2])) * (B.SIDE / (ebox[2] - ebox[0]))
        d = float(np.linalg.norm(eyes[0] - eyes[1]))
        target = photo.target.numpy()
        contours = E.eye_contours(target, eyes, d)
        smooth = sam_outlines(target, eyes, d) if a.tracer == "sam" else smooth_outlines(target, eyes, d)
        face = face_on_crop(name, ebox)
        table, _ = variants(target, eyes, d, contours)              # the outline as traced, ray by ray
        table_s, extra = variants(target, eyes, d, smooth)          # the outline as one smooth curve
        table["smooth"], table["fine"] = table_s["blended"], table_s["fine"]
        for k in table_s:                                           # digital and its shorter versions: smooth outline
            if k.startswith(("digital", "lite")):
                table[k] = table_s[k]
        contours = smooth                                           # measured and saved with the smooth outline
        inside = extra["inside"]
        dist_out = distance_transform_edt(~inside)
        band = (dist_out > 1.5) & (dist_out <= 5)
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        disc = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < 0.35 * d
        images = {"photo": target, "face": face}
        row = {"id": rel, "eye_dist_px": round(d, 1), "photo_rim": round(E.rim_strength(target, eyes, d), 4),
               "inside_px": int(inside.sum()), "outline_points": [len(c) for c in contours]}
        chosen = a.styles.split(",")
        for style in ["face"] + chosen:
            if style == "face":
                specs = []
            elif style.endswith("_to_digital"):                      # fitted to the digital painting, not the photo
                goal = torch.from_numpy(np.asarray(images["digital"], np.float32))
                specs = [{**sp, "target": goal} for sp in table[style[:-len("_to_digital")]]]
            else:
                specs = table[style]
            t0 = time.time()
            final = E.paint(photo.target, photo.mask, specs, start=face)[-1] if specs else face
            images[style] = final
            err = np.abs(final - target)
            row[style] = {"l1": round(float(err.mean()), 4), "eye_l1": round(float(err[disc].mean()), 4),
                          "overflow_l1": round(float(err[band].mean()), 4), "rim": round(E.rim_strength(final, eyes, d), 4),
                          "strokes": sum(s["n"] for s in specs), "seconds": round(time.time() - t0, 1)}
            if "digital" in images:                                 # how far from the digital painting, in the eyes
                row[style]["to_digital_eye"] = round(float(np.abs(final - images["digital"])[disc].mean()), 4)
        light = (E.luminance(target) - uniform_filter(E.luminance(target), 9)).clip(0)
        light *= np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < 0.25 * d
        light /= max(np.percentile(light[light > 0], 99), 1e-6) if (light > 0).any() else 1.0
        np.savez_compressed(a.out / f"{name}.npz", eyes=eyes, contour_0=contours[0], contour_1=contours[1], inside=inside,
                            highlights=E.peaks(light.clip(0, 1), E.HIGHLIGHT[2], spacing=4.0, floor=0.35), **images)
        rows.append(row)
        print(json.dumps(row), flush=True)
    styles = ["face"] + chosen
    summary = {s: {k: round(float(np.mean([r[s][k] for r in rows])), 4) for k in ("l1", "eye_l1", "overflow_l1", "rim")}
               for s in styles}
    summary["photo_rim"] = round(float(np.mean([r["photo_rim"] for r in rows])), 4)
    if parts > 1:                                                   # parallel parts: summary_<i>of<k>.json
        (a.out / f"summary_{part}of{parts}.json").write_text(json.dumps({"cats": rows, "summary": summary}, indent=1))
        print(json.dumps(summary))
        return
    (a.out / "summary.json").write_text(json.dumps({"cats": rows, "summary": summary}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
