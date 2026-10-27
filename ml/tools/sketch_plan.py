"""The sketch a painter makes before painting (owner, 2026-10-19: "a painter first decides the angle of the face, then
draws the outlines; nine points are not enough to mark where the eye sockets are"). For each cat of the repaint
inputs (tools/repaint_prep.py), from the photo and the nine face points:

  pose     yaw, pitch, roll of the head. A mean 3D cat head (the nine points) is recovered from the 2D points of all
           first 12 input shards (AFHQ 5,547 and CAT 453) by Tomasi-Kanade factorisation (scaled orthographic cameras; rank 3; metric upgrade), so the
           template's depths come from the data, not from hand-set numbers. Each cat's pose is the scaled rotation
           that best maps the template onto its points (least squares), as Euler angles (roll about the view axis,
           then pitch, then yaw).
  head     the head's outline where the cat meets the background (tri 1 next to tri 0); where it meets the
           uncertain region (the chin and neck, tri 2) there is no line, because the head's end there is not known.
           Each point is labelled ear or head by the ear triangles.
  eyes     each eye's outline from SAM (facebook/sam-vit-base, as tools/eye_layers) on a zoomed crop around the eye,
           prompted from several points near the eye point, the most eye-like mask kept (eye_outlines).
  pupils   in each eye (its outline shrunk by PUPIL_SHRINK towards its centre, so the dark rim is not taken for the
           pupil), the dark blob around the darkest point, as an ellipse (centre, two radii, angle).
  nose     SAM prompted with points near the nose (NOSE_AT of the way from the eyes to the mouth point), not the eyes,
           the bridge (BRIDGE_AT) or below the mouth; the best mask of a nose's size.
  mouth    the darkest thin line across the muzzle below the nose (one point per column of the face frame).

Two forms (owner: do both):
  A  plan tokens: the pose (three angle bins) and each outline as a short list of points on the 256 grid, written
     before the strokes (format and token counts in summary.json);
  B  a faint sketch layer rendered from the same outlines, for display under the painting.

    PYTHONPATH=.:tools python tools/sketch_plan.py --n 6 --out runs/sketch_plan/six
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.ndimage import label as cc_label
from scipy.spatial.transform import Rotation
from skimage.measure import find_contours

import eye_crop_styles as E
import eye_layers as EL
from repaint_tidy import face_frame

ML = Path(__file__).resolve().parents[1]
INPUTS = ML / "data" / "processed" / "repaint" / "inputs"
AFHQ_SHARDS = range(0, 12)          # all AFHQ (shards 0-10 and the first 47 cats of 11) plus 453 CAT cats of 11
SIDE = 256
NOSE_AT = 0.85                      # the nose prompt, as a fraction of the way from the eyes' midpoint to the mouth
BRIDGE_AT = 0.45                    # a negative prompt on the bridge of the nose (SAM otherwise takes the whole bridge)
HEAD_STEP = 6.0                     # px between plan points along the head's outline
EYE_POINTS = 16
NOSE_POINTS = 12
MOUTH_HALF = 0.4                    # half the mouth's width searched, in eye distances
EYE_CROP = 1.2                      # eye crop side for SAM, in eye distances
PUPIL_SHRINK = 0.75
EYE_PROMPTS = ((0, 0), (0, 0.08), (0, 0.15), (0, -0.08))
ANGLE_BINS = 64                     # pose bins over -90..90 degrees


# ---------------------------------------------------------------- pose
def template(points: np.ndarray) -> np.ndarray:
    """(3, 9) mean head from (F, 9, 2) views by factorisation; eye distance 1, x along the eyes, y towards the mouth,
    z away from the viewer (the ears behind the mouth)."""
    w = points - points.mean(1, keepdims=True)
    W = np.concatenate([w[..., 0], w[..., 1]], 0)                       # (2F, 9): all x rows, then all y rows
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    M, S = U[:, :3] * np.sqrt(s[:3]), np.sqrt(s[:3])[:, None] * Vt[:3]
    f = len(points)
    m, n = M[:f], M[f:]

    def row(a: np.ndarray, b: np.ndarray) -> np.ndarray:                 # a L b^T as linear in L's 6 entries
        return np.stack([a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0], a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
                         a[:, 1] * b[:, 1], a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1], a[:, 2] * b[:, 2]], 1)
    A = np.concatenate([row(m, m) - row(n, n), row(m, n)], 0)
    ell = np.linalg.svd(A)[2][-1]
    L = np.array([[ell[0], ell[1], ell[2]], [ell[1], ell[3], ell[4]], [ell[2], ell[4], ell[5]]])
    if np.trace(L) < 0:
        L = -L
    vals, vecs = np.linalg.eigh(L)
    Q = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
    shape = np.linalg.solve(Q, S)                                        # (3, 9), metric up to rotation
    ex = shape[:, 1] - shape[:, 0]
    d = np.linalg.norm(ex)
    ex /= d
    to_mouth = shape[:, 2] - shape[:, :2].mean(1)
    ey = to_mouth - ex * (ex @ to_mouth)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    t = np.stack([ex, ey, ez]) @ (shape - shape[:, :2].mean(1, keepdims=True)) / d
    if t[2, [3, 4, 5, 6, 7, 8]].mean() < t[2, 2]:                        # reflection: ears must be behind the mouth
        t[2] = -t[2]
    return t


def pose(tmpl: np.ndarray, pts: np.ndarray) -> dict:
    """Scaled-orthographic fit of the template to one cat's points: yaw, pitch, roll (degrees), scale, residual."""
    w = (pts - pts.mean(0)).T                                            # (2, 9)
    s = tmpl - tmpl.mean(1, keepdims=True)
    Mx = w @ np.linalg.pinv(s)                                           # (2, 3) affine
    u, sig, vt = np.linalg.svd(Mx, full_matrices=False)
    r12 = u @ vt
    R = np.vstack([r12, np.cross(r12[0], r12[1])])
    scale = sig.mean()
    resid = np.linalg.norm(scale * r12 @ s - w, axis=0).mean() / np.linalg.norm(pts[0] - pts[1])
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("ZXY", degrees=True)
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "scale": float(scale),
            "residual": float(resid), "R": R}


# ---------------------------------------------------------------- outlines
def head_outline(tri: np.ndarray, points: np.ndarray) -> list[np.ndarray]:
    """Runs of the cat's outline that touch the background (no line along the uncertain chin), resampled every
    HEAD_STEP px; each point labelled 1 if inside an ear triangle."""
    cat = tri == 1
    near_bg = binary_dilation(tri == 0, iterations=2)
    runs = []
    for c in find_contours(np.pad(cat, 1).astype(float), 0.5):
        xy = c[:, ::-1] - 1                                               # (x, y) in pixels
        if len(xy) < 20:
            continue
        ix, iy = np.clip(xy[:, 0].round().astype(int), 0, SIDE - 1), np.clip(xy[:, 1].round().astype(int), 0, SIDE - 1)
        on = near_bg[iy, ix]
        if on.all():
            pieces = [xy]
        else:                                                             # rotate so a run never wraps around
            k = int(np.argmin(on))
            xy, on = np.roll(xy, -k, 0), np.roll(on, -k)
            lab, n = cc_label(on)
            pieces = [xy[lab == j] for j in range(1, n + 1)]
        for p in pieces:
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            if seg[-1] < 3 * HEAD_STEP:
                continue
            at = np.arange(0, seg[-1] + 1e-6, HEAD_STEP)
            q = np.c_[np.interp(at, seg, p[:, 0]), np.interp(at, seg, p[:, 1])]
            ear = np.zeros(len(q), bool)
            for t in ((3, 4, 5), (6, 7, 8)):
                big = points[list(t)].mean(0) + 1.25 * (points[list(t)] - points[list(t)].mean(0))
                ear |= Polygon(big).contains_points(q)
            runs.append(np.c_[q, ear])
    return runs


def resample_closed(xy: np.ndarray, n: int) -> np.ndarray:
    p = np.r_[xy, xy[:1]]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    at = np.linspace(0, seg[-1], n, endpoint=False)
    return np.c_[np.interp(at, seg, p[:, 0]), np.interp(at, seg, p[:, 1])]


def eye_outlines(target: np.ndarray, eyes: np.ndarray, d: float) -> list[np.ndarray]:
    """Each eye's outline from SAM on a crop of EYE_CROP eye distances around the eye point, zoomed to 256 px. The face
    points are not always on the eye (on 2 of the first 6 cats the point sits on the fur just above it, 2026-10-19),
    so SAM is prompted from several points near it (EYE_PROMPTS, in eye distances along and across the eye line;
    negatives 0.6 eye distances above and below) and, of all the masks, the best-scoring eye-like one is kept: the
    size of a disc of 0.10-0.35 eye distances, compact (at least 0.85 of its convex hull), no point farther than 0.4
    eye distances from its centre, and its centre within 0.25 eye distances of the eye point. (n, 2) on the canvas;
    an ellipse of 0.2 x 0.12 eye distances around the point when no mask qualifies."""
    import torch
    from skimage.morphology import convex_hull_image
    if not EL._SAM:
        EL.sam_outlines(target, eyes, d)                                  # loads the model
    model, proc = EL._SAM["model"], EL._SAM["proc"]
    ex = (eyes[1] - eyes[0]) / d
    ey = np.array([-ex[1], ex[0]])
    img = Image.fromarray((target.clip(0, 1) * 255).astype(np.uint8))
    out = []
    for e in eyes:
        side = EYE_CROP * d
        lo = e - side / 2
        z = SIDE / side
        crop = np.asarray(img.transform((SIDE, SIDE), Image.EXTENT, (*lo, *(lo + side)), Image.BICUBIC,
                                        fillcolor=(255, 255, 255)))
        dz = d * z
        to = lambda q: (q - lo) * z                                      # noqa: E731
        best, best_score = None, -1.0
        emb = None                                                        # the crop's image embedding, computed once
        for u, v in EYE_PROMPTS:
            c = e + d * (u * ex + v * ey)
            pos = [to(c), to(c + 0.1 * d * ex), to(c - 0.1 * d * ex)]
            neg = [to(e - 0.6 * d * ey), to(e + 0.6 * d * ey)]
            inp = proc(images=crop, input_points=[[[list(map(float, q)) for q in pos + neg]]],
                       input_labels=[[[1, 1, 1, 0, 0]]], return_tensors="pt")
            with torch.no_grad():
                if emb is None:
                    emb = model.get_image_embeddings(inp["pixel_values"])
                res = model(image_embeddings=emb, input_points=inp["input_points"], input_labels=inp["input_labels"],
                            multimask_output=True)
            masks = proc.image_processor.post_process_masks(res.pred_masks, inp["original_sizes"],
                                                            inp["reshaped_input_sizes"])[0][0].numpy()
            for m, sc in zip(masks, res.iou_scores[0, 0].numpy()):
                area = m.sum()
                if not (np.pi * (0.10 * dz) ** 2 < area < np.pi * (0.35 * dz) ** 2) or sc <= best_score:
                    continue
                ys, xs = np.nonzero(m)
                cen = np.array([xs.mean(), ys.mean()])
                if (np.hypot(xs - cen[0], ys - cen[1]).max() > 0.4 * dz or np.linalg.norm(cen - to(e)) > 0.25 * dz
                        or area < 0.85 * convex_hull_image(m).sum()):
                    continue
                best, best_score = m, float(sc)
        if best is None:
            t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
            out.append(e + d * (0.2 * np.cos(t)[:, None] * ex + 0.12 * np.sin(t)[:, None] * ey))
            continue
        c = max(find_contours(np.pad(best, 1).astype(float), 0.5), key=len)[:, ::-1] - 1
        out.append(c / z + lo)
    return out


def pupil(lum: np.ndarray, outline: np.ndarray) -> dict | None:
    """The dark blob around the darkest (smoothed) point inside the eye, as an ellipse from its second moments."""
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    c0 = outline.mean(0)
    outline = c0 + PUPIL_SHRINK * (outline - c0)
    inside = Polygon(outline).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(SIDE, SIDE)
    if inside.sum() < 12:
        return None
    sm = gaussian_filter(lum, 1.0)
    vals = sm[inside]
    lo, mid = vals.min(), np.median(vals)
    dark = inside & (sm < lo + 0.4 * (mid - lo))
    lab, _ = cc_label(dark)
    iy, ix = np.unravel_index(np.argmin(np.where(inside, sm, np.inf)), sm.shape)
    if lab[iy, ix] == 0:
        return None
    blob = lab == lab[iy, ix]
    if blob.sum() < 3:                                     # one or two pixels: no shape to fit (2026-10-19: NaN radii)
        return None
    ys, xs = np.nonzero(blob)
    c = np.array([xs.mean(), ys.mean()]) + 0.5
    cov = np.cov(np.c_[xs, ys].T) + np.eye(2) * 1e-3
    vals2, vecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(vals2)):
        return None
    return {"centre": c, "radii": 2 * np.sqrt(vals2[::-1]), "angle": float(np.arctan2(vecs[1, 1], vecs[0, 1])),
            "share": float(blob.sum() / inside.sum())}


def nose_outline(target: np.ndarray, points: np.ndarray, d: float) -> np.ndarray | None:
    """SAM's mask for the nose (positive: the nose point and a little either side; negative: both eyes, the mouth
    point), the best-scoring mask of a nose's size, as its outer contour resampled to NOSE_POINTS."""
    import torch
    EL.sam_outlines(target, points[:2], d) if not EL._SAM else None              # loads the model once
    model, proc = EL._SAM["model"], EL._SAM["proc"]
    mid = points[:2].mean(0)
    nose = mid + NOSE_AT * (points[2] - mid)
    ex = (points[1] - points[0]) / d
    pos = [nose, nose + 0.08 * d * ex, nose - 0.08 * d * ex]
    neg = [points[0], points[1], mid + BRIDGE_AT * (points[2] - mid),
           points[2] + 0.15 * d * (points[2] - mid) / np.linalg.norm(points[2] - mid)]
    img = (target.clip(0, 1) * 255).astype(np.uint8)
    inp = proc(images=img, input_points=[[[list(map(float, q)) for q in pos + neg]]],
               input_labels=[[[1, 1, 1, 0, 0, 0, 0]]], return_tensors="pt")
    with torch.no_grad():
        res = model(**inp, multimask_output=True)
    masks = proc.image_processor.post_process_masks(res.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])[0][0].numpy()
    scores = res.iou_scores[0, 0].numpy()
    area = masks.reshape(len(masks), -1).sum(1)
    ok = (area > np.pi * (0.06 * d) ** 2) & (area < np.pi * (0.25 * d) ** 2)
    if not ok.any():
        return None
    m = masks[int(np.argmax(np.where(ok, scores, -1)))]
    cs = find_contours(np.pad(m, 1).astype(float), 0.5)
    c = max(cs, key=len)[:, ::-1] - 1
    return resample_closed(c, NOSE_POINTS)


def mouth_line(lum: np.ndarray, points: np.ndarray, nose: np.ndarray | None, d: float) -> np.ndarray:
    """The mouth as one smooth curve: across the columns of the face frame (x within MOUTH_HALF eye distances of the
    mouth point), between the nose's lowest point and 0.3 eye distances below the mouth point, the path of strongest
    thin dark line that moves at most one row per column (dynamic programming), cut to the columns around the middle
    where the line is at least CONTOUR_FLOOR strong."""
    lines = E.dark_lines(lum)
    mid = points[:2].mean(0)
    ex = (points[1] - points[0]) / d
    ey = np.array([-ex[1], ex[0]])
    if np.dot(points[2] - mid, ey) < 0:
        ey = -ey
    m = face_frame(points, points[2:3])[0]
    top = face_frame(points, nose).max(0)[1] if nose is not None else m[1] - 0.15
    xs = np.arange(m[0] - MOUTH_HALF, m[0] + MOUTH_HALF, 1.5 / d)
    ys = np.arange(top, m[1] + 0.3, 0.5 / d)
    if len(ys) < 2:
        return np.zeros((0, 2))
    q = mid + d * (xs[:, None, None] * ex + ys[None, :, None] * ey)        # (columns, rows, 2)
    ix, iy = np.clip(q[..., 0].astype(int), 0, SIDE - 1), np.clip(q[..., 1].astype(int), 0, SIDE - 1)
    v = lines[iy, ix]
    score, back = v[0].copy(), np.zeros(v.shape, int)
    for c in range(1, len(xs)):
        prev = np.stack([np.r_[-np.inf, score[:-1]], score, np.r_[score[1:], -np.inf]])
        k = np.argmax(prev, 0)
        back[c] = np.arange(len(ys)) + k - 1
        score = prev[k, np.arange(len(ys))] + v[c]
    path = [int(np.argmax(score))]
    for c in range(len(xs) - 1, 0, -1):
        path.append(back[c, path[-1]])
    path = path[::-1]
    strong = v[np.arange(len(xs)), path] >= E.CONTOUR_FLOOR
    centre = int(np.argmin(np.abs(xs - m[0])))
    lo = hi = centre
    gap = 3                                                                 # allow a few weak columns in a row
    while lo > 0 and strong[max(0, lo - gap):lo].any():
        lo -= 1
    while hi < len(xs) - 1 and strong[hi + 1:hi + 1 + gap].any():
        hi += 1
    return q[np.arange(lo, hi + 1), np.array(path[lo:hi + 1])]


# ---------------------------------------------------------------- the two forms
def plan_tokens(p: dict, head: list, eyes: list, pupils: list, nose, mouth) -> list[str]:
    """A: the plan as tokens. Angles in ANGLE_BINS bins over -90..90 degrees; every point as two grid tokens (x, y on
    the 256 grid, the strokes' own position bins); each outline opens with its kind and ends with [end]."""
    def ang(a: float) -> str:
        return f"a{int(np.clip((a + 90) / 180 * ANGLE_BINS, 0, ANGLE_BINS - 1))}"

    def pts(xy: np.ndarray) -> list[str]:
        return [t for x, y in np.clip(np.round(xy), 0, SIDE - 1).astype(int) for t in (f"x{x}", f"y{y}")]
    tok = ["[plan]", "[pose]", ang(p["yaw"]), ang(p["pitch"]), ang(p["roll"])]
    for run in head:
        tok += ["[head]"] + pts(run[:, :2]) + ["[end]"]
    for e in eyes:
        tok += ["[eye]"] + pts(e) + ["[end]"]
    for pu in pupils:
        if pu is not None:
            tok += ["[pupil]"] + pts(pu["centre"][None]) + [f"r{int(min(pu['radii'][0], 63))}", f"r{int(min(pu['radii'][1], 63))}",
                                                            ang(np.degrees(pu["angle"]) % 180 - 90)]
    if nose is not None:
        tok += ["[nose]"] + pts(nose) + ["[end]"]
    if len(mouth):
        tok += ["[mouth]"] + pts(mouth) + ["[end]"]
    return tok + ["[/plan]"]


PART_COLOURS = {"head": (40, 90, 220), "ear": (240, 140, 0), "eye": (0, 170, 60), "pupil": (220, 0, 0),
                "nose": (200, 0, 200), "mouth": (0, 180, 200)}


def render_sketch(head: list, eyes: list, pupils: list, nose, mouth, under: np.ndarray | None = None,
                  grey: int = 150, scale: int = 2, colours: bool = False) -> Image.Image:
    """B: a faint pencil sketch of the plan (grey lines on white, or over `under`); `colours` marks each part in its
    own colour (for checking the plan, not part of the sketch)."""
    base = np.full((SIDE, SIDE, 3), 255, np.uint8) if under is None else under
    img = Image.fromarray(base).resize((SIDE * scale, SIDE * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(img)

    def line(xy: np.ndarray, part: str, closed: bool = False) -> None:
        q = [tuple(map(float, v * scale)) for v in xy]
        if len(q) > 1:
            dr.line(q + ([q[0]] if closed else []), fill=PART_COLOURS[part] if colours else (grey,) * 3, width=scale)
    for run in head:
        for j in range(len(run) - 1):
            line(run[j:j + 2, :2], "ear" if run[j, 2] and run[j + 1, 2] else "head")
    for e in eyes:
        line(e, "eye", closed=True)
    for pu in pupils:
        if pu is None:
            continue
        t = np.linspace(0, 2 * np.pi, 40)
        c, (a, b), th = pu["centre"], pu["radii"], pu["angle"]
        xy = np.c_[c[0] + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th),
                   c[1] + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)]
        line(xy, "pupil", closed=True)
    if nose is not None:
        line(nose, "nose", closed=True)
    if len(mouth):
        line(mouth, "mouth")
    return img


def plan_one(tmpl: np.ndarray, target_u8: np.ndarray, tri: np.ndarray, pts: np.ndarray) -> dict:
    target = target_u8.astype(np.float32) / 255
    d = float(np.linalg.norm(pts[0] - pts[1]))
    p = pose(tmpl, pts)
    head = head_outline(tri, pts)
    eyes = [resample_closed(o, EYE_POINTS) for o in eye_outlines(target, pts[:2], d)]
    pupils = [pupil(E.luminance(target), e) for e in eyes]
    nose = nose_outline(target, pts, d)
    mouth = mouth_line(E.luminance(target), pts, nose, d)
    return {"pose": p, "head": head, "eyes": eyes, "pupils": pupils, "nose": nose, "mouth": mouth,
            "tokens": plan_tokens(p, head, eyes, pupils, nose, mouth)}


def batch(tmpl: np.ndarray, a: argparse.Namespace) -> None:
    """Every cat of the given shards: the plan alone (pose, outlines, tokens) as {source}__{id}.npz per shard folder."""
    import time
    from repaint import parse_shards
    for sh in parse_shards(a.batch):
        z = np.load(INPUTS / f"shard_{sh:03d}.npz")
        d = a.out / f"shard_{sh:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(len(z["id"])):
            if a.source and str(z["source"][i]) != a.source:
                continue
            path = d / f"{z['source'][i]}__{z['id'][i]}.npz"
            if path.exists():
                continue
            t0 = time.time()
            pts = z["points"][i].astype(np.float64)
            try:
                r = plan_one(tmpl, z["target"][i], z["tri"][i], pts)
            except Exception as e:                     # one bad cat must not stop the batch (it did once, 2026-10-19)
                with open(a.out / "failed.jsonl", "a") as f:
                    f.write(json.dumps({"shard": sh, "id": str(z["id"][i]), "error": repr(e)[:300]}) + "\n")
                continue
            pu = [np.r_[q["centre"], q["radii"], q["angle"], q["share"]] if q is not None else np.full(6, np.nan) for q in r["pupils"]]
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, yaw=r["pose"]["yaw"], pitch=r["pose"]["pitch"], roll=r["pose"]["roll"],
                                residual=r["pose"]["residual"], R=r["pose"]["R"],
                                head=np.array(r["head"] + [None], dtype=object)[:-1], eyes=np.array(r["eyes"]),
                                pupils=np.array(pu), nose=r["nose"] if r["nose"] is not None else np.zeros((0, 2)),
                                mouth=r["mouth"], tokens=np.array(r["tokens"]), version=np.array("sketch-plan-v1 (2026-10-19)"))
            tmp.rename(path)
            with open(a.out / f"shard_{sh:03d}.jsonl", "a") as f:
                f.write(json.dumps({"id": str(z["id"][i]), "tokens": len(r["tokens"]), "nose": r["nose"] is not None,
                                    "seconds": round(time.time() - t0, 1)}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ids", nargs="*", help="cat ids (default: the first n of the shard)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", default="", help="shards to plan in full (e.g. 0-11), saving only the plan, no pictures")
    ap.add_argument("--source", default="", help="with --batch: only this source (afhq or cat)")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    tpath = a.out.parent / "template.npz"
    if tpath.exists():
        tmpl = np.load(tpath)["template"]
    else:
        allp = np.concatenate([np.load(INPUTS / f"shard_{k:03d}.npz")["points"] for k in AFHQ_SHARDS])
        tmpl = template(allp)
        poses = [pose(tmpl, p_) for p_ in allp]
        np.savez(tpath, template=tmpl, views=len(allp),
                 yaw=[p_["yaw"] for p_ in poses], pitch=[p_["pitch"] for p_ in poses], roll=[p_["roll"] for p_ in poses],
                 residual=[p_["residual"] for p_ in poses])
    if a.batch:
        batch(tmpl, a)
        return
    z = np.load(INPUTS / f"shard_{a.shard:03d}.npz")
    ids = [str(i) for i in z["id"]]
    pick = [ids.index(i) for i in a.ids] if a.ids else list(range(a.n))
    rows = []
    for k in pick:
        target, tri, pts = z["target"][k].astype(np.float32) / 255, z["tri"][k], z["points"][k].astype(np.float64)
        d = float(np.linalg.norm(pts[0] - pts[1]))
        lum = E.luminance(target)
        p = pose(tmpl, pts)
        head = head_outline(tri, pts)
        eyes = [resample_closed(o, EYE_POINTS) for o in eye_outlines(target, pts[:2], d)]
        pupils = [pupil(lum, e) for e in eyes]
        nose = nose_outline(target, pts, d)
        mouth = mouth_line(lum, pts, nose, d)
        tok = plan_tokens(p, head, eyes, pupils, nose, mouth)
        name = Path(ids[k]).stem
        u8 = (target * 255).astype(np.uint8)
        render_sketch(head, eyes, pupils, nose, mouth).save(a.out / f"{name}_sketch.png")
        render_sketch(head, eyes, pupils, nose, mouth, under=u8, colours=True).save(a.out / f"{name}_overlay.png")
        np.savez_compressed(a.out / f"{name}.npz", target=z["target"][k], tri=tri, points=pts, R=p["R"],
                            head=np.array(head, dtype=object), eyes=np.array(eyes), nose=nose if nose is not None else np.zeros((0, 2)),
                            mouth=mouth, tokens=np.array(tok))
        row = {"id": ids[k], **{q: round(p[q], 1) for q in ("yaw", "pitch", "roll")}, "residual": round(p["residual"], 3),
               "head_runs": len(head), "head_points": int(sum(len(r) for r in head)),
               "pupils": [None if q is None else {"radii": [round(float(r), 1) for r in q["radii"]], "share": round(q["share"], 2)}
                          for q in pupils],
               "nose": nose is not None, "mouth_points": len(mouth), "tokens": len(tok)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "summary.json").write_text(json.dumps({"template": np.round(tmpl, 3).tolist(), "cats": rows}, indent=1))


if __name__ == "__main__":
    main()
