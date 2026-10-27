"""Cat faces as painting steps: the sequence a GPT will learn to draw stroke by stroke and layer by layer.

Each face (catface.py's aligned, cut-out face) becomes an ordered list of steps, the order the owner set:
lines first, part by part (outline → left ear → right ear → left eye → right eye → nose and mouth → forehead →
left cheek → right cheek → muzzle and chin, largest piece first within a step), then colour at 85% opacity over
everything before (face base → inner ears → iris → pupil → nose and mouth → fur lighter than the base, nearest the
base first → shadows darker than the base, the darkest last). How many tone steps there are depends on the cat.

Lines are catface's contour drawing: ink thicker than about 6 px stays a filled shape (a black nose, a slit pupil),
thinner ink becomes centreline strokes with one width each (centerline.py), which costs about half the points of
tracing both edges. Colours are 3 fur colours plus 3 eye colours, pieces under 200 px left out, outlines simplified
to 3 px; lines keep pieces down to 12 px at 1.5 px. Measured on 16 cats (devlog 2026-10-09): 1,288 tokens median,
2,688 max.

Tokens:   [BOS] x y × 9 points   then per step: [STEP_k], and per item
          a filled shape:   [COLOUR c] (colour steps) or [SHAPE] (line steps), then [RING] x y x y … per
                            outline (several outlines of one shape are filled even-odd: holes)
          a stroke:         [STROKE] [WIDTH w] x y x y …
          [EOS]
x and y in 2 px bins of the 256 face; colours 8 levels a channel over [DARKEST, 255]; widths 1–16 px.

    python -m sellanything_ml.sketch.catshape data --workers 6     # every face of every split (CPU)
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from functools import lru_cache

import numpy as np

from . import catface as C

OUT_DIR = C.DATA_DIR / "shapes"
OPACITY = 0.85
SETTINGS = {"k": 3, "min_area": 200, "tol": 3.0, "line_min_area": 12, "line_tol": 1.5}

LINE_STEPS = ["outline", "ear L", "ear R", "eye L", "eye R", "nose/mouth", "forehead", "cheek L", "cheek R", "muzzle/chin"]
COLOUR_STEPS = ["base", "inner ears", "iris", "pupil", "nose/mouth colour", "light", "dark"]
STEPS = LINE_STEPS + COLOUR_STEPS
NBIN, NW, NLEVEL = 128, 16, 8
X0, Y0 = 0, NBIN
W0 = 2 * NBIN
C0 = W0 + NW
RING = C0 + NLEVEL ** 3
STROKE = RING + 1
SHAPE = STROKE + 1
STEP0 = SHAPE + 1
BOS = STEP0 + len(STEPS)
EOS = BOS + 1
VOCAB = EOS + 1


# --- one face → steps -----------------------------------------------------------------------------

def _kmeans(px, k, iters=15):
    c = px[np.random.default_rng(0).choice(len(px), min(k, len(px)), replace=False)]
    for _ in range(iters):
        lab = ((px[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([px[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(len(c))])
    return lab, c


def _lifted(c):
    return C.DARKEST + np.asarray(c, float) * (255 - C.DARKEST) / 255


def _lum(c):
    return float(np.asarray(c) @ [0.299, 0.587, 0.114])


def _circle(xy, r, shape):
    from skimage.draw import disk
    m = np.zeros(shape, bool)
    rr, cc = disk((xy[1], xy[0]), r, shape=shape)
    m[rr, cc] = True
    return m


@lru_cache(maxsize=1)
def _net():
    import torch

    from . import infdraw
    torch.set_num_threads(1)
    return infdraw.load(C.STYLES["contour"], "cpu")


@lru_cache(maxsize=2)
def _zip(stem: str):
    return zipfile.ZipFile(C.RAW_DIR / f"{stem}.zip")


def prepare(full_id: str) -> dict:
    """Everything that does not depend on the simplification settings: the cut-out face, its lines, the regions."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion, median_filter

    from . import infdraw
    zstem, m = full_id.split("/", 1)
    name = f"{zstem}__{m.replace('/', '__')}"
    photo = Image.open(C.RAW_DIR / "photos" / name)
    v = [float(x) for x in _zip(zstem).read(m + ".cat").decode().split()]
    p = np.array(v[1:1 + 2 * int(v[0])]).reshape(-1, 2)
    fg = C._subject_mask(name, p, (photo.height, photo.width))
    orig, inside, q = C.align(photo, p)
    keep = (C.align(Image.fromarray((fg * 255).astype(np.uint8)), p)[0][..., 0] > 127) & inside & C.face_region(q, C.BIG)
    face = np.where(keep[..., None], orig, 255).astype(np.uint8)
    d = q[1, 0] - q[0, 0]
    sh = keep.shape
    ears = []
    for ear in ((3, 4, 5), (8, 7, 6)):
        tri = Image.new("L", (C.BIG, C.BIG), 0)
        ImageDraw.Draw(tri).polygon([tuple(q[i]) for i in ear], fill=255)
        ears.append((np.asarray(tri) > 0) & keep)
    nose = q[:2].mean(0) * 0.35 + q[2] * 0.65
    lines = np.where(keep, infdraw.draw(_net(), face[None])[0], 1.0)
    return {"q": q, "d": d, "keep": keep, "face": face, "ears": ears,
            "eyes": [_circle(xy, C.EYE_R * d, sh) & keep for xy in q[:2]],
            "eyes_big": [_circle(xy, 1.4 * C.EYE_R * d, sh) for xy in q[:2]],
            "muzzle": _circle((nose[0], nose[1] + 0.1 * d), 0.35 * d, sh),
            "border": keep & ~binary_erosion(keep, iterations=4), "ink": keep & (lines < 0.5),
            "sm": median_filter(face, size=(7, 7, 1)).astype(np.float64)}


def _shapes(mask, min_area, tol):
    """Connected pieces as outline lists (even-odd), largest first."""
    from scipy.ndimage import label as cc_label
    from skimage.measure import approximate_polygon, find_contours
    lab, n = cc_label(mask)
    res = []
    for i in range(1, n + 1):
        piece = lab == i
        if piece.sum() < min_area:
            continue
        cs = find_contours(np.pad(piece, 1).astype(float), 0.5)
        res.append((piece.sum(), [approximate_polygon(c - 1, tol)[:, ::-1] for c in cs if len(c) > 3]))
    return [r for _, r in sorted(res, key=lambda t: -t[0])]


def _lines(m, min_area, tol):
    """Solid ink thicker than about 6 px stays a filled shape (its skeleton would be a hollow loop); thinner ink
    becomes centreline strokes."""
    from scipy.ndimage import binary_dilation, binary_opening
    from skimage.morphology import disk

    from .centerline import strokes
    thick = binary_opening(m, structure=disk(3)) if m.any() else m
    thin = m & ~binary_dilation(thick, iterations=1)
    return _shapes(thick, min_area, tol) + [("stroke", pts, w) for pts, w in strokes(thin, tol, min_area)]


def build(pr: dict, k=3, min_area=200, tol=3.0, line_min_area=12, line_tol=1.5) -> list:
    """[(step name, [(colour or None, item)])] in painting order; an item is a list of outlines or a stroke."""
    keep, q, d = pr["keep"], pr["q"], pr["d"]
    S = lambda m: _shapes(m, min_area, tol)  # noqa: E731
    eye = pr["eyes"][0] | pr["eyes"][1]
    fur = keep & ~eye
    flab = np.full(keep.shape, -1)
    lab, cent = _kmeans(pr["sm"][fur], k)
    flab[fur] = lab
    elab = np.full(keep.shape, -1)
    el, ecent = _kmeans(pr["face"][eye].astype(np.float64), 3)
    elab[eye] = el

    steps, claimed = [], np.zeros(keep.shape, bool)
    yy, xx = np.mgrid[:C.BIG, :C.BIG]
    brow, mx = q[:2, 1].mean() - 0.15 * d, q[2, 0]
    regions = [pr["border"], pr["ears"][0], pr["ears"][1], pr["eyes_big"][0], pr["eyes_big"][1], pr["muzzle"],
               yy < brow, xx < mx - 0.3 * d, xx > mx + 0.3 * d, np.ones(keep.shape, bool)]
    for name, region in zip(LINE_STEPS, regions):        # each ink pixel goes to the first part that claims it
        mine = pr["ink"] & region & ~claimed
        claimed |= mine
        steps.append((name, [(None, it) for it in _lines(mine, line_min_area, line_tol)]))
    base = np.bincount(lab).argmax()
    steps.append(("base", [(_lifted(cent[base]), s) for s in S(keep)]))
    steps.append(("inner ears", [(_lifted(pr["sm"][m].mean(0)), s) for m in pr["ears"] if m.any() for s in S(m)]))
    dark = ecent.sum(1).argmin()
    iris_items, pupil_items = [], []
    for disc in pr["eyes"]:
        others = disc & (elab != dark)
        if others.any():
            iris = np.bincount(elab[others], minlength=3).argmax()
            iris_items += [(_lifted(ecent[iris]), s) for s in S(disc & (elab == iris))]
        pupil_items += [(_lifted(ecent[dark]), s) for s in S(disc & (elab == dark))]
    steps.append(("iris", iris_items))
    steps.append(("pupil", pupil_items))
    nm, tones = [], []
    for j in range(len(cent)):
        if j != base:
            m = flab == j
            nm += [(_lifted(cent[j]), s) for s in S(m & pr["muzzle"])]
            tones.append((j, m & ~pr["muzzle"]))
    steps.append(("nose/mouth colour", nm))
    b = _lum(cent[base])
    for j, m in sorted([t for t in tones if _lum(cent[t[0]]) >= b], key=lambda t: _lum(cent[t[0]]) - b):
        steps.append(("light", [(_lifted(cent[j]), s) for s in S(m)]))
    for j, m in sorted([t for t in tones if _lum(cent[t[0]]) < b], key=lambda t: b - _lum(cent[t[0]])):
        steps.append(("dark", [(_lifted(cent[j]), s) for s in S(m)]))
    return steps


# --- drawing --------------------------------------------------------------------------------------

def raster(item) -> np.ndarray:
    from PIL import Image, ImageDraw
    if isinstance(item, tuple):                                    # stroke: round caps, one width
        _, pts, w = item
        img = Image.new("L", (C.BIG, C.BIG), 0)
        dr = ImageDraw.Draw(img)
        if len(pts) > 1:
            dr.line([tuple(p) for p in pts], fill=255, width=max(1, int(round(w))), joint="curve")
        for x, y in (pts[0], pts[-1]):
            dr.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
        return np.asarray(img) > 0
    m = np.zeros((C.BIG, C.BIG), bool)
    for ring in item:
        if len(ring) >= 3:
            one = Image.new("1", (C.BIG, C.BIG), 0)
            ImageDraw.Draw(one).polygon([tuple(p) for p in ring], fill=1)
            m ^= np.asarray(one)
    return m


def render(steps, opacity=OPACITY, frames=False):
    """Lines in near-black; colour laid over everything before it at the given opacity."""
    canvas = np.full((C.BIG, C.BIG, 3), 255.0)
    out = []
    for name, items in steps:
        for colour, item in items:
            m = raster(item)
            canvas[m] = 25.0 if colour is None else (1 - opacity) * canvas[m] + opacity * np.asarray(colour, float)
        out.append(canvas.copy())
    return out if frames else canvas


# --- tokens ---------------------------------------------------------------------------------------

def _xy(pts) -> list[int]:
    b = np.clip((np.asarray(pts) / 2).astype(int), 0, NBIN - 1)
    return [int(t) for xy in b for t in (X0 + xy[0], Y0 + xy[1])]


def _colour_token(c) -> int:
    lv = np.clip(np.round((np.asarray(c, float) - C.DARKEST) / (255 - C.DARKEST) * (NLEVEL - 1)), 0, NLEVEL - 1).astype(int)
    return C0 + int(lv[0] * NLEVEL * NLEVEL + lv[1] * NLEVEL + lv[2])


def encode(points: np.ndarray, steps) -> np.ndarray:
    toks = [BOS] + _xy(points)
    for name, items in steps:
        toks.append(STEP0 + STEPS.index(name))
        for colour, item in items:
            if isinstance(item, tuple):
                toks += [STROKE, W0 + int(np.clip(round(item[2]), 1, NW)) - 1] + _xy(item[1])
                continue
            toks.append(SHAPE if colour is None else _colour_token(colour))
            for ring in item:
                toks += [RING] + _xy(ring)
    return np.array(toks + [EOS], np.int16)


def decode(toks) -> tuple[np.ndarray, list, int]:
    """Tokens → (points, steps, tokens that did not fit the grammar). Coordinates come back at bin centres."""
    toks = [int(t) for t in toks]
    bad, i = 0, 1 if toks and toks[0] == BOS else 0

    def coords(i):
        pts = []
        while i + 1 < len(toks) and X0 <= toks[i] < X0 + NBIN and Y0 <= toks[i + 1] < Y0 + NBIN:
            pts.append(((toks[i] - X0) * 2 + 1, (toks[i + 1] - Y0) * 2 + 1))
            i += 2
        return np.array(pts, float).reshape(-1, 2), i

    points, i = coords(i)
    steps, colour = [], None
    while i < len(toks) and toks[i] != EOS:
        t = toks[i]
        if STEP0 <= t < STEP0 + len(STEPS):
            steps.append((STEPS[t - STEP0], []))
            colour, i = None, i + 1
        elif C0 <= t < RING and steps:
            lv = np.array([(t - C0) // (NLEVEL * NLEVEL), (t - C0) // NLEVEL % NLEVEL, (t - C0) % NLEVEL])
            colour, i = C.DARKEST + lv * (255 - C.DARKEST) / (NLEVEL - 1), i + 1
            steps[-1][1].append((colour, []))
        elif t == SHAPE and steps:
            steps[-1][1].append((None, []))
            i += 1
        elif t == RING and steps and steps[-1][1] and isinstance(steps[-1][1][-1][1], list):
            pts, i = coords(i + 1)
            steps[-1][1][-1][1].append(pts)
        elif t == STROKE and steps and i + 1 < len(toks) and W0 <= toks[i + 1] < C0:
            pts, j = coords(i + 2)
            if len(pts):
                steps[-1][1].append((None, ("stroke", pts, float(toks[i + 1] - W0 + 1))))
            else:                                                  # a stroke with no points: nothing to draw
                bad += 1
            i = j
        else:
            bad, i = bad + 1, i + 1
    return points, steps, bad


# --- the whole dataset ----------------------------------------------------------------------------

def _one(full_id: str):
    try:
        pr = prepare(full_id)
        pts = pr["q"]
        return full_id, encode(pts, build(pr, **SETTINGS)), None
    except Exception as e:                                         # reported per face, never silently dropped
        return full_id, None, f"{type(e).__name__}: {e}"


def stage_data(workers: int = 6) -> None:
    from concurrent.futures import ProcessPoolExecutor
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report, t0 = {"settings": SETTINGS, "vocab": VOCAB}, time.time()
    for split in ("train", "validation", "test"):
        ids = list(C.load(split)["ids"])
        seqs, errors = {}, {}
        with ProcessPoolExecutor(workers) as ex:
            for n, (fid, toks, err) in enumerate(ex.map(_one, ids, chunksize=8), 1):
                if err:
                    errors[fid] = err
                else:
                    seqs[fid] = toks
                if n % 500 == 0:
                    print(f"  {split}: {n}/{len(ids)}, {time.time() - t0:.0f} s", flush=True)
        kept = [f for f in ids if f in seqs]
        lens = np.array([len(seqs[f]) for f in kept])
        np.savez_compressed(OUT_DIR / f"{split}.npz", tokens=np.concatenate([seqs[f] for f in kept]),
                            offsets=np.r_[0, np.cumsum(lens)], ids=np.array(kept))
        report[split] = {"faces": len(ids), "encoded": len(kept), "errors": len(errors),
                         "tokens_median": int(np.median(lens)), "tokens_p95": int(np.percentile(lens, 95)),
                         "tokens_max": int(lens.max()), "tokens_total": int(lens.sum())}
        if errors:
            (OUT_DIR / f"errors_{split}.json").write_text(json.dumps(errors, indent=1) + "\n")
        print(split, json.dumps(report[split]), flush=True)
    report["seconds"] = round(time.time() - t0)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def load(split: str) -> list[np.ndarray]:
    z = np.load(OUT_DIR / f"{split}.npz")
    t, o = z["tokens"], z["offsets"]
    return [t[o[i]:o[i + 1]] for i in range(len(o) - 1)]


def round_trip(n: int = 32) -> dict:
    """Validation faces: draw from the steps directly and from encode → decode; the share of face pixels that
    differ by less than 40 grey levels in every channel (quantisation of positions, colours and widths)."""
    ids = list(C.load("validation")["ids"])[:n]
    same, bad = [], 0
    for fid in ids:
        pr = prepare(fid)
        steps = build(pr, **SETTINGS)
        _, back, b = decode(encode(pr["q"], steps))
        bad += b
        same.append(float((np.abs(render(steps) - render(back)).max(-1) < 40)[pr["keep"]].mean()))
    rec = {"faces": n, "pixels_within_40_mean": round(float(np.mean(same)), 4),
           "pixels_within_40_min": round(float(np.min(same)), 4), "grammar_errors": bad}
    print(json.dumps(rec), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["data", "roundtrip"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.workers)
    else:
        rec = round_trip()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "roundtrip.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
