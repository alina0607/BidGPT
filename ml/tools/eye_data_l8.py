"""The eye model's training data (owner, 2026-10-21): on every cat's L8 base (tools/gen_l8_bases.py: L8 painting from the
cat's real plan), the eye painter's four layers (tools/eye_layers.py "layers": outline 120 x 10 x 1.5 px, iris 40 x 12 x 6,
pupil 24 x 8 x 3, highlight 8 x 4 x 3) are fitted to the photo inside the plan's eye outlines, on the zoomed eye crop
(as tools/eye_upper_bound_l8.py's fit stage, which measured the upper bound in zh notebook 17.24). Two changes make the
fit something a sequence model can learn, and both are measured against the upper-bound painter (--compare):

  order    each layer's strokes are ordered before fitting, eye by eye (the plan's first eye first), by angle around
           that eye's plan centre (from the left, clockwise on screen): the order the model will write them in;
           --order inside_out (owner, 2026-10-22, run E3): the iris, pupil and highlight layers go eye by eye from
           the plan centre outwards instead (nearest first, ties by angle), so a dark centre fading lighter outwards
           is a sequence that changes step by step; the outline keeps going around (it is a line along the rim).
           Strokes move while they are fitted, so each of these layers is sorted again by where its snapped strokes
           ended up, and that sorted layer is what is painted, stored and fitted on (24 val cats: eye-disc L1 0.1829
           -> 0.1838, mean pixel change 0.34/255; order-distance rank correlation 0.32 before the sort, 1.0 after);
  bins     each fitted layer is snapped to the token grid (crop x/y 256, angle 32, size 6 x 6, colour 12^3, opacity 8
           over catbrush's 0.3-1.0) and the next layer is fitted on the snapped canvas, so the stored strokes paint
           exactly what is stored (as tools/prepare_l.py does for the face).

--style digital (owner, 2026-10-24: the eye material should look like runs/eye_layers/digital6/steps_000031.png):
tools/eye_layers.py's digital painter instead of its four layers (lid line, soft base, light and shade, blending, radial
fibres, blending, pupil, highlight; each layer its own brush: edge softness, opacity range, bristle), or one of its
shorter versions (lite_*, eye_layers.LITE). Each layer is snapped with its own opacity range; blending layers carry
no paint (their colour bins are kept but mean nothing) and are rendered as blending. --device runs the fit on a GPU
(the stroke placement stays on the CPU's random generator); --pack reads the cats from tools/eye_pack.py's files
instead of the photos (Kaggle), --reverse goes through them from the last.

One file per cat in data/processed/brush_l/eye_l8/ (written whole; a new run skips the cats already there): the crop
(ebox, fbox), the eye points and plan outlines in the crop, per layer the bin indices (n, 9: x, y, angle, length,
width, r, g, b, opacity) and the unsnapped fitted values, the base and the painted crop (uint8), and the measures
(eye-disc L1 to the photo, rim strength) for the base and the painted crop. CPU only (one GPU job at a time).

    PYTHONPATH=.:tools python tools/eye_data_l8.py --names afhq__flickr_cat_000008 ... --compare --out runs/eye_data_l8/trial
    PYTHONPATH=.:tools python tools/eye_data_l8.py --part 0/2            (every cat with a base, half of them)
    PYTHONPATH=.:tools python tools/eye_data_l8.py --order inside_out --out data/processed/brush_l/eye_l8_inside --part 0/3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import eye_layers as EL
import eye_layers_l as ELL
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
BASES = ML / "data" / "processed" / "brush_l" / "l8_gen"
OUT = ML / "data" / "processed" / "brush_l" / "eye_l8"
ASSIGN = ML / "data" / "processed" / "review" / "assignments.csv"
NXY, NANGLE, LEVELS_SIZE, LEVELS_RGB, LEVELS_ALPHA = 256, 32, 6, 12, 8
LAYER_NAMES = ("outline", "iris", "pupil", "highlight")
BRUSHES = ((10, 1.5), (12, 6), (8, 3), (4, 3))     # each layer's (length, width) on the eye crop (eye_layers "layers")
DIGITAL_NAMES = ("lid", "base", "shade", "blend", "fibres", "blend2", "pupil", "highlight")


def layer_names(style: str) -> tuple[str, ...]:
    """The layers of a style, in painting order (lite styles drop the steps with no strokes)."""
    if style == "layers":
        return LAYER_NAMES
    if style in ("digital", "digital_rim", "digital_rim_long"):
        return DIGITAL_NAMES
    return tuple(n for n, c in zip(DIGITAL_NAMES, EL.LITE[style]) if c)


BRUSH_KEYS = ("length", "width", "n", "edge_px", "alpha_range", "bristle", "blend")


def brush_of(spec: dict) -> dict:
    """A layer's brush as stored with the material (catbrush's defaults where the layer keeps them)."""
    return {"length": spec["length"], "width": spec["width"], "n": spec["n"], "edge_px": spec.get("edge_px", B.EDGE_PX),
            "alpha_range": list(spec.get("alpha_range", B.ALPHA_RANGE)), "bristle": spec.get("bristle", B.BRISTLE),
            "blend": bool(spec.get("blend", False))}


def _bin(x, lo, hi, n):
    return np.clip(((np.asarray(x, float) - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _centre(i, lo, hi, n):
    return lo + (np.asarray(i, float) + 0.5) * (hi - lo) / n


def to_bins(pr: B.StrokeParams, base: tuple) -> np.ndarray:
    """(n, 9) bin indices of fitted strokes: x, y, angle, length, width, r, g, b, opacity."""
    xy, lw = pr.xy.detach().cpu().numpy(), pr.lw.detach().cpu().numpy()
    scale = lw / np.array(base, float)
    return np.column_stack([_bin(xy[:, 0], 0, B.SIDE, NXY), _bin(xy[:, 1], 0, B.SIDE, NXY),
                            _bin(np.mod(pr.angle.detach().cpu().numpy(), math.pi), 0, math.pi, NANGLE),
                            _bin(scale[:, 0], *B.SIZE_RANGE, LEVELS_SIZE), _bin(scale[:, 1], *B.SIZE_RANGE, LEVELS_SIZE),
                            _bin(pr.colour.detach().cpu().numpy(), 0, 1, LEVELS_RGB),
                            _bin(pr.alpha.detach().cpu().numpy(), *B.ALPHA_RANGE, LEVELS_ALPHA)]).astype(np.int16)


def from_bins(bins: np.ndarray, base: tuple) -> dict:
    """Stroke values at the bin centres (Strokes.from_stage's input)."""
    b = np.asarray(bins, int)
    return {"xy": np.column_stack([_centre(b[:, 0], 0, B.SIDE, NXY), _centre(b[:, 1], 0, B.SIDE, NXY)]),
            "angle": _centre(b[:, 2], 0, math.pi, NANGLE),
            "length_width": np.column_stack([_centre(b[:, 3], *B.SIZE_RANGE, LEVELS_SIZE),
                                             _centre(b[:, 4], *B.SIZE_RANGE, LEVELS_SIZE)]) * np.array(base, float),
            "colour": _centre(b[:, 5:8], 0, 1, LEVELS_RGB), "alpha": _centre(b[:, 8], *B.ALPHA_RANGE, LEVELS_ALPHA)}


def render_layer(canvas: torch.Tensor, spec: dict, bins: np.ndarray, grid: torch.Tensor) -> torch.Tensor:
    """A layer's snapped strokes painted on the canvas, with the layer's own brush (bristle, edge, opacity range) as
    when fitted; a blending layer smudges instead of painting (eye_crop_styles.blend_composite)."""
    if not len(bins):
        return canvas
    with E.brush_settings(spec), torch.no_grad():
        st = B.Strokes.from_stage(from_bins(bins, (spec["length"], spec["width"])), (spec["length"], spec["width"]))
        st = st.to(canvas.device)
        return E.blend_composite(canvas, st) if spec.get("blend") else B.composite(canvas, st, grid)


def eyes_on_face(face: np.ndarray, bins: list[np.ndarray], ebox: tuple, fbox: np.ndarray) -> np.ndarray:
    """The eye strokes painted straight onto the face canvas (no pixels moved between canvases): every stroke's centre,
    length, width and soft edge scaled from the eye crop to the face canvas, each layer with its own brush."""
    k = float((ebox[2] - ebox[0]) / (fbox[2] - fbox[0]))              # face px per crop px
    shift = (np.array(ebox[:2], float) - fbox[:2]) / (fbox[2] - fbox[0]) * B.SIDE
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.from_numpy(np.asarray(face, np.float32))
    with torch.no_grad():
        for (length, width), b in zip(BRUSHES, bins):
            if not len(b):
                continue
            v = from_bins(b, (length, width))
            v["xy"] = v["xy"] * k + shift
            v["length_width"] = v["length_width"] * k
            v = {key: np.asarray(val, np.float32) for key, val in v.items()}
            base = (float(length * k), float(width * k))
            with E.brush_settings({"edge_px": B.EDGE_PX * k}):
                canvas = B.composite(canvas, B.Strokes.from_stage(v, base), grid)
    return canvas.numpy().clip(0, 1)


def hybrid_base(cat: Cat) -> np.ndarray:
    """The base for the hybrid material (owner, 2026-10-25): outside the plan's eye outlines the whole-face digital
    repaint of the cat (tools/repaint.py; the painting L8 learns from, so the fur's colours are the photo's), inside
    them L8's painting (eye_layers_l.l8_on_crop: the eye the eye model has to paint over). The outlines' inside is
    widened by 2 px and its edge softened (Gaussian 1.5 px) so the two meet without a seam."""
    import repaint as R
    from matplotlib.path import Path as Polygon
    from scipy import ndimage
    rp = R.load(ELL.repaint_file(cat.name))
    fb, eb = cat.fbox, cat.ebox
    k = B.SIDE / (fb[2] - fb[0])
    face = R.render_zoomed(rp["layers"], ((eb[0] - fb[0]) * k, (eb[1] - fb[1]) * k, (eb[2] - eb[0]) * k)).clip(0, 1)
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    pts = np.c_[xx.ravel(), yy.ravel()]
    inside = np.zeros((B.SIDE, B.SIDE), bool)
    for o in cat.outlines:
        inside |= Polygon(o[:, :2]).contains_points(pts).reshape(B.SIDE, B.SIDE)
    w = ndimage.gaussian_filter(ndimage.binary_dilation(inside, iterations=2).astype(np.float32), 1.5)[..., None]
    return (w * np.asarray(cat.base, np.float32) + (1 - w) * face.astype(np.float32)).astype(np.float32)


def around_eyes(centres: np.ndarray):
    """A stroke order: eye by eye (nearest plan eye centre, the plan's order), then by angle around that centre,
    starting from the left (angle pi) and going clockwise on screen (y down)."""
    def order(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        d = np.stack([np.hypot(px - x, py - y) for x, y in centres])
        eye = d.argmin(0)
        c = centres[eye]
        ang = np.mod(np.arctan2(py - c[:, 1], px - c[:, 0]) - math.pi, 2 * math.pi)
        return np.lexsort((ang, eye))
    return order


def inside_out(centres: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A stroke order: eye by eye (nearest plan eye centre, the plan's order), then by distance from that centre
    (nearest first), ties by angle as around_eyes."""
    def order(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        d = np.stack([np.hypot(px - x, py - y) for x, y in centres])
        eye = d.argmin(0)
        c = centres[eye]
        ang = np.mod(np.arctan2(py - c[:, 1], px - c[:, 0]) - math.pi, 2 * math.pi)
        return np.lexsort((ang, d.min(0), eye))
    return order


ORDERS = ("around", "inside_out")


def photo_paths() -> dict[str, str]:
    """Every AFHQ cat photo, stem -> path under data/raw/afhq (all 5,558 stems are unique; the old split file of runs
    D-G lists only 4,085 of them)."""
    return {f.stem: str(f.relative_to(E.AFHQ)) for f in E.AFHQ.glob("*/cat/*.png")}


def eyes_ok() -> set[str]:
    """AFHQ cats whose eyes the owner's review keeps (assignments.csv: eyes == used)."""
    return {f"afhq__{r['id']}" for r in csv.DictReader(open(ASSIGN)) if r["source"] == "afhq" and r["eyes"] == "used"}


class Cat:
    """Everything the fit needs for one cat: its L8 base on the eye crop, the plan's outlines, the photo crop."""

    @classmethod
    def from_pack(cls, name: str, p: dict) -> Cat:
        """A cat from one entry of tools/eye_pack.py's files (no photo, repaint file or L8 base needed)."""
        cat = cls.__new__(cls)
        cat.name = name
        target = torch.from_numpy(p["target"].astype(np.float32) / 255)
        cat.photo = B.Photo(target, torch.from_numpy(p["mask"].astype(np.float32) / 255))
        cat.target = target.numpy()
        cat.eyes, cat.d = p["eyes"].astype(np.float64), float(p["d"])
        cat.ebox, cat.fbox = tuple(p["ebox"]), p["fbox"].astype(float)
        cat.base = p["base"].astype(np.float32) / 255
        cat.outlines = [p["outline_0"].astype(np.float64), p["outline_1"].astype(np.float64)]
        cat.centres = np.array([o[:, :2].mean(0) for o in cat.outlines])
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        cat.disc = np.min([np.hypot(xx - x, yy - y) for x, y in cat.eyes], 0) < 0.35 * cat.d
        return cat

    def __init__(self, name: str, rel_of: dict):
        self.name = name
        self.seq = np.load(BASES / f"{name}.npz")["tokens"]
        rp = np.load(ELL.repaint_file(name))
        self.fbox = rp["box"].astype(float)
        pts = self.fbox[:2] + rp["points"] * (self.fbox[2] - self.fbox[0]) / B.SIDE
        # the eye canvas is placed by the eye outlines in L8's plan (their centres), as it will be when the eye model
        # is used without a photo (2026-10-21; the first 1,000 cats were placed by the photo's eye points, kept in
        # eye_l8_landmarkbox/); the photo's eye points still set the painter's zones (eyes, d)
        k = (self.fbox[2] - self.fbox[0]) / B.SIDE
        centres = np.array([e.mean(0) + 0.5 for e in L.decode_plan(self.seq)["eyes"]]) * k + self.fbox[:2]
        self.ebox = E.eye_box(np.r_[centres, pts[2:]])
        stem = name.split("__", 1)[1]
        self.photo = B.load(str(E.AFHQ / rel_of[stem]), str(E.MASKS / f"{stem}_mask1.png"), self.ebox)
        self.target = self.photo.target.numpy()
        self.eyes = (pts[:2] - np.array(self.ebox[:2])) * (B.SIDE / (self.ebox[2] - self.ebox[0]))
        self.d = float(np.linalg.norm(self.eyes[0] - self.eyes[1]))
        self.base = ELL.l8_on_crop(self.seq, self.fbox, self.ebox)
        self.outlines = ELL.plan_outlines(self.seq, self.fbox, self.ebox)
        self.centres = np.array([o[:, :2].mean(0) for o in self.outlines])
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        self.disc = np.min([np.hypot(xx - x, yy - y) for x, y in self.eyes], 0) < 0.35 * self.d

    def measure(self, img: np.ndarray) -> dict:
        return {"eye_l1": round(float(np.abs(img - self.target)[self.disc].mean()), 4),
                "rim": round(E.rim_strength(img, self.eyes, self.d), 4)}

    def specs(self, ordered: bool, order: str = "around", style: str = "layers") -> list[dict]:
        specs = EL.variants(self.target, self.eyes, self.d, self.outlines)[0][style]
        if not ordered:
            return specs
        around, inside = around_eyes(self.centres), inside_out(self.centres)
        return [{**s, "order": around if order == "around" or k == 0 else inside}   # the first layer is the rim
                for k, s in enumerate(specs)]


def fit(cat: Cat, seed: int = 0, ordered: bool = True, order: str = "around", style: str = "layers",
        dev: str = "cpu") -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """The four layers, ordered around the eyes and snapped layer by layer; E.paint's loop with the snapping inserted
    (same random generator and seeds, so with neither change it is E.paint exactly)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    target, mask = cat.photo.target.to(dev), cat.photo.mask.to(dev)
    ys, xs = torch.meshgrid(torch.arange(B.SIDE, device=dev) + 0.5, torch.arange(B.SIDE, device=dev) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    angles = B.orientation(target.cpu()).to(dev)                     # catbrush builds its filter on the CPU
    canvas = torch.from_numpy(np.asarray(cat.base, np.float32)).to(dev)
    bins, values = [], []
    specs = cat.specs(ordered, order, style)
    for spec in specs:
        with E.brush_settings(spec):                                 # the layer's opacity range also sets its bins
            _, s = E._fit_stage(spec, target, mask, canvas, grid, angles, rng)
            pr = s.params()
            b = to_bins(pr, (spec["length"], spec["width"]))
        v = {"xy": pr.xy.detach().cpu().numpy(), "lw": pr.lw.detach().cpu().numpy(),
             "angle": pr.angle.detach().cpu().numpy(), "colour": pr.colour.detach().cpu().numpy(),
             "alpha": pr.alpha.detach().cpu().numpy()}
        if ordered and order == "inside_out" and spec is not specs[0]:   # re-sort by where the strokes ended up
            perm = inside_out(cat.centres)(b[:, 0] + 0.5, b[:, 1] + 0.5)
            b, v = b[perm], {k: a[perm] for k, a in v.items()}
        bins.append(b)
        values.append(v)
        canvas = render_layer(canvas, spec, b, grid)
    return canvas.cpu().numpy().clip(0, 1), bins, values


def save(cat: Cat, final: np.ndarray, bins: list[np.ndarray], values: list[dict], seconds: float, out: Path,
         style: str = "layers", specs: list[dict] | None = None) -> dict:
    names = layer_names(style)
    row = {"name": cat.name, "style": style, "seconds": round(seconds, 1), "strokes": [len(b) for b in bins],
           "base": cat.measure(cat.base), "painted": cat.measure(final), "photo_rim": round(E.rim_strength(cat.target, cat.eyes, cat.d), 4)}
    arrays = {f"bins_{k}": b for k, b in zip(names, bins)}
    arrays["base_kind"] = np.array(getattr(cat, "base_kind", "l8"))
    if style != "layers":                                           # each layer's brush, to render the bins again
        arrays["style"] = np.array(style)
        arrays["layers"] = np.array(json.dumps([{"name": k, **brush_of(sp)} for k, sp in zip(names, specs)]))
    for k, v in zip(names, values):
        arrays.update({f"val_{k}_{q}": a.astype(np.float32) for q, a in v.items()})
    tmp = out / f"{cat.name}.tmp.npz"
    np.savez_compressed(tmp, ebox=np.array(cat.ebox), fbox=cat.fbox, eyes=cat.eyes, eye_dist=cat.d,
                        plan_0=cat.outlines[0][:, :2].astype(np.float32), plan_1=cat.outlines[1][:, :2].astype(np.float32),
                        base=(cat.base * 255).round().astype(np.uint8), painted=(final * 255).round().astype(np.uint8),
                        info=json.dumps(row), **arrays)
    tmp.rename(out / f"{cat.name}.npz")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="*", default=None, help="these cats only (default: every cat with a base)")
    ap.add_argument("--part", default="0/1", help="i/n: the i-th of n interleaved parts of the cats")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--compare", action="store_true",
                    help="also paint each cat with the upper-bound painter (no order, no bins) and with bins but no order")
    ap.add_argument("--max-cats", type=int, default=0)
    ap.add_argument("--order", choices=ORDERS, default="around", help="each layer's stroke order (inside_out: run E3)")
    ap.add_argument("--style", default="layers", help="layers, digital or one of eye_layers.LITE")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pack", type=Path, default=None, help="tools/eye_pack.py's folder: the cats come from it")
    ap.add_argument("--reverse", action="store_true", help="from the last cat backwards (a second machine)")
    ap.add_argument("--stop-after-min", type=float, default=0.0, help="stop cleanly after this many minutes")
    ap.add_argument("--base", choices=("l8", "hybrid"), default="l8",
                    help="hybrid: the whole-face repaint outside the eye outlines, L8 inside (hybrid_base)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.style not in ("layers", "digital", "digital_rim", "digital_rim_long") and a.style not in EL.LITE:
        raise SystemExit(f"unknown style {a.style}")
    packed: dict[str, tuple[Path, int]] = {}
    if a.pack:                                                     # name -> (pack file, row); loaded lazily
        for f in sorted(a.pack.glob("pack_*.npz")):
            if not f.name.endswith(".tmp.npz"):
                packed.update({str(x): (f, j) for j, x in enumerate(np.load(f)["names"])})
        names = a.names if a.names else sorted(packed)
        keep = set(packed)
        rel_of = {}
    else:
        rel_of = photo_paths()
        keep = eyes_ok()
        names = a.names if a.names else sorted(f.name[:-4] for f in BASES.glob("afhq__*.npz") if not f.name.endswith(".tmp.npz"))
    i, n = map(int, a.part.split("/"))
    names = [x for j, x in enumerate(names) if j % n == i and x in keep]
    if a.reverse:
        names = names[::-1]
    cache: dict[Path, dict] = {}

    def load_cat(name: str) -> Cat:
        if not a.pack:
            return Cat(name, rel_of)
        f, j = packed[name]
        if f not in cache:
            cache.clear()
            z = np.load(f)
            cache[f] = {k: z[k] for k in z.files}
        return Cat.from_pack(name, {k: v[j] for k, v in cache[f].items() if k != "names"})
    t_start = time.time()
    done = 0
    log = open(a.out / f"progress_{i}of{n}.jsonl", "a")
    for name in names:
        if (a.out / f"{name}.npz").exists():
            continue
        if a.stop_after_min and time.time() - t_start > 60 * a.stop_after_min:
            print(json.dumps({"stopped": "time", "painted": done}), flush=True)
            return
        t0 = time.time()
        cat = load_cat(name)
        if a.base == "hybrid":
            cat.base, cat.base_kind = hybrid_base(cat), "hybrid"
        final, bins, values = fit(cat, order=a.order, style=a.style, dev=a.device)
        row = save(cat, final, bins, values, time.time() - t0, a.out, a.style, cat.specs(True, a.order, a.style))
        if a.compare:                                        # the upper-bound painter, and bins without the order
            t1 = time.time()
            ub = E.paint(cat.photo.target, cat.photo.mask, cat.specs(ordered=False), start=cat.base)[-1]
            row["upper_bound"] = {**cat.measure(ub), "seconds": round(time.time() - t1, 1)}
            bo = fit(cat, ordered=False)[0]
            row["bins_only"] = cat.measure(bo)
            np.savez_compressed(a.out / f"{name}.compare.npz", upper_bound=(ub * 255).round().astype(np.uint8),
                                bins_only=(bo * 255).round().astype(np.uint8),
                                photo=(cat.target * 255).round().astype(np.uint8))
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        done += 1
        if a.max_cats and done >= a.max_cats:
            print(json.dumps({"stopped": "max_cats", "painted": done}), flush=True)
            return
    print(json.dumps({"done": len(names)}), flush=True)


if __name__ == "__main__":
    main()
