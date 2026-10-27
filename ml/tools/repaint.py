"""The digital repaint (owner, 2026-10-18): every cat of tools/repaint_prep.py's shards painted once with the
whole-face digital painter (tools/digital_face.py's eight layers: smooth brushes, soft edges, blending, the finer and
higher-contrast the later), recording everything later models need, so no model variant needs another repaint:

  * one model painting everything: all eight layers' strokes, in order (replay(..) reproduces the painting);
  * a whole-face model plus part models: each stroke's layer is stored, and its region (eyes, ears, nose and mouth,
    fur) follows from the stored face points (regions(..), stroke_regions(..)); strokes are vectors, so they can be
    re-rendered on a zoomed part canvas (render_zoomed(..)), on which the part models paint;
  * learning nothing where the data says nothing (section 15.6): the painter fits only cat and background pixels
    (tri 1 and 0); uncertain pixels (tri 2: the chest under the chin, the back behind the ears, beyond the photo) have
    weight 0, no stroke starts there, and the tri map is stored for the masked loss;
  * appearance conditions (section 13): source and id lead back to the photo.

Background pixels are protected (changing them costs PROTECT times the change: they stay white paper). Each cat is
saved as <out>/<shard>/<source>__<id>.npz; cats already saved are skipped, so a run can stop and resume (Kaggle
sessions end after 12 hours). One progress line per cat goes to <out>/<shard>.jsonl.

    PYTHONPATH=.:tools python tools/repaint.py --shards 0-3 --device mps --out data/processed/repaint/mac
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from scipy.ndimage import binary_erosion

import ear_strands as S
import eye_crop_styles as E
from digital_face import EDGE_KEEP, LAYERS, PROTECT
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
INPUTS = ML / "data" / "processed" / "repaint" / "inputs"
VERSION = "repaint-digital-v1 (2026-10-18)"
PARAMS = ("xy", "size", "angle", "colour", "alpha")


def layer_specs(target: np.ndarray, tri: np.ndarray) -> list[dict]:
    """digital_face's eight layers with the tri map: weight 1 on cat and background, 0 on uncertain; background
    protected; strokes start on the cat; hairs start on the photo's fine lines at least EDGE_KEEP px inside the cat."""
    cat = tri == 1
    lum = E.luminance(target)
    inner = binary_erosion(cat, iterations=EDGE_KEEP)
    dark = E.dark_lines(lum) * inner
    light = np.maximum(S.bright_lines(lum, 1), S.bright_lines(lum, 2)) * inner
    weight = torch.from_numpy((tri != 2).astype(np.float32))
    protect = (torch.from_numpy((tri == 0).astype(np.float32)), PROTECT)
    out = []
    for _name, length, width, n, edge, kind in LAYERS:
        spec = {"length": length, "width": width, "n": n, "edge_px": edge, "bristle": 0.0, "alpha_range": (0.2, 1.0),
                "weight": weight, "protect": protect, "order_by_contrast": kind != "blend"}
        if kind == "blend":
            spec["blend"] = True
        if kind == "dark":
            spec["place"] = dark ** 2 + 1e-6 * cat
        if kind == "light":
            spec["place"] = light ** 2 + 1e-6 * cat
        out.append(spec)
    return out


def brush_of(spec: dict) -> dict:
    """The settings that define how a layer's strokes are drawn (stored with them)."""
    return {"length": spec["length"], "width": spec["width"], "edge_px": spec["edge_px"], "bristle": spec["bristle"],
            "alpha_range": list(spec["alpha_range"]), "blend": bool(spec.get("blend", False))}


def replay(layers: list[dict], start: torch.Tensor | None = None, only: list[int] | None = None) -> np.ndarray:
    """Render stored layers (all, or the indices in `only`) on white paper or `start`. Blending strokes of a layer act
    together, as when they were fitted, so the replay of every layer reproduces the stored painting."""
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(B.SIDE, B.SIDE, 3) if start is None else start.clone()
    with torch.no_grad():
        for k, layer in enumerate(layers):
            if only is not None and k not in only:
                continue
            br = layer["brush"]
            s = B.Strokes(base=(br["length"], br["width"]), **{p: torch.as_tensor(layer[p]) for p in PARAMS})
            with E.brush_settings({"edge_px": br["edge_px"], "bristle": br["bristle"], "alpha_range": tuple(br["alpha_range"])}):
                canvas = E.blend_composite(canvas, s) if br["blend"] else B.composite(canvas, s, grid)
    return canvas.numpy()


def regions(points: np.ndarray, side: int = B.SIDE) -> dict[str, np.ndarray]:
    """Part regions on the canvas from the nine face points (CAT order): eyes (discs of 0.35 eye distances), ears
    (each ear's triangle), nose and mouth (a disc of 0.45 eye distances around the point 0.6 of the way from the
    eyes' midpoint to the mouth); everything else is fur. The same rules as the part crops of tools/review_export.py
    and the eye zone of brushgpt."""
    d = float(np.linalg.norm(points[0] - points[1]))
    yy, xx = np.mgrid[:side, :side] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]
    eyes = np.min([np.hypot(xx - x, yy - y) for x, y in points[:2]], 0) < 0.35 * d
    ears = np.zeros((side, side), bool)
    for tri in ((3, 4, 5), (6, 7, 8)):
        ears |= Polygon(points[list(tri)]).contains_points(q).reshape(side, side)
    nm = points[:2].mean(0) + 0.6 * (points[2] - points[:2].mean(0))
    nose_mouth = np.hypot(xx - nm[0], yy - nm[1]) < 0.45 * d
    return {"eyes": eyes, "ears": ears & ~eyes, "nose_mouth": nose_mouth & ~eyes & ~ears}


def stroke_regions(layers: list[dict], points: np.ndarray) -> list[np.ndarray]:
    """Per layer, each stroke's region by its centre: 'eyes', 'ears', 'nose_mouth' or 'fur'."""
    reg = regions(points)
    out = []
    for layer in layers:
        xy = layer["xy_px"]
        ix = np.clip(xy[:, 0].astype(int), 0, B.SIDE - 1)
        iy = np.clip(xy[:, 1].astype(int), 0, B.SIDE - 1)
        lab = np.full(len(xy), "fur", dtype=object)
        for name in ("nose_mouth", "ears", "eyes"):
            lab[reg[name][iy, ix]] = name
        out.append(lab)
    return out


def render_zoomed(layers: list[dict], box: tuple[float, float, float], only: list[int] | None = None) -> np.ndarray:
    """The stored strokes re-rendered on a zoomed canvas: box = (x, y, side) of a square on the 256 canvas, drawn at
    256 px (the part models' canvases, e.g. an eye crop). Strokes are vectors: centre and size scale, angle stays."""
    x0, y0, side = box
    k = B.SIDE / side
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(B.SIDE, B.SIDE, 3)
    with torch.no_grad():
        for i, layer in enumerate(layers):
            if only is not None and i not in only:
                continue
            br = layer["brush"]
            p = B.StrokeParams(torch.as_tensor((layer["xy_px"] - np.array([x0, y0])) * k, dtype=torch.float32),
                               torch.as_tensor(layer["lw_px"] * k, dtype=torch.float32),
                               torch.as_tensor(layer["angle_rad"]), torch.as_tensor(layer["colour_rgb"]),
                               torch.as_tensor(layer["alpha_val"]))
            with E.brush_settings({"edge_px": br["edge_px"] * k, "bristle": br["bristle"], "alpha_range": tuple(br["alpha_range"])}):
                if br["blend"]:
                    continue                      # blending needs the canvas at its own scale; parts repaint their own
                canvas = B.composite_dense(canvas, p, grid)
    return canvas.numpy()


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    n = int(z["layers"])
    layers = []
    for k in range(n):
        layer = {p: z[f"l{k}_{p}"] for p in PARAMS}
        layer.update({q: z[f"l{k}_{q}"] for q in ("xy_px", "lw_px", "angle_rad", "colour_rgb", "alpha_val")})
        layer["brush"] = json.loads(str(z[f"l{k}_brush"]))
        layers.append(layer)
    return {"layers": layers, **{k: z[k] for k in ("target", "tri", "points", "box", "painting")},
            "meta": json.loads(str(z["meta"]))}


def paint_one(target_u8: np.ndarray, tri: np.ndarray, dev: str) -> tuple[list, np.ndarray]:
    target = torch.from_numpy(target_u8.astype(np.float32) / 255)
    cat = torch.from_numpy((tri == 1).astype(np.float32))
    record = []
    final = E.paint(target, cat, layer_specs(target.numpy(), tri), record=record, dev=dev)[-1]
    return record, final


def save(path: Path, record: list, final: np.ndarray, item: dict, meta: dict) -> None:
    arrays = {"layers": np.array(len(record)), "painting": (final.clip(0, 1) * 255).round().astype(np.uint8),
              "target": item["target"], "tri": item["tri"], "points": item["points"], "box": item["box"],
              "meta": np.array(json.dumps(meta))}
    for k, (spec, s) in enumerate(record):
        with E.brush_settings(spec):
            pr = s.params()
        for p in PARAMS:
            arrays[f"l{k}_{p}"] = getattr(s, p).detach().cpu().numpy()
        arrays[f"l{k}_xy_px"] = pr.xy.detach().cpu().numpy()
        arrays[f"l{k}_lw_px"] = pr.lw.detach().cpu().numpy()
        arrays[f"l{k}_angle_rad"] = pr.angle.detach().cpu().numpy()
        arrays[f"l{k}_colour_rgb"] = pr.colour.detach().cpu().numpy()
        arrays[f"l{k}_alpha_val"] = pr.alpha.detach().cpu().numpy()
        arrays[f"l{k}_brush"] = np.array(json.dumps(brush_of(spec)))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)                                  # a cat is saved whole or not at all


def rss_mb() -> int:
    """This process's resident memory in MB (to watch for memory that is not given back)."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        return -1


def parse_shards(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", required=True, help="e.g. 0-9 or 3,5,7")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-after-min", type=float, default=0, help="stop cleanly after this many minutes (0: never)")
    ap.add_argument("--reverse", action="store_true", help="go through the shards from the last")
    ap.add_argument("--limit", type=int, default=0, help="at most N cats per shard (a test)")
    ap.add_argument("--source", default="", help="only this source (afhq or cat)")
    ap.add_argument("--max-cats", type=int, default=0,
                    help="exit after painting this many cats (0: never); a wrapper restarts the process, so memory that a "
                         "long run fails to give back cannot pile up (on the Mac it grew to 7 GB and 28 GB of swap after "
                         "about 100 cats, and a cat took 1,500 s instead of 35)")
    ap.add_argument("--reverse-cats", action="store_true",
                    help="go through each shard's cats from the last (two machines meeting in the middle)")
    ap.add_argument("--only", type=Path, default=None,
                    help="a text file of source__id names (one per line): paint only these (a session that picks up the "
                         "cats other machines have not painted)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    only = set(a.only.read_text().split()) if a.only else None
    t_start = time.time()
    painted = 0
    shards = parse_shards(a.shards)
    for sh in (reversed(shards) if a.reverse else shards):
        z = np.load(a.inputs / f"shard_{sh:03d}.npz")
        d = a.out / f"shard_{sh:03d}"
        d.mkdir(parents=True, exist_ok=True)
        n = len(z["id"]) if not a.limit else min(a.limit, len(z["id"]))
        for i in (reversed(range(n)) if a.reverse_cats else range(n)):
            if a.source and str(z["source"][i]) != a.source:
                continue
            if only is not None and f"{z['source'][i]}__{z['id'][i]}" not in only:
                continue
            if a.stop_after_min and time.time() - t_start > 60 * a.stop_after_min:
                print(json.dumps({"stopped": "time", "minutes": round((time.time() - t_start) / 60, 1)}), flush=True)
                return
            item = {k: z[k][i] for k in ("source", "id", "target", "tri", "points", "box")}
            path = d / f"{item['source']}__{item['id']}.npz"
            if path.exists():
                continue
            t0 = time.time()
            record, final = paint_one(item["target"], item["tri"], a.device)
            tg = item["target"].astype(np.float32) / 255
            cat = item["tri"] == 1
            meta = {"version": VERSION, "source": str(item["source"]), "id": str(item["id"]), "device": a.device,
                    "seconds": round(time.time() - t0, 1), "l1_cat": round(float(np.abs(final - tg)[cat].mean()), 4),
                    "strokes": int(sum(len(s.xy) for _, s in record)), "rss_mb": rss_mb()}
            save(path, record, final, item, meta)
            with open(a.out / f"shard_{sh:03d}.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            del record, final
            gc.collect()
            if a.device == "mps":
                torch.mps.empty_cache()
            elif a.device.startswith("cuda"):
                torch.cuda.empty_cache()
            painted += 1
            if a.max_cats and painted >= a.max_cats:
                print(json.dumps({"stopped": "max_cats", "painted": painted}), flush=True)
                return
    print(json.dumps({"done": shards, "minutes": round((time.time() - t_start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
