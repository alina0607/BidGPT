"""The hybrid eye material's base, L8 kept further inside (owner, 2026-10-26). eye_data_l8.hybrid_base keeps L8's
painting inside the plan's eye outlines widened by 2 px (then softened), but the eye layers may only paint inside the
outlines themselves, so a ring of L8's colours stayed in the material where no stroke could cover it (median 90 px a
cat, 76% of it outside the outlines; tools/eye_material_flaws.py). Here L8 is kept inside the outlines narrowed by
ERODE px (softened the same way) and the whole-face repaint is everywhere else; the stored strokes are drawn again on
that base with each layer's brush (tools/eye_render.py; drawn on the old base they give the stored painting within
1/255). The strokes are not fitted again: outside the outlines they could not paint before either.

Each file of --dir whose base_kind is "hybrid" gets the new base, painting, base_kind "hybrid_in3" and its measures
in info; written next to it first and renamed, so a stopped run leaves every file whole. The photo crop, L8 base and
outlines come from tools/eye_pack.py's files, the whole-face repaint from tools/repaint.py's files.

    PYTHONPATH=.:tools python tools/eye_rebase.py --dir data/processed/brush_l/eye_l8_rim_hybrid --max-cats 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import eye_crop_styles as E
import eye_layers as EL
import eye_layers_l as ELL
import eye_material_compare as C
import eye_render as ER
import repaint as R
from sellanything_ml.sketch import catbrush as B

ERODE, KIND = 3, "hybrid_in3"


def new_base(p: dict, name: str) -> np.ndarray:
    """L8 inside the outlines narrowed by ERODE px, the whole-face repaint outside (0-1, (256, 256, 3))."""
    fb, eb = p["fbox"].astype(float), p["ebox"]
    k = B.SIDE / (fb[2] - fb[0])
    face = R.render_zoomed(R.load(ELL.repaint_file(name))["layers"],
                           ((eb[0] - fb[0]) * k, (eb[1] - fb[1]) * k, (eb[2] - eb[0]) * k)).clip(0, 1).astype(np.float32)
    inside = EL.interior([p["outline_0"].astype(np.float64), p["outline_1"].astype(np.float64)])
    w = ndimage.gaussian_filter(ndimage.binary_erosion(inside, iterations=ERODE).astype(np.float32), 1.5)[..., None]
    return w * (p["base"].astype(np.float32) / 255) + (1 - w) * face


def measures(img: np.ndarray, target: np.ndarray, eyes: np.ndarray, d: float) -> dict:
    """eye_data_l8.Cat.measure: L1 on the eye discs and the rim strength."""
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    disc = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < 0.35 * d
    return {"eye_l1": round(float(np.abs(img - target)[disc].mean()), 4), "rim": round(E.rim_strength(img, eyes, d), 4)}


def rebase(path: Path, p: dict) -> dict:
    """One material file on the new base; its new measures."""
    z = dict(np.load(path))
    layers = json.loads(str(z["layers"]))
    base = new_base(p, path.stem)
    painted = ER.layers_on_crop(base, [z[f"bins_{layer['name']}"] for layer in layers], layers)
    target = p["target"].astype(np.float32) / 255
    info = json.loads(str(z["info"]))
    info.update(base=measures(base, target, z["eyes"], float(z["eye_dist"])),
                painted=measures(painted, target, z["eyes"], float(z["eye_dist"])), rebase={"erode": ERODE})
    z.update(base=(base * 255).round().astype(np.uint8), painted=(painted * 255).round().astype(np.uint8),
             base_kind=np.array(KIND), info=np.array(json.dumps(info)))
    tmp = path.with_name(f"{path.stem}.rebase{os.getpid()}.tmp.npz")   # two processes may meet on one cat
    np.savez_compressed(tmp, **z)
    tmp.rename(path)
    return info["painted"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--max-cats", type=int, default=0, help="stop after this many (a fresh process each batch)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reverse", action="store_true", help="from the last cat backwards (a second process from the other end)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    todo = sorted(f for f in a.dir.glob("afhq__*.npz") if not f.name.endswith(".tmp.npz")
                  and str(np.load(f)["base_kind"]) == "hybrid")
    if not todo:
        print(json.dumps({"done": True, "left": 0}), flush=True)
        return
    idx = C.pack_index()
    if a.reverse:
        todo = todo[::-1]
    by_pack: dict[Path, list[tuple[Path, int]]] = {}
    for f in todo:
        pf, j = idx[f.stem]
        by_pack.setdefault(pf, []).append((f, j))
    done = 0
    for pf, items in sorted(by_pack.items(), reverse=a.reverse):
        z = np.load(pf)
        packed = {k: z[k] for k in z.files if k != "names"}
        for f, j in items:
            if str(np.load(f)["base_kind"]) != "hybrid":                 # done meanwhile by the other process
                continue
            rebase(f, {k: v[j] for k, v in packed.items()})
            done += 1
            if a.max_cats and done >= a.max_cats:
                print(json.dumps({"stopped": "max_cats", "rebased": done, "left": len(todo) - done}), flush=True)
                return
    print(json.dumps({"done": True, "rebased": done, "left": 0}), flush=True)


if __name__ == "__main__":
    main()
