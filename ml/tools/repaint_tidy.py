"""Turn a fixed-length digital repaint (tools/repaint.py, 1,848 strokes) into a variable-length one (owner, 2026-10-19):
each layer uses only the strokes it needs, and the detail layers finish one region at a time.

  * Regions (face frame: origin the eyes' midpoint, x along the eyes, y towards the mouth, unit the eye distance):
    ears (each ear's triangle), forehead (above the eye line), left cheek, right cheek (|x| > CHEEK_X below the eye
    line), chin and mouth (the rest below the eye line). Detail layers (fur, dark hairs, light hairs) are laid in
    REGION_ORDER; the block-in layers (washes, blocks, small blocks, blending) stay whole-face.
  * Fine strokes whose centre is uncertain (tri 2) are dropped (section 15.6); big ones stay (their loss is masked).
  * "Enough": within each paint layer and region, strokes are ranked by their contribution (opacity x area x how far
    their colour is from the canvas under them) and kept until they hold the fraction --q of the region's total
    ("share"), or kept when their own contribution is at least --tau ("abs"; px x colour, the same measure as the
    pruning test of 2026-10-19). The share rule keeps about the same number of strokes for every cat (6 cats at
    q 0.85: 1,083-1,166), so it does not say when a plain cat is done sooner than a tabby; the absolute rule does.
    Blending layers are kept whole in this first version.
  * Hair layers' own "enough" (--gain, owner 2026-10-19): the area-based contribution favours wide strokes, so at
    tau 1 most hairs were dropped (dark hairs 52-92 of 500). With --gain, the fur layer is judged by the photo's
    texture (|luminance - its 2 px blur|), the dark and light hairs by the photo's thin dark and light lines (the maps
    their strokes were started on), each map less LINE_FLOOR and only EDGE_KEEP px or more inside the cat. In each
    region, strokes are taken greedily by how much not-yet-covered line mass their footprint (the stroke dilated by
    1 px) lies on, as long as that is at least --gain. A share-of-the-lines target does not work: all 500 strokes of
    a hair layer together cover only 44-77% of the line mass (6 cats), so any target above that keeps every stroke.
    With an absolute gain, a cat with few fine lines needs few hairs (gain 1: dark hairs 32 for a white cat, 211 for
    a calico).
  * Order: in each detail layer, region by region; within a region (and in block-in layers) lowest contrast first.
  * Refit: the kept strokes, in the new order, are fitted a little longer (REFIT_ITERS per layer) to the photo with
    the painter's own loss, because dropping strokes without refitting raised the L1 by about 0.006 (2026-10-19).

    PYTHONPATH=.:tools python tools/repaint_tidy.py --files data/processed/repaint/mac/shard_010 --n 6 --q 0.93 --out runs/repaint_tidy/q93
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from scipy.ndimage import binary_erosion, gaussian_filter

import eye_crop_styles as E
import ear_strands as S
import repaint as R
from digital_face import EDGE_KEEP, PROTECT
from sellanything_ml.sketch import catbrush as B

DETAIL = {5, 6, 7}                     # fur, dark hairs, light hairs (digital_face.LAYERS)
REGION_ORDER = ("ears", "forehead", "left_cheek", "right_cheek", "chin_mouth")
CHEEK_X = 0.35
FINE_LW = 3.0                          # a stroke this wide or narrower is "fine" (dropped where uncertain)
REFIT_ITERS = 40
LINE_FLOOR = 0.02                      # catches dark_lines' floor (eye_crop_styles.CONTOUR_FLOOR)


def face_frame(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    d = float(np.linalg.norm(points[0] - points[1]))
    mid = points[:2].mean(0)
    ex = (points[1] - points[0]) / d
    ey = np.array([-ex[1], ex[0]])
    if np.dot(points[2] - mid, ey) < 0:
        ey = -ey
    rel = xy - mid
    return np.stack([rel @ ex, rel @ ey], -1) / d


def region_of(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Region name for each point xy (n, 2) on the canvas."""
    q = face_frame(points, xy)
    lab = np.where(q[:, 1] < 0, "forehead", "chin_mouth").astype(object)
    lab[(q[:, 1] >= 0) & (q[:, 0] < -CHEEK_X)] = "left_cheek"
    lab[(q[:, 1] >= 0) & (q[:, 0] > CHEEK_X)] = "right_cheek"
    for tri in ((3, 4, 5), (6, 7, 8)):
        lab[Polygon(points[list(tri)]).contains_points(xy)] = "ears"
    return lab


def contribution(layer: dict, canvas: np.ndarray) -> np.ndarray:
    xy = layer["xy_px"]
    ix, iy = np.clip(xy[:, 0].astype(int), 0, B.SIDE - 1), np.clip(xy[:, 1].astype(int), 0, B.SIDE - 1)
    return layer["alpha_val"] * layer["lw_px"][:, 0] * layer["lw_px"][:, 1] * np.abs(layer["colour_rgb"] - canvas[iy, ix]).mean(1)


def keep_enough(c: np.ndarray, q: float, tau: float | None = None) -> np.ndarray:
    """Indices of the strokes, by decreasing contribution, that hold the fraction q of the total (or, with tau, whose
    contribution is at least tau)."""
    if tau is not None:
        order = np.argsort(-c)
        return order[c[order] >= tau]
    if len(c) == 0 or c.sum() <= 0:
        return np.zeros(0, int)
    order = np.argsort(-c)
    n = int(np.searchsorted(np.cumsum(c[order]) / c.sum(), q) + 1)
    return order[:min(n, len(c))]


def line_maps(target: np.ndarray, tri: np.ndarray) -> dict[int, np.ndarray]:
    """Per hair layer, the photo's map the layer is judged by (see the module notes)."""
    lum = E.luminance(target.astype(np.float32) / 255)
    inner = binary_erosion(tri == 1, iterations=EDGE_KEEP)
    maps = {5: np.abs(lum - gaussian_filter(lum, 2)), 6: E.dark_lines(lum),
            7: np.maximum(S.bright_lines(lum, 1), S.bright_lines(lum, 2))}
    return {k: np.clip(m - LINE_FLOOR, 0, None) * inner for k, m in maps.items()}


def footprints(layer: dict) -> list[np.ndarray]:
    """Flat pixel indices under each stroke (its length x width, dilated by 1 px)."""
    out = []
    for (x, y), (ln, w), a in zip(layer["xy_px"], layer["lw_px"], layer["angle_rad"]):
        u = np.arange(-ln / 2, ln / 2 + 0.01, 0.5)
        v = np.arange(-w / 2 - 1, w / 2 + 1.01, 0.5)
        uu, vv = np.meshgrid(u, v)
        px = (x + uu * np.cos(a) - vv * np.sin(a)).astype(int).ravel()
        py = (y + uu * np.sin(a) + vv * np.cos(a)).astype(int).ravel()
        ok = (px >= 0) & (px < B.SIDE) & (py >= 0) & (py < B.SIDE)
        out.append(np.unique(py[ok] * B.SIDE + px[ok]))
    return out


def cover_enough(members: np.ndarray, feet: list[np.ndarray], mass: np.ndarray, gain: float) -> np.ndarray:
    """Greedy: the stroke on the most not-yet-covered line mass first, while that mass is at least `gain`."""
    left = mass.copy()
    keep, cand = [], list(members)
    while cand:
        gains = np.array([left[feet[i]].sum() for i in cand])
        j = int(np.argmax(gains))
        if gains[j] < gain:
            break
        i = cand.pop(j)
        left[feet[i]] = 0
        keep.append(i)
    return np.array(keep, int)


def refit(canvas: torch.Tensor, s: B.Strokes, br: dict, goal: torch.Tensor, weight: torch.Tensor,
          protect: torch.Tensor, grid: torch.Tensor, iters: int) -> torch.Tensor:
    blend = br["blend"]
    draw = (lambda c, st: E.blend_composite(c, st)) if blend else (lambda c, st: B.composite(c, st, grid))
    params = [s.xy, s.size, s.angle, s.alpha] + ([] if blend else [s.colour])
    for t in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=0.02)
    for _ in range(iters):
        painted = draw(canvas, s)
        loss = ((painted - goal).abs() * weight[..., None]).mean() + PROTECT * ((painted - canvas).abs().mean(-1) * protect).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return draw(canvas, s)


def tidy(cat: dict, q: float, iters: int = REFIT_ITERS, tau: float | None = None, gain: float | None = None) -> dict:
    layers, points, tri = cat["layers"], cat["points"], cat["tri"]
    goal = torch.from_numpy(cat["target"].astype(np.float32) / 255)
    weight = torch.from_numpy((tri != 2).astype(np.float32))
    protect = torch.from_numpy((tri == 0).astype(np.float32))
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(B.SIDE, B.SIDE, 3)
    out_layers, counts = [], []
    maps = line_maps(cat["target"], tri) if gain is not None else {}
    for k, layer in enumerate(layers):
        br = layer["brush"]
        n_all = len(layer["xy_px"])
        if br["blend"]:
            idx, regions = np.arange(n_all), None
        else:
            xy = layer["xy_px"]
            ix, iy = np.clip(xy[:, 0].astype(int), 0, B.SIDE - 1), np.clip(xy[:, 1].astype(int), 0, B.SIDE - 1)
            fine = layer["lw_px"][:, 1] <= FINE_LW
            usable = ~(fine & (tri[iy, ix] == 2))
            c = contribution(layer, canvas.numpy())
            reg = region_of(points, xy)
            if k in DETAIL:
                idx, regions = [], {}
                feet = footprints(layer) if k in maps else None
                for r in REGION_ORDER:
                    members = np.flatnonzero(usable & (reg == r))
                    if feet is not None:
                        keep = cover_enough(members, feet, maps[k].ravel(), gain)
                    else:
                        keep = members[keep_enough(c[members], q, tau)]
                    keep = keep[np.argsort(c[keep])]                     # lowest contrast first within the region
                    regions[r] = int(len(keep))
                    idx.extend(keep.tolist())
                idx = np.array(idx, int)
            else:
                members = np.flatnonzero(usable)
                keep = members[keep_enough(c[members], q, tau)]
                idx, regions = keep[np.argsort(c[keep])], None
        s = B.Strokes(base=(br["length"], br["width"]), **{p: torch.as_tensor(layer[p][idx]).clone() for p in R.PARAMS})
        with E.brush_settings({"edge_px": br["edge_px"], "bristle": br["bristle"], "alpha_range": tuple(br["alpha_range"])}):
            canvas = refit(canvas, s, br, goal, weight, protect, grid, iters) if len(idx) else canvas
            pr = s.params()
        new = {p: getattr(s, p).detach().numpy() for p in R.PARAMS}
        new.update({"xy_px": pr.xy.detach().numpy(), "lw_px": pr.lw.detach().numpy(), "angle_rad": pr.angle.detach().numpy(),
                    "colour_rgb": pr.colour.detach().numpy(), "alpha_val": pr.alpha.detach().numpy(), "brush": br,
                    "regions": regions})
        out_layers.append(new)
        counts.append({"layer": k, "kept": int(len(idx)), "of": n_all, "regions": regions})
    painting = canvas.numpy()
    cat_px = tri == 1
    return {"layers": out_layers, "painting": painting, "counts": counts,
            "l1_cat": float(np.abs(painting - cat["target"].astype(np.float32) / 255)[cat_px].mean()),
            "strokes": int(sum(c["kept"] for c in counts))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=Path, required=True, help="a folder of repaint .npz files")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--q", type=float, default=0.93)
    ap.add_argument("--tau", type=float, default=None, help="absolute rule: keep strokes contributing at least this")
    ap.add_argument("--gain", type=float, default=None, help="hair layers: least new line mass a kept stroke covers")
    ap.add_argument("--iters", type=int, default=REFIT_ITERS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(a.files.glob("*.npz"))[:a.n]:
        cat = R.load(f)
        t0 = time.time()
        res = tidy(cat, a.q, a.iters, a.tau, a.gain)
        full_l1 = float(np.abs(cat["painting"].astype(np.float32) / 255 - cat["target"].astype(np.float32) / 255)[cat["tri"] == 1].mean())
        row = {"file": f.name, "q": a.q, "tau": a.tau, "gain": a.gain, "strokes": res["strokes"], "l1_cat": round(res["l1_cat"], 4),
               "full_strokes": 1848, "full_l1_cat": round(full_l1, 4), "seconds": round(time.time() - t0, 1),
               "counts": res["counts"]}
        np.savez_compressed(a.out / f.name, painting=(res["painting"].clip(0, 1) * 255).round().astype(np.uint8),
                            target=cat["target"], tri=cat["tri"], points=cat["points"])
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("file", "strokes", "l1_cat", "full_l1_cat", "seconds")}), flush=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
