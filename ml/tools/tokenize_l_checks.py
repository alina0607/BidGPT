"""The measurements behind Brush GPT L's tokenizer (sellanything_ml/sketch/brushtokens_l.py), rerun and saved for the
zh notebook, section 17.9 (2026-10-19). On repainted AFHQ cats of shard 0 whose sketch plan is ready, CPU:

  fields      binning one field at a time (the others exact), first five layers, 10 cats: which field costs most
  levels      colour x opacity levels (12x4 ... 20x8), first five layers, 10 cats
  refit       binning each layer and then refitting the next one on top (tools/prepare_l.py), 6 cats, 12x4 and 12x8
  eight       all eight layers: unbinned, binned without refit, and the first five binned+refit with the hair layers
              binned without refit (8 cats), and the L8 sequence lengths
  sequences   the prepared L5 sequences (data/processed/brush_l/cats): lengths, grammar check, an example

    PYTHONPATH=.:tools python tools/tokenize_l_checks.py --out runs/brush_l_tokenize
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import repaint as R
import repaint_tidy as RT
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
REPAINT = ML / "data/processed/repaint/kaggle_t4_run1/out/shard_000"
PLANS = ML / "data/processed/sketch_plan/afhq/shard_000"
YS, XS = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
GRID = torch.stack([XS, YS], -1)


def cats(n: int) -> list[tuple]:
    out = []
    for p in sorted(PLANS.glob("*.npz"))[:n]:
        out.append((p.stem, *L.load_cat(REPAINT / p.name, p)))
    return out


def l1(img: np.ndarray, ex: dict) -> float:
    return float(np.abs(img - ex["target"] / 255)[ex["tri"] == 1].mean())


def render_values(vals: list[dict]) -> np.ndarray:
    c = torch.ones(B.SIDE, B.SIDE, 3)
    with torch.no_grad():
        for k, v in enumerate(vals):
            br = L.brush(k)
            with E.brush_settings(br):
                s = B.Strokes.from_stage(v, (br["length"], br["width"]))
                c = E.blend_composite(c, s) if br["blend"] else B.composite(c, s, GRID)
    return c.numpy()


def exact(lay: dict) -> dict:
    """Copies: Strokes.from_stage shares the angle array, and a refit would change the stored layer in place (it did
    once, 2026-10-19: later measurements on the same cats then used angles fitted for an earlier setting)."""
    return {"xy": lay["xy_px"].copy(), "angle": lay["angle_rad"].copy(), "length_width": lay["lw_px"].copy(),
            "colour": lay["colour_rgb"].copy(), "alpha": lay["alpha_val"].copy()}


def set_levels(rgb: int, alpha: int) -> None:
    L.LEVELS_RGB, L.LEVELS_ALPHA = rgb, alpha
    L.FIELDS["paint"] = rgb ** 3 * alpha
    L.OFFSET.update(L._offsets())


def fields_check() -> dict:
    out = {f: [] for f in ("none", "xy", "angle", "length_width", "colour", "alpha", "all")}
    for _, _, layers, ex in cats(10):
        ex_vals = [exact(layers[k]) for k in range(5)]
        binned = [L.stroke_values(L.stroke_tokens(layers[k], k), k) for k in range(5)]
        for f in out:
            vals = []
            for k in range(5):
                v = dict(binned[k]) if f == "all" else dict(ex_vals[k])
                if f not in ("none", "all"):
                    v[f] = binned[k][f]
                if L.BRUSHES[k][3]:
                    v["colour"] = ex_vals[k]["colour"]
                vals.append(v)
            out[f].append(l1(render_values(vals), ex))
    return {f: round(float(np.mean(v)), 4) for f, v in out.items()}


def levels_check() -> dict:
    data, out = cats(10), {}
    for rgb, al in ((12, 4), (12, 8), (16, 4), (16, 8), (20, 8)):
        set_levels(rgb, al)
        e = [l1(L.render([L.stroke_tokens(layers[k], k) for k in range(5)]), ex) for _, _, layers, ex in data]
        out[f"{rgb}x{al}"] = {"paint_tokens": rgb ** 3 * al, "l1": round(float(np.mean(e)), 4)}
    set_levels(12, 8)
    out["unbinned"] = round(float(np.mean([l1(render_values([exact(layers[k]) for k in range(5)]), ex)
                                           for _, _, layers, ex in data])), 4)
    return out


def snap_refit(layers: list[dict], ex: dict, iters: int, n_layers: int = 5) -> np.ndarray:
    goal = torch.from_numpy(ex["target"].astype(np.float32) / 255)
    w = torch.from_numpy((ex["tri"] != 2).astype(np.float32))
    prot = torch.from_numpy((ex["tri"] == 0).astype(np.float32))
    c, rows_all = torch.ones(B.SIDE, B.SIDE, 3), []
    for k in range(n_layers):
        lay, br = layers[k], L.brush(k)
        with E.brush_settings(br):
            v = exact(lay)
            v["colour"] = np.clip(v["colour"], 0.005, 0.995)
            s = B.Strokes.from_stage(v, (br["length"], br["width"]))
            if iters:
                RT.refit(c, s, br, goal, w, prot, GRID, iters)
            pr = s.params()
        rows_all.append(L.stroke_tokens({"xy_px": pr.xy.detach().numpy(), "lw_px": pr.lw.detach().numpy(),
                                         "angle_rad": pr.angle.detach().numpy(), "colour_rgb": pr.colour.detach().numpy(),
                                         "alpha_val": pr.alpha.detach().numpy()}, k))
        c = torch.from_numpy(L.render(rows_all).astype(np.float32))
    return c.numpy()


def refit_check() -> dict:
    data, out = cats(6), {}
    out["unbinned"] = round(float(np.mean([l1(R.replay(R.load(REPAINT / f"{n}.npz")["layers"], only=list(range(5))), ex)
                                           for n, _, _, ex in data])), 4)
    for rgb, al in ((12, 4), (12, 8)):
        set_levels(rgb, al)
        for it in (0, 50):
            out[f"{rgb}x{al}_refit{it}"] = round(float(np.mean([l1(snap_refit(layers, ex, it), ex) for _, _, layers, ex in data])), 4)
    set_levels(12, 8)
    return out


def eight_check() -> dict:
    data = cats(8)
    unb, binned, hairs_norefit, lengths = [], [], [], []
    for name, words, layers, ex in data:
        full = R.load(REPAINT / f"{name}.npz")
        unb.append(l1(full["painting"] / 255, ex))
        binned.append(l1(L.render([L.stroke_tokens(layers[k], k) for k in range(8)]), ex))
        c = torch.from_numpy(snap_refit(layers, ex, 50).astype(np.float32))
        with torch.no_grad():
            for k in (5, 6, 7):
                br = L.brush(k)
                with E.brush_settings(br):
                    c = B.composite(c, B.Strokes.from_stage(L.stroke_values(L.stroke_tokens(layers[k], k), k),
                                                            (br["length"], br["width"])), GRID)
        hairs_norefit.append(l1(c.numpy(), ex))
        regions = {k: [np.flatnonzero(RT.region_of(ex["points"], layers[k]["xy_px"]) == r) for r in RT.REGION_ORDER]
                   for k in L.DETAIL}
        seq = L.encode(words, layers, 8, regions=regions)
        L.allowed_groups(seq, 8)
        lengths.append(len(seq))
    return {"unbinned": round(float(np.mean(unb)), 4), "binned_no_refit": round(float(np.mean(binned)), 4),
            "five_refit_hairs_binned_no_refit": round(float(np.mean(hairs_norefit)), 4),
            "l8_tokens": [int(min(lengths)), int(max(lengths))], "cats": len(data)}


def sequences_check() -> dict:
    files = sorted(glob.glob(str(ML / "data/processed/brush_l/cats/*.npz")))
    lens, ok = [], 0
    for f in files:
        t = np.load(f)["tokens"]
        L.allowed_groups(t, 5)
        ok += 1
        lens.append(len(t))
    t = np.load(ML / "data/processed/brush_l/cats/afhq__flickr_cat_000018.npz")["tokens"]
    words = [L.group_of(x) if L.group_of(x) not in L.FIELDS else f"{L.group_of(x)}:{int(x - L.OFFSET[L.group_of(x)])}" for x in t]
    k = words.index("LAYER0")
    first = L.stroke_values(t[k + 1:k + 6][None], 0)
    return {"cats": len(files), "valid": ok, "tokens": [int(min(lens)), int(np.median(lens)), int(max(lens))],
            "example_cat": "flickr_cat_000018", "example_len": int(len(t)), "example_start": words[:22],
            "example_layer0": words[k:k + 6], "example_first_stroke": {a: np.round(b[0], 2).tolist() for a, b in first.items()},
            "vocab": L.VOCAB, "fields": L.FIELDS, "specials": list(L.SPECIALS) + list(L.EXTRA)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["fields", "levels", "refit", "eight", "sequences"])
    a = ap.parse_args()
    torch.set_num_threads(4)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only:
        res = globals()[f"{name}_check"]()
        (a.out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(name, json.dumps(res)[:400], flush=True)


if __name__ == "__main__":
    main()
