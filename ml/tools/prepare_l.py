"""Training sequences for Brush GPT L (owner, 2026-10-19): a repainted cat (tools/repaint.py) and its sketch plan
(tools/sketch_plan.py --batch) become one token sequence of sellanything_ml.sketch.brushtokens_l.

Layer by layer, on the canvas of the layers already binned:
  1. keep the strokes that contribute at least TAU (opacity x area x how far their colour is from the canvas under
     them; repaint_tidy, absolute rule tau 1 chosen by the owner); blending layers keep all their strokes;
  2. refit the kept strokes REFIT_ITERS steps with the painter's own loss (weight 0 on the uncertain region,
     PROTECT on the background), so this layer makes up for the binning of the layers under it;
  3. bin them to the token grid; the binned layer is the canvas the next layer is fitted on.
Binning the five layers without this raised the L1 to the photo by 0.0041 (12 x 8, 6 cats); with it, by 0.0022 (12 colour x 8
opacity levels; 2026-10-19).

Each stroke also gets a loss weight: 0 if its centre is in the uncertain region (tri 2; section 15.6: those strokes are
kept for the picture, but the model is not taught to paint what the data does not know), else 1.

    PYTHONPATH=.:tools python tools/prepare_l.py --repaint data/processed/repaint/kaggle_t4_run1/out \
        --plans data/processed/sketch_plan/afhq --out data/processed/brush_l/cats --n 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import repaint_tidy as RT
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

TAU = 1.0
REFIT_ITERS = 50


def prepare(repaint_npz: Path, plan_npz: Path, dev: str = "cpu") -> dict:
    words, layers, ex = L.load_cat(repaint_npz, plan_npz)
    tri = ex["tri"]
    goal = torch.from_numpy(ex["target"].astype(np.float32) / 255).to(dev)
    weight = torch.from_numpy((tri != 2).astype(np.float32)).to(dev)
    protect = torch.from_numpy((tri == 0).astype(np.float32)).to(dev)
    ys, xs = torch.meshgrid(torch.arange(B.SIDE, device=dev) + 0.5, torch.arange(B.SIDE, device=dev) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas = torch.ones(B.SIDE, B.SIDE, 3, device=dev)
    rows_all, kept, binned_layers = [], [], []
    for k in range(L.LAYERS):
        lay, br = layers[k], L.brush(k)
        if br["blend"]:
            idx = np.arange(len(lay["xy_px"]))
        else:
            c = RT.contribution(lay, canvas.cpu().numpy())
            idx = np.flatnonzero(c >= TAU)                                  # stored (contrast) order kept
        with E.brush_settings(br):
            s = B.Strokes.from_stage({"xy": lay["xy_px"][idx], "angle": lay["angle_rad"][idx], "length_width": lay["lw_px"][idx],
                                      "colour": np.clip(lay["colour_rgb"][idx], 0.005, 0.995), "alpha": lay["alpha_val"][idx]},
                                     (br["length"], br["width"])).to(dev)
            if len(idx) and REFIT_ITERS:
                RT.refit(canvas, s, br, goal, weight, protect, grid, REFIT_ITERS)
            pr = s.params()
        fitted = {"xy_px": pr.xy.detach().cpu().numpy(), "lw_px": pr.lw.detach().cpu().numpy(),
                  "angle_rad": pr.angle.detach().cpu().numpy(), "colour_rgb": pr.colour.detach().cpu().numpy(),
                  "alpha_val": pr.alpha.detach().cpu().numpy()}
        rows = L.stroke_tokens(fitted, k)
        rows_all.append(rows)
        kept.append(len(idx))
        binned_layers.append(fitted)
        canvas = torch.from_numpy(L.render(rows_all, dev=dev).astype(np.float32)).to(dev)
    seq = [L.SP["BOS"]] + L.plan_ids(words)
    stroke_w = []
    for k, rows in enumerate(rows_all):
        cx = np.clip(L.stroke_values(rows, k)["xy"].astype(int), 0, B.SIDE - 1)
        w = (tri[cx[:, 1], cx[:, 0]] != 2).astype(np.float32)
        seq += [L.SP[f"LAYER{k}"]] + rows.reshape(-1).tolist() + [L.SP["LAYER_END"]]
        stroke_w.append(w)
    seq.append(L.SP["EOS"])
    seq = np.array(seq, np.int32)
    L.allowed_groups(seq)                                                    # the grammar holds
    loss_w = np.ones(len(seq), np.float32)                                   # per token; strokes' 5 tokens share w
    pos = 1 + len(words)
    for k, rows in enumerate(rows_all):
        pos += 1
        loss_w[pos:pos + 5 * len(rows)] = np.repeat(stroke_w[k], 5)
        pos += 5 * len(rows) + 1
    painting = canvas.cpu().numpy()
    cat = tri == 1
    return {"tokens": seq, "loss_w": loss_w, "kept": kept, "painting": painting,
            "l1_cat": float(np.abs(painting - ex["target"].astype(np.float32) / 255)[cat].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repaint", type=Path, nargs="+", required=True, help="repaint output folders (shard_NNN/ inside)")
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-after-min", type=float, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    plans = {p.stem: p for p in a.plans.glob("shard_*/*.npz") if not p.name.endswith(".tmp.npz")}
    seen, todo = set(), []                    # a cat painted on two machines: the first folder given wins
    for root in a.repaint:
        for f in sorted(root.glob("shard_*/*.npz")):
            if f.name.endswith(".tmp.npz") or f.stem in seen:
                continue
            seen.add(f.stem)
            if f.stem in plans and not (a.out / f.name).exists():
                todo.append((f, plans[f.stem]))
    failed = {json.loads(x)["file"][:-4] for x in open(a.out / "failed.jsonl")} if (a.out / "failed.jsonl").exists() else set()
    todo = [(f, p) for f, p in todo if f.stem not in failed]
    if a.n:
        todo = todo[:a.n]
    t_start = time.time()
    log = open(a.out / "progress.jsonl", "a")
    for f, p in todo:
        if a.stop_after_min and time.time() - t_start > 60 * a.stop_after_min:
            break
        t0 = time.time()
        if f.stem in failed:
            continue
        try:
            r = prepare(f, p, a.device)
        except Exception as e:                         # skip and record: a retry loop must not stick on one cat
            with open(a.out / "failed.jsonl", "a") as fl:
                fl.write(json.dumps({"file": f.name, "error": repr(e)[:300]}) + "\n")
            continue
        tmp = a.out / f"{f.stem}.tmp.npz"
        np.savez_compressed(tmp, tokens=r["tokens"], loss_w=r["loss_w"], kept=np.array(r["kept"]),
                            painting=(r["painting"].clip(0, 1) * 255).round().astype(np.uint8))
        tmp.rename(a.out / f.name)
        row = {"file": f.name, "tokens": int(len(r["tokens"])), "kept": r["kept"], "l1_cat": round(r["l1_cat"], 4),
               "seconds": round(time.time() - t0, 1), "device": a.device}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
