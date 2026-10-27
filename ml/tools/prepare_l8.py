"""Training sequences for Brush GPT L8 (owner, 2026-10-20): each L5 sequence (tools/prepare_l.py) with the three hair
layers added, so an L8 sequence starts with its cat's L5 sequence token for token (only the final EOS moves) and L8
can be trained from L5's weights.

Hair layer by hair layer (fur, dark hairs, light hairs), on the canvas of the L5 sequence's own binned layers:
  1. drop the strokes whose centre is in the uncertain region (tri 2): the data does not know what is there, and
     unlike L5's strokes a hair there adds nothing the picture needs; every other hair stays (no contribution
     threshold: tau 1 removed fine lines the photo has, section 17.3 of the zh notebook);
  2. refit the kept strokes REFIT_ITERS steps with the painter's own loss, as prepare_l.py does;
  3. bin them to the token grid; the binned layer is the canvas the next one is fitted on.
The strokes are then written region by region (repaint_tidy.REGION_ORDER, each region closed by REGION_END), in the
order the repaint stored them within a region. Every hair token gets loss weight 1.

    PYTHONPATH=.:tools python tools/prepare_l8.py --repaint data/processed/repaint/kaggle_t4_run1/out \
        data/processed/repaint/kaggle_t4_run2/out data/processed/repaint/mac --plans data/processed/sketch_plan/afhq \
        --l5 data/processed/brush_l/cats --out data/processed/brush_l/cats8 --n 20 --device mps
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

REFIT_ITERS = 50


def prepare(l5_npz: Path, repaint_npz: Path, plan_npz: Path, dev: str = "cpu") -> dict:
    l5 = np.load(l5_npz)
    words, layers, ex = L.load_cat(repaint_npz, plan_npz)
    tri = ex["tri"]
    goal = torch.from_numpy(ex["target"].astype(np.float32) / 255).to(dev)
    weight = torch.from_numpy((tri != 2).astype(np.float32)).to(dev)
    protect = torch.from_numpy((tri == 0).astype(np.float32)).to(dev)
    ys, xs = torch.meshgrid(torch.arange(B.SIDE, device=dev) + 0.5, torch.arange(B.SIDE, device=dev) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    rows_all = L.decode_layers(l5["tokens"])[:L.LAYERS]
    canvas = torch.from_numpy(L.render(rows_all, dev=dev).astype(np.float32)).to(dev)
    l5_gap = float(np.abs(canvas.cpu().numpy() * 255 - l5["painting"]).max())      # the stored L5 painting, rounded
    hair, kept = [], []
    for k in L.DETAIL:
        lay, br = layers[k], L.brush(k)
        cx = np.clip(lay["xy_px"].astype(int), 0, B.SIDE - 1)
        idx = np.flatnonzero(tri[cx[:, 1], cx[:, 0]] != 2)
        with E.brush_settings(br):
            s = B.Strokes.from_stage({"xy": lay["xy_px"][idx], "angle": lay["angle_rad"][idx], "length_width": lay["lw_px"][idx],
                                      "colour": np.clip(lay["colour_rgb"][idx], 0.005, 0.995), "alpha": lay["alpha_val"][idx]},
                                     (br["length"], br["width"])).to(dev)
            if len(idx) and REFIT_ITERS:
                RT.refit(canvas, s, br, goal, weight, protect, grid, REFIT_ITERS)
            pr = s.params()
        rows = L.stroke_tokens({"xy_px": pr.xy.detach().cpu().numpy(), "lw_px": pr.lw.detach().cpu().numpy(),
                                "angle_rad": pr.angle.detach().cpu().numpy(), "colour_rgb": pr.colour.detach().cpu().numpy(),
                                "alpha_val": pr.alpha.detach().cpu().numpy()}, k)
        rows_all.append(rows)
        hair.append(rows)
        kept.append(len(idx))
        canvas = torch.from_numpy(L.render(rows_all, dev=dev).astype(np.float32)).to(dev)
    t5 = l5["tokens"]
    assert t5[-1] == L.SP["EOS"]
    seq = list(t5[:-1])
    for k, rows in zip(L.DETAIL, hair):
        region = RT.region_of(ex["points"], L.stroke_values(rows, k)["xy"])
        seq.append(L.SP[f"LAYER{k}"])
        for r in RT.REGION_ORDER:
            seq += rows[np.flatnonzero(region == r)].reshape(-1).tolist() + [L.SP["REGION_END"]]
        seq.append(L.SP["LAYER_END"])
    seq = np.array(seq + [L.SP["EOS"]], np.int32)
    L.allowed_groups(seq, L.MAX_LAYERS)                                      # the grammar holds
    assert (seq[:len(t5) - 1] == t5[:-1]).all()                              # starts with the L5 sequence
    loss_w = np.concatenate([l5["loss_w"][:-1], np.ones(len(seq) - len(t5) + 1, np.float32)])
    painting = canvas.cpu().numpy()
    cat = tri == 1
    return {"tokens": seq, "loss_w": loss_w, "kept": list(l5["kept"]) + kept, "painting": painting, "l5_gap": l5_gap,
            "l1_cat_l5": float(np.abs(l5["painting"] / 255 - ex["target"] / 255)[cat].mean()),
            "l1_cat": float(np.abs(painting - ex["target"].astype(np.float32) / 255)[cat].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repaint", type=Path, nargs="+", required=True, help="repaint output folders (shard_NNN/ inside)")
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--l5", type=Path, required=True, help="the L5 sequences (tools/prepare_l.py --out)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    plans = {p.stem: p for p in a.plans.glob("shard_*/*.npz") if not p.name.endswith(".tmp.npz")}
    repaint = {}
    for root in a.repaint:                    # a cat painted on two machines: the first folder given wins (as L5)
        for f in sorted(root.glob("shard_*/*.npz")):
            if not f.name.endswith(".tmp.npz"):
                repaint.setdefault(f.stem, f)
    failed = {json.loads(x)["file"][:-4] for x in open(a.out / "failed.jsonl")} if (a.out / "failed.jsonl").exists() else set()
    todo = [f for f in sorted(a.l5.glob("*.npz")) if not f.name.endswith(".tmp.npz") and f.stem not in failed
            and not (a.out / f.name).exists()]
    if a.n:
        todo = todo[:a.n]
    log = open(a.out / "progress.jsonl", "a")
    for f in todo:
        t0 = time.time()
        try:
            r = prepare(f, repaint[f.stem], plans[f.stem], a.device)
        except Exception as e:                         # skip and record: a retry loop must not stick on one cat
            with open(a.out / "failed.jsonl", "a") as fl:
                fl.write(json.dumps({"file": f.name, "error": repr(e)[:300]}) + "\n")
            continue
        tmp = a.out / f"{f.stem}.tmp.npz"
        np.savez_compressed(tmp, tokens=r["tokens"], loss_w=r["loss_w"], kept=np.array(r["kept"]),
                            painting=(r["painting"].clip(0, 1) * 255).round().astype(np.uint8))
        tmp.rename(a.out / f.name)
        row = {"file": f.name, "tokens": int(len(r["tokens"])), "kept": [int(x) for x in r["kept"]],
               "l1_cat_l5": round(r["l1_cat_l5"], 4), "l1_cat": round(r["l1_cat"], 4), "l5_gap": round(r["l5_gap"], 2),
               "seconds": round(time.time() - t0, 1), "device": a.device}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
