"""Repaint only the three hair layers (fur, dark hairs, light hairs) of finished repaints, with --times as many strokes
(owner, 2026-10-19: four times the hairs brought back much more of the photo's fine lines, tools/hair_count_test.py).
The first five layers (washes, blocks, blend, small blocks, blend) are kept as they are: they are replayed from the
stored strokes, and the hair layers are fitted on top, with the same specs as tools/repaint.py except the counts.
So no cat is repainted from scratch, and a cat painted on any machine can be finished on any other.

Each output keeps the input's keys for layers 0-4 and replaces layers 5-7; meta gains "hairs_times".

    PYTHONPATH=.:tools python tools/repaint_hairs.py --files data/processed/repaint/mac/shard_010 --n 3 --times 4 --out runs/repaint_hairs/x4
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import repaint as R

KEEP = 5                                   # layers 0-4 are kept
HAIR_KINDS = {"fur", "dark hairs", "light hairs"}


def repaint_hairs(path: Path, times: int, dev: str) -> tuple[dict, list, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    old = R.load(path)
    target = old["target"]
    start = R.replay(old["layers"], only=list(range(KEEP)))
    specs = R.layer_specs(target.astype(np.float32) / 255, old["tri"])[KEEP:]
    for spec in specs:
        spec["n"] *= times
    record = []
    final = E.paint(torch.from_numpy(target.astype(np.float32) / 255), torch.from_numpy((old["tri"] == 1).astype(np.float32)),
                    specs, start=start, record=record, dev=dev)[-1]
    return {k: z[k] for k in z.files}, record, final


def save(path: Path, kept: dict, record: list, final: np.ndarray, meta: dict) -> None:
    arrays = {k: v for k, v in kept.items() if not any(k.startswith(f"l{j}_") for j in range(KEEP, 99))}
    arrays["layers"] = np.array(KEEP + len(record))
    arrays["painting"] = (final.clip(0, 1) * 255).round().astype(np.uint8)
    arrays["meta"] = np.array(json.dumps(meta))
    for j, (spec, s) in enumerate(record):
        k = KEEP + j
        with E.brush_settings(spec):
            pr = s.params()
        for p in R.PARAMS:
            arrays[f"l{k}_{p}"] = getattr(s, p).detach().cpu().numpy()
        arrays[f"l{k}_xy_px"] = pr.xy.detach().cpu().numpy()
        arrays[f"l{k}_lw_px"] = pr.lw.detach().cpu().numpy()
        arrays[f"l{k}_angle_rad"] = pr.angle.detach().cpu().numpy()
        arrays[f"l{k}_colour_rgb"] = pr.colour.detach().cpu().numpy()
        arrays[f"l{k}_alpha_val"] = pr.alpha.detach().cpu().numpy()
        arrays[f"l{k}_brush"] = np.array(json.dumps(R.brush_of(spec)))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=Path, nargs="+", required=True, help="repaint .npz files or folders of them")
    ap.add_argument("--n", type=int, default=0, help="first N files only (a test)")
    ap.add_argument("--times", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-after-min", type=float, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for p in a.files for f in ([p] if p.suffix == ".npz" else p.glob("*.npz")) if not f.name.endswith(".tmp.npz"))
    if a.n:
        files = files[:a.n]
    t_start = time.time()
    log = open(a.out / "progress.jsonl", "a")
    for f in files:
        if (a.out / f.name).exists():
            continue
        if a.stop_after_min and time.time() - t_start > 60 * a.stop_after_min:
            print(json.dumps({"stopped": "time"}), flush=True)
            break
        t0 = time.time()
        kept, record, final = repaint_hairs(f, a.times, a.device)
        meta = {**json.loads(str(kept["meta"])), "hairs_times": a.times, "hairs_device": a.device, "hairs_version": "repaint-hairs-v1 (2026-10-19)"}
        save(a.out / f.name, kept, record, final, meta)
        tri = kept["tri"]
        l1 = float(np.abs(final - kept["target"].astype(np.float32) / 255)[tri == 1].mean())
        row = {"file": f.name, "times": a.times, "device": a.device, "seconds": round(time.time() - t0, 1), "l1_cat": round(l1, 4),
               "strokes": int(sum(len(kept[f"l{k}_xy"]) for k in range(KEEP)) + sum(len(s.xy) for _, s in record))}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        del kept, record, final
        gc.collect()
        if a.device == "cuda":
            torch.cuda.empty_cache()
        elif a.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
