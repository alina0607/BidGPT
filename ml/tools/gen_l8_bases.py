"""The eye model's bases (owner, 2026-10-21): for every cat of data_L8.npz, L8 is handed the cat's real plan (its tokens
up to PLAN_END) and paints the rest, as in tools/eye_upper_bound_l8.py; the eye layers are later fitted over these
paintings, so the eye model learns on the same kind of painting it will be given (L8's own, not L8's training data).
Temperature 1; chunk k of the cats in name order is sampled with seed k, so a chunk is reproducible on its own.
One file per cat, written whole (a temporary file renamed); a new run skips the cats already there.

    PYTHONPATH=.:tools python tools/gen_l8_bases.py --device mps --max-chunks 4      (tools/run_gen_l8_bases_mac.sh loops)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
DATA = ML / "data" / "processed" / "brush_l" / "data_L8.npz"
CATS8 = ML / "data" / "processed" / "brush_l" / "cats8"
OUT = ML / "data" / "processed" / "brush_l" / "l8_gen"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-chunks", type=int, default=0, help="exit after this many chunks (a wrapper restarts the process)")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    os.environ["BRUSHGPT_DEVICE"] = a.device
    from sellanything_ml.sketch import brushgpt_l as G
    from sellanything_ml.sketch import brushtokens_l as L
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(str(x) for x in np.load(DATA)["names"])
    chunks = [names[i:i + a.chunk] for i in range(0, len(names), a.chunk)]
    model = None
    done_chunks = 0
    log = open(OUT / "progress.jsonl", "a")
    for k, chunk in enumerate(chunks):
        todo = [n for n in chunk if not (OUT / f"{n}.npz").exists()]
        if not todo:
            continue
        if model is None:
            model = G.load_model(G.RUN_DIR / "L8" / "best.pt", G.device())
        prefix = []
        for n in todo:
            s = np.load(CATS8 / f"{n}.npz")["tokens"]
            prefix.append(s[:int(np.flatnonzero(s == L.SP["PLAN_END"])[0]) + 1])
        t0 = time.time()
        seqs = G.sample(model, len(todo), a.temperature, seed=k, max_len=G.MAX_SEQ, prefix=prefix)
        for n, s in zip(todo, seqs):
            tmp = OUT / f"{n}.tmp.npz"
            np.savez_compressed(tmp, tokens=np.asarray(s, np.int32), seed=k, temperature=a.temperature)
            tmp.rename(OUT / f"{n}.npz")
        row = {"chunk": k, "cats": len(todo), "seconds": round(time.time() - t0, 1),
               "ended": int(sum(s[-1] == L.SP["EOS"] for s in seqs)), "time": time.strftime("%H:%M")}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        done_chunks += 1
        if a.max_chunks and done_chunks >= a.max_chunks:
            print(json.dumps({"stopped": "max_chunks"}), flush=True)
            return
    print(json.dumps({"done": len(names)}), flush=True)


if __name__ == "__main__":
    main()
