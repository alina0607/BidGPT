"""How long a sequence the eye model can train on, on one GPU (owner, 2026-10-24: measure Kaggle's T4, then decide how many
tokens the eye material may use). The E5 network (eyegpt_l8, split paint: width 384, 6 layers) with its 256 pixel
patches in front of L tokens; random tokens and pixels (memory and time do not depend on the values); one optimizer
step of AdamW at each batch size, fp16 autocast on CUDA as eyegpt_l8.train. Per (L, batch): peak memory and seconds
per forward + backward of one batch (after one warm-up), or "oom". Prints one JSON line per setting and a summary.

    PYTHONPATH=. python tools/eye_t4_bench.py --lengths 2048,4096,6144 --batches 2,4,8
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from sellanything_ml.sketch import eyegpt_l8 as EG


def bench(length: int, batch: int, dev: str, reps: int) -> dict:
    cfg = EG.Config(paint="split")
    base = EG.model_config

    def longer(c: EG.Config):                                   # room for the measured length (RoPE tables only)
        m = base(c)
        m.max_seq_len = EG.NPATCH + length
        return m

    EG.model_config = longer
    try:
        model = EG.make_model(cfg).to(dev)
    finally:
        EG.model_config = base
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    vocab = EG.Vocab("split").size
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for r in range(reps + 1):
            tok = torch.randint(0, vocab, (batch, length), device=dev)
            pix = torch.randint(0, 256, (batch, EG.NPATCH, EG.PATCH_DIM), dtype=torch.uint8, device=dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with EG.autocast(dev):
                logits = model(tok[:, :-1], pix)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab), tok[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            if r:
                times.append(time.time() - t0)
        out = {"length": length, "positions": EG.NPATCH + length, "batch": batch,
               "seconds_per_batch": round(sum(times) / len(times), 3)}
        if dev == "cuda":
            out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return out
    except torch.OutOfMemoryError:
        return {"length": length, "positions": EG.NPATCH + length, "batch": batch, "oom": True}
    finally:
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lengths", default="2048,3072,4096,5120,6144,7168,8192")
    ap.add_argument("--batches", default="2,4,8")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    info = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
            "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if dev == "cuda" else None,
            "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    rows = []
    for length in map(int, a.lengths.split(",")):
        for batch in map(int, a.batches.split(",")):
            r = bench(length, batch, dev, a.reps)
            rows.append(r)
            print(json.dumps(r), flush=True)
            if r.get("oom"):
                break                                            # a larger batch at this length will not fit either
    print(json.dumps({"summary": rows, **info}), flush=True)


if __name__ == "__main__":
    main()
