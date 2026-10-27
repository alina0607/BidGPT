"""The shape GPT: learns p(token | tokens before it) over catshape.py's painting sequences, then draws new cats
token by token, which the game can paint as they arrive.

Sequences longer than CAP tokens are left out (1.7% of train, 2.2% of validation at 3,072, measured
2026-10-09). A batch is padded to its longest sequence with EOS; padding is not in the loss. Each optimiser step adds up
the gradients of `accum` micro-batches of `batch` sequences: 1 × 8 = 8 sequences a step, as planned. A micro-batch
of 2 at 3,072 tokens needs more than 15 GB of GPU memory; this Mac has 32 GB in all, and above that the MPS limit
(42 GB by default) lets it swap instead of failing: 0 epochs in 3 h 46 min (2026-10-09). run_shape.sh caps it with
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 (15 GB), so running short fails at once. At 1 × 8, lengths padded to
multiples of BUCKET and the cache freed every step: 2.84 s a step, 41.1 min an epoch, peak 13.3 GB; the 40 longest
sequences interleaved with random ones peak at 12.2 GB (measured 2026-10-09).

    python -m sellanything_ml.sketch.catshape_gpt speed            # time 20 training steps (GPU)
    python -m sellanything_ml.sketch.catshape_gpt train --epochs 8
    python -m sellanything_ml.sketch.catshape_gpt sample --n 16 --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import catshape as K
from .catface import RUN_DIR as CATFACE_RUNS
from .pixsketch import _log, _save, device

RUN_DIR = CATFACE_RUNS / "shape_gpt"
CAP = 3072
BUCKET = 512                     # batches are padded up to a multiple of this: few shapes for the GPU allocator


@dataclass
class Config:
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 1
    accum: int = 8
    epochs: int = 8
    warmup_steps: int = 200
    seed: int = 0


def gpt_config(cfg: Config):
    from ..llm.model import GPTConfig
    return GPTConfig(vocab_size=K.VOCAB, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                     n_kv_heads=cfg.n_kv_heads, max_seq_len=CAP, dropout=cfg.dropout, abs_pos=True)


def rows(split: str) -> list[np.ndarray]:
    return [s.astype(np.int64) for s in K.load(split) if len(s) <= CAP]


def batch(seqs: list[np.ndarray], dev):
    """(inputs, targets, mask): padded with EOS to the longest, rounded up to BUCKET; the mask is 1 on real target
    positions. Every new length is a new set of buffers that MPS keeps cached, so free lengths ran past the
    memory cap within the first minute (2026-10-09)."""
    n = min(CAP, -(-max(len(s) for s in seqs) // BUCKET) * BUCKET)
    x = np.full((len(seqs), n), K.EOS, np.int64)
    m = np.zeros((len(seqs), n - 1), np.float32)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s) - 1] = 1
    t = torch.from_numpy(x).to(dev)
    return t[:, :-1], t[:, 1:], torch.from_numpy(m).to(dev)


@torch.no_grad()
def val_loss(model, seqs, dev, bs: int = 1) -> float:
    model.eval()
    tot, n = 0.0, 0.0
    for i in range(0, len(seqs), bs):
        xb, yb, mb = batch(seqs[i:i + bs], dev)
        _, loss = model(xb, yb, loss_mask=mb)
        tot += loss.item() * mb.sum().item()
        n += mb.sum().item()
    return tot / n


def _model(cfg: Config, dev):
    from ..llm.model import GPT
    torch.manual_seed(cfg.seed)
    return GPT(gpt_config(cfg)).to(dev)


def _optimizer(model, cfg: Config):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": other, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))


def stage_speed(cfg: Config, steps: int = 20) -> dict:
    """Seconds per training step on random batches, and the epoch time that implies."""
    dev = device()
    train = rows("train")
    model = _model(cfg, dev).train()
    opt = _optimizer(model, cfg)
    rng = np.random.default_rng(0)
    times = []
    for s in range(steps + 3):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.accum):
            xb, yb, mb = batch([train[i] for i in rng.choice(len(train), cfg.batch, replace=False)], dev)
            _, loss = model(xb, yb, loss_mask=mb)
            (loss / cfg.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss.item()                                                    # wait for the GPU
        if s >= 3:
            times.append(time.time() - t0)
    per = float(np.mean(times))
    per_step = cfg.batch * cfg.accum
    rec = {"parameters": model.num_params(), "batch": cfg.batch, "accum": cfg.accum, "seconds_per_step": round(per, 3),
           "steps_per_epoch": len(train) // per_step, "minutes_per_epoch": round(per * len(train) / per_step / 60, 1),
           "peak_gpu_gb": round(torch.mps.driver_allocated_memory() / 2 ** 30, 1) if dev.type == "mps" else None}
    print(json.dumps(rec), flush=True)
    return rec


def stage_train(cfg: Config) -> None:
    dev = device()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train, val = rows("train"), rows("validation")
    model = _model(cfg, dev)
    opt = _optimizer(model, cfg)
    per_step = cfg.batch * cfg.accum
    steps = len(train) // per_step
    total = steps * cfg.epochs
    lr_at = lambda st: cfg.lr * min(1.0, (st + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / total)))  # noqa: E731
    log, best, step = RUN_DIR / "metrics.jsonl", float("inf"), 0
    log.write_text("")
    (RUN_DIR / "config.json").write_text(json.dumps({"config": asdict(cfg), "parameters": model.num_params(),
                                                     "train_rows": len(train), "val_rows": len(val), "cap": CAP}, indent=1) + "\n")
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        order, t0, run, ntok = rng.permutation(len(train)), time.time(), 0.0, 0.0
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            for a in range(cfg.accum):
                lo = s * per_step + a * cfg.batch
                xb, yb, mb = batch([train[i] for i in order[lo:lo + cfg.batch]], dev)
                _, loss = model(xb, yb, loss_mask=mb)
                (loss / cfg.accum).backward()
                k = mb.sum().item()
                run += loss.item() * k
                ntok += k
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if dev.type == "mps":
                torch.mps.empty_cache()
        rec = {"epoch": epoch + 1, "train_loss": round(run / ntok, 4), "val_loss": round(val_loss(model, val, dev), 4),
               "lr": round(lr_at(step), 7), "seconds": round(time.time() - t0)}
        _log(log, rec)
        if rec["val_loss"] < best:
            best = rec["val_loss"]
            _save({"model": model.state_dict(), "config": asdict(cfg), **rec}, RUN_DIR / "best.pt")


def stage_sample(n: int = 16, temperature: float = 1.0, seed: int = 1, dev_name: str = "") -> dict:
    """n new cats from [BOS] alone: tokens until EOS (or CAP), decoded, painted; a sheet of the finished cats and
    one cat's frames step by step."""
    from PIL import Image, ImageDraw

    from ..llm.model import GPT
    dev = torch.device(dev_name) if dev_name else device()
    st = torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    cfg = Config(**st["config"])
    model = GPT(gpt_config(cfg))
    model.load_state_dict(st["model"])
    model = model.to(dev).eval()
    torch.manual_seed(seed)
    out = model.generate(torch.full((n, 1), K.BOS, device=dev), CAP - 1, temperature=temperature, stop_id=K.EOS).cpu()
    cats, lengths, bad, ended = [], [], 0, 0
    for row in out.tolist():
        row = row[:row.index(K.EOS) + 1] if K.EOS in row else row
        ended += row[-1] == K.EOS
        lengths.append(len(row))
        _, steps, b = K.decode(row)
        bad += b
        cats.append(steps)
    side, cols = 160, 8
    sheet = Image.new("RGB", (cols * side, math.ceil(n / cols) * side), "white")
    for i, steps in enumerate(cats):
        im = Image.fromarray(K.render(steps).clip(0, 255).astype(np.uint8)).resize((side - 6, side - 6), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * side + 3, (i // cols) * side + 3))
    tag = f"t{temperature}_s{seed}"
    sheet.save(RUN_DIR / f"sample_{tag}.png")
    frames = K.render(cats[0], frames=True)
    fs = Image.new("RGB", (6 * side, math.ceil(len(frames) / 6) * (side + 20)), "white")
    dr = ImageDraw.Draw(fs)
    for i, (f, (name, _)) in enumerate(zip(frames, cats[0])):
        x, y = (i % 6) * side, (i // 6) * (side + 20)
        dr.text((x + 4, y + 2), f"{i + 1}. {name}", fill=(0, 0, 0))
        fs.paste(Image.fromarray(f.clip(0, 255).astype(np.uint8)).resize((side - 6, side - 6), Image.LANCZOS), (x + 3, y + 18))
    fs.save(RUN_DIR / f"sample_{tag}_steps.png")
    rec = {"n": n, "temperature": temperature, "seed": seed, "ended_with_eos": ended, "grammar_errors_total": bad,
           "tokens_median": int(np.median(lengths)), "tokens_max": int(max(lengths)),
           "steps_median": float(np.median([len(c) for c in cats])), "epoch": st["epoch"], "val_loss": st["val_loss"]}
    (RUN_DIR / f"sample_{tag}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["speed", "train", "sample"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    cfg = Config()
    if a.epochs:
        cfg.epochs = a.epochs
    if a.batch:
        cfg.batch = a.batch
    if a.accum:
        cfg.accum = a.accum
    if a.stage == "speed":
        stage_speed(cfg)
    elif a.stage == "train":
        stage_train(cfg)
    else:
        stage_sample(a.n, a.temperature, a.seed, a.device)


if __name__ == "__main__":
    main()
