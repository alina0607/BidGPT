"""Drawing GPT: a decoder-only transformer that draws a chosen Quick, Draw! category stroke by stroke
(docs/design-v2.md §5).

A drawing becomes tokens:  [category] [start] move move … [lift] move … [end]
  - a *move* is one of K = 1,000 typical pen moves (dx, dy), found by K-means on 100,000 moves from
    the training split, 20% of them pen-up moves (Sketchformer §3.1);
  - moves are chosen with *error feedback*: each token aims at the true point from where the decoded
    drawing currently is, so rounding errors do not pile up. Measured on 450 training drawings
    (2026-10-05): mean error 2.1 px on the 256-px canvas, against 12.9 px without feedback;
  - [lift] follows the last point of every stroke but the last, which is followed by [end].
Drawings longer than MAX_LEN tokens (0.8% of them) are left out rather than cut, so every training
sequence ends with [end] (Sketch-RNN §3.2: learning when to stop is the hard part).

The model is llm/model.py's GPT; training reuses pipeline.train (checkpoints every 10 minutes,
resumable). How many epochs fit is decided once by timing the first steps against a time budget.

Versions (`version()`; each trains into runs/<name>/):
  v1  from scratch on every training drawing (2026-10-05, 29 epochs, 10 h).
  v2  v1's weights trained further, with two changes (2026-10-07):
      - only drawings the Quick, Draw! game recognised. 8.3% of the drawings were not, and the v1 CNN
        recognises only 20.8% of those (77.1% of the others), so they teach shapes that are not the class;
      - 10% of the class tokens are replaced by [no class], so the same model also learns to draw
        without a class. Sampling can then use classifier-free guidance (Ho & Salimans 2021).
      v1's train and validation losses were equal (3.73 / 3.75: no overfitting) and its learning rate
      had decayed to zero, so training further is expected to help; warm-starting reuses its 10 hours.
      Peak lr 3e-4: a first try at 5e-4 (runs/drawgpt_v2/attempt1) pushed validation loss from v1's
      3.75 to 3.86 after one epoch, too much to win back in the ~10 epochs the 3.5 h budget allows.

    python -m sellanything_ml.sketch.drawgpt dict                   # pen-move dictionary (CPU)
    python -m sellanything_ml.sketch.drawgpt tokens                 # every drawing as tokens (CPU)
    python -m sellanything_ml.sketch.drawgpt train --version v2     # benchmark once, then train (GPU)
    python -m sellanything_ml.sketch.drawgpt all                    # the three in order; safe to re-run
    python -m sellanything_ml.sketch.drawgpt sample cat --version v2 --guidance 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..llm.model import GPT, GPTConfig
from . import pipeline
from .data import ML_DIR
from .strokes import load_cached

RUNS_DIR = ML_DIR / "runs"
DATA_DIR = ML_DIR / "data" / "processed" / "drawgpt_v1"

K = 1000
PAD, START, LIFT, END = K, K + 1, K + 2, K + 3
FIRST_CLASS = K + 4
MAX_LEN = 160                       # 99.2% of drawings fit (measured on 20,700 training drawings)


@dataclass
class DrawConfig:
    per_class: int = 15_500          # must match pipeline.Config so splits are the same drawings
    classes: int = 0                 # first N classes only (smoke tests)
    dict_samples: int = 100_000
    pen_up_share: float = 0.2
    per_class_per_epoch: int = 2_000 # each epoch draws a fresh random subset of the training split
    val_per_class: int = 20
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.1
    batch: int = 256
    budget_hours: float = 10.0       # training time to fill; epochs are chosen to fit it
    max_epochs: int = 40
    seed: int = 0
    checkpoint_minutes: float = 10.0
    name: str = "drawgpt_v1"          # folder under runs/
    recognized_only: bool = False     # train only on drawings the Quick, Draw! game recognised
    class_dropout: float = 0.0        # share of training drawings whose class token becomes [no class]
    init_from: str = ""               # warm start from this run folder's best weights


def version(name: str) -> DrawConfig:
    """The settings of each trained version (module docstring)."""
    if name == "v1":
        return DrawConfig()
    if name == "v2":
        return DrawConfig(name="drawgpt_v2", recognized_only=True, class_dropout=0.1, init_from="drawgpt_v1",
                          lr=3e-4, budget_hours=3.5)
    raise ValueError(f"unknown drawing GPT version {name!r}")


def run_dir(cfg: DrawConfig) -> Path:
    return RUNS_DIR / cfg.name


def _pcfg(cfg: DrawConfig) -> pipeline.Config:
    return pipeline.Config(per_class=cfg.per_class, classes=cfg.classes)


def categories(cfg: DrawConfig) -> list[str]:
    return pipeline.categories(_pcfg(cfg))


def vocab_size(n_classes: int, no_class: bool = False) -> int:
    """Moves and [pad] [start] [lift] [end], one token per class, and [no class] when the model learns it."""
    return FIRST_CLASS + n_classes + int(no_class)


def no_class_token(cfg: DrawConfig) -> int | None:
    """The [no class] token, right after the class tokens; None for a model trained without it."""
    return FIRST_CLASS + len(categories(cfg)) if cfg.class_dropout > 0 else None


# --- tokens -------------------------------------------------------------------------------------

def moves(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dx, dy) from the previous point (the first from (0, 0)) and whether each is a pen-up move."""
    xy = points[:, :2].astype(np.int16)
    delta = np.diff(np.vstack([[0, 0], xy]), axis=0)
    up = np.r_[True, points[:-1, 2] == 1]
    return delta, up


def encode_batch(drawings: list[np.ndarray], centers: np.ndarray) -> list[np.ndarray]:
    """Error-feedback tokens for many drawings at once (all advance one point per step)."""
    n = len(drawings)
    lens = np.array([len(d) for d in drawings])
    xy = np.zeros((n, lens.max(), 2), np.float32)
    for i, d in enumerate(drawings):
        xy[i, :len(d)] = d[:, :2]
    cur = np.zeros((n, 2), np.float32)
    toks = np.zeros((n, lens.max()), np.int64)
    c2 = (centers ** 2).sum(1)
    for t in range(lens.max()):
        target = xy[:, t] - cur
        j = np.argmin(c2[None, :] - 2 * target @ centers.T, axis=1)   # nearest center
        toks[:, t] = j
        cur = cur + centers[j]
    return [toks[i, :lens[i]] for i in range(n)]


def sequence(class_id: int, points: np.ndarray, move_tokens: np.ndarray) -> np.ndarray:
    ends = points[:, 2] == 1
    out = [FIRST_CLASS + class_id, START]
    for k, tok in enumerate(move_tokens):
        out.append(int(tok))
        if k == len(move_tokens) - 1:
            out.append(END)
        elif ends[k]:
            out.append(LIFT)
    return np.array(out, np.int64)


def decode(tokens: np.ndarray, centers: np.ndarray) -> list[tuple[list[float], list[float]]]:
    """Tokens back to strokes ([xs], [ys]); stops at [end]; ignores anything that is not a move."""
    cur = np.zeros(2)
    strokes, xs, ys = [], [], []
    for tok in tokens:
        tok = int(tok)
        if tok == END:
            break
        if tok == LIFT:
            if xs:
                strokes.append((xs, ys))
            xs, ys = [], []
        elif tok < K:
            cur = cur + centers[tok]
            xs.append(float(cur[0]))
            ys.append(float(cur[1]))
    if xs:
        strokes.append((xs, ys))
    return strokes


TINY_PX = 4                         # a stroke this small (width and height) is a dot
MAX_DOT_RUN = 10                    # 99.9% of people's drawings have at most this many dots in a row


def _is_dot(stroke: tuple[list[float], list[float]]) -> bool:
    xs, ys = stroke
    return max(max(xs) - min(xs), max(ys) - min(ys)) <= TINY_PX


def tidy(strokes: list[tuple[list[float], list[float]]], ended: bool) -> list[tuple[list[float], list[float]]]:
    """Clean up a drawing that ran out of tokens before [end]. At low temperature the model sometimes
    gets stuck lifting the pen in place: 4.9% of drawings at 0.6 never ended, and their last 40
    strokes (median) are dots of at most TINY_PX. People draw dots too (eyes), but in 6,900 training
    drawings the longest run was 20 and 99.9% had at most MAX_DOT_RUN (2026-10-06). So only an
    unfinished drawing is touched: its last stroke was cut off, and a run of dots longer than any
    person would draw is the stuck part. Shorter runs stay (they may be eyes)."""
    if ended or not strokes:
        return strokes
    out = strokes[:-1]
    run = 0
    while run < len(out) and _is_dot(out[len(out) - 1 - run]):
        run += 1
    if run > MAX_DOT_RUN:
        out = out[:len(out) - run]
    return out or strokes


def _paths(cfg: DrawConfig) -> dict[str, Path]:
    tag = f"{len(categories(cfg))}x{cfg.per_class}"
    return {"dict": DATA_DIR / f"dict{K}.npy", "tokens": DATA_DIR / f"tokens{MAX_LEN}_{tag}.u16",
            "lengths": DATA_DIR / f"lengths{MAX_LEN}_{tag}.i16", "progress": DATA_DIR / f"progress_{tag}.json"}


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    """k-means++ start, then Lloyd's iterations. NumPy only: loading scikit-learn and PyTorch in one
    process crashes on this Mac (OpenMP clash, checked 2026-10-06)."""
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    centers = np.empty((k, 2))
    centers[0] = x[rng.integers(len(x))]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        centers[j] = x[rng.choice(len(x), p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((x - centers[j]) ** 2).sum(1))
    x2 = (x ** 2).sum(1)
    for _ in range(iters):
        near = np.argmin(x2[:, None] - 2 * x @ centers.T + (centers ** 2).sum(1)[None, :], axis=1)
        counts = np.bincount(near, minlength=k)
        for dim in range(2):
            sums = np.bincount(near, weights=x[:, dim], minlength=k)
            centers[counts > 0, dim] = sums[counts > 0] / counts[counts > 0]
        empty = np.flatnonzero(counts == 0)
        centers[empty] = x[rng.choice(len(x), len(empty), replace=False)]
    return centers.astype(np.float32)


def stage_dict(cfg: DrawConfig) -> np.ndarray:
    path = _paths(cfg)["dict"]
    if path.exists():
        return np.load(path)
    rng = np.random.default_rng(cfg.seed)
    tr = pipeline.splits(_pcfg(cfg))["train"]
    down, up = [], []
    for c, name in enumerate(categories(cfg)):
        s = load_cached(name, cfg.per_class)
        idx = tr[(tr >= c * cfg.per_class) & (tr < (c + 1) * cfg.per_class)] - c * cfg.per_class
        for k in rng.choice(idx, min(60, len(idx)), replace=False):
            delta, is_up = moves(s.drawing(int(k)))
            down.append(delta[~is_up])
            up.append(delta[is_up])
    down, up = np.vstack(down), np.vstack(up)
    n_up = int(cfg.pen_up_share * cfg.dict_samples)
    n_up, n_down = min(n_up, len(up)), min(cfg.dict_samples - n_up, len(down))
    x = np.vstack([down[rng.choice(len(down), n_down, replace=False)],
                   up[rng.choice(len(up), n_up, replace=False)]]).astype(np.float32)
    found = kmeans(x, min(K, len(x)), cfg.seed)
    centers = np.zeros((K, 2), np.float32)
    centers[:len(found)] = found
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centers)
    print(f"dict: {K} pen moves from {len(x):,} samples ({n_up:,} pen-up)", flush=True)
    return centers


def stage_tokens(cfg: DrawConfig) -> None:
    p = _paths(cfg)
    cats = categories(cfg)
    n_total = len(cats) * cfg.per_class
    centers = stage_dict(cfg)
    toks = pipeline._memmap(p["tokens"], np.uint16, (n_total, MAX_LEN), "r+")
    lens = pipeline._memmap(p["lengths"], np.int16, (n_total,), "r+")
    prog = json.loads(p["progress"].read_text()) if p["progress"].exists() else {}
    done, dropped = set(prog.get("tokens", [])), prog.get("dropped", {})
    t0 = time.time()
    for c, name in enumerate(cats):
        if name in done:
            continue
        s = load_cached(name, cfg.per_class)
        drawings = [s.drawing(i) for i in range(cfg.per_class)]
        # a sequence has points + strokes + 2 tokens; skip encoding what cannot fit anyway
        fits = [len(d) + int(d[:, 2].sum()) + 2 <= MAX_LEN for d in drawings]
        kept = [i for i, f in enumerate(fits) if f]
        mt = dict(zip(kept, encode_batch([drawings[i] for i in kept], centers)))
        base, n_drop = c * cfg.per_class, 0
        for i, d in enumerate(drawings):
            if i not in mt:
                lens[base + i] = 0          # left out, never cut
                n_drop += 1
                continue
            seq = sequence(c, d, mt[i])
            toks[base + i, :len(seq)] = seq
            toks[base + i, len(seq):] = PAD
            lens[base + i] = len(seq)
        toks.flush()
        lens.flush()
        done.add(name)
        dropped[name] = n_drop
        tmp = p["progress"].with_suffix(".tmp")
        tmp.write_text(json.dumps({"tokens": sorted(done), "dropped": dropped}, indent=1))
        os.replace(tmp, p["progress"])
        if len(done) % 25 == 0:
            print(f"  tokens: {len(done)}/{len(cats)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"tokens: {len(cats)} classes; {sum(dropped.values()):,} drawings longer than {MAX_LEN} tokens "
          f"left out", flush=True)


# --- training -----------------------------------------------------------------------------------

def model_config(cfg: DrawConfig) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size(len(categories(cfg)), cfg.class_dropout > 0), d_model=cfg.d_model, n_layers=cfg.n_layers,
                     n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, max_seq_len=MAX_LEN)


def _data(cfg: DrawConfig):
    p = _paths(cfg)
    n_total = len(categories(cfg)) * cfg.per_class
    toks = np.memmap(p["tokens"], np.uint16, "r", shape=(n_total, MAX_LEN))
    lens = np.asarray(np.memmap(p["lengths"], np.int16, "r", shape=(n_total,))).astype(np.int64)
    return toks, lens


def recognized(cfg: DrawConfig) -> np.ndarray:
    """Per drawing (same index as the tokens): did the Quick, Draw! game recognise it at the time."""
    out = [load_cached(name, cfg.per_class).recognized for name in categories(cfg)]
    assert all(len(r) == cfg.per_class for r in out), "cached stroke files are shorter than per_class"
    return np.concatenate(out).astype(bool)


def warm_start(model: GPT, path: Path) -> None:
    """Load a trained model's weights. When this model has more tokens (here [no class]), the old rows
    are copied and each new row starts as the mean of the class-token rows: "some class, unknown"."""
    old = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    for key, w in old.items():
        if w.shape != own[key].shape:
            assert key in ("embed.weight", "head.weight") and w.shape[1:] == own[key].shape[1:], key
            grown = torch.empty_like(own[key])
            grown[:len(w)] = w
            grown[len(w):] = w[FIRST_CLASS:].mean(0)
            old[key] = grown
    model.load_state_dict(old)


def _per_class(idx: np.ndarray, per_class: int, n: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    for c in np.unique(idx // per_class):
        mine = idx[idx // per_class == c]
        out.append(rng.choice(mine, min(n, len(mine)), replace=False))
    return np.concatenate(out)


def make_loader(toks, lens):
    def load(sub):
        sub = np.asarray(sub)
        width = int(lens[sub].max())
        return (torch.from_numpy(toks[sub, :width].astype(np.int64)),)
    return load


def _device() -> torch.device:
    """DRAWGPT_DEVICE=cpu forces the CPU (smoke tests while the GPU is busy)."""
    return torch.device(os.environ["DRAWGPT_DEVICE"]) if os.environ.get("DRAWGPT_DEVICE") else pipeline._device()


def stage_train(cfg: DrawConfig) -> Path:
    run = run_dir(cfg)
    if (run / "done.json").exists():
        return run / "drawgpt_best.pt"
    run.mkdir(parents=True, exist_ok=True)
    pipeline.RUN_DIR = run                          # pipeline.train writes its checkpoints here
    toks, lens = _data(cfg)
    sp = pipeline.splits(_pcfg(cfg))
    train_idx = sp["train"][lens[sp["train"]] > 0]
    if cfg.recognized_only:
        train_idx = train_idx[recognized(cfg)[train_idx]]
    # validation stays every drawing of the same 20 per class, so losses compare across versions
    val_idx = sp["val"][lens[sp["val"]] > 0]
    val = _per_class(val_idx, cfg.per_class, cfg.val_per_class, np.random.default_rng(cfg.seed + 1))
    device = _device()
    mcfg = model_config(cfg)
    no_class = no_class_token(cfg)

    def new_model() -> GPT:
        m = GPT(mcfg)
        if cfg.init_from:
            warm_start(m, RUNS_DIR / cfg.init_from / "drawgpt_best.pt")
        return m.to(device)

    model = new_model()
    load = make_loader(toks, lens)

    def step(m, batch, gen):
        (t,) = batch
        if no_class is not None:                    # the class token is position 0 of every row
            drop = (torch.rand(len(t), generator=gen) < cfg.class_dropout).to(t.device)
            t = torch.cat([torch.where(drop, no_class, t[:, 0])[:, None], t[:, 1:]], 1)
        with pipeline._autocast(device):
            _, loss = m(t[:, :-1], t[:, 1:], loss_mask=(t[:, 1:] != PAD))
        return loss

    def order(epoch):
        rng = np.random.default_rng(cfg.seed * 1000 + 700 + epoch)
        sub = _per_class(train_idx, cfg.per_class, cfg.per_class_per_epoch, rng)
        return pipeline.bucket_order(rng.permutation(sub), lens, cfg.batch, rng)

    @torch.no_grad()
    def evaluate(m):
        """Validation loss and next-token top-1 / top-3 accuracy (stored as val_top1 / val_top3)."""
        tot = n = top1 = top3 = 0.0
        srt = val[np.argsort(lens[val], kind="stable")]
        for s in range(0, len(srt), 512):
            (t,) = pipeline.to_device(load(srt[s:s + 512]), device)
            with pipeline._autocast(device):
                logits, _ = m(t[:, :-1])
            y, mask = t[:, 1:], t[:, 1:] != PAD
            lp = F.log_softmax(logits.float(), -1)
            tot += float(-(lp.gather(-1, y[..., None])[..., 0] * mask).sum())
            top = lp.topk(3, -1).indices
            top1 += float(((top[..., 0] == y) & mask).sum())
            top3 += float(((top == y[..., None]).any(-1) & mask).sum())
            n += float(mask.sum())
        return tot / n, top1 / n, top3 / n

    plan_path = run / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:                                           # time the step once, then fix the schedule
        # Time exactly what pipeline.train does per step (clipping included) on batches spread over
        # the epoch. v2's first try timed only forward, backward and the optimiser step: 0.31 s/step
        # planned, 0.45 s/step run (runs/drawgpt_v2/attempt1), so it planned 15 epochs for 3.5 h of 5.
        epoch0 = order(0)
        n_batches = len(epoch0) // cfg.batch
        n_bench = min(60, n_batches)
        picks = np.linspace(0, n_batches - 1, n_bench).astype(int)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        times = []
        for s in picks:
            (t,) = pipeline.to_device(load(epoch0[s * cfg.batch:(s + 1) * cfg.batch]), device)
            t0 = time.time()
            loss = step(model, (t,), torch.Generator().manual_seed(int(s)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if device.type == "mps":
                torch.mps.synchronize()
            times.append(time.time() - t0)
        sec = float(np.median(times[min(10, len(times) // 3):]))
        steps_per_epoch = len(order(0)) // cfg.batch
        epochs = int(max(1, min(cfg.max_epochs, cfg.budget_hours * 3600 // (sec * steps_per_epoch))))
        plan = {"seconds_per_step": sec, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
                "parameters": model.num_params(), "model": asdict(mcfg), "config": asdict(cfg),
                "train_drawings": int(len(train_idx)), "val_drawings": int(len(val))}
        plan_path.write_text(json.dumps(plan, indent=2))
        model = new_model()                         # the benchmark's updates are thrown away
        print(f"drawgpt plan: {json.dumps(plan)}", flush=True)

    def make_sched(o, total):
        warm = max(1, total // 50)
        return torch.optim.lr_scheduler.LambdaLR(
            o, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    pcfg = pipeline.Config(per_class=cfg.per_class, classes=cfg.classes, batch=cfg.batch, seed=cfg.seed,
                           checkpoint_minutes=cfg.checkpoint_minutes)
    best = pipeline.train("drawgpt", pcfg, model, step, plan["epochs"],
                          lambda m: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
                          make_sched, order, load, evaluate, device)
    (run / "done.json").write_text(json.dumps({"best": str(best)}) + "\n")
    print("drawgpt: training done", flush=True)
    return best


# --- drawing ------------------------------------------------------------------------------------

@torch.no_grad()
def draw(model: GPT, class_id: int, n: int, temperature: float, seed: int = 0, guidance: float = 1.0,
         no_class: int | None = None) -> list[np.ndarray]:
    """n token sequences for one category, sampled at `temperature`, each ending at [end] if reached.
    guidance ≠ 1 uses classifier-free guidance against the [no class] prompt (GPT.generate), which
    only a model trained with class dropout has."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    start = torch.tensor([[FIRST_CLASS + class_id, START]] * n, device=device)
    uncond = None
    if guidance != 1.0:
        if no_class is None:
            raise ValueError("guidance needs a model trained with class dropout (a [no class] token)")
        uncond = torch.tensor([[no_class, START]] * n, device=device)
    out = model.generate(start, MAX_LEN - 2, temperature=temperature, stop_id=END, uncond=uncond,
                         guidance=guidance)
    seqs = []
    for row in out.cpu().numpy():
        body = row[2:]
        stop = np.flatnonzero(body == END)
        seqs.append(body[:stop[0] + 1] if len(stop) else body)
    return seqs


def load_model(cfg: DrawConfig, device):
    st = torch.load(run_dir(cfg) / "drawgpt_best.pt", map_location=device, weights_only=False)
    model = GPT(model_config(cfg)).to(device)
    model.load_state_dict(st["model"])
    return model.eval()


def stage_sample(cfg: DrawConfig, name: str, n: int, temperature: float, guidance: float = 1.0) -> Path:
    from PIL import Image, ImageDraw
    cats = categories(cfg)
    device = _device()
    model = load_model(cfg, device)
    centers = np.load(_paths(cfg)["dict"])
    seqs = draw(model, cats.index(name), n, temperature, guidance=guidance, no_class=no_class_token(cfg))
    side, pad = 256, 8
    img = Image.new("L", (n * (side + pad) + pad, side + 2 * pad), 255)
    g = ImageDraw.Draw(img)
    for i, seq in enumerate(seqs):
        ox = pad + i * (side + pad)
        for xs, ys in decode(seq, centers):
            pts = [(ox + x, pad + y) for x, y in zip(xs, ys)]
            if len(pts) > 1:
                g.line(pts, fill=0, width=3)
    out = run_dir(cfg) / f"sample_{name.replace(' ', '_')}_t{temperature}_g{guidance}.png"
    img.save(out)
    print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["dict", "tokens", "train", "all", "sample"])
    ap.add_argument("category", nargs="?", default="cat")
    ap.add_argument("--version", default="v1", help="v1 or v2 (see the module docstring)")
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="tiny model, for testing the code path")
    args = ap.parse_args()
    global RUNS_DIR, DATA_DIR
    if args.run_dir:
        RUNS_DIR, DATA_DIR = args.run_dir, args.run_dir / "data"
    cfg = version(args.version)
    cfg.classes = args.classes
    if args.budget_hours is not None:
        cfg.budget_hours = args.budget_hours
    if args.smoke:                       # tiny model and epochs: exercises the code path on a CPU
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads = 64, 2, 4, 2
        cfg.per_class_per_epoch, cfg.batch, cfg.val_per_class, cfg.max_epochs = 256, 64, 8, 2
        cfg.init_from = ""               # the real v1 weights do not fit the tiny model
    if args.stage in ("dict", "all"):
        stage_dict(cfg)
    if args.stage in ("tokens", "all"):
        stage_tokens(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg)
    if args.stage == "sample":
        stage_sample(cfg, args.category, args.n, args.temperature, args.guidance)


if __name__ == "__main__":
    main()
