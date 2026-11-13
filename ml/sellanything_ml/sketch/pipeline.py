"""Sketch classifier v2: a bigger, higher-resolution CNN, a stroke-order model, and their fusion.

    python -m sellanything_ml.sketch.pipeline all        # every stage, in order
    python -m sellanything_ml.sketch.pipeline status     # what is done
    python -m sellanything_ml.sketch.pipeline final      # the one final measurement

Every stage can be stopped at any time (power cut, crash, Ctrl-C) and the same command continues
where it stopped: downloads and renders are written one class at a time, atomically; training
saves a checkpoint every few minutes, including where it was in the epoch, and replays exactly the
same batches and random numbers after a restart.

Stages (all on the first 15,500 drawings of each of the 345 classes):

  strokes   download the drawings as strokes, in drawing order          (strokes.py)
  check     confirm drawing k of the stroke file is row k of the bitmap file, for every class
  render    draw every drawing at 48x48 once, into one file on disk     (raster.rasterize)
  pack      every drawing as a fixed-length point sequence, into one file on disk
  cnn       train the v2 CNN on the 48x48 images
  seq       train the stroke-order Transformer
  stack     learn how to combine the models, on held-out drawings no model trained on
  final     score every model and combination once, on drawings 15,000-15,499 of every class,
            which nothing above has seen

Splits match train.py exactly (same seed and order), so the v1 CNN's validation and test drawings
are the same drawings here: per class, the first 15,000 are shuffled with seed 0 into
14,000 train / 500 val / 500 'stack' (train.py's test split); 15,000-15,499 are the final set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ML_DIR, download_first_n, load_categories
from .model import random_affine
from .raster import rasterize
from .strokes import download_first_n_strokes

RUN_DIR = ML_DIR / "runs" / "sketch_v2"
DATA_DIR = ML_DIR / "data" / "processed" / "sketch_v2"


@dataclass
class Config:
    per_class: int = 15_500          # drawings fetched per class
    n_train: int = 14_000
    n_val: int = 500
    n_stack: int = 500               # = train.py's test split
    side: int = 48
    max_len: int = 128               # points per sequence (99% of cat drawings have fewer than 120)
    seed: int = 0
    batch: int = 512
    cnn_epochs: int = 16
    cnn_max_lr: float = 0.2
    cnn_weight_decay: float = 1e-4
    cnn_dropout: float = 0.1
    seq_epochs: int = 10
    seq_lr: float = 1e-3
    seq_weight_decay: float = 0.05
    stack_epochs: int = 30
    checkpoint_minutes: float = 10.0
    classes: int = 0                 # use only the first N classes (smoke tests)


# --- data layout --------------------------------------------------------------------------------

def categories(cfg: Config) -> list[str]:
    cats = load_categories()
    return cats[:cfg.classes] if cfg.classes else cats


def splits(cfg: Config) -> dict[str, np.ndarray]:
    """Global drawing indices (class * per_class + k) of each split; same shuffle as data.load_splits."""
    rng = np.random.default_rng(cfg.seed)
    first = cfg.n_train + cfg.n_val + cfg.n_stack
    out: dict[str, list] = {"train": [], "val": [], "stack": [], "final": []}
    for c in range(len(categories(cfg))):
        perm = rng.permutation(first)
        base = c * cfg.per_class
        out["train"].append(base + perm[:cfg.n_train])
        out["val"].append(base + perm[cfg.n_train:cfg.n_train + cfg.n_val])
        out["stack"].append(base + perm[cfg.n_train + cfg.n_val:first])
        out["final"].append(base + np.arange(first, cfg.per_class))
    return {k: np.concatenate(v) for k, v in out.items()}


def labels(cfg: Config, idx: np.ndarray) -> np.ndarray:
    return idx // cfg.per_class


def _paths(cfg: Config) -> dict[str, Path]:
    tag = f"{len(categories(cfg))}x{cfg.per_class}"
    return {"images": DATA_DIR / f"images{cfg.side}_{tag}.u8", "seq": DATA_DIR / f"seq{cfg.max_len}_{tag}.u8",
            "lengths": DATA_DIR / f"len{cfg.max_len}_{tag}.i16", "progress": DATA_DIR / f"progress_{tag}.json"}


def _progress(cfg: Config) -> dict:
    p = _paths(cfg)["progress"]
    return json.loads(p.read_text()) if p.exists() else {}


def _mark(cfg: Config, key: str, value) -> None:
    p = _paths(cfg)["progress"]
    p.parent.mkdir(parents=True, exist_ok=True)
    state = _progress(cfg)
    state[key] = value
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, p)


def _memmap(path: Path, dtype, shape, mode: Literal["r", "r+"]):
    if mode == "r+" and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.memmap(path, dtype=dtype, mode="w+", shape=shape).flush()
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


# --- stages: data -------------------------------------------------------------------------------

def stage_strokes(cfg: Config) -> None:
    cats = categories(cfg)
    done = [0]

    def one(c):
        download_first_n_strokes(c, cfg.per_class)
        download_first_n(c, cfg.per_class)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  strokes: {done[0]}/{len(cats)}", flush=True)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, cats))
    print(f"strokes: {len(cats)} classes x {cfg.per_class:,}", flush=True)


def stage_check(cfg: Config) -> None:
    """Drawing k of the stroke file must be row k of the bitmap file, or fusing a bitmap model with a
    stroke model would compare different drawings. 20 drawings per class, spread over the file."""
    if _progress(cfg).get("check"):
        return
    worst = []
    for c in categories(cfg):
        s = download_first_n_strokes(c, cfg.per_class)
        bm = download_first_n(c, cfg.per_class)
        corrs = []
        for k in np.linspace(0, cfg.per_class - 1, 20).astype(int):
            a = rasterize(s.strokes(int(k))).astype(np.float32).ravel()
            b = bm[k].astype(np.float32).ravel()
            a, b = a - a.mean(), b - b.mean()
            corrs.append(float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-9)))
        worst.append((float(np.median(corrs)), c))
    worst.sort()
    print("check: lowest median correlations", [(round(v, 3), c) for v, c in worst[:5]], flush=True)
    bad = [c for v, c in worst if v < 0.95]
    if bad:
        raise SystemExit(f"stroke and bitmap files disagree for {bad}")
    _mark(cfg, "check", {"min_median_corr": worst[0][0]})


def _render_class(args) -> str:
    cfg_dict, c_index, c, path = args
    cfg = Config(**cfg_dict)
    s = download_first_n_strokes(c, cfg.per_class)
    n_total = len(categories(cfg)) * cfg.per_class
    images = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n_total, cfg.side, cfg.side))
    base = c_index * cfg.per_class
    for k in range(cfg.per_class):
        images[base + k] = rasterize(s.strokes(k), side=cfg.side)
    images.flush()
    return c


def stage_render(cfg: Config) -> None:
    p = _paths(cfg)
    cats = categories(cfg)
    n_total = len(cats) * cfg.per_class
    _memmap(p["images"], np.uint8, (n_total, cfg.side, cfg.side), "r+")
    done = set(_progress(cfg).get("render", []))
    todo = [(asdict(cfg), i, c, str(p["images"])) for i, c in enumerate(cats) if c not in done]
    if not todo:
        return
    print(f"render: {len(todo)} classes to draw at {cfg.side}x{cfg.side}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for c in pool.map(_render_class, todo):
            done.add(c)
            _mark(cfg, "render", sorted(done))                 # only after the class is flushed
            if len(done) % 25 == 0:
                print(f"  render: {len(done)}/{len(cats)} ({time.time() - t0:.0f}s)", flush=True)


def stage_pack(cfg: Config) -> None:
    p = _paths(cfg)
    cats = categories(cfg)
    n_total = len(cats) * cfg.per_class
    seq = _memmap(p["seq"], np.uint8, (n_total, cfg.max_len, 3), "r+")
    lengths = _memmap(p["lengths"], np.int16, (n_total,), "r+")
    done = set(_progress(cfg).get("pack", []))
    for i, c in enumerate(cats):
        if c in done:
            continue
        s = download_first_n_strokes(c, cfg.per_class)
        pts, lens = s.padded(np.arange(cfg.per_class), cfg.max_len)
        base = i * cfg.per_class
        seq[base:base + cfg.per_class] = pts
        lengths[base:base + cfg.per_class] = lens
        seq.flush()
        lengths.flush()
        done.add(c)
        _mark(cfg, "pack", sorted(done))
    print(f"pack: {len(cats)} classes, {cfg.max_len} points each", flush=True)


# --- models -------------------------------------------------------------------------------------

def _block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU())


class CNN2(nn.Module):
    """VGG-style, four blocks of two 3x3 convolutions (48, 96, 192, 384 channels)."""

    def __init__(self, num_classes: int, side: int = 48, widths=(48, 96, 192, 384), hidden: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        layers, c = [], 1
        for w in widths:
            layers += [_block(c, w), _block(w, w), nn.MaxPool2d(2)]
            c = w
        f = side // 2 ** len(widths)
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c * f * f, hidden, bias=False),
                                  nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.out(self.head(self.features(x)))


class StrokeTransformer(nn.Module):
    """Points in drawing order -> class scores. Each point is (x, y, dx, dy, pen lifts after it,
    pen just went down); a Transformer encoder reads the sequence and the outputs are averaged.
    (The Google tutorial uses a bidirectional LSTM here; on this Mac's GPU it trains 2.3x slower.)"""

    def __init__(self, num_classes: int, d: int = 192, layers: int = 4, heads: int = 6, max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(6, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, num_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        pad = torch.arange(L, device=x.device)[None] >= lengths[:, None]
        h = self.enc(self.inp(x) + self.pos[:, :L], src_key_padding_mask=pad)
        h = self.norm(h).masked_fill(pad[..., None], 0).sum(1) / lengths[:, None].clamp(min=1)
        return self.out(h)


def seq_features(points: torch.Tensor, lengths: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """(B, L, 3) uint8 (x, y, end) -> (B, L, 6) float. Coordinates fitted into a unit box; with `gen`,
    x and y are scaled by independent random factors in 0.9-1.1 (Sketch-RNN's augmentation)."""
    B, L, _ = points.shape
    xy = points[..., :2].float()
    valid = torch.arange(L, device=xy.device)[None] < lengths[:, None]
    big = 1e9
    lo = torch.where(valid[..., None], xy, torch.full_like(xy, big)).amin(1)
    hi = torch.where(valid[..., None], xy, torch.full_like(xy, -big)).amax(1)
    xy = (xy - (lo + hi)[:, None] / 2) / (hi - lo).amax(1).clamp(min=1)[:, None, None]    # [-0.5, 0.5]
    if gen is not None:
        scale = torch.empty(B, 1, 2).uniform_(0.9, 1.1, generator=gen).to(xy.device)
        xy = xy * scale
    d = torch.zeros_like(xy)
    d[:, 1:] = xy[:, 1:] - xy[:, :-1]
    end = points[..., 2].float()
    start = torch.zeros_like(end)
    start[:, 0] = 1
    start[:, 1:] = end[:, :-1]
    f = torch.cat([xy * 2, d * 8, end[..., None], start[..., None]], -1)
    return f.masked_fill(~valid[..., None], 0)


# --- resumable training -------------------------------------------------------------------------

def to_device(batch, device):
    return tuple(t.to(device, non_blocking=False) for t in batch)


class Batches:
    """The training batches of one epoch in a fixed order (seeded by epoch), loaded one ahead on a
    background thread. Starting at `step` skips what an interrupted run already did. The thread only
    reads from disk into CPU memory: Metal (the Mac GPU) crashes when two threads use it at once."""

    def __init__(self, order: np.ndarray, batch: int, load, step: int = 0):
        self.order, self.batch, self.load = order, batch, load
        self.steps = len(order) // batch
        self.q: queue.Queue = queue.Queue(maxsize=3)
        self.t = threading.Thread(target=self._fill, args=(step,), daemon=True)
        self.t.start()

    def _fill(self, step):
        for s in range(step, self.steps):
            self.q.put((s, self.load(self.order[s * self.batch:(s + 1) * self.batch])))
        self.q.put(None)

    def __iter__(self):
        while (item := self.q.get()) is not None:
            yield item


def bucket_order(order: np.ndarray, lengths: np.ndarray, batch: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffled, but each batch holds drawings of similar length (less padding): sort chunks of 50
    batches by length, cut them into batches, then shuffle the batches."""
    chunk = batch * 50
    batches = []
    for s in range(0, len(order) - batch + 1, chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch] for i in range(0, len(part) - batch + 1, batch)]
    batches = [batches[i] for i in rng.permutation(len(batches))]
    return np.concatenate(batches) if batches else order[:0]


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(name: str, cfg: Config, model: nn.Module, step_fn, epochs: int, make_opt, make_sched,
          order_fn, load_fn, evaluate_fn, device) -> Path:
    """Generic resumable loop. Saves <run>/<name>_last.pt every cfg.checkpoint_minutes and at each
    epoch end, and <run>/<name>_best.pt when validation loss improves."""
    run = RUN_DIR
    run.mkdir(parents=True, exist_ok=True)
    last, best_path = run / f"{name}_last.pt", run / f"{name}_best.pt"
    log = run / f"{name}_metrics.jsonl"
    opt = make_opt(model)
    steps_per_epoch = len(order_fn(0)) // cfg.batch
    sched = make_sched(opt, epochs * steps_per_epoch)
    epoch, step, best = 0, 0, float("inf")
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        epoch, step, best = st["epoch"], st["step"], st["best"]
        print(f"{name}: resumed at epoch {epoch + 1}, step {step}/{steps_per_epoch}", flush=True)
    if epoch >= epochs:
        return best_path

    def checkpoint(e, s):
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "epoch": e, "step": s, "best": best}, last)

    while epoch < epochs:
        model.train()
        t0, t_ck, running, seen = time.time(), time.time(), 0.0, 0
        for s, batch in Batches(order_fn(epoch), cfg.batch, load_fn, step):
            batch = to_device(batch, device)
            torch.manual_seed(cfg.seed * 1_000_003 + epoch * steps_per_epoch + s)   # dropout etc.
            gen = torch.Generator().manual_seed(cfg.seed * 7_919 + epoch * steps_per_epoch + s)
            loss = step_fn(model, batch, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if s % 50 == 0:
                running, seen = running + loss.item(), seen + 1
            if time.time() - t_ck > cfg.checkpoint_minutes * 60:
                checkpoint(epoch, s + 1)
                t_ck = time.time()
                print(f"  {name}: epoch {epoch + 1} step {s + 1}/{steps_per_epoch}, "
                      f"loss {running / max(seen, 1):.4f}", flush=True)
        model.eval()
        val_loss, val_top1, val_top3 = evaluate_fn(model)
        rec = {"epoch": epoch + 1, "train_loss_sampled": running / seen if seen else None, "val_loss": val_loss,
               "val_top1": val_top1, "val_top3": val_top3, "lr": sched.get_last_lr()[0],
               "seconds": time.time() - t0}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name}: " + json.dumps(rec), flush=True)
        if val_loss < best:
            best = val_loss
            _save({"model": model.state_dict(), "epoch": epoch + 1, **rec}, best_path)
        epoch, step = epoch + 1, 0
        checkpoint(epoch, 0)
    return best_path


@torch.no_grad()
def predict(model: nn.Module, idx: np.ndarray, load_fn, forward, batch: int = 2048) -> torch.Tensor:
    """log-probabilities (N, classes) on the CPU, in idx order."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for s in range(0, len(idx), batch):
        b = to_device(load_fn(idx[s:s + batch]), device)
        out.append(F.log_softmax(forward(model, b).float(), -1).cpu())
    return torch.cat(out)


def scores(logp: torch.Tensor, y: np.ndarray) -> tuple[float, float, float]:
    yt = torch.from_numpy(y)
    top = logp.topk(3, dim=1).indices
    return (float(F.nll_loss(logp, yt)), float((top[:, 0] == yt).float().mean()),
            float((top == yt[:, None]).any(1).float().mean()))


# --- stages: models -----------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda"))


def _image_loader(cfg: Config, device):
    p = _paths(cfg)
    n = len(categories(cfg)) * cfg.per_class
    images = np.memmap(p["images"], dtype=np.uint8, mode="r", shape=(n, cfg.side, cfg.side))

    def load(idx):
        srt = np.sort(idx)                                   # sorted reads are faster on disk
        x = torch.from_numpy(np.ascontiguousarray(images[srt])).view(-1, 1, cfg.side, cfg.side)
        inv = np.argsort(np.argsort(idx))
        return x[torch.from_numpy(inv)], torch.from_numpy(labels(cfg, idx))
    return load


def _seq_loader(cfg: Config, device):
    p = _paths(cfg)
    n = len(categories(cfg)) * cfg.per_class
    seq = np.memmap(p["seq"], dtype=np.uint8, mode="r", shape=(n, cfg.max_len, 3))
    lengths = np.memmap(p["lengths"], dtype=np.int16, mode="r", shape=(n,))

    def load(idx):
        lens = np.asarray(lengths[idx]).astype(np.int64)
        L = max(1, int(lens.max()))
        pts = np.ascontiguousarray(seq[idx][:, :L])
        return torch.from_numpy(pts), torch.from_numpy(lens), torch.from_numpy(labels(cfg, idx))
    return load, lengths


def build_cnn(cfg: Config) -> CNN2:
    return CNN2(len(categories(cfg)), cfg.side, dropout=cfg.cnn_dropout)


def build_seq(cfg: Config) -> StrokeTransformer:
    return StrokeTransformer(len(categories(cfg)), max_len=cfg.max_len)


def cnn_forward(model, batch):
    x, _ = batch
    return model(x.float().div_(255.0))


def seq_forward(model, batch):
    pts, lens, _ = batch
    return model(seq_features(pts, lens, None), lens)


def stage_cnn(cfg: Config) -> None:
    device = _device()
    sp = splits(cfg)
    load = _image_loader(cfg, device)
    model = build_cnn(cfg).to(device)

    def step(m, batch, gen):
        x, y = batch
        x = random_affine(x.float().div_(255.0), max_degrees=10, max_shift=0.06)
        with _autocast(device):
            return F.cross_entropy(m(x).float(), y)

    def order(epoch):
        return np.random.default_rng(cfg.seed * 1000 + epoch).permutation(sp["train"])

    def evaluate(m):
        with _autocast(device):
            return scores(predict(m, sp["val"], load, cnn_forward), labels(cfg, sp["val"]))

    train("cnn", cfg, model, step, cfg.cnn_epochs,
          lambda m: torch.optim.SGD(m.parameters(), lr=cfg.cnn_max_lr / 25, momentum=0.9, nesterov=True,
                                    weight_decay=cfg.cnn_weight_decay),
          lambda o, total: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=cfg.cnn_max_lr, total_steps=total),
          order, load, evaluate, device)


def stage_seq(cfg: Config) -> None:
    device = _device()
    sp = splits(cfg)
    load, lengths = _seq_loader(cfg, device)
    lens_all = np.asarray(lengths).astype(np.int64)
    model = build_seq(cfg).to(device)

    def step(m, batch, gen):
        pts, lens, y = batch
        with _autocast(device):
            return F.cross_entropy(m(seq_features(pts, lens, gen), lens).float(), y)

    def order(epoch):
        rng = np.random.default_rng(cfg.seed * 1000 + 500 + epoch)
        return bucket_order(rng.permutation(sp["train"]), lens_all, cfg.batch, rng)

    def evaluate(m):
        val = sp["val"][np.argsort(lens_all[sp["val"]], kind="stable")]
        with _autocast(device):
            logp = predict(m, val, load, seq_forward)
        return scores(logp, labels(cfg, val))

    def make_sched(o, total):
        warm = max(1, total // 50)
        return torch.optim.lr_scheduler.LambdaLR(
            o, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    train("seq", cfg, model, step, cfg.seq_epochs,
          lambda m: torch.optim.AdamW(m.parameters(), lr=cfg.seq_lr, weight_decay=cfg.seq_weight_decay),
          make_sched, order, load, evaluate, device)


# --- stacking and the final measurement ---------------------------------------------------------

def _v1(path: Path, device):
    from .infer import SketchClassifier
    return SketchClassifier(path, str(device)) if path.exists() else None


def model_logprobs(cfg: Config, idx: np.ndarray, cache_name: str) -> dict[str, torch.Tensor]:
    """log-probabilities of every model on the drawings idx, cached on disk per split."""
    cache = RUN_DIR / f"logp_{cache_name}.pt"
    if cache.exists():
        return torch.load(cache)
    device = _device()
    out = {}
    load_img = _image_loader(cfg, device)
    load_seq, lengths = _seq_loader(cfg, device)
    cnn = build_cnn(cfg).to(device)
    cnn.load_state_dict(torch.load(RUN_DIR / "cnn_best.pt", map_location=device)["model"])
    seq = build_seq(cfg).to(device)
    seq.load_state_dict(torch.load(RUN_DIR / "seq_best.pt", map_location=device)["model"])
    with _autocast(device):
        out["cnn_v2"] = predict(cnn, idx, load_img, cnn_forward)
        lens = np.asarray(lengths[idx]).astype(np.int64)
        srt = np.argsort(lens, kind="stable")
        lp = predict(seq, idx[srt], load_seq, seq_forward)
        out["seq"] = lp[torch.from_numpy(np.argsort(srt))]
    cats = categories(cfg)
    for name, ck in (("cnn_v1_6k", "sketch_cnn_345_6k.pt"), ("cnn_v1_15k", "sketch_cnn_345_15k.pt")):
        clf = _v1(ML_DIR / "checkpoints" / ck, device)
        if clf is None or clf.categories != cats:
            continue
        bitmaps = {}

        def load_bm(sub):
            rows = []
            for g in sub:
                c, k = divmod(int(g), cfg.per_class)
                if c not in bitmaps:
                    bitmaps[c] = download_first_n(cats[c], cfg.per_class)
                rows.append(bitmaps[c][k])
            return (torch.from_numpy(np.stack(rows)).view(-1, 1, 28, 28),)
        out[name] = predict(clf.model, idx, load_bm, lambda m, b: m(b[0].float().div_(255.0)))
    _save(out, cache)
    return out


class Stacker(nn.Module):
    """Learns per-class weights for combining models' log-probabilities (plus a bias)."""

    def __init__(self, n_models: int, n_classes: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_models, n_classes) / n_models)
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logps: torch.Tensor) -> torch.Tensor:     # (N, models, classes)
        return (logps * self.w).sum(1) + self.b


def stage_stack(cfg: Config) -> None:
    """Fit the combinations on the 'stack' split (no model trained on it), choose on 'val'."""
    out_path = RUN_DIR / "stack.pt"
    if out_path.exists():
        return
    sp = splits(cfg)
    fit, val = model_logprobs(cfg, sp["stack"], "stack"), model_logprobs(cfg, sp["val"], "val")
    y_fit, y_val = torch.from_numpy(labels(cfg, sp["stack"])), torch.from_numpy(labels(cfg, sp["val"]))
    results = {}
    for combo in (("cnn_v2", "seq"), ("cnn_v1_15k", "seq"), ("cnn_v2", "cnn_v1_15k", "seq")):
        if not all(m in fit for m in combo):
            continue
        X, Xv = torch.stack([fit[m] for m in combo], 1), torch.stack([val[m] for m in combo], 1)
        # (a) one weight for the whole model, chosen on val
        best_w, best_v = None, float("inf")
        for w in np.linspace(0, 1, 21):
            ws = torch.tensor([w] + [(1 - w) / (len(combo) - 1)] * (len(combo) - 1))
            v = F.cross_entropy((Xv * ws[None, :, None]).sum(1), y_val).item()
            if v < best_v:
                best_w, best_v = ws, v
        # (b) per-class weights, learned on the stack split
        st = Stacker(len(combo), X.shape[2])
        opt = torch.optim.Adam(st.parameters(), lr=0.01)
        best_state, best_sv = None, float("inf")
        for _ in range(cfg.stack_epochs):
            for s in range(0, len(X), 4096):
                loss = F.cross_entropy(st(X[s:s + 4096]), y_fit[s:s + 4096])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = F.cross_entropy(st(Xv), y_val).item()
            if v < best_sv:
                best_sv, best_state = v, {k: t.clone() for k, t in st.state_dict().items()}
        results["+".join(combo)] = {"weights": cast(torch.Tensor, best_w).tolist(), "weights_val_loss": best_v,
                                    "stacker": best_state, "stacker_val_loss": best_sv}
        print(f"stack {'+'.join(combo)}: weights {[round(x, 2) for x in cast(torch.Tensor, best_w).tolist()]} "
              f"(val loss {best_v:.4f}); per-class stacker val loss {best_sv:.4f}", flush=True)
    _save(results, out_path)


def stage_final(cfg: Config, force: bool = False) -> None:
    """The one measurement: every model and combination on drawings 15,000-15,499 of every class."""
    report = RUN_DIR / "final_report.json"
    if report.exists() and not force:
        print(f"final: already measured, see {report} (use --force to measure again)")
        print(report.read_text())
        return
    missing = [f for f in ("cnn_best.pt", "seq_best.pt", "stack.pt") if not (RUN_DIR / f).exists()]
    if missing:
        raise SystemExit(f"final: training is not finished ({', '.join(missing)} missing); run 'all' first")
    sp = splits(cfg)
    idx = sp["final"]
    y = labels(cfg, idx)
    lp = model_logprobs(cfg, idx, "final")
    stack = torch.load(RUN_DIR / "stack.pt", weights_only=False)
    rows = {}
    for name, logp in lp.items():
        rows[name] = logp
    for combo, r in stack.items():
        X = torch.stack([lp[m] for m in combo.split("+")], 1)
        rows[f"{combo} (one weight)"] = (X * torch.tensor(r["weights"])[None, :, None]).sum(1)
        st = Stacker(X.shape[1], X.shape[2])
        st.load_state_dict(r["stacker"])
        with torch.no_grad():
            rows[f"{combo} (per-class)"] = st(X)
    cats = categories(cfg)
    rec = np.concatenate([download_first_n_strokes(c, cfg.per_class).recognized[cfg.per_class - len(idx) // len(cats):]
                          for c in cats])
    out: dict[str, Any] = {"drawings": len(idx), "recognized_share": float(rec.mean()), "models": {}}
    yt = torch.from_numpy(y)
    for name, logits in rows.items():
        loss, t1, t3 = scores(F.log_softmax(logits, -1), y)
        top1 = logits.argmax(1) == yt
        per = {c: float(top1[torch.from_numpy(y == i)].float().mean()) for i, c in enumerate(cats)
               if c in ("cat", "dog", "raccoon", "bicycle")}
        out["models"][name] = {"top1": t1, "top3": t3, "loss": loss,
                               "top1_recognized": float(top1[torch.from_numpy(rec)].float().mean()),
                               "top1_unrecognized": float(top1[torch.from_numpy(~rec)].float().mean()),
                               "classes": per}
    report.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


STAGES = {"strokes": stage_strokes, "check": stage_check, "render": stage_render, "pack": stage_pack,
          "cnn": stage_cnn, "seq": stage_seq, "stack": stage_stack}


def status(cfg: Config) -> None:
    prog = _progress(cfg)
    n = len(categories(cfg))
    print(f"check:  {'done' if prog.get('check') else '-'}")
    print(f"render: {len(prog.get('render', []))}/{n} classes")
    print(f"pack:   {len(prog.get('pack', []))}/{n} classes")
    for name in ("cnn", "seq"):
        log = RUN_DIR / f"{name}_metrics.jsonl"
        lines = log.read_text().splitlines() if log.exists() else []
        print(f"{name}:    {len(lines)} epochs done" + (f", last {lines[-1]}" if lines else ""))
    print(f"stack:  {'done' if (RUN_DIR / 'stack.pt').exists() else '-'}")
    print(f"final:  {'measured' if (RUN_DIR / 'final_report.json').exists() else 'not yet'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=[*STAGES, "all", "final", "status"])
    ap.add_argument("--classes", type=int, default=0, help="first N classes only (smoke tests)")
    ap.add_argument("--per-class", type=int, default=15_500)
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests)")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--checkpoint-minutes", type=float, default=10.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    global RUN_DIR, DATA_DIR
    if args.run_dir:
        RUN_DIR = args.run_dir
        DATA_DIR = args.run_dir / "data"
    cfg = Config(classes=args.classes, per_class=args.per_class, checkpoint_minutes=args.checkpoint_minutes)
    if args.per_class != 15_500:                         # smoke tests: shrink the splits to fit
        cfg.n_train, cfg.n_val, cfg.n_stack = int(args.per_class * 0.8), args.per_class // 10, args.per_class // 20
    if args.epochs_scale != 1.0:
        cfg.cnn_epochs = max(1, round(cfg.cnn_epochs * args.epochs_scale))
        cfg.seq_epochs = max(1, round(cfg.seq_epochs * args.epochs_scale))
        cfg.stack_epochs = max(1, round(cfg.stack_epochs * args.epochs_scale))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if args.stage == "status":
        status(cfg)
    elif args.stage == "final":
        stage_final(cfg, args.force)
    elif args.stage == "all":
        for name, fn in STAGES.items():
            print(f"=== {name} ===", flush=True)
            fn(cfg)
        print("all training stages done; run the 'final' stage once to measure", flush=True)
    else:
        STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()
