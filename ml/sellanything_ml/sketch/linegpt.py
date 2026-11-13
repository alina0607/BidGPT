"""Line GPT: draw a clean line sketch of a class, stroke by stroke, with a decoder-only transformer.

Data: ControlSketch (Arar et al., SwiftSketch, SIGGRAPH 2025), as released with part annotations in
ControlSketch-Part (Du et al. 2026, CC BY 4.0). Every sketch is exactly 32 cubic Bézier strokes on a
512×512 canvas, made by the authors' ControlSketch optimisation from SDXL images, in one consistent style;
the strokes are stored in the authors' order, coarse structure first. Splits: train 15 classes × 1,000,
validation the same 15 classes × 200, test 85 other classes × 200.

A sketch becomes 257 tokens:  [class] x0 y0 x1 y1 x2 y2 x3 y3  × 32 strokes
  - each stroke is its start point and three control points (`M x0 y0 C x1 y1 x2 y2 x3 y3`);
  - a coordinate v is put in one of 512 bins of 2 px covering [-256, 768): control points can lie off
    the canvas (the curve itself stays on it), and this range holds all but 0.0041% of the coordinates
    of all 34,989 sketches (measured 2026-10-07); those few are clipped;
  - x and y use separate tokens (x: 0–511, y: 512–1023), so the model always knows which one it writes.
With `rel` (relative), only a stroke's start point is absolute; its three control points are written as
offsets from the start (dx: 1024–1535, dy: 1536–2047, 2 px bins over [-512, 512), which hold all but
0.0023% of the offsets), so one shape of stroke is the same tokens wherever it is drawn.
With a part `order` (ControlSketch-Part's stroke→part labels, 2–5 parts per sketch, listed head first for
animals), every stroke is preceded by its part token P0–P4 (9 tokens a stroke, 289 a sketch) and the
strokes are reordered:
  - "part":   all strokes of part 0, then part 1, … (each part finished before the next);
  - "layout": first the most important stroke of every part (the whole figure is laid in), then the
              remaining strokes part by part — how a person blocks in a drawing before refining each region;
  - "chain":  parts as in "part"; inside a part, the authors' first stroke, then each time the stroke with an end
              nearest to anything drawn so far, turned so that end is its start (same curve, drawn backwards).
              In the authors' order 41% of the held-out cats' strokes 2–32 start within 8 px of an earlier
              stroke; in this order 68% (measured 2026-10-08), so "a new stroke grows out of a drawn line" is
              the usual case the model sees instead of the exception.
In "part" and "layout" the authors' order (coarse first) is kept within a part. "orig" is the authors' order
with no part tokens.
With `oncurve` (part orders only), a stroke is written as the 4 points the curve passes through at t = 0, 1/3,
2/3, 1 instead of its control points. The two are the same curve (4 points at fixed t determine a cubic
Bézier exactly), and the drawn curve stays within 1.27 px on average of the original after 2 px binning
(control points: 1.17 px). The difference is what the model sees: a control point usually lies off the line,
so where an earlier stroke runs is never in the context, while 68% of the held-out cats' strokes that start on
an earlier stroke start in its middle, not at an end (measured 2026-10-08). With on-curve points, 52% of those
starts lie within 8 px of a point the model has already written (control points: 37.5%).
The GPT (llm/model.py) learns p(token | class, tokens before it); sampling gives a new sketch each time.

    python -m sellanything_ml.sketch.linegpt data                # parquet → token arrays, round-trip check
    python -m sellanything_ml.sketch.linegpt train --classes cat  # train (GPU); best weights by validation loss
    python -m sellanything_ml.sketch.linegpt sample --classes cat --n 24 --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data import ML_DIR

RAW_DIR = ML_DIR / "data" / "raw" / "sketch_extra" / "controlsketch_part" / "data"
DATA_DIR = ML_DIR / "data" / "processed" / "linegpt"
RUN_DIR = ML_DIR / "runs" / "linegpt"
SPLITS = ("train", "validation", "test")

CANVAS = 512
LO, BIN, NB = -256, 2, 512          # bins of BIN px from LO; NB per axis
X0, Y0 = 0, NB                      # first x token, first y token
FIRST_CLASS = 2 * NB                # class tokens follow the coordinate tokens (absolute layout)
DLO = -512                          # relative layout: offset bins of BIN px from DLO
DX0, DY0 = 2 * NB, 3 * NB           # first dx token, first dy token
FIRST_CLASS_REL = 4 * NB
STROKES, PER_STROKE = 32, 8
SEQ = 1 + STROKES * PER_STROKE      # 257
FIRST_PART = 2 * NB                 # part orders: part tokens P0–P4 follow the coordinate tokens
NPARTS = 5                          # the most parts any sketch has (all splits, measured 2026-10-07)
FIRST_CLASS_PART = FIRST_PART + NPARTS
ORDERS = ("orig", "part", "layout", "chain")
ONCURVE_T = np.array([0.0, 1 / 3, 2 / 3, 1.0])
BERNSTEIN = np.stack([(1 - ONCURVE_T) ** 3, 3 * (1 - ONCURVE_T) ** 2 * ONCURVE_T,
                      3 * (1 - ONCURVE_T) * ONCURVE_T ** 2, ONCURVE_T ** 3], axis=1)     # control points → on-curve points


def seq_len(order: str = "orig") -> int:
    return SEQ if order == "orig" else 1 + STROKES * (PER_STROKE + 1)


# --- coordinates ↔ tokens -------------------------------------------------------------------------

def quantize(v: np.ndarray) -> np.ndarray:
    return np.clip((v - LO) // BIN, 0, NB - 1).astype(np.int64)


def dequantize(b: np.ndarray) -> np.ndarray:
    return LO + b * BIN + BIN / 2          # bin centre: at most 1 px from any coordinate inside the range


def first_class(rel: bool, order: str = "orig") -> int:
    if order != "orig":
        return FIRST_CLASS_PART
    return FIRST_CLASS_REL if rel else FIRST_CLASS


def slot_ranges(rel: bool) -> list[tuple[int, int]]:
    """The token range each of a stroke's 8 slots must hold."""
    if not rel:
        return [(X0, X0 + NB), (Y0, Y0 + NB)] * 4
    return [(X0, X0 + NB), (Y0, Y0 + NB)] + [(DX0, DX0 + NB), (DY0, DY0 + NB)] * 3


def encode(path_data: np.ndarray, class_token: int, rel: bool = False) -> np.ndarray:
    """(32, 8) control points → 257 tokens."""
    p = np.asarray(path_data, dtype=np.float64)
    q = quantize(p)
    q[:, 1::2] += Y0                       # y coordinates use the y tokens
    if rel:
        d = p[:, 2:] - np.tile(p[:, :2], 3)
        dq = np.clip((d - DLO) // BIN, 0, NB - 1).astype(np.int64)
        q[:, 2:] = dq + np.tile([DX0, DY0], 3)
    return np.concatenate([[class_token], q.reshape(-1)])


def decode(tokens: np.ndarray, rel: bool = False) -> tuple[np.ndarray, int]:
    """Tokens after the class token → (strokes, 8) control points, and how many tokens were in the wrong
    place (an x token where a y belongs, a class token, …); a stroke with one is left out whole."""
    t = np.asarray(tokens)[:STROKES * PER_STROKE]
    n = len(t) // PER_STROKE
    t = t[:n * PER_STROKE].reshape(n, PER_STROKE)
    lo = np.array([a for a, _ in slot_ranges(rel)])
    hi = np.array([b for _, b in slot_ranges(rel)])
    wrong = (t < lo) | (t >= hi)
    keep = ~wrong.any(axis=1)
    b = (t - lo)[keep].astype(np.float64)
    pts = dequantize(b)
    if rel:
        pts[:, 2:] = DLO + b[:, 2:] * BIN + BIN / 2 + np.tile(pts[:, :2], 3)
    return pts, int(wrong.sum())


def stroke_order(assign: np.ndarray, order: str) -> np.ndarray:
    """The order the 32 strokes are written in (indices into the authors' order)."""
    assign = np.asarray(assign)
    idx = np.arange(len(assign))
    if order == "orig":
        return idx
    by_part = [idx[assign == k] for k in range(NPARTS)]                    # each already coarse first
    if order == "part":
        return np.concatenate(by_part)
    firsts = [g[0] for g in by_part if len(g)]
    return np.concatenate([firsts] + [g[1:] for g in by_part])


def stroke_plan(path_data: np.ndarray, assign: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray]:
    """The order the strokes are written in (indices into the authors' order) and which are drawn backwards."""
    if order != "chain":
        perm = stroke_order(assign, order)
        return perm, np.zeros(len(perm), bool)
    p = np.asarray(path_data, dtype=np.float64)
    ends = p.reshape(-1, 4, 2)[:, [0, 3]]                                   # (n, 2 ends, xy)
    perm, back, drawn = [], [], []
    for k in range(NPARTS):
        rest = [i for i in range(len(p)) if assign[i] == k]
        while rest:
            if not perm or assign[perm[-1]] != k:
                i, e = rest.pop(0), 0                                       # a part opens with the authors' first
            else:
                d = np.sqrt(((ends[rest][:, :, None, :] - np.concatenate(drawn)[None, None]) ** 2).sum(-1)).min(-1)
                j, e = np.unravel_index(np.argmin(d), d.shape)              # d: (remaining strokes, 2 ends)
                i = rest.pop(j)
            perm.append(i)
            back.append(bool(e == 1))
            drawn.append(bezier_points(p[i]))
    return np.array(perm), np.array(back)


def planned(path_data: np.ndarray, plan: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Control points in writing order, the backwards strokes reversed."""
    perm, back = plan
    p = np.asarray(path_data, dtype=np.float64)[perm].reshape(-1, 4, 2).copy()
    p[back] = p[back, ::-1]
    return p.reshape(-1, 8)


def to_oncurve(path_data: np.ndarray) -> np.ndarray:
    """(n, 8) control points → (n, 8) points on each curve at ONCURVE_T."""
    p = np.asarray(path_data, dtype=np.float64).reshape(-1, 4, 2)
    return np.einsum("ij,njk->nik", BERNSTEIN, p).reshape(-1, 8)


def from_oncurve(points: np.ndarray) -> np.ndarray:
    """Inverse of to_oncurve."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 4, 2)
    return np.einsum("ij,njk->nik", np.linalg.inv(BERNSTEIN), p).reshape(-1, 8)


def encode_parts(path_data: np.ndarray, assign: np.ndarray, class_token: int, order: str,
                 oncurve: bool = False, plan: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """(32, 8) control points and their part labels → 289 tokens: [class] then [P_k] x0 y0 … y3 per stroke.
    A plan from stroke_plan can be passed in to reuse it (a mirror image has the same one)."""
    plan = plan or stroke_plan(path_data, assign, order)
    p = planned(path_data, plan)
    q = quantize(to_oncurve(p) if oncurve else p)
    q[:, 1::2] += Y0
    rows = np.concatenate([FIRST_PART + np.asarray(assign)[plan[0], None], q], axis=1)
    return np.concatenate([[class_token], rows.reshape(-1)])


def decode_parts(tokens: np.ndarray, oncurve: bool = False) -> tuple[np.ndarray, int, np.ndarray]:
    """Part-order tokens after the class token → strokes (in drawing order), misplaced tokens, part of each."""
    t = np.asarray(tokens)[:STROKES * (PER_STROKE + 1)]
    n = len(t) // (PER_STROKE + 1)
    t = t[:n * (PER_STROKE + 1)].reshape(n, PER_STROKE + 1)
    lo = np.array([FIRST_PART] + [a for a, _ in slot_ranges(False)])
    hi = np.array([FIRST_PART + NPARTS] + [b for _, b in slot_ranges(False)])
    wrong = (t < lo) | (t >= hi)
    keep = ~wrong.any(axis=1)
    pts = dequantize((t[:, 1:] - lo[1:])[keep].astype(np.float64))
    if oncurve:
        pts = from_oncurve(pts)
    return pts, int(wrong.sum()), (t[keep, 0] - FIRST_PART)


def mirror(path_data: np.ndarray) -> np.ndarray:
    """The same sketch flipped left–right."""
    p = np.asarray(path_data, dtype=np.float64).copy()
    p[:, 0::2] = CANVAS - p[:, 0::2]
    return p


# --- drawing --------------------------------------------------------------------------------------

def bezier_points(stroke: np.ndarray, n: int = 40) -> np.ndarray:
    p0, p1, p2, p3 = stroke.reshape(4, 2)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def render(strokes: np.ndarray, side: int = 256, width: float = 5.0):
    """Strokes on the 512 canvas → a white PIL image of side×side with black lines (width in canvas px)."""
    from PIL import Image, ImageDraw
    scale = 4                                            # draw large, then shrink: smooth lines
    s = side * scale / CANVAS
    img = Image.new("L", (side * scale, side * scale), 255)
    d = ImageDraw.Draw(img)
    w = max(1, round(width * s))
    for stroke in strokes:
        pts = [tuple(p) for p in bezier_points(stroke) * s]
        d.line(pts, fill=0, width=w, joint="curve")
        for p in (pts[0], pts[-1]):                      # round caps, as in the SVGs
            d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=0)
    return img.resize((side, side), Image.LANCZOS)


# --- data -----------------------------------------------------------------------------------------

def _read(split: str):
    import pandas as pd
    return pd.read_parquet(RAW_DIR / f"{split}-00000-of-00001.parquet")


def categories() -> list[str]:
    """The 15 train classes first (as in the train split, sorted), then the 85 test classes (sorted)."""
    train = sorted(set(_read("train").category))
    test = sorted(set(_read("test").category))
    return train + test


def stage_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cats = categories()
    index = {c: i for i, c in enumerate(cats)}
    (DATA_DIR / "categories.json").write_text(json.dumps(cats) + "\n")
    report = {}
    for split in SPLITS:
        df = _read(split)
        pts = np.stack([np.stack(p) for p in df.path_data]).astype(np.float64)        # (N, 32, 8)
        cls = np.array([index[c] for c in df.category])
        toks = np.stack([encode(p, FIRST_CLASS + c) for p, c in zip(pts, cls)]).astype(np.int16)
        flip = np.stack([encode(mirror(p), FIRST_CLASS + c) for p, c in zip(pts, cls)]).astype(np.int16)
        rtoks = np.stack([encode(p, FIRST_CLASS_REL + c, rel=True) for p, c in zip(pts, cls)]).astype(np.int16)
        rflip = np.stack([encode(mirror(p), FIRST_CLASS_REL + c, rel=True) for p, c in zip(pts, cls)]).astype(np.int16)
        assign = np.stack([np.asarray(a) for a in df.path_assignment]).astype(np.int64)
        extra = {}
        plans = {order: [stroke_plan(p, a, order) for p, a in zip(pts, assign)] for order in ORDERS[1:]}
        for order in ORDERS[1:]:
            for on, tag in ((False, ""), (True, "_on")):
                extra[f"tokens_{order}{tag}"] = np.stack([encode_parts(p, a, FIRST_CLASS_PART + c, order, on, pl)
                                                          for p, a, c, pl in zip(pts, assign, cls, plans[order])]).astype(np.int16)
                extra[f"mirrored_{order}{tag}"] = np.stack([encode_parts(mirror(p), a, FIRST_CLASS_PART + c, order, on, pl)
                                                            for p, a, c, pl in zip(pts, assign, cls, plans[order])]).astype(np.int16)
        np.savez_compressed(DATA_DIR / f"{split}.npz", tokens=toks, mirrored=flip, tokens_rel=rtoks,
                            mirrored_rel=rflip, labels=cls, ids=np.array(df.sketch_id), assign=assign, **extra)
        # round trip: tokens back to control points
        report[split] = {"sketches": len(df), "values": int(pts.size)}
        for name, rows, rel in (("abs", toks, False), ("rel", rtoks, True)):
            back = np.stack([decode(t[1:], rel)[0] for t in rows])
            err = np.abs(back - pts)
            report[split][name] = {"error_px_p99_99": float(np.percentile(err, 99.99)),
                                   "mean_error_px": round(float(err.mean()), 4),
                                   "values_off_by_more_than_2px": int((err > 2).sum())}
        for order in ORDERS[1:]:
            back = [decode_parts(t[1:]) for t in extra[f"tokens_{order}"]]
            perms = [pl[0] for pl in plans[order]]
            want = np.stack([planned(p, pl) for p, pl in zip(pts, plans[order])])
            err = np.abs(np.stack([b[0] for b in back]) - want)
            parts_ok = all((b[2] == a[q]).all() for b, a, q in zip(back, assign, perms))
            report[split][order] = {"error_px_p99_99": float(np.percentile(err, 99.99)),
                                    "parts_round_trip": parts_ok,
                                    "part_switches_median": float(np.median([(np.diff(a[q]) != 0).sum()
                                                                             for a, q in zip(assign, perms)]))}
            # on-curve: compare the drawn curves, since binning error is amplified when solving for control points
            back = np.stack([decode_parts(t[1:], oncurve=True)[0] for t in extra[f"tokens_{order}_on"]])
            dist = np.linalg.norm(np.stack([bezier_points(b, 41) for b in back.reshape(-1, 8)])
                                  - np.stack([bezier_points(w, 41) for w in want.reshape(-1, 8)]), axis=-1)
            report[split][f"{order}_on"] = {"curve_error_px_mean": round(float(dist.max(1).mean()), 3),
                                            "curve_error_px_p99": round(float(np.percentile(dist.max(1), 99)), 3)}
        print(split, json.dumps(report[split]), flush=True)
    (DATA_DIR / "roundtrip.json").write_text(json.dumps(report, indent=1) + "\n")


def load(split: str) -> dict[str, np.ndarray]:
    z = np.load(DATA_DIR / f"{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


# --- training -------------------------------------------------------------------------------------

@dataclass
class Config:
    name: str = "cat_v1"
    classes: list = field(default_factory=lambda: ["cat"])
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 64
    epochs: int = 200
    warmup_steps: int = 100
    seed: int = 0
    abs_pos: bool = True             # cat_v1 (RoPE only) lost track of x/y in 21 of 24 samples
    rel: bool = False                # control points as offsets from the stroke's start
    init_from: str = ""              # run name whose best.pt to start from (fine-tuning)
    order: str = "orig"              # stroke order: "orig", "part" or "layout" (see the module docstring)
    pretrain: bool = False           # train on every class: train split (15) + test split (85 other classes)
    oncurve: bool = False            # strokes as 4 points on the curve instead of control points (part orders)


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def vocab_size(rel: bool = False, order: str = "orig") -> int:
    return first_class(rel, order) + len(categories())


def model_config(cfg: Config):
    from ..llm.model import GPTConfig
    return GPTConfig(vocab_size=vocab_size(cfg.rel, cfg.order), d_model=cfg.d_model, n_layers=cfg.n_layers,
                     n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, max_seq_len=seq_len(cfg.order), dropout=cfg.dropout,
                     abs_pos=cfg.abs_pos)


def class_rows(split: str, classes: list[str] | None, mirrored: bool, rel: bool = False,
               order: str = "orig", oncurve: bool = False) -> np.ndarray:
    """Token rows of the given classes (None: all); with mirrored=True the mirror images are added."""
    cats = json.loads((DATA_DIR / "categories.json").read_text())
    z = load(split)
    keep = np.ones(len(z["labels"]), bool) if classes is None else np.isin(z["labels"], [cats.index(c) for c in classes])
    suffix = "_rel" if rel else ("" if order == "orig" else f"_{order}") + ("_on" if oncurve else "")
    rows = z["tokens" + suffix][keep]
    if mirrored:
        rows = np.concatenate([rows, z["mirrored" + suffix][keep]])
    return rows.astype(np.int64)


def run_dir(cfg: Config) -> Path:
    return RUN_DIR / cfg.name


def _lr(cfg: Config, step: int, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))))


def evaluate_loss(model, rows: np.ndarray, dev, batch: int = 256) -> float:
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            t = torch.from_numpy(rows[i:i + batch]).to(dev)
            _, loss = model(t[:, :-1], t[:, 1:])
            tot += loss.item() * len(t)
            n += len(t)
    return tot / n


def stage_train(cfg: Config) -> Path:
    """Train on the train split of cfg.classes (and their mirror images); after every epoch, measure the
    loss on the validation split (never trained on) and keep the weights with the lowest one."""
    import torch
    from ..llm.model import GPT
    dev = device()
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.pretrain:      # the 85 test-split classes are never evaluated on, so they can be trained on
        train = np.concatenate([class_rows("train", None, True, cfg.rel, cfg.order, cfg.oncurve),
                                class_rows("test", None, True, cfg.rel, cfg.order, cfg.oncurve)])
        val = class_rows("validation", None, False, cfg.rel, cfg.order, cfg.oncurve)
    else:
        train = class_rows("train", cfg.classes, True, cfg.rel, cfg.order, cfg.oncurve)
        val = class_rows("validation", cfg.classes, False, cfg.rel, cfg.order, cfg.oncurve)
    torch.manual_seed(cfg.seed)
    model = GPT(model_config(cfg)).to(dev)
    if cfg.init_from:
        model.load_state_dict(torch.load(RUN_DIR / cfg.init_from / "best.pt", map_location=dev)["model"])
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    steps = len(train) // cfg.batch
    total = steps * cfg.epochs
    (out / "config.json").write_text(json.dumps({"config": asdict(cfg), "model": asdict(model_config(cfg)),
                                                 "parameters": model.num_params(), "train_rows": len(train),
                                                 "val_rows": len(val), "steps_per_epoch": steps}, indent=1) + "\n")
    print(f"{cfg.name}: {model.num_params():,} parameters, {len(train)} train rows, {len(val)} val rows, "
          f"{steps} steps/epoch", flush=True)
    log = out / "metrics.jsonl"
    log.write_text("")
    best, rng, step = float("inf"), np.random.default_rng(cfg.seed), 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train))
        t0, run = time.time(), 0.0
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = _lr(cfg, step, total)
            t = torch.from_numpy(train[order[s * cfg.batch:(s + 1) * cfg.batch]]).to(dev)
            _, loss = model(t[:, :-1], t[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
        val_loss = evaluate_loss(model, val, dev)
        rec = {"epoch": epoch + 1, "train_loss": round(run / steps, 4), "val_loss": round(val_loss, 4),
               "lr": round(_lr(cfg, step, total), 7), "seconds": round(time.time() - t0, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(json.dumps(rec), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": asdict(cfg), **rec}, out / "best.pt")
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out / "last.pt")
    return out / "best.pt"


def load_model(path: Path, dev=None):
    import torch
    from ..llm.model import GPT
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**({"abs_pos": False} | st["config"]))      # runs saved before abs_pos existed had none
    st["config"] = asdict(cfg)
    model = GPT(model_config(cfg))
    model.load_state_dict(st["model"])
    model.line_cfg = cfg                                      # draw() reads the token layout from it
    return model.to(dev or "cpu").eval(), cfg


def draw(model, class_name: str, n: int, temperature: float, seed: int, dev=None) -> list[tuple[np.ndarray, int]]:
    """n new sketches of a class: (strokes, misplaced tokens) each."""
    import torch
    cats = json.loads((DATA_DIR / "categories.json").read_text())
    torch.manual_seed(seed)
    dev = dev or next(model.parameters()).device
    cfg = getattr(model, "line_cfg", None) or Config()
    idx = torch.full((n, 1), first_class(cfg.rel, cfg.order) + cats.index(class_name), dtype=torch.long, device=dev)
    out = model.generate(idx, seq_len(cfg.order) - 1, temperature=temperature)
    if cfg.order != "orig":
        return [decode_parts(row[1:].cpu().numpy(), cfg.oncurve)[:2] for row in out]
    return [decode(row[1:].cpu().numpy(), cfg.rel) for row in out]


def sheet(strokes: list[np.ndarray], cols: int = 6, side: int = 256, labels: list[str] | None = None):
    from PIL import Image, ImageDraw
    rows = (len(strokes) + cols - 1) // cols
    img = Image.new("L", (cols * side, rows * side), 200)
    d = ImageDraw.Draw(img)
    for k, s in enumerate(strokes):
        img.paste(render(s, side), ((k % cols) * side + 1, (k // cols) * side + 1))
        if labels:
            d.text(((k % cols) * side + 6, (k // cols) * side + 4), labels[k], fill=0)
    return img


# --- evaluation -----------------------------------------------------------------------------------
# The judge is a CNN that never sees generated sketches: trained on the train split of the 15 classes,
# checked on the validation split. Copying is measured by how close each sketch is to its nearest
# training sketch, next to the same number for real validation sketches (which the model never saw).

JUDGE_SIDE = 64


def raster(strokes: np.ndarray) -> np.ndarray:
    """A sketch as a 64×64 float image, ink = 1, for the judge and the distance measures."""
    img = render(strokes, side=JUDGE_SIDE * 2, width=8.0).resize((JUDGE_SIDE, JUDGE_SIDE))
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0


def rasters(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Every sketch of a split (15 train classes only) as 64×64 images, cached."""
    path = DATA_DIR / f"{split}_raster{JUDGE_SIDE}.npz"
    if path.exists():
        z = np.load(path)
        return z["images"], z["labels"]
    z = load(split)
    keep = z["labels"] < 15
    imgs = np.stack([raster(decode(t[1:])[0]) for t in z["tokens"][keep]]).astype(np.float16)
    np.savez_compressed(path, images=imgs, labels=z["labels"][keep])
    return imgs, z["labels"][keep]


def build_judge():
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 15))


def stage_judge(epochs: int = 12, dev_name: str = "cpu") -> Path:
    import torch
    import torch.nn.functional as F
    dev = torch.device(dev_name)
    xtr, ytr = rasters("train")
    xva, yva = rasters("validation")
    torch.manual_seed(0)
    net = build_judge().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    out = RUN_DIR / "judge"
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(xtr))
        for i in range(0, len(order), 128):
            ix = order[i:i + 128]
            x = torch.from_numpy(xtr[ix].astype(np.float32)).unsqueeze(1).to(dev)
            flip = torch.rand(len(x), device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            loss = F.cross_entropy(net(x), torch.from_numpy(ytr[ix]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = judge_accuracy(net, xva, yva, dev)
        print(json.dumps({"epoch": ep + 1, "val_top1": round(acc, 4)}), flush=True)
    torch.save({"model": net.state_dict(), "val_top1": acc}, out / "judge.pt")
    (out / "judge.json").write_text(json.dumps({"val_top1": acc, "epochs": epochs,
                                                "val_sketches": len(yva)}, indent=1) + "\n")
    return out / "judge.pt"


def judge_probs(net, imgs: np.ndarray, dev="cpu") -> np.ndarray:
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 512):
            x = torch.from_numpy(np.asarray(imgs[i:i + 512], dtype=np.float32)).unsqueeze(1).to(dev)
            out.append(net(x).softmax(-1).cpu().numpy())
    return np.concatenate(out)


def judge_accuracy(net, imgs, labels, dev="cpu") -> float:
    return float((judge_probs(net, imgs, dev).argmax(1) == labels).mean())


def load_judge():
    import torch
    net = build_judge()
    net.load_state_dict(torch.load(RUN_DIR / "judge" / "judge.pt", map_location="cpu")["model"])
    return net.eval()


def nearest(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every image in a, the index of and distance to its nearest image in b (mean absolute pixel
    difference after a 3×3 blur, so a line moved by a pixel or two still counts as close)."""
    from scipy.ndimage import uniform_filter

    def blur(x: np.ndarray) -> np.ndarray:
        return uniform_filter(np.asarray(x, dtype=np.float32), size=(1, 3, 3)).reshape(len(x), -1)

    A, B = blur(a), blur(b)
    d = np.stack([np.abs(B - row).mean(1) for row in A])
    return d.argmin(1), d.min(1)


def diversity(imgs: np.ndarray, n: int = 60) -> float:
    """Mean pixel difference between pairs of images (first n): how different the sketches are."""
    f = np.asarray(imgs[:n], dtype=np.float32).reshape(min(n, len(imgs)), -1)
    return float(np.mean([np.abs(f[i] - f[j]).mean() for i in range(len(f)) for j in range(i + 1, len(f))]))


def stage_eval(name: str, class_name: str = "cat", temperatures=(1.0, 0.8), n: int = 100, seed: int = 1) -> dict:
    """Score a run's best weights with the judge, next to the real held-out sketches of the class:
    how often the judge sees the class, misplaced tokens, distance to the nearest training sketch
    (copying), and diversity. Also saves a sheet of the first 24 sketches at each temperature."""
    cats = json.loads((DATA_DIR / "categories.json").read_text())
    ci = cats.index(class_name)
    judge = load_judge()
    xva, yva = rasters("validation")
    xtr, ytr = rasters("train")
    real, train = xva[yva == ci].astype(np.float32), xtr[ytr == ci].astype(np.float32)
    report = {"real_heldout": {"judge_says_class": round(float((judge_probs(judge, real).argmax(1) == ci).mean()), 3),
                               "nearest_train_median": round(float(np.median(nearest(real, train)[1])), 4),
                               "diversity": round(diversity(real), 4), "n": len(real)}}
    model, cfg = load_model(run_dir(Config(name=name)) / "best.pt", "cpu")
    for t in temperatures:
        res = draw(model, class_name, n, t, seed=seed, dev="cpu")
        imgs = np.stack([raster(r[0]) for r in res])
        report[f"t{t}"] = {"judge_says_class": round(float((judge_probs(judge, imgs).argmax(1) == ci).mean()), 3),
                           "misplaced_tokens_per_sketch": round(sum(r[1] for r in res) / n, 2),
                           "nearest_train_median": round(float(np.median(nearest(imgs, train)[1])), 4),
                           "diversity": round(diversity(imgs), 4), "n": n}
        sheet([r[0] for r in res[:24]]).save(run_dir(cfg) / f"eval_{class_name}_t{t}.png")
    (run_dir(cfg) / f"eval_{class_name}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return report


def stage_score(name: str, class_name: str = "cat", t: float = 0.8, seeds=(2, 3, 4, 5, 6), per_seed: int = 100) -> dict:
    """The judge's rate on 500 new sketches (5 seeds × 100, none of them the seed stage_eval uses), with its
    standard error: 100 sketches give about ±5 points, too few to tell close models apart."""
    cats = json.loads((DATA_DIR / "categories.json").read_text())
    ci = cats.index(class_name)
    judge = load_judge()
    model, cfg = load_model(run_dir(Config(name=name)) / "best.pt", "cpu")
    hits, bad = [], 0
    for sd in seeds:
        res = draw(model, class_name, per_seed, t, seed=sd, dev="cpu")
        bad += sum(r[1] for r in res)
        hits += list(judge_probs(judge, np.stack([raster(r[0]) for r in res])).argmax(1) == ci)
    p = float(np.mean(hits))
    rec = {"judge_says_class": round(p, 4), "se": round(float(np.sqrt(p * (1 - p) / len(hits))), 4), "n": len(hits),
           "t": t, "seeds": list(seeds), "misplaced_tokens_per_sketch": round(bad / len(hits), 2)}
    (run_dir(cfg) / f"score_{class_name}_t{t}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(name, json.dumps(rec), flush=True)
    return rec


def stage_compare(rows: list[tuple[str, str]], class_name: str = "cat", per_row: int = 6, t: float = 0.8,
                  seed: int = 1, out: Path | None = None) -> Path:
    """One row per model (label, run name), first per_row sketches drawn exactly as stage_eval draws them
    (same seed and temperature), with the judge score from that run's eval_<class>.json; the first row is
    real held-out sketches of the class. A run name of "" means the real sketches."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 20)
    small = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 16)
    cats = json.loads((DATA_DIR / "categories.json").read_text())
    side, left = 200, 260
    img = Image.new("L", (left + per_row * side, len(rows) * side), 255)
    d = ImageDraw.Draw(img)
    for r, (label, name) in enumerate(rows):
        if name:
            model, cfg = load_model(run_dir(Config(name=name)) / "best.pt", "cpu")
            sketches = [x[0] for x in draw(model, class_name, 100, t, seed=seed, dev="cpu")[:per_row]]
            ev = json.loads((run_dir(Config(name=name)) / f"eval_{class_name}.json").read_text())[f"t{t}"]
            score = f"評審認得出:{ev['judge_says_class']:.0%}"
        else:
            z = load("validation")
            rows_real = z["tokens"][z["labels"] == cats.index(class_name)][:per_row]
            sketches = [decode(x[1:])[0] for x in rows_real]
            ev = json.loads((run_dir(Config(name=rows[1][1])) / f"eval_{class_name}.json").read_text())["real_heldout"]
            score = f"評審認得出:{ev['judge_says_class']:.0%}"
        y = r * side
        for k, line in enumerate(label.split("\n")):
            d.text((12, y + 60 + k * 28), line, fill=0, font=font)
        d.text((12, y + 60 + 28 * len(label.split("\n")) + 6), score, fill=80, font=small)
        for k, sk in enumerate(sketches):
            img.paste(render(sk, side - 8), (left + k * side + 4, y + 4))
        d.line([(0, y + side - 1), (img.width, y + side - 1)], fill=180)
    out = out or RUN_DIR / f"compare_{class_name}.png"
    img.save(out)
    return out


# --- CLI ------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["data", "train", "sample", "judge", "eval", "score"])
    ap.add_argument("--name", default="cat_v1")
    ap.add_argument("--classes", default="cat")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-abs-pos", action="store_true")
    ap.add_argument("--rel", action="store_true")
    ap.add_argument("--order", default="orig", choices=ORDERS)
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--oncurve", action="store_true")
    a = ap.parse_args()
    if a.oncurve and (a.order == "orig" or a.rel):
        ap.error("--oncurve needs --order part or layout, without --rel")
    cfg = Config(name=a.name, classes=a.classes.split(","), abs_pos=not a.no_abs_pos, rel=a.rel,
                 pretrain=a.pretrain, init_from=a.init_from, order=a.order, oncurve=a.oncurve, seed=a.seed)
    for key, val in (("epochs", a.epochs), ("lr", a.lr), ("d_model", a.d_model), ("n_layers", a.layers)):
        if val:
            setattr(cfg, key, val)
    if a.stage == "data":
        stage_data()
    elif a.stage == "train":
        stage_train(cfg)
    elif a.stage == "judge":
        stage_judge()
    elif a.stage == "eval":
        stage_eval(a.name, cfg.classes[0])
    elif a.stage == "score":
        stage_score(a.name, cfg.classes[0], a.temperature if a.temperature != 1.0 else 0.8)
    elif a.stage == "sample":
        model, cfg = load_model(run_dir(cfg) / "best.pt", device())
        res = draw(model, cfg.classes[0], a.n, a.temperature, a.seed)
        bad = sum(r[1] for r in res)
        path = run_dir(cfg) / f"sample_t{a.temperature}_s{a.seed}.png"
        sheet([r[0] for r in res]).save(path)
        print(f"{a.n} sketches, misplaced tokens {bad}, -> {path}")


if __name__ == "__main__":
    main()
