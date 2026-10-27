"""Partial-drawing classifier for live guessing (docs/design-v2.md §4).

    python -m sellanything_ml.sketch.partial finetune     # GPU: fine-tune the v2 stroke Transformer on prefixes
    python -m sellanything_ml.sketch.partial evaluate     # validation curves: by share of strokes and by stroke index
    python -m sellanything_ml.sketch.partial export       # copy the chosen model to ml/checkpoints/sketch_partial/

Fine-tuning starts from the sketch v2 stroke Transformer (`runs/sketch_v2/seq_best.pt`) and trains every
layer on drawings cut at a stroke boundary: for each training drawing a completeness fraction f is drawn
uniformly from (0, 1] and the drawing keeps its first max(1, ceil(f × n_strokes)) strokes, so complete
drawings are included. The label never changes. Only live guesses use this model; the final guess
("Sell it!") keeps the v2 ensemble, so it cannot get worse.

Everything is measured on the validation split; the test split is not touched.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import pipeline
from .data import ML_DIR

RUN = ML_DIR / "runs" / "sketch_v2"
OUT = ML_DIR / "checkpoints" / "sketch_partial"
EPOCHS = 3
LR = 3e-4
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
STROKE_INDEX = (1, 2, 3, 4, 5)
EVAL_PER_CLASS = 20


def cut(pts: torch.Tensor, lens: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """New lengths that keep the first k[i] strokes of each drawing (k ≥ 1; more than it has = all).
    A stroke ends at a point whose third value is 1."""
    L = pts.shape[1]
    valid = torch.arange(L, device=pts.device)[None] < lens[:, None]
    ends = (pts[..., 2] == 1) & valid
    n = ends.sum(1).clamp(min=1)
    k = torch.minimum(k.clamp(min=1), n)
    reached = ends.long().cumsum(1) >= k[:, None]                       # True from the k-th end on
    first = torch.where(reached.any(1), reached.float().argmax(1), lens - 1)
    return torch.minimum(first + 1, lens)


def strokes_of(pts: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    L = pts.shape[1]
    valid = torch.arange(L, device=pts.device)[None] < lens[:, None]
    return (((pts[..., 2] == 1) & valid).sum(1)).clamp(min=1)


def by_fraction(pts, lens, f: torch.Tensor) -> torch.Tensor:
    """Keep max(1, ceil(f × n_strokes)) strokes (f per drawing)."""
    n = strokes_of(pts, lens)
    return cut(pts, lens, torch.ceil(f * n).long().clamp(min=1))


def finetune(cfg: pipeline.Config | None = None, epochs: int = EPOCHS, lr: float = LR, device=None) -> Path:
    cfg = cfg or pipeline.Config()
    device = device or pipeline._device()
    sp = pipeline.splits(cfg)
    load, lengths = pipeline._seq_loader(cfg, device)
    lens_all = np.asarray(lengths).astype(np.int64)
    model = pipeline.build_seq(cfg).to(device)
    model.load_state_dict(torch.load(RUN / "seq_best.pt", map_location=device, weights_only=False)["model"])

    def step(m, batch, gen):
        pts, lens, y = batch
        f = 1 - torch.rand(len(lens), generator=gen).to(lens.device)      # (0, 1]
        new = by_fraction(pts, lens, f)
        L = int(new.max())
        with pipeline._autocast(device):
            return F.cross_entropy(m(pipeline.seq_features(pts[:, :L], new, gen), new).float(), y)

    def order(epoch):
        rng = np.random.default_rng(cfg.seed * 1000 + 900 + epoch)
        return pipeline.bucket_order(rng.permutation(sp["train"]), lens_all, cfg.batch, rng)

    # Validation: every drawing cut at its own fixed fraction, so successive epochs are comparable.
    val = sp["val"][np.argsort(lens_all[sp["val"]], kind="stable")]
    val_f = torch.from_numpy(1 - np.random.default_rng(cfg.seed + 77).random(len(val))).float()

    def evaluate(m):
        out = []
        m.eval()
        with torch.no_grad(), pipeline._autocast(device):
            for i in range(0, len(val), 2048):
                pts, lens, _ = pipeline.to_device(load(val[i:i + 2048]), device)
                new = by_fraction(pts, lens, val_f[i:i + 2048].to(device))
                L = int(new.max())
                out.append(F.log_softmax(m(pipeline.seq_features(pts[:, :L], new, None), new).float(), -1).cpu())
        return pipeline.scores(torch.cat(out), pipeline.labels(cfg, val))

    def make_sched(o, total):
        warm = max(1, total // 50)
        return torch.optim.lr_scheduler.LambdaLR(
            o, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    return pipeline.train("partial", cfg, model, step, epochs,
                          lambda m: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=cfg.seq_weight_decay),
                          make_sched, order, load, evaluate, device)


def _rows_to_strokes(pts: np.ndarray, n: int) -> list[tuple[list[int], list[int]]]:
    """Packed points (x, y, end) back to Quick, Draw! simplified strokes, the first n points."""
    out, xs, ys = [], [], []
    for x, y, e in pts[:n]:
        xs.append(int(x))
        ys.append(int(y))
        if e:
            out.append((xs, ys))
            xs, ys = [], []
    if xs:
        out.append((xs, ys))
    return out


@torch.no_grad()
def evaluate(per_class: int = EVAL_PER_CLASS, boot: int = 1000, out: Path | None = None,
             tuned_path: Path | None = None) -> dict:
    """Top-1 / top-3 with bootstrap 95% CIs on a fixed validation sample, through the same code the
    server uses (raster + CNN v2, stroke points + Transformer), for three live-guess candidates:
    the v2 ensemble used now, the fine-tuned Transformer alone, and the ensemble with the fine-tuned
    Transformer in place of the original."""
    from .infer import SketchEnsemble, stroke_points
    from .raster import rasterize
    cfg = pipeline.Config()
    sp = pipeline.splits(cfg)
    p = pipeline._paths(cfg)
    n_all = len(pipeline.categories(cfg)) * cfg.per_class
    seq = np.memmap(p["seq"], dtype=np.uint8, mode="r", shape=(n_all, cfg.max_len, 3))
    lengths = np.memmap(p["lengths"], dtype=np.int16, mode="r", shape=(n_all,))
    rng = np.random.default_rng(cfg.seed + 2026)
    y_all = pipeline.labels(cfg, sp["val"])
    pick = np.concatenate([rng.choice(sp["val"][y_all == c], per_class, replace=False)
                           for c in range(len(pipeline.categories(cfg)))])
    y = pipeline.labels(cfg, pick)
    ens = SketchEnsemble()
    tuned = pipeline.build_seq(cfg)
    tuned.load_state_dict(torch.load(tuned_path or RUN / "partial_best.pt", map_location="cpu", weights_only=False)["model"])
    tuned.eval()
    w_cnn, w_seq = ens.weights
    settings = [("share", f) for f in FRACTIONS] + [("stroke", s) for s in STROKE_INDEX]
    hits = {name: {s: [] for s in settings} for name in ("ensemble now", "tuned Transformer", "ensemble, tuned Transformer")}
    for idx, label in zip(pick, y):
        pts_full = np.asarray(seq[idx])
        n_pts = int(lengths[idx])
        strokes = _rows_to_strokes(pts_full, n_pts)
        n = len(strokes)
        for kind, v in settings:
            k = max(1, math.ceil(v * n)) if kind == "share" else min(int(v), n)
            prefix = strokes[:k]
            img = torch.from_numpy(rasterize(prefix, side=ens.side).astype(np.float32) / 255.0).view(1, 1, ens.side, ens.side)
            lp_cnn = ens.cnn(img).log_softmax(-1)
            pts, m = stroke_points(prefix, ens.max_len)
            pts_t, lens = torch.from_numpy(pts[None, :max(m, 1)]), torch.tensor([m])
            feats = pipeline.seq_features(pts_t, lens, None)
            lp_seq = ens.seq(feats, lens).log_softmax(-1)
            lp_tuned = tuned(feats, lens).log_softmax(-1)
            for name, lp in (("ensemble now", w_cnn * lp_cnn + w_seq * lp_seq), ("tuned Transformer", lp_tuned),
                             ("ensemble, tuned Transformer", w_cnn * lp_cnn + w_seq * lp_tuned)):
                top = lp[0].topk(3).indices.tolist()
                hits[name][(kind, v)].append((top[0] == label, label in top))
    report = {"per_class": per_class, "drawings": int(len(pick)), "bootstrap": boot, "results": {}}
    brng = np.random.default_rng(1)
    samples = [brng.integers(0, len(pick), len(pick)) for _ in range(boot)]
    for name, by in hits.items():
        report["results"][name] = {}
        for (kind, v), hs in by.items():
            a = np.array(hs, dtype=float)
            row = {}
            for j, metric in enumerate(("top1", "top3")):
                means = np.array([a[s, j].mean() for s in samples])
                row[metric] = [round(float(a[:, j].mean()), 4), round(float(np.percentile(means, 2.5)), 4),
                               round(float(np.percentile(means, 97.5)), 4)]
            report["results"][name][f"{kind} {v}"] = row
    out = out or RUN / "partial_eval.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


def export(candidate: str) -> None:
    """Copy the fine-tuned Transformer (and, for the ensemble candidate, the weight) for the server."""
    OUT.mkdir(parents=True, exist_ok=True)
    st = torch.load(RUN / "partial_best.pt", map_location="cpu", weights_only=False)
    torch.save({"model": st["model"], "epoch": st["epoch"]}, OUT / "seq.pt")
    meta = json.loads((ML_DIR / "checkpoints" / "sketch_v2" / "meta.json").read_text())
    meta.update({"mode": candidate, "source": str((RUN / "partial_best.pt").relative_to(ML_DIR)),
                 "val_partial": json.loads((RUN / "partial_eval.json").read_text())["results"][candidate]})
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"exported {candidate} to {OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "finetune":
        print(finetune())
    elif cmd == "evaluate":
        print(json.dumps(evaluate()["results"], indent=1))
    elif cmd == "export" and len(sys.argv) == 3:
        export(sys.argv[2])
    else:
        raise SystemExit("usage: python -m sellanything_ml.sketch.partial finetune | evaluate | export <candidate>")
