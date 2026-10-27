"""Evaluate the drawing GPT (sketch/drawgpt.py): can people's classifier recognise what it draws,
does it copy training drawings, and how varied are its drawings (docs/design-v2.md §5).

Every number is measured against the same quantity for real drawings from the validation split, so
"good" means "close to what people draw", not a fixed threshold:

  1. Recognisability: each drawing goes through exactly the game's path (simplify, then the
     CNN + Transformer ensemble that serves `classify`); top-1 / top-3 against the asked category.
     The same drawings are also scored by the v1 CNN (`v1_top1` / `v1_top3`), a separate model that
     is never used to tune anything, so comparisons between versions and settings stay honest.
  2. Copying: exact token-sequence matches against the category's training drawings, and the
     distance from each drawing's 48 x 48 bitmap to its nearest training bitmap.
  3. Variety: mean pairwise bitmap distance within a category.

Each temperature (and guidance, for a model trained with class dropout) is scored twice on the same
samples: as drawn, and after drawgpt.tidy (which only changes drawings that never reached [end]).

    python -m sellanything_ml.sketch.drawgpt_eval                       # runs/drawgpt_v1/eval_report.json
    python -m sellanything_ml.sketch.drawgpt_eval --version v2 --temperatures 0.6 --guidance 1 1.5 2 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import drawgpt, pipeline
from .infer import DEFAULT_CHECKPOINTS, SketchClassifier, SketchEnsemble
from .raster import Stroke, rasterize, simplify
from .strokes import load_cached


def _bitmap(simple: list[Stroke], side: int) -> np.ndarray:
    return rasterize(simple, side=side).astype(np.float32).ravel() / 255.0


def _pairwise_mean(x: np.ndarray) -> float:
    """Mean Euclidean distance over all pairs of rows."""
    d = np.sqrt(np.maximum(((x[:, None] - x[None]) ** 2).sum(-1), 0))
    n = len(x)
    return float(d.sum() / (n * (n - 1))) if n > 1 else 0.0


def _nearest(queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    q2, r2 = (queries ** 2).sum(1)[:, None], (refs ** 2).sum(1)[None]
    return np.sqrt(np.maximum(q2 + r2 - 2 * queries @ refs.T, 0)).min(1)


class Judge:
    """The game's classifier, the independent v1 CNN, and the bitmaps used for copying and variety."""

    def __init__(self, side: int):
        self.ens = SketchEnsemble()
        self.v1 = SketchClassifier(next(p for p in DEFAULT_CHECKPOINTS if p.exists()))
        self.side = side

    def score(self, strokes: list[Stroke], target: int) -> tuple[int, int, int, int, np.ndarray]:
        """Game classifier top-1 / top-3 hit, v1 CNN top-1 / top-3 hit, bitmap."""
        simple = simplify(strokes)
        if not simple:
            return 0, 0, 0, 0, np.zeros(self.side * self.side, np.float32)
        top = self.ens.logits_simplified(simple).topk(3).indices.tolist()
        name = self.ens.categories[target]
        v1 = [p["label"] for p in self.v1.predict(simple, k=3)]
        return (int(top[0] == target), int(target in top), int(v1[0] == name), int(name in v1),
                _bitmap(simple, self.side))


def generated(model, centers, class_id: int, n: int, temperature: float, seed: int, guidance: float = 1.0,
              no_class: int | None = None):
    seqs = drawgpt.draw(model, class_id, n, temperature, seed=seed, guidance=guidance, no_class=no_class)
    return [(s, drawgpt.decode(s, centers)) for s in seqs]


def _group(t: float, g: float) -> str:
    return f"t{t}" if g == 1.0 else f"t{t}g{g}"


def evaluate(n: int, temperatures: list[float], seed: int, limit: int = 0, version: str = "v1",
             guidances: tuple[float, ...] = (1.0,)) -> dict:
    cfg = drawgpt.version(version)
    pcfg = drawgpt._pcfg(cfg)
    cats = drawgpt.categories(cfg)
    device = drawgpt._device()
    model = drawgpt.load_model(cfg, device)
    centers = np.load(drawgpt._paths(cfg)["dict"])
    judge = Judge(pcfg.side)
    assert judge.ens.categories == cats, "classifier and drawing GPT list the categories differently"
    toks, lens = drawgpt._data(cfg)
    sp = pipeline.splits(pcfg)
    images = np.memmap(pipeline._paths(pcfg)["images"], np.uint8, "r",
                       shape=(len(cats) * cfg.per_class, pcfg.side * pcfg.side))
    rng = np.random.default_rng(seed)
    settings = [(t, g) for t in temperatures for g in guidances]
    no_class = drawgpt.no_class_token(cfg)
    groups = ["real"] + [k for t, g in settings for k in (_group(t, g), _group(t, g) + "+tidy")]
    rows = {g: [] for g in groups}            # per drawing: class, top1, top3, nearest, ended, strokes
    variety = {g: [] for g in groups}
    t0 = time.time()
    used = cats[:limit] if limit else cats
    for c, name in enumerate(used):
        base = c * cfg.per_class
        train = sp["train"][(sp["train"] >= base) & (sp["train"] < base + cfg.per_class)]
        refs = np.asarray(images[np.sort(train)], np.float32) / 255.0
        seen = {bytes(toks[i, 2:lens[i]].astype(np.uint16)) for i in train if lens[i] > 0}
        val = sp["val"][(sp["val"] >= base) & (sp["val"] < base + cfg.per_class)]
        strokes_db = load_cached(name, cfg.per_class)
        batches = {"real": [(None, strokes_db.strokes(int(k - base))) for k in rng.choice(val, n, replace=False)]}
        for t, g in settings:                  # same seed for every setting: paired comparisons
            items = generated(model, centers, c, n, t, seed=seed * 1000 + c, guidance=g, no_class=no_class)
            batches[_group(t, g)] = items
            batches[_group(t, g) + "+tidy"] = [(s, drawgpt.tidy(st, bool(len(s) and s[-1] == drawgpt.END)))
                                               for s, st in items]
        for g, items in batches.items():
            bits = []
            for seq, strokes in items:
                top1, top3, v1_top1, v1_top3, bm = judge.score(strokes, c)
                bits.append(bm)
                ended = True if seq is None else bool(len(seq) and seq[-1] == drawgpt.END)
                copied = False if seq is None else bytes(np.asarray(seq).astype(np.uint16)) in seen
                rows[g].append((c, top1, top3, 0.0, ended, len(strokes), copied, v1_top1, v1_top3))
            bits = np.stack(bits)
            near = _nearest(bits, refs)
            for j in range(len(items)):
                r = rows[g][-len(items) + j]
                rows[g][-len(items) + j] = (r[0], r[1], r[2], float(near[j]), *r[4:])
            variety[g].append(_pairwise_mean(bits))
        if (c + 1) % 25 == 0:
            print(f"  eval: {c + 1}/{len(used)} classes ({time.time() - t0:.0f}s)", flush=True)
    out = summarise(rows, variety, used, n, temperatures, seed)
    out.update({"version": version, "guidances": list(guidances)})
    return out


def summarise(rows: dict, variety: dict, cats: list[str], n: int, temperatures: list[float], seed: int) -> dict:
    real_near = np.array([r[3] for r in rows["real"]])
    close = float(np.percentile(real_near, 5))
    out = {"drawings_per_class": n, "classes": len(cats), "temperatures": temperatures, "seed": seed,
           "copy_threshold": {"what": "5th percentile of real validation drawings' nearest-training distance",
                              "value": close},
           "groups": {}}
    for g, rs in rows.items():
        a = np.array([r[1:5] for r in rs], np.float64)
        per_class = np.array([r[1] for r in rs]).reshape(len(cats), n).mean(1)
        order = np.argsort(per_class)
        out["groups"][g] = {
            "top1": float(a[:, 0].mean()), "top3": float(a[:, 1].mean()),
            "v1_top1": float(np.mean([r[7] for r in rs])), "v1_top3": float(np.mean([r[8] for r in rs])),
            "nearest_train_distance_median": float(np.median(a[:, 2])),
            "closer_than_threshold": float((a[:, 2] < close).mean()),
            "exact_copies": int(sum(r[6] for r in rs)),
            "ended": float(a[:, 3].mean()),
            "strokes_mean": float(np.mean([r[5] for r in rs])),
            "variety_mean_pairwise": float(np.mean(variety[g])),
            "classes_at_0": int((per_class == 0).sum()),
            "worst_classes": [[cats[i], float(per_class[i])] for i in order[:10]],
            "best_classes": [[cats[i], float(per_class[i])] for i in order[::-1][:10]],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="drawings per class and group")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 1.0])
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N classes only (smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<version>/eval_report.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    report = evaluate(args.n, args.temperatures, args.seed, args.limit, args.version, tuple(args.guidance))
    args.out = args.out or drawgpt.run_dir(drawgpt.version(args.version)) / "eval_report.json"
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({g: {k: v for k, v in s.items() if not k.endswith("classes")}
                      for g, s in report["groups"].items()}, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
