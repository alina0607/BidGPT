"""Where in the coarse-to-fine painting does a Brush GPT run go wrong? For each validation cat, keep the first k real
stages (k = 0: only BOS and the face points), let the model paint the rest, and ask the eye judge (tools/eye_judge.py)
whether the eyes show. Next to it: the first k real stages alone (nothing painted after them), the whole real painting,
and the model painting the whole cat by itself, face points too ("free"; its paired difference pairs it with cats it
was not given, so read only its mean).

If the model's eye score falls mostly when it has to paint stage s itself, that stage is where its errors come from.
The same cats, the same seed and the same temperature for every k, so the scores differ only by where the model takes
over; each cat's difference from its own whole real painting is reported too (paired, so a small n still says something).
Also the teacher-forced loss of each stage's stroke tokens.

    nice -n 19 env BRUSHGPT_DEVICE=cpu PYTHONPATH=. python tools/stage_continuation.py --run I_aligned_points_eyes \
        --data data/processed/brushgpt/tokens_runs_HI.npz --n 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brushgpt_by_source import source_of  # noqa: E402
from eye_judge import EYES, judge  # noqa: E402

from sellanything_ml.sketch import brushgpt as G  # noqa: E402
from sellanything_ml.sketch import brushtokens as T  # noqa: E402


def stage_marks(points: bool) -> list[int]:
    """Positions of the four STAGE tokens and of EOS: row[:, :marks[k]] is the painting up to the end of stage k."""
    lo, _, _ = G.LAYOUTS[points]
    return [int(np.flatnonzero(lo == T.STAGE0 + k)[0]) for k in range(len(G.STROKES))] + [len(lo) - 1]


def eye_conf(rows: np.ndarray, dev: str) -> np.ndarray:
    """Each row's mean eye confidence from the judge (an empty painting is blank paper)."""
    paper = np.ones((G.B.SIDE, G.B.SIDE, 3))
    painted = (rows[0] == T.STAGE0).any()                          # a cut before stage 0 has no strokes
    _, c = judge(np.stack([G.render(r) if painted else paper for r in rows]), dev)
    return c[:, EYES].mean(1)


def summary(conf: np.ndarray, whole: np.ndarray) -> dict[str, float]:
    d = conf - whole
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))  # noqa: E731
    return {"eye_conf_mean": round(float(conf.mean()), 4), "eye_conf_se": round(se(conf), 4),
            "vs_whole_real": round(float(conf.mean() / whole.mean()), 3),
            "paired_diff": round(float(d.mean()), 4), "paired_diff_se": round(se(d), 4)}


def stage_loss(model: "G.GPT", rows: np.ndarray, marks: list[int], dev: str, batch: int = 8) -> list[float]:
    """Mean teacher-forced cross-entropy of each stage's stroke tokens (the STAGE token itself left out)."""
    import torch
    import torch.nn.functional as F
    mask = G.allowed(dev, model.brush_cfg.points) if model.brush_cfg.heads else None
    total = np.zeros(rows.shape[1] - 1)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            t = torch.from_numpy(rows[i:i + batch]).to(dev)
            logits = G.field_logits(model(t[:, :-1])[0].float(), 1, mask)
            ce = F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction="none")
            total += ce.sum(0).cpu().numpy()
    per_pos = total / len(rows)                           # per_pos[p - 1] is the loss of predicting position p
    return [round(float(per_pos[marks[k]:marks[k + 1] - 1].mean()), 4) for k in range(len(G.STROKES))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=Path, required=True, help="the run's own token file")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--source", choices=["afhq", "cat", "all"], default="afhq")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ids-from", type=Path, default=None,
                    help="an earlier run's stage_continuation .npz: score the same cats, in the same order")
    ap.add_argument("--out", type=Path, default=None, help="default: the run's stage_continuation_<source>_t<t>.json")
    a = ap.parse_args()
    dev = G.device()
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", dev)
    cfg = model.brush_cfg
    rows, ids = G.load_rows(a.data, "val", 0, cfg.order, points=cfg.points)
    keep = np.ones(len(rows), bool) if a.source == "all" else source_of(ids) == a.source
    rows, ids = rows[keep][:a.n], ids[keep][:a.n]
    if a.ids_from is not None:
        all_rows, all_ids = G.load_rows(a.data, "val", 0, cfg.order, points=cfg.points)
        where = {str(i): k for k, i in enumerate(all_ids)}
        ids = np.load(a.ids_from)["ids"]
        rows = all_rows[[where[str(i)] for i in ids]]
    marks = stage_marks(cfg.points)
    t0 = time.time()

    whole = eye_conf(rows, dev)
    report = {"run": a.run, "data": str(a.data), "source": a.source, "n": len(rows), "temperature": a.temperature,
              "seed": a.seed, "device": dev, "marks": marks, "whole_real": summary(whole, whole),
              "real_first": {}, "model_after": {}}
    tokens, conf = {}, {"whole_real": whole}
    for k in range(len(G.STROKES)):                 # only the first k real stages (k = 0: blank paper)
        c = eye_conf(rows[:, :marks[k]], dev)
        report["real_first"][str(k)] = summary(c, whole)
        conf[f"real_first_{k}"] = c
    for k in range(len(G.STROKES)):                 # the first k real stages, then the model paints the rest
        gen = G.sample(model, len(rows), a.temperature, seed=a.seed, batch=a.batch, prefix=rows[:, :marks[k]])
        assert (gen[:, :marks[k]] == rows[:, :marks[k]]).all()
        c = eye_conf(gen, dev)
        report["model_after"][str(k)] = summary(c, whole)
        tokens[f"model_after_{k}"], conf[f"model_after_{k}"] = gen, c
        print(json.dumps({"model_after": k, **report["model_after"][str(k)], "min": round((time.time() - t0) / 60, 1)}),
              flush=True)
    free = G.sample(model, len(rows), a.temperature, seed=a.seed, batch=a.batch)
    conf["free"] = eye_conf(free, dev)
    tokens["free"] = free
    report["free"] = summary(conf["free"], whole)
    report["stage_loss"] = stage_loss(model, rows, marks, dev)
    report["minutes"] = round((time.time() - t0) / 60, 1)

    out = a.out or G.RUN_DIR / a.run / f"stage_continuation_{a.source}_t{a.temperature}.json"
    out.write_text(json.dumps(report, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), ids=ids, real=rows, **tokens, **{"conf_" + k: v for k, v in conf.items()})
    print(json.dumps(report))


if __name__ == "__main__":
    main()
