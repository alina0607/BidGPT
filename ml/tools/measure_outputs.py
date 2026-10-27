"""Measurements on a buyer model's validation outputs (paper-ideas B1, B2, B12), from `eval_buyer_b1.py --outputs`.

    PYTHONPATH=. python tools/measure_outputs.py runs/measure_b1/b4_val_new_outputs.jsonl \
        --train data/processed/buyers_b4/train.jsonl --out runs/measure_b1/b4_val_new_measure.json

Only the spoken part of each line is compared (open: the greeting; ask: the question; react: what the buyer says;
imagine: the scene sentence), lower-cased, with every dollar amount written as "$n".

B1, near and far: a validation example is "near" when the sale's own line has a close twin among the training lines
of the same task. Closeness is the best bigram F1 (the share of word pairs two lines have in common, like Rouge-2 on
sets of pairs) against every training line; the examples are split at the median of each task. The eval checks are
then reported for each half, so a score that only holds on lines like the training ones shows up.

B2, copying: the share of the model's 4-grams and 5-grams that occur anywhere in the training lines of the task, the
share of its lines that are word for word a training line, and the median best bigram F1 against the training lines.
The same numbers for the sale's own (held-out, human-checked) lines are the baseline: a model that copies more than
the held-out lines are "new" is reciting.

B12, repetition: per task, the share of the model's lines taken by its 3 most common lines, and distinct-1/2 (unique
words and word pairs over all words and pairs), next to the same numbers for the sale's own lines.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

WORD = re.compile(r"\$n|[a-z0-9]+(?:'[a-z]+)?")
DOLLARS = re.compile(r"\$\d+")
TASKS = ("open", "ask", "react", "imagine")
CHECKS = ("readable", "spoken_ok", "new_question", "new_in_meaning", "rules_ok", "same_direction", "offer_error")
CHUNK = 256


def spoken(task: str, line: str) -> str:
    """The words a buyer says in a target or output line, as compared here."""
    parts = [p.strip() for p in line.replace("<|end|>", "").split(" | ")]
    if task == "react":
        text = parts[1] if len(parts) >= 2 else parts[0]
    elif task == "imagine":
        text = parts[1] if len(parts) >= 2 else parts[0]
    elif task == "open":
        text = parts[0]
    else:
        text = " | ".join(parts)
    return DOLLARS.sub("$n", text.lower())


def words(text: str) -> list[str]:
    return WORD.findall(text)


def grams(ws: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def best_f1(queries: list[list[str]], train: list[list[str]]) -> list[float]:
    """For each query, the best bigram-set F1 against any training line (0 when a line has no word pair)."""
    vocab: dict[tuple[str, ...], int] = {}

    def matrix(lines: list[list[str]], grow: bool) -> sparse.csr_matrix:
        rows, cols = [], []
        for r, ws in enumerate(lines):
            for g in set(grams(ws, 2)):
                if g not in vocab and grow:
                    vocab[g] = len(vocab)
                if g in vocab:
                    rows.append(r)
                    cols.append(vocab[g])
        return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                 shape=(len(lines), max(1, len(vocab))))

    t = matrix(train, grow=True)
    q = matrix(queries, grow=False)
    t_size = np.asarray(t.sum(axis=1)).ravel()
    q_size = np.array([len(set(grams(ws, 2))) for ws in queries], dtype=np.float32)
    out = []
    tt = t.T.tocsc()
    for start in range(0, len(queries), CHUNK):
        inter = (q[start:start + CHUNK] @ tt).toarray()
        denom = q_size[start:start + CHUNK, None] + t_size[None, :]
        f1 = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 0)
        out += f1.max(axis=1).tolist()
    return out


def copying(lines: list[list[str]], train: list[list[str]], f1: list[float]) -> dict[str, float]:
    """B2: shares of 4-grams and 5-grams seen in training, share of exact copies, median best bigram F1."""
    out: dict[str, float] = {}
    for n in (4, 5):
        seen = {g for ws in train for g in grams(ws, n)}
        mine = [g for ws in lines for g in grams(ws, n)]
        out[f"{n}gram_in_train"] = round(sum(g in seen for g in mine) / max(1, len(mine)), 3)
    whole = {" ".join(ws) for ws in train}
    out["exact_copy"] = round(sum(" ".join(ws) in whole for ws in lines if ws) / max(1, len(lines)), 3)
    out["median_best_f1"] = round(statistics.median(f1), 3) if f1 else 0.0
    return out


def repetition(lines: list[list[str]]) -> dict[str, float]:
    """B12: share of the 3 most common lines, distinct-1 and distinct-2."""
    joined = Counter(" ".join(ws) for ws in lines)
    uni = [w for ws in lines for w in ws]
    bi = [g for ws in lines for g in grams(ws, 2)]
    return {"top3_share": round(sum(c for _, c in joined.most_common(3)) / max(1, len(lines)), 3),
            "distinct_1": round(len(set(uni)) / max(1, len(uni)), 3),
            "distinct_2": round(len(set(bi)) / max(1, len(bi)), 3)}


def check_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The eval checks over a group of examples (mean offer error over the readable reactions)."""
    out: dict[str, float] = {"n": len(rows)}
    for key in CHECKS:
        vals = [r["checks"][key] for r in rows if key in r["checks"] and r["checks"][key] is not None
                and (key != "offer_error" or r["checks"].get("readable"))]
        if vals:
            out[key] = round(sum(vals) / len(vals), 3 if key != "offer_error" else 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outputs", type=Path)
    ap.add_argument("--train", type=Path, required=True, help="the training examples the near/far split is made on")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.outputs.read_text().splitlines()]
    train_rows = [json.loads(line) for line in a.train.read_text().splitlines()]
    report: dict[str, Any] = {"outputs": str(a.outputs), "train": str(a.train), "tasks": {}}
    near_of: dict[int, float] = {}
    for task in TASKS:
        mine = [r for r in rows if r["task"] == task]
        if not mine:
            continue
        train = [words(spoken(task, t["target"])) for t in train_rows if t["task"] == task]
        teacher = [words(spoken(task, r["target"])) for r in mine]
        model = [words(spoken(task, r["output"])) for r in mine]
        f1_teacher = best_f1(teacher, train)
        f1_model = best_f1(model, train)
        cut = statistics.median(f1_teacher)
        near = [r for r, f in zip(mine, f1_teacher, strict=True) if f > cut]
        far = [r for r, f in zip(mine, f1_teacher, strict=True) if f <= cut]
        for r, f in zip(mine, f1_teacher, strict=True):
            near_of[r["index"]] = f
        report["tasks"][task] = {
            "n": len(mine), "train_lines": len(train),
            "B1": {"median_teacher_f1": round(cut, 3), "near": check_rates(near), "far": check_rates(far)},
            "B2": {"model": copying(model, train, f1_model), "teacher": copying(teacher, train, f1_teacher)},
            "B12": {"model": repetition(model), "teacher": repetition(teacher)},
        }
    a.out.write_text(json.dumps(report, indent=1))
    a.out.with_suffix(".near.json").write_text(json.dumps({str(k): round(v, 4) for k, v in sorted(near_of.items())}))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
