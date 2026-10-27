"""Pre-training data for the knowledge experiment: a slice of FineWeb-Edu with the cat knowledge texts mixed in, and the
same slice with nothing mixed in (the control), so the two continued pre-training runs differ only in the knowledge.

    cd ml && python tools/make_knowledge_mix.py              # → data/processed/pretrain_knowledge{,_control}/

The knowledge texts are data/kaggle_out/dialogues_v5/knowledge.jsonl (13 topics × 6 forms, written from
buyers/cat_knowledge.py by the Qwen teacher on Kaggle, 2026-10-10). Each text, ending with EOS, is put REPEATS times at
random document boundaries of the slice; both folders use FineWeb-Edu's own val.bin, which measures forgetting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
KNOWLEDGE = ML.parent / "data" / "kaggle_out" / "dialogues_v5" / "knowledge.jsonl"
BASE = ML / "data" / "processed" / "pretrain"
SLICE = 5_000_000                  # FineWeb-Edu tokens in the mix (the first ones of train.bin)
REPEATS = 50                       # copies of every knowledge text


def write(folder: Path, tokens: np.ndarray, meta: dict) -> None:
    """train.bin, a link to FineWeb-Edu's val.bin, and meta.json."""
    folder.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(folder / "train.bin")
    val = folder / "val.bin"
    if not val.exists():
        val.symlink_to(BASE / "val.bin")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    """Build the mix and the control."""
    from sellanything_ml.llm.tokenizer import TOKENIZER_DIR, Tokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = Tokenizer(TOKENIZER_DIR / "sp8192.model")
    base = np.memmap(BASE / "train.bin", dtype=np.uint16, mode="r")[:SLICE]
    texts = [json.loads(line)["text"] for line in KNOWLEDGE.read_text().splitlines()]
    docs = [np.array(tok.encode(t) + [tok.eos_id], dtype=np.uint16) for t in texts]
    rng = np.random.default_rng(a.seed)
    bounds = np.flatnonzero(base == tok.eos_id) + 1                  # positions right after an EOS
    copies = [d for d in docs for _ in range(REPEATS)]
    where = np.sort(rng.choice(bounds, size=len(copies), replace=False))
    order = rng.permutation(len(copies))
    pieces, last = [], 0
    for pos, k in zip(where, order, strict=True):
        pieces += [base[last:pos], copies[k]]
        last = pos
    pieces.append(base[last:])
    mix = np.concatenate(pieces)
    knowledge = sum(len(d) for d in docs)
    meta = {"source": str(KNOWLEDGE.relative_to(ML.parent)), "texts": len(texts), "knowledge_tokens": knowledge,
            "repeats": REPEATS, "fineweb_tokens": SLICE, "total_tokens": len(mix),
            "knowledge_share": round(knowledge * REPEATS / len(mix), 4)}
    write(ML / "data" / "processed" / "pretrain_knowledge", mix, meta)
    control = np.asarray(np.memmap(BASE / "train.bin", dtype=np.uint16, mode="r")[:len(mix)])
    write(ML / "data" / "processed" / "pretrain_knowledge_control", control,
          {"fineweb_tokens": len(control), "note": "the same number of tokens as the mix, FineWeb-Edu only"})
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
