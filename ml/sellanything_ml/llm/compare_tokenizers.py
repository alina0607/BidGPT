"""Compare trained tokenizers on held-out text (docs/plan.md D7).

    python -m sellanything_ml.llm.compare_tokenizers
"""

from __future__ import annotations

import random

import pyarrow.parquet as pq
import tiktoken

from ..buyers.datagen import load_split
from ..buyers.format import SPECIAL_TOKENS
from .tokenizer import FINEWEB_DIR, TOKENIZER_DIR, Tokenizer


def samples(n_fineweb: int = 400, n_buyer: int = 2000, seed: int = 0) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    held_out = sorted(FINEWEB_DIR.glob("shard_*.parquet"))[-1]   # never used for training
    docs = pq.read_table(held_out, columns=["text"]).column("text").to_pylist()
    fineweb = rng.sample(docs, n_fineweb)
    rows = load_split("val")
    buyer = [r["prompt"] + r["target"] for r in rng.sample(rows, n_buyer)]
    return fineweb, buyer


def main() -> None:
    fineweb, buyer = samples()
    fw_bytes = sum(len(t.encode()) for t in fineweb)
    by_bytes = sum(len(t.encode()) for t in buyer)
    print(f"held-out text: FineWeb {fw_bytes:,} bytes, buyer {by_bytes:,} bytes\n")
    print(f"{'tokenizer':<14}{'vocab':>8}{'FineWeb B/tok':>15}{'buyer B/tok':>13}"
          f"{'round trip':>12}{'emb @512':>10}")

    gpt2 = tiktoken.get_encoding("gpt2")
    fw = sum(len(gpt2.encode(t)) for t in fineweb)
    print(f"{'tiktoken gpt2':<14}{gpt2.n_vocab:>8,}{fw_bytes / fw:>15.3f}{'n/a':>13}{'—':>12}"
          f"{gpt2.n_vocab * 512 / 1e6:>9.1f}M")

    for model in sorted(TOKENIZER_DIR.glob("sp*.model"), key=lambda p: int(p.stem[2:])):
        tok = Tokenizer(model)
        fw = sum(len(tok.encode(t)) for t in fineweb)
        by = sum(len(tok.encode(t)) for t in buyer)
        exact = all(tok.decode(tok.encode(t)) == t for t in fineweb[:100] + buyer[:500])
        singles = all(len(tok.encode(s)) == 1 for s in SPECIAL_TOKENS + ["\n"])
        print(f"{model.stem:<14}{tok.vocab_size:>8,}{fw_bytes / fw:>15.3f}{by_bytes / by:>13.3f}"
              f"{str(exact and singles):>12}{tok.vocab_size * 512 / 1e6:>9.1f}M")


if __name__ == "__main__":
    main()
