"""B15 (paper-ideas): the same SFT data with half of the training sales, for a data-size curve against the full set.

    cd ml && python tools/make_half_set.py data/processed/buyers_b4 data/processed/buyers_b4_half

Half of each author's training sales is kept (the half whose seed hash is lower, so the choice is fixed and does not
depend on the order of the file); every example of a kept sale is kept. The validation file is copied unchanged, so the
two models are measured on exactly the same held-out sales.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (src / "train.jsonl").read_text().splitlines()]
    by_author: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        by_author[r["author"]].add(r["seed"])
    keep = set()
    for seeds in by_author.values():
        ranked = sorted(seeds, key=lambda s: hashlib.sha1(f"half{s}".encode()).hexdigest())
        keep |= set(ranked[: len(ranked) // 2])
    kept = [r for r in rows if r["seed"] in keep]
    (dst / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    shutil.copy(src / "val.jsonl", dst / "val.jsonl")
    meta = json.loads((src / "meta.json").read_text())
    meta |= {"half_of": str(src), "sales": {"train": len(keep), "val": meta["sales"]["val"]},
             "authors_train": {a: len(s & keep) for a, s in by_author.items()}}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(keep)} of {len({r['seed'] for r in rows})} sales, {len(kept)} of {len(rows)} examples",
          meta["authors_train"])


if __name__ == "__main__":
    main()
