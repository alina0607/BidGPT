"""B4f9's data: B4's SFT files with every scene rewritten line first (prompt format 9; docs/b4f9-line-first-plan.md).

    cd ml && PYTHONPATH=. python tools/make_line_first.py                 # → data/processed/buyers_b4f9/

The files are converted, not rebuilt from the sales: sales_b2.jsonl has grown since B4 was built (2026-10-13), so a
rebuild gives 49,809 training examples instead of B4's 48,832. Converting keeps B4's examples and split exactly, and
changes only the target of each imagine example, "things | line" → "line | things" with the things in the order the
line names them (sales_sft.scene_target). Checked before anything is written:
  1. every example other than imagine is the same line, byte for byte;
  2. every imagine example keeps its prompt and every other field;
  3. every rewritten scene, read back by the game's format-9 parser, has the same three things and the same line,
     and passes the scene rules (stories.imagine_problem).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sellanything_ml.buyers.cat_market import parse
from sellanything_ml.buyers.format import END, LINE_FIRST_FORMATS
from sellanything_ml.buyers.sales_sft import scene_target
from sellanything_ml.buyers.schema import DATA_DIR
from sellanything_ml.buyers.stories import imagine_problem

PROCESSED = DATA_DIR / "processed"
FORMAT = LINE_FIRST_FORMATS[0]
B4_SHA256 = {"train": "b738e4484b798c2f", "val": "b7ee1c0a3ff5879e"}     # B4's files (docs/b4f9-line-first-plan.md)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def convert(line: str, counts: Counter[str]) -> str:
    """One example: unchanged unless it is a scene, which is rewritten line first and checked."""
    ex = json.loads(line)
    if ex["task"] != "imagine":
        return line
    old = parse("imagine", ex["target"].removesuffix(END))
    if old is None or imagine_problem(old) is not None:
        raise ValueError(f"a B4 scene that does not pass the rules: {ex['target']!r}")
    target = scene_target(old, FORMAT) + END
    new = parse("imagine", target.removesuffix(END), FORMAT)
    if new is None or sorted(new["things"]) != sorted(old["things"]) or new["says"] != old["says"] \
            or imagine_problem(new) is not None:
        raise ValueError(f"a rewritten scene does not read back: {target!r}")
    counts["imagine"] += 1
    counts["reordered"] += new["things"] != old["things"]
    out = json.dumps({**ex, "target": target})
    if {k: v for k, v in json.loads(out).items() if k != "target"} != {k: v for k, v in ex.items() if k != "target"}:
        raise ValueError("a field other than the target changed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=PROCESSED / "buyers_b4")
    ap.add_argument("--out", type=Path, default=PROCESSED / "buyers_b4f9")
    a = ap.parse_args()
    report: dict[str, Any] = {"prompt_format": FORMAT, "derived_from": str(a.src), "source_sha256": {},
                              "sha256": {}, "examples": {}}
    converted = {}
    for split in ("train", "val"):
        src = a.src / f"{split}.jsonl"
        lines = src.read_text().splitlines()
        counts: Counter[str] = Counter()
        new = [convert(line, counts) for line in lines]
        same = sum(old == n for old, n in zip(lines, new, strict=True))
        if same != len(lines) - counts["imagine"]:          # check 1: only the scenes changed
            raise ValueError(f"{split}: {len(lines) - same} lines changed, {counts['imagine']} scenes")
        converted[split] = new
        report["source_sha256"][split] = sha16(src)
        if a.src == PROCESSED / "buyers_b4" and report["source_sha256"][split] != B4_SHA256[split]:
            raise ValueError(f"{src} is not the file B4 was trained on")
        report["examples"][split] = {"all": len(lines), "imagine": counts["imagine"], "reordered": counts["reordered"],
                                     "unchanged": same}
    a.out.mkdir(parents=True, exist_ok=True)
    for split, lines in converted.items():
        path = a.out / f"{split}.jsonl"
        path.write_text("".join(line + "\n" for line in lines))
        report["sha256"][split] = sha16(path)
    meta = json.loads((a.src / "meta.json").read_text())
    meta |= {"prompt_format": FORMAT, "line_first": report}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
