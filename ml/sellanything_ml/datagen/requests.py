"""Write the requests the teacher LLM is asked to fulfil (docs/design-v2.md §6.3d).

    python -m sellanything_ml.datagen.requests parse --items cat bicycle "cell phone" --variants 8

One request per (item, attribute, option). The request id is a hash of those fields and the task, so
re-running this never changes ids, and batches can skip requests that already have an output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..buyers.schema import DATA_DIR, load_items

GEN_DIR = DATA_DIR / "raw" / "teacher_gen" / "v6"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SPLITS = DATA_DIR / "item_splits.json"


def request_id(task: str, *fields: str) -> str:
    return hashlib.sha256("\x1f".join((task, *fields)).encode()).hexdigest()[:16]


def prompt_sha256(name: str) -> str:
    return hashlib.sha256((PROMPTS_DIR / name).read_bytes()).hexdigest()


def item_split(item: str) -> str:
    """'heldout' for items never trained on (ml/data/item_splits.json), else 'train'."""
    splits = json.loads(SPLITS.read_text())
    return "heldout" if item in splits["buy_heldout"] + splits["refuse_heldout"] else "train"


def parse_requests(items: list[str], variants: int) -> list[dict]:
    schema = load_items()
    out = []
    for key in items:
        item = schema[key]
        for attr in item.attributes.values():
            options = [{"id": o.id, "answer": o.answer} for o in attr.options]
            for o in attr.options:
                out.append({
                    "request_id": request_id("parse", key, attr.name, o.id),
                    "task": "parse",
                    "item": key,
                    "noun": item.noun,
                    "split": item_split(key),
                    "attribute": attr.name,
                    "question": attr.question,
                    "options": options,
                    "target": o.id,
                    "n": variants,
                    "prompt": "parse_v1.md",
                    "prompt_sha256": prompt_sha256("parse_v1.md"),
                })
    return out


def write_new(path: Path, rows: list[dict]) -> int:
    """Append requests whose id is not in the file yet; existing lines are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        seen = {json.loads(line)["request_id"] for line in path.read_text().splitlines() if line}
    new = [r for r in rows if r["request_id"] not in seen]
    with path.open("a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["parse"])
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--variants", type=int, default=8)
    args = ap.parse_args()
    rows = parse_requests(args.items, args.variants)
    added = write_new(GEN_DIR / "requests" / f"{args.task}.jsonl", rows)
    print(f"{args.task}: {len(rows)} requests for {len(args.items)} items, {added} new")


if __name__ == "__main__":
    main()
