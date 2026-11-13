"""Blind review, round 2 (prompt review_parse_v2.md): reviewers cannot see or look up the target.

    python -m sellanything_ml.datagen.blind make --chunk 250 --out <dir outside the repo>
    python -m sellanything_ml.datagen.blind collect --out <same dir>

Why a second round (2026-10-07): in round 1 (review_parse_v1.md) every line carried the answer id,
which is "<request id>#<n>", and the request file in the repo names the target option. Reviewers of
batches 0004–0017 agreed with the writer on 99.7–100% of answers, took 74–107 s for 900 lines and
wrote one templated reason for nearly every line, so those reviews cannot be shown to be independent.
They are kept on disk unchanged and not used.

Here each line gets a random key, the options are shuffled and shown only as lettered texts (no
option ids), and the key file stays outside the repository. `collect` maps letters back to
option ids and writes reviews/parse_r2/batch_*.jsonl in round 1's schema, which build.py reads.

Honesty probes: each chunk also has PROBES lines where a real answer is shown with the question and
options of another attribute of the same item (sharing no content word with it), so the right reading
is "none". A reviewer who reads says none; one who looks answers up by their text gives the writer's
option, which is not on the list. `collect` reports the probe score per chunk and keeps a chunk only
if at least PROBE_PASS probes were answered "none" or "ambiguous". Not all 8: sharing no word is not
sharing no meaning, and an honest reader can rightly answer a probe (chunk 003: "a cheap Chinese-made
one from Amazon", written about the brand, read as "a student model" for the student-or-professional
question). In the first 41 chunks honest reviewers (98-99.5% agreement on the real lines, reasons
about each line) answered 5-8 probes "none"; the probes they answered were answerable by meaning
("It's a Swarovski NL Pure pair." for "How clear is the view?", "Strike on a brick, a zipper,
anything." for "Have they stayed dry?"). A reviewer who looks answers up finds a target that is
not among the letters and can only guess, so 5 of 8 still separates the two.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from pathlib import Path

from ..buyers.format import LETTERS
from ..buyers.schema import load_items
from .build import _content
from .check import answers
from .requests import GEN_DIR, prompt_sha256

PROMPT = "review_parse_v2.md"
PROBES = 8
PROBE_PASS = 5
ROUND_DIR = GEN_DIR / "reviews" / "parse_r2"


def _probe(r: dict, items, rng: random.Random):
    """(attribute, its options as dicts) of another attribute that this answer says nothing about, or None."""
    item = items[r["request"]["item"]]
    others = [a for a in item.attributes.values() if a.name != r["request"]["attribute"]]
    rng.shuffle(others)
    for a in others:
        words = set().union(_content(a.question), *(_content(o.answer) for o in a.options))
        if not (_content(r["text"]) & words):
            return a, [{"id": o.id, "answer": o.answer} for o in a.options]
    return None


def make(out: Path, chunk: int, keys_dir: Path, seed: int = 7) -> int:
    out.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)
    items = load_items()
    reviewed = set()
    for p in ROUND_DIR.glob("batch_*.jsonl"):
        reviewed |= {json.loads(line)["answer_id"] for line in p.read_text().splitlines() if line.strip()}
    old_keys = keys_dir / "keys.jsonl"
    keyed = set()
    if old_keys.exists():
        keyed = {json.loads(line)["answer_id"] for line in old_keys.read_text().splitlines()}
    _, rows, _ = answers("parse")
    pool = [r for r in rows if r["style"] != "extra"]
    rows = [r for r in rows if r["answer_id"] not in reviewed and r["answer_id"] not in keyed]
    rng = random.Random(seed + len(keyed))
    rng.shuffle(rows)                                   # chunks mix items and writers
    start = len(list(out.glob("chunk_*.tsv")))
    with open(old_keys, "a") as keys:
        for c in range(0, len(rows), chunk):
            name = f"chunk_{start + c // chunk + 1:03d}"
            lines = [(r, r["request"]["question"], r["request"]["options"], False) for r in rows[c:c + chunk]]
            while sum(p for *_, p in lines) < PROBES:
                r = rng.choice(pool)
                found = _probe(r, items, rng)
                if found:
                    lines.append((r, found[0].question, found[1], True))
            rng.shuffle(lines)
            with open(out / f"{name}.tsv", "w") as f:
                for r, question, options, probe in lines:
                    order = [o["id"] for o in options]
                    rng.shuffle(order)
                    texts = {o["id"]: o["answer"] for o in options}
                    key = secrets.token_hex(4)
                    listed = " | ".join(f"{LETTERS[i]}. {texts[o]}" for i, o in enumerate(order))
                    said = " ".join(r["text"].split())
                    f.write(f"{key}\t{r['request']['noun']}\t{question}\t{listed}\t{said}\n")
                    keys.write(json.dumps({"key": key, "chunk": name, "answer_id": r["answer_id"],
                                           "order": order, "probe": probe}) + "\n")
    print(f"{len(rows)} answers in {(len(rows) + chunk - 1) // chunk} chunks -> {out}")
    return len(rows)


def _maps_to(verdict: str, order: list[str]) -> str | None:
    """A reviewer's letter back to the option id; "ambiguous" / "none" as they are; None if unreadable."""
    if len(verdict) == 1 and verdict in LETTERS[:len(order)]:
        return order[LETTERS.index(verdict)]
    return verdict.lower() if verdict.lower() in ("ambiguous", "none") else None


def _read_chunk(res: Path, name: str, keys: dict, stats: dict) -> tuple[list[tuple[dict, list[str]]], list[bool]]:
    """(key, reviewer's fields) for the real lines of one reviewed chunk, and whether each probe read as none."""
    lines, probes = [], []
    for line in res.read_text().splitlines():
        parts = [x.strip() for x in line.rstrip("\n").split("\t")]
        if len(parts) < 3 or not parts[0]:
            continue
        k = keys.get(parts[0])
        if k is None or k["chunk"] != name:
            stats["unknown key"] += 1
        elif k.get("probe"):
            probes.append(parts[1].lower() in ("none", "ambiguous"))
        else:
            lines.append((k, parts))
    return lines, probes


def collect(out: Path, keys_dir: Path, model: str, date: str) -> dict:
    keys = {}
    for line in (keys_dir / "keys.jsonl").read_text().splitlines():
        k = json.loads(line)
        keys[k["key"]] = k
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    sha = prompt_sha256(PROMPT)
    stats = {"lines": 0, "unknown key": 0, "bad": 0, "chunks": 0, "rejected chunks": {}}
    for res in sorted(out.glob("chunk_*.out.tsv")):
        name = res.name.replace(".out.tsv", "")
        target = ROUND_DIR / f"batch_{name.split('_')[1]}.jsonl"
        if target.exists():
            continue
        lines, probes = _read_chunk(res, name, keys, stats)
        if len(probes) < PROBES or sum(probes) < PROBE_PASS:
            stats["rejected chunks"][name] = f"probes read as none: {sum(probes)}/{len(probes)}"
            continue
        rows = []
        for k, parts in lines:
            maps_to = _maps_to(parts[1], k["order"])
            if maps_to is None:
                stats["bad"] += 1
                continue
            conf = parts[2].lower()
            rows.append({"answer_id": k["answer_id"], "maps_to": maps_to,
                         "confidence": conf if conf in ("high", "low") else "low",
                         "reason": parts[3] if len(parts) > 3 else "",
                         "generator": {"model": model, "role": "reviewer", "date": date},
                         "prompt": PROMPT, "prompt_sha256": sha})
        with open(target, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["lines"] += len(rows)
        stats["chunks"] += 1
    print(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["make", "collect"])
    ap.add_argument("--out", type=Path, required=True, help="where reviewers read and write chunks")
    ap.add_argument("--keys", type=Path, required=True, help="key file folder, never shown to reviewers")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--model", required=True, help="id of the reviewer model, stored with each verdict")
    ap.add_argument("--date", default="2026-10-07")
    a = ap.parse_args()
    make(a.out, a.chunk, a.keys) if a.command == "make" else collect(a.out, a.keys, a.model, a.date)


if __name__ == "__main__":
    main()
