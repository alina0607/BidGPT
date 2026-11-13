"""Check the teacher's parse outputs, prepare blind reviews, and report (docs/design-v2.md §6.3d).

    python -m sellanything_ml.datagen.check review-input   # answers without targets, for reviewers
    python -m sellanything_ml.datagen.check report         # checks + agreement + rule baseline + MANIFEST

Raw files are never modified. An answer is accepted only if a reviewer who did not see the target
maps it to the target with high confidence; everything else is kept and reported, not deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..buyers.answers import match
from ..buyers.schema import load_items
from .requests import GEN_DIR, prompt_sha256

REVIEW_PROMPT = "review_parse_v1.md"

STYLES = {"plain", "casual", "specific", "indirect", "extra"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answers(task: str = "parse") -> tuple[dict[str, dict], list[dict], list[str]]:
    """Requests by id, every written answer with its id, and format problems found."""
    requests = {r["request_id"]: r for r in read_jsonl(GEN_DIR / "requests" / f"{task}.jsonl")}
    out, problems = [], []
    for path in sorted((GEN_DIR / "batches" / task).glob("batch_*.jsonl")):
        for row in read_jsonl(path):
            req = requests.get(row.get("request_id"))
            if req is None:
                problems.append(f"{path.name}: unknown request_id {row.get('request_id')}")
                continue
            if row.get("prompt_sha256") != req["prompt_sha256"]:
                problems.append(f"{path.name}: {req['request_id']} written with another prompt version")
            if len(row["outputs"]) != req["n"]:
                problems.append(f"{path.name}: {req['request_id']} has {len(row['outputs'])} outputs")
            for i, o in enumerate(row["outputs"]):
                if o.get("style") not in STYLES or not o.get("text", "").strip():
                    problems.append(f"{path.name}: {req['request_id']}#{i} bad style or empty text")
                out.append({"answer_id": f"{req['request_id']}#{i}", "request": req, "batch": path.name,
                            "text": o["text"].strip(), "style": o.get("style")})
    return requests, out, problems


def review_input(task: str = "parse") -> None:
    _, rows, _ = answers(task)
    path = GEN_DIR / "reviews" / task / "review_input.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:   # derived file, rebuilt each time; never contains the target
        for r in rows:
            q = r["request"]
            f.write(json.dumps({"answer_id": r["answer_id"], "item": q["noun"], "question": q["question"],
                                "options": q["options"], "player_answer": r["text"],
                                "prompt": REVIEW_PROMPT, "prompt_sha256": prompt_sha256(REVIEW_PROMPT)},
                               ensure_ascii=False) + "\n")
    print(f"{len(rows)} answers -> {path}")


def manifest() -> dict:
    files = {}
    for path in sorted(GEN_DIR.rglob("*.jsonl")):
        data = path.read_bytes()
        files[str(path.relative_to(GEN_DIR))] = {"rows": data.count(b"\n"),
                                                 "sha256": hashlib.sha256(data).hexdigest()}
    (GEN_DIR / "MANIFEST.json").write_text(json.dumps(files, indent=2) + "\n")
    return files


def verdict(review: dict | None, target: str) -> str:
    """How a blind review relates to the writer's target option."""
    if review is None:
        return "not reviewed"
    if review["maps_to"] == target:
        return "accepted" if review.get("confidence") == "high" else "agreed, low confidence"
    if review["maps_to"] in ("ambiguous", "none"):
        return f"disagreed ({review['maps_to']})"
    return "disagreed (other option)"


def rule_result(option: str | None, target: str) -> str:
    """How the current rule matcher (buyers/answers.py) does on the same answer."""
    if option == target:
        return "right"
    return "can't tell" if option is None else "wrong option"


def read_reviews(task: str, folder: str | None = None) -> dict[str, dict]:
    """Reviews by answer id from reviews/<folder or task>/batch_*.jsonl."""
    reviews = {}
    for path in sorted((GEN_DIR / "reviews" / (folder or task)).glob("batch_*.jsonl")):
        for v in read_jsonl(path):
            reviews[v["answer_id"]] = v
    return reviews


def report(task: str = "parse") -> dict:
    _, rows, problems = answers(task)
    reviews = read_reviews(task)
    items = load_items()
    dup = Counter(r["text"].lower() for r in rows)
    stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        q = r["request"]
        verdict_ = verdict(reviews.get(r["answer_id"]), q["target"])
        rule_res = rule_result(match(r["text"], items[q["item"]].attributes[q["attribute"]]).option, q["target"])
        for key in ("all", q["item"], f"style:{r['style']}"):
            stats[key]["answers"] += 1
            stats[key][verdict_] += 1
            if verdict_ == "accepted":
                stats[key][f"rules {rule_res} (accepted)"] += 1
        if dup[r["text"].lower()] > 1:
            stats["all"]["exact duplicates"] += 1
    result = {"problems": problems, "stats": {k: dict(v) for k, v in stats.items()},
              "manifest_files": len(manifest())}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["review-input", "report"])
    args = ap.parse_args()
    review_input() if args.command == "review-input" else report()


if __name__ == "__main__":
    main()
