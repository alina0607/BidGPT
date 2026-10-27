"""Convert reviewed Qwen cat-sale dialogues into buyer-Transformer SFT examples.

The teacher stories are free-form ten-buyer sales, while the game model is trained on small,
guardrailed tasks (ask, react and parse).  This adapter keeps only examples the game can actually
use at inference time:

* structurally consistent reactions become ``react`` examples with the teacher's +/0/- direction;
* questions become ``ask`` examples only when the existing on-topic guardrail would accept them;
* seller replies become ``parse`` examples when the cat facts map unambiguously to a game option.

    cd ml
    python -m sellanything_ml.buyers.cat_dialogues
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import format as fmt
from .schema import load_items

ML_DIR = Path(__file__).resolve().parents[2]
REPO = ML_DIR.parent
DEFAULT_SALES = REPO / "data" / "kaggle_out" / "dialogues_v5" / "sales.jsonl"
DEFAULT_REVIEWS = REPO / "data" / "kaggle_out" / "review_v6" / "reviews.jsonl"
DEFAULT_OUT = ML_DIR / "data" / "processed" / "buyers_cat_v1"
MAX_LOG_LINES = 12       # ten-buyer stories otherwise exceed the model's 768-token context

KEYWORDS = {
    "age": ("how old", " age", "month", "year", "kitten", "senior"),
    "breed": ("breed", "purebred", "pedigree", "papers", "maine coon", "ragdoll", "persian",
              "siamese", "bengal", "sphynx", "scottish"),
    "litter": ("litter", "house trained", "potty"),
    "temperament": ("personality", "temperament", "cuddly", "affection", "friendly", "shy", "scratch",
                    "feisty", "independent", "people", "children", "kids", "dog", "other cat", "social"),
    "health": ("health", "healthy", "vaccin", "shots", "vet", "fiv", "felv", "virus", "disease",
               "medication", "spay", "neuter", "fixed", "bloodwork", "kidney", "thyroid"),
}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def attribute_for(question: str) -> str | None:
    """The game-cat attribute a free teacher question is chiefly about, if any."""
    text = " " + question.lower()
    scores = {name: sum(word in text for word in words) for name, words in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else None


def _option_for(attr: str, cat: dict, answer: str, option_ids: list[str]) -> int | None:
    """Map true cat facts to an option only when the game's coarser schema has an honest match."""
    option = None
    if attr == "age":
        months = int(cat["months"])
        option = "kitten" if months < 12 else "senior" if months >= 84 else "adult"
    elif attr == "breed":
        said = answer.lower()
        if any(word in said for word in ("mixed", "domestic shorthair", "unknown breed", "calico", "tabby")):
            option = "mixed"
        elif any(word in said for word in ("pedigree", "purebred", "with papers", "certificate")):
            option = "pedigree"
    elif attr == "litter":
        said = answer.lower()
        if "litter" in said and any(word in said for word in ("never misses", "fully trained", "always uses")):
            option = "yes"
        elif "litter" in said and any(word in said for word in ("not trained", "misses", "accident", "unreliable")):
            option = "no"
    elif attr == "temperament":
        temperament = cat["temperament"]
        option = {"very cuddly": "cuddly", "follows people around like a dog": "cuddly",
                  "independent": "independent", "shy with strangers": "shy",
                  "a bit feisty, scratches when annoyed": "feisty"}.get(temperament)
    elif attr == "health":
        health = cat["health"]
        if health == "fully vaccinated, vet check last month":
            option = "vaccinated"
        elif health in ("no vaccination records", "no vaccines yet"):
            option = "unknown"
    return option_ids.index(option) if option in option_ids else None


def _persona(sale: dict, name: str) -> str:
    card = sale["cards"][name]
    return f"{name}, who {card['persona']}."


def _view(sale: dict, name: str, log: list[str], attrs: list[str]) -> fmt.BuyerView:
    return fmt.BuyerView("cat", attrs, _persona(sale, name), log[-MAX_LOG_LINES:], ("cat",))


def _public_line(event: dict) -> str | None:
    kind = event["kind"]
    if kind in ("open", "ask", "react") and event.get("says"):
        return f"{event['name']}: {event['says']}"
    if kind == "answer" and event.get("says"):
        return f"You: {event['says']}"
    return None


def convert_sale(sale: dict) -> list[dict]:
    """Examples from one sale, filtered only by checks whose answer is mechanically knowable."""
    item = load_items()["cat"]
    attrs = list(item.attributes)
    names = fmt.item_names("cat", item.noun)
    examples: list[dict] = []
    log: list[str] = []
    answer_snapshot: list[str] | None = None
    asked_attr: str | None = None

    for index, event in enumerate(sale["events"]):
        kind = event["kind"]
        if kind == "ask":
            asked_attr = attribute_for(event["says"])
            if asked_attr is not None:
                attr = item.attributes[asked_attr]
                # This is the exact inference guardrail: examples it would reject are not learned.
                if fmt.on_topic(event["says"], attr.question, item.noun):
                    target = fmt.to_slot(event["says"], names)
                    examples.append({"task": "ask", "item": "cat", "buyer": event["name"], "traits": [],
                                     "show_attrs": True, "attr": asked_attr, "source_seed": sale["seed"],
                                     "source_event": index,
                                     "prompt": fmt.ask_prompt(_view(sale, event["name"], log, attrs), asked_attr,
                                                              fmt.to_slot(attr.question, names)),
                                     "target": fmt.ask_target(event["name"], target)})
        line = _public_line(event)
        if line:
            log.append(line)

        if kind == "answer":
            answer_snapshot = list(log)
            if asked_attr is not None:
                attr = item.attributes[asked_attr]
                option_ids = [option.id for option in attr.options]
                option_index = _option_for(asked_attr, sale["sale"]["cat"], event["says"], option_ids)
                if option_index is not None:
                    examples.append({"task": "parse", "item": "cat", "attr": asked_attr,
                                     "option": option_ids[option_index], "said": event["says"],
                                     "source_seed": sale["seed"], "source_event": index,
                                     "prompt": fmt.parse_prompt(item.noun, attr.question,
                                                                [option.answer for option in attr.options],
                                                                event["says"]),
                                     "target": fmt.parse_target(option_index)})
        elif kind == "react" and answer_snapshot is not None:
            delta = {"-": -1, "0": 0, "+": 1}.get(str(event.get("sign")))
            movement = int(event["offer"]) - int(event["old"])
            # A positive/negative reaction may already be capped at the buyer's budget/floor, so no
            # offer movement is still valid; only movement in the opposite direction is inconsistent.
            if delta is None or (delta == 0 and movement != 0) or (delta and movement * delta < 0):
                continue
            target = fmt.to_slot(event.get("says", ""), names) or None
            # Silent zeroes overwhelm useful lines; the game already has plenty from engine data.
            if target is None:
                continue
            examples.append({"task": "react", "item": "cat", "buyer": event["name"], "traits": [],
                             "show_attrs": True, "delta": delta, "speaks": True,
                             "source_seed": sale["seed"], "source_event": index,
                             "prompt": fmt.react_prompt(_view(sale, event["name"], answer_snapshot, attrs)),
                             "target": fmt.react_target(event["name"], target, delta)})
    return examples


def build(sales: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    """Stable 80/10/10 split by sale, so one conversation never leaks between splits."""
    out = {"train": [], "val": [], "test_seen": [], "test_heldout": []}
    counts: Counter = Counter()
    for position, sale in enumerate(sorted(sales, key=lambda row: row["seed"])):
        split = "val" if position % 10 == 8 else "test_seen" if position % 10 == 9 else "train"
        rows = convert_sale(sale)
        out[split].extend(rows)
        counts.update((split, row["task"]) for row in rows)
    stats = {"sales": len(sales), "examples": {f"{split}/{task}": n for (split, task), n in sorted(counts.items())},
             "total_examples": sum(counts.values()),
             "review_policy": "diagnostic only; low-precision Qwen review flags do not filter training data"}
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # The review file remains recorded for audit, but its hand-checked precision is too low to delete examples.
    # Only deterministic checks in convert_sale filter reactions.
    splits, stats = build(_read_jsonl(args.sales))
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    def source_name(path: Path) -> str:
        resolved = path.resolve()
        return str(resolved.relative_to(REPO)) if resolved.is_relative_to(REPO) else str(path)

    (args.out / "meta.json").write_text(json.dumps({"prompt_format": fmt.PROMPT_FORMAT,
                                                     "tasks": ["ask", "react", "parse"],
                                                     "source": source_name(args.sales),
                                                     "reviews": source_name(args.reviews)}, indent=2) + "\n")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
