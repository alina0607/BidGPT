"""Does a buyer model hear the seller right? Takes every reaction prompt of the two B2 validation sets whose latest
seller answer says the cat's vaccines are done, or gives the cat's age, and samples several reactions to each on the
CPU (temperature 0.7, top-k 50, as tools/eval_buyer_b1.py). A reaction misreads the answer when it says the cat has no
vaccines, or names an age in the same unit (years or months) that the seller did not say.

    cd ml && python tools/probe_misread.py runs/sft_b2/best.pt --out runs/sft_b2/misread/b2.jsonl
    cd ml && python tools/probe_misread.py runs/sft_b2f6/best.pt --sets runs/eval_sets_b2f6 --out …/b2f6.jsonl

The word matching is rough (a buyer may talk about the age of their own old cat), so the flagged lines are saved with
the answer and are meant to be read, not only counted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]
SETS = ("old", "new")
SAMPLES = 5                        # reactions per prompt; seeds 1000 * prompt index + k
DONE = re.compile(r"fully vaccinated|(vaccines|vaccinations|shots|jabs) (are )?(all )?(up to date|done)"
                  r"|all (her|his) (shots|vaccines|jabs)|vaccinated and", re.I)
NONE = re.compile(r"\bno (vaccin|shots|jabs)|\bnot (been )?(vaccinated|jabbed)|\bunvaccinated|\bmissing (vaccin|shots)"
                  r"|\bwithout (any )?(vaccin|shots)|\bno vaccinated", re.I)
WORDS = "one two three four five six seven eight nine ten eleven twelve".split()
AGE = re.compile(r"\b(\d+|" + "|".join(WORDS) + r")[- ](year|month)s?[- ]old\b", re.I)


def ages(text: str) -> set[tuple[int, str]]:
    """(number, "year" or "month") for every "N years old" / "two-month-old" in the text."""
    return {(int(n) if n.isdigit() else WORDS.index(n.lower()) + 1, unit.lower()) for n, unit in AGE.findall(text)}


def misread(answer: str, line: str) -> str | None:
    """Why the line misreads the answer, or None."""
    if DONE.search(answer) and NONE.search(line):
        return "says no vaccines"
    said = ages(answer)
    wrong = {(n, u) for n, u in ages(line) if (n, u) not in said and any(u == su for _, su in said)}
    return f"wrong age {sorted(wrong)}" if wrong else None


def prompts(path: Path, context: Callable[[dict], dict]) -> list[tuple[int, dict, str, str]]:
    """(index in the file, example, "vaccines" or "age", the latest answer) for every reaction prompt to test."""
    out = []
    for i, ex in enumerate(json.loads(line) for line in path.read_text().splitlines()):
        answer = context(ex)["answer"] if ex["task"] == "react" else ""
        kind = "vaccines" if DONE.search(answer) else "age" if ages(answer) else None
        if kind:
            out.append((i, ex, kind, answer))
    return out


def main() -> None:
    """Sample SAMPLES reactions to every prompt to test, mark the misreads and write one row per reaction."""
    import torch

    import eval_buyer_b1 as E
    from sellanything_ml.buyers.cat_market import MAX_NEW, parse
    from sellanything_ml.buyers.format import END
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sets", type=Path, default=ML / "runs/eval_sets_b2", help="folder of val_{old,new}_format*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))
    rows = []
    for which in SETS:
        [path] = a.sets.glob(f"val_{which}_format*.jsonl")
        for i, ex, kind, answer in prompts(path, E.context):
            ids = torch.tensor([tok.encode(ex["prompt"])])
            for k in range(SAMPLES):
                torch.manual_seed(1000 * i + k)
                with torch.no_grad():
                    gen = model.generate(ids, MAX_NEW["react"], temperature=E.TEMPERATURE, top_k=E.TOP_K,
                                         stop_id=tok.eos_id)
                text = tok.decode(gen[0, ids.size(1):].tolist()).replace(END, "").strip()
                row = parse("react", text)
                rows.append({"set": which, "index": i, "sample": k, "kind": kind, "name": ex["name"],
                             "answer": answer, "model": text, "misread": misread(answer, row["says"] if row else text)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for kind in ("vaccines", "age"):
        got = [r for r in rows if r["kind"] == kind]
        print(f"{kind}: {len(got)} reactions, {sum(bool(r['misread']) for r in got)} misread")


if __name__ == "__main__":
    main()
