"""Where does a format-6 buyer model put its probability when it writes a scene's list? A teacher-forced test on the
written scenes of the two B2 validation sets (prompt format 6), on the CPU, nothing sampled.

    cd ml && python tools/probe_list_tokens.py runs/sft_b2f6/best.pt --out runs/sft_b2f6/list_tokens.json

A written scene starts with three thing tokens. At each of the three places the model is given the prompt and the
written things before it, and we add up the probability it gives to all thing tokens, to the written thing, and to
everything else (the ordinary word pieces). Sampling at temperature 0.7 with top-k 50, any mass left outside the thing
tokens is a chance of a list that is not three things.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]
SETS = ("old", "new")
LIST_LENGTH = 3                    # things in a scene's list
FULL = 0.9                         # a place counts as leaking when the thing tokens get less than this together
TOP_OTHER = 3                      # other pieces counted at each place
SHOWN = 15                         # most likely other pieces to report


def main() -> None:
    """Score every place of every written list and write the report."""
    import torch

    import eval_buyer_b1 as E
    from sellanything_ml.buyers.format import THING_TOKENS
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sets", type=Path, default=ML / "runs/eval_sets_b2f6")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))
    ids = torch.tensor([tok.encode(t)[0] for t in THING_TOKENS.values()])
    thing_token = re.compile(r"<\|thing:[^|]+\|>")
    places: list[dict] = []
    outside: Counter[str] = Counter()
    for which in SETS:
        [path] = a.sets.glob(f"val_{which}_format*.jsonl")
        for ex in (json.loads(line) for line in path.read_text().splitlines()):
            if ex["task"] != "imagine":
                continue
            written = thing_token.findall(ex["target"].split(" | ")[0])
            for k, thing in enumerate(written):
                prefix = tok.encode(ex["prompt"] + "".join(written[:k]))
                with torch.no_grad():
                    logits, _ = model(torch.tensor([prefix]))
                p = torch.softmax(logits[0, -1].float() / E.TEMPERATURE, -1)       # as the game samples
                rest = p.clone()
                rest[ids] = 0
                for t in rest.topk(TOP_OTHER).indices.tolist():
                    outside[tok.decode([t])] += 1
                places.append({"set": which, "place": k, "things": float(p[ids].sum()),
                               "written": float(p[tok.encode(thing)[0]])})
    report = {"places": len(places)}
    for which in SETS:
        got = [x for x in places if x["set"] == which]
        report[which] = {"mean p(any thing token)": round(sum(x["things"] for x in got) / len(got), 4),
                         f"places under {FULL}": sum(x["things"] < FULL for x in got),
                         "median p(written thing)": round(sorted(x["written"] for x in got)[len(got) // 2], 4),
                         "by place": [round(sum(x["things"] for x in got if x["place"] == k)
                                            / sum(x["place"] == k for x in got), 4) for k in range(LIST_LENGTH)]}
    report["most likely other pieces"] = dict(outside.most_common(SHOWN))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
