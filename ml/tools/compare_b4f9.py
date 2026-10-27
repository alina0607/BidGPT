"""B4 against B4f9 over three sampling seeds (docs/b4f9-line-first-plan.md), and the verdict the plan fixed before
the run: on the new set's scenes, B4f9 helps only when its mean is at least 10 points above B4's AND its worst seed
is above B4's best seed. Otherwise the report says it did not help.

    cd ml && python tools/compare_b4f9.py runs/sft_b4f9/eval --out runs/sft_b4f9/eval/compare.json

Reads <model>_val_<set>_s<k>.json written by tools/eval_buyer_b1.py (--seed-offset k).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODELS, SETS, SEEDS = ("b4", "b4f9"), ("old", "new"), (0, 1, 2)
MARGIN = 0.10                      # the plan's threshold: 10 points
RATES = ("readable", "spoken_ok", "new_in_meaning", "rules_ok", "same_direction", "said_equals_offer", "reasonable")


def spread(values: list[float]) -> dict[str, float]:
    return {"mean": round(sum(values) / len(values), 3), "min": min(values), "max": max(values)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    reports = {(m, s, k): json.loads((a.eval_dir / f"{m}_val_{s}_s{k}.json").read_text())["tasks"]
               for m in MODELS for s in SETS for k in SEEDS}
    table: dict[str, Any] = {}
    for s in SETS:
        for task in reports[MODELS[0], s, 0]:
            for rate in RATES:
                if rate not in reports[MODELS[0], s, 0][task]:
                    continue
                table[f"{s} {task} {rate}"] = {m: spread([reports[m, s, k][task][rate] for k in SEEDS]) for m in MODELS}
        table[f"{s} imagine kinds"] = {m: {kind: [reports[m, s, k]["imagine"]["kinds"][kind] for k in SEEDS]
                                           for kind in reports[m, s, 0]["imagine"]["kinds"]} for m in MODELS}
    b4, f9 = table["new imagine rules_ok"]["b4"], table["new imagine rules_ok"]["b4f9"]
    helped = f9["mean"] >= b4["mean"] + MARGIN and f9["min"] > b4["max"]
    result = {"verdict": "helped" if helped else "did not help",
              "rule": f"new-set scenes: B4f9 mean >= B4 mean + {MARGIN:.2f} and B4f9 worst seed > B4 best seed",
              "new_imagine_rules_ok": {"b4": b4, "b4f9": f9}, "table": table}
    a.out.write_text(json.dumps(result, indent=1))
    width = max(map(len, table))
    for key, row in table.items():
        if "kinds" in key:
            print(f"{key:<{width}}  " + "   ".join(f"{m}: {row[m]}" for m in MODELS))
        else:
            print(f"{key:<{width}}  " + "   ".join(f"{m} {row[m]['mean']:.3f} [{row[m]['min']:.3f}-{row[m]['max']:.3f}]"
                                                    for m in MODELS))
    print(f"\nverdict: {result['verdict']} ({result['rule']})")


if __name__ == "__main__":
    main()
