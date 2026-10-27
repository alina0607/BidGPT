"""How often each drawable thing is pictured in the buyers' scenes of the written sales, and how many more scenes it
needs to reach the quotas (notebook 6 §10: a thing seen fewer than 10 times is not learned; 30 is safer).

    cd ml && python tools/scene_quota.py                     # B3's sources → data/written_sales/scene_quota.md / .json

Run it again after every batch of new sales: the counts are read from the sales files, never kept by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
QUOTAS = (10, 30)
# Things that do not fit a buyer's home scene with a cat or dog (owner, 2026-10-13: quotas count only the rest; these
# wait for other item families). Proposed by reading the 248 things seen fewer than 30 times.
NOT_HOME = frozenset("""The Great Wall of China|aircraft carrier|animal migration|bulldozer|camel|camouflage|cannon|
crocodile|flamingo|flying saucer|giraffe|helicopter|hurricane|jail|lion|monkey|octagon|octopus|panda|parachute|
rhinoceros|rifle|roller coaster|scorpion|sea turtle|shark|submarine|tornado|whale|zebra|squiggle|zigzag|speedboat|
hot air balloon|dolphin|lobster|arm|beard|ear|elbow|eye|face|finger|foot|goatee|knee|leg|moustache|mouth|nose|toe|
tooth|bear|waterslide|diving board|boomerang|bat|palm tree|angel|brain|castle|circle|cruise ship|kangaroo|line|
mermaid|penguin|tiger|triangle|The Eiffel Tower|elephant|skull|square|snake|smiley face""".replace("\n", "").split("|"))


def counts(sources: tuple[Path, ...]) -> tuple[Counter[str], int, int]:
    """Scenes per thing, number of sales, number of scenes."""
    picks: Counter[str] = Counter()
    sales = scenes = 0
    for path in sources:
        for line in path.read_text().splitlines():
            sale = json.loads(line)
            sales += 1
            for e in sale["events"]:
                if e["kind"] == "imagine":
                    scenes += 1
                    picks.update(e["things"])
    return picks, sales, scenes


def main() -> None:
    """Count the picks of every drawable thing and write the quota table."""
    from sellanything_ml.buyers.sales_sft import SOURCES_B3, WRITTEN_SALES
    from sellanything_ml.buyers.stories import DRAWABLE_THINGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=WRITTEN_SALES / "scene_quota")
    a = ap.parse_args()
    newer = tuple(p for p in (WRITTEN_SALES / "sales_b3.jsonl",) if p.exists())   # written after B3, not trained on yet
    picks, sales, scenes = counts(SOURCES_B3 + newer)
    assert NOT_HOME <= DRAWABLE_THINGS, sorted(NOT_HOME - DRAWABLE_THINGS)
    rows = sorted(((t, picks[t]) for t in DRAWABLE_THINGS), key=lambda r: (r[1], r[0]))
    need = {q: sum(max(0, q - n) for t, n in rows if t not in NOT_HOME) for q in QUOTAS}
    bands = Counter("0" if n == 0 else "1-9" if n < 10 else "10-29" if n < 30 else "30+" for _, n in rows)
    summary = {"sales": sales, "scenes": scenes, "picks": sum(picks.values()), "things": len(rows),
               "home_things": len(rows) - len(NOT_HOME),
               "bands": {b: bands[b] for b in ("0", "1-9", "10-29", "30+")},
               "picks_needed": {f"to {q}": need[q] for q in QUOTAS},
               "scenes_needed_at_3_per_scene": {f"to {q}": -(-need[q] // 3) for q in QUOTAS}}
    a.out.with_suffix(".json").write_text(json.dumps({"summary": summary, "counts": dict(rows),
                                                      "not_home": sorted(NOT_HOME)}, indent=1))
    lines = [f"# Scene quota ({sales} sales, {scenes} scenes, {sum(picks.values())} picks)", "",
             f"Things: {len(rows)}; seen 0 times {bands['0']}, 1-9 {bands['1-9']}, 10-29 {bands['10-29']}, "
             f"30+ {bands['30+']}. {len(NOT_HOME)} things do not fit a home scene and are left out of the quotas. "
             f"Picks still needed for the other {len(rows) - len(NOT_HOME)}: {need[10]:,} to reach 10 each, "
             f"{need[30]:,} to reach 30 each "
             f"(at 3 things a scene: {summary['scenes_needed_at_3_per_scene']['to 10']:,} and "
             f"{summary['scenes_needed_at_3_per_scene']['to 30']:,} scenes).", "",
             "| thing | scenes now | to 10 | to 30 |", "|---|---|---|---|"]
    lines += [f"| {t} | {n} | {max(0, 10 - n)} | {max(0, 30 - n)} |" if t not in NOT_HOME
              else f"| {t} | {n} | (not a home thing) | |" for t, n in rows]
    a.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
