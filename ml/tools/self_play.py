"""The model on both sides of a cat sale (owner, 2026-10-28): a prompt-format-8 model (B5) is the seller and every
buyer, through the game's own sale (buyers/cat_market.py, `seller=`), so it runs exactly what the game would run.

    cd ml && PYTHONPATH=. python3 tools/self_play.py runs/sft_b5/best.pt --sales 10 --out runs/sft_b5/self_play

Each sale's cat is the sheet stories.sale_for_seed gives its seed, priced as the written sales are (asking.asking).
The seeds start at 5000 by default, away from every written sale (2048-2999), so the cats are not ones the model
trained on. Writes a readable transcript of every sale (`sales.md`) and a summary (`summary.json`): unreadable lines,
the seller's decisions and how many break the written decisions' rules (cat_market.decide_problem), the replies to
counter-offers, and the deals. Nothing is picked or retried: every line is the model's first sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sellanything_ml.buyers.cat_market import BuyerGPT, CatMarket, seller_card, seller_sheet


def transcript(seed: int, m: CatMarket) -> str:
    """One sale for reading: the seller's sheet, the buyers' cards, every line in order, the decisions and the deal."""
    lines = [f"## Sale {seed}", "", f"- seller: {seller_card(m.seller)}"]
    lines += [f"- {n}, who {c['persona']}: cares about {', '.join(c['cares_about'])}" for n, c in m.cards.items()]
    lines += ["", *[f"{i + 1}. {line}" + ("  *(unreadable)*" if src == "model_unreadable" else "")
                    for i, (line, src) in enumerate(zip(m.log, m.sources, strict=True))], ""]
    for name, d in m.decisions.items():
        reply = d.get("reply")
        lines.append(f"- {name}: final ${d['offer']} → {d.get('action') or 'unreadable'}"
                     + (f" ${d['price']}" if d.get("price") else "")
                     + (f" — rule: {d['problem']}" if d.get("problem") else "")
                     + (f" — reply: {reply['action']}" if reply else ""))
    sold = m.sold or {}
    lines += ["", f"**Sold to {sold.get('buyer') or 'nobody'}**"
              + (f" at ${sold['price']} (asking ${sold['asking']})" if sold.get("buyer") else ""), ""]
    return "\n".join(lines)


def summarize(sales: list[tuple[int, CatMarket]]) -> dict[str, Any]:
    """Counts over all sales: lines and unreadable lines, decisions by action and rule problems, replies, deals."""
    decisions = [d for _, m in sales for d in m.decisions.values()]
    read = [d for d in decisions if d.get("action")]
    replies = [d["reply"] for d in read if d.get("reply")]
    sold = [m.sold for _, m in sales if m.sold and m.sold["buyer"]]
    return {
        "sales": len(sales),
        "lines": sum(m.stats["lines"] for _, m in sales),
        "unreadable": sum(m.stats["unreadable"] for _, m in sales),
        "decisions": {"n": len(decisions), "unreadable": len(decisions) - len(read),
                      "actions": dict(Counter(d["action"] for d in read)),
                      "rule_problems": sum(bool(d.get("problem")) for d in read),
                      "problems": [d["problem"] for d in read if d.get("problem")]},
        "replies": dict(Counter(r["action"] for r in replies)),
        "sold": len(sold),
        "price_over_asking": [round(s["price"] / s["asking"], 2) for s in sold],
    }


def main() -> None:
    """Play the sales, writing the transcripts as they finish and the summary at the end."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sales", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--buyers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gpt = BuyerGPT(a.checkpoint, a.device)
    if gpt.prompt_format != 8:
        raise SystemExit(f"{a.checkpoint} is prompt format {gpt.prompt_format}; self-play needs format 8")
    a.out.mkdir(parents=True, exist_ok=True)
    sales = []
    with (a.out / "sales.md").open("w") as md:      # each sale as soon as it is done, so a stopped run keeps them
        for seed in range(a.first_seed, a.first_seed + a.sales):
            m = CatMarket.start(gpt, a.buyers, seed=seed, seller=seller_sheet(seed))
            sales.append((seed, m))
            md.write(transcript(seed, m) + "\n")
            md.flush()
            print(f"sale {seed}: {m.stats['lines']} lines, sold to {m.sold['buyer'] or 'nobody'}", flush=True)
    report = {"checkpoint": str(a.checkpoint), "first_seed": a.first_seed, **summarize(sales)}
    (a.out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
