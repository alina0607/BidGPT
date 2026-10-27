"""Does a per-token bias at sampling time (the `logit_bias` of an OpenAI-style API) fix a buyer model's scenes?
Held-out imagine prompts, sampled the way the game samples (temperature 0.7, top-k 50), checked by the same program
rule as the evaluation (stories.imagine_problem). Nothing is trained; only the sampling changes.

    cd ml && PYTHONPATH=. python tools/probe_logit_bias.py runs/sft_b4/best.pt \
        --data runs/eval_sets_b2/val_new_format5.jsonl --out runs/sft_b4/logit_bias/new

Variants (each adds to the one before; a banned token gets probability 0 before the top-k cut):
  base   model.generate as the evaluation runs it, the same seed per prompt (reproduces its rules_ok)
  base2  the same, seed + 1: how much the rate moves from sampling alone
  list   in the list part, only tokens that keep it "name, name, name |" with names on the drawing GPT's list
  line   in the spoken part, no token that makes the line mention an unpicked drawable thing or a NOT_DRAWABLE word
  end    <|end|> is banned until the line names every picked thing (the rule's own _named)

A ban looks only at the text so far, so a word that would go on ("bowl" in "bowling") is banned too.
Every problem of a scene is counted (imagine_problem returns the first only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]

from eval_buyer_b1 import TEMPERATURE, TOP_K  # noqa: E402

from sellanything_ml.buyers import stories  # noqa: E402
from sellanything_ml.buyers.cat_market import MAX_NEW, parse  # noqa: E402
from sellanything_ml.buyers.format import END  # noqa: E402
from sellanything_ml.buyers.stories import (DRAWABLE_THINGS, EVERYDAY_WORDS, MAX_THINGS,  # noqa: E402
                                            NOT_DRAWABLE, _mentions, _named, imagine_problem)
from sellanything_ml.llm.evaluate import load_model  # noqa: E402
from sellanything_ml.llm.pretrain import pick_device  # noqa: E402

VARIANTS = ("base", "base2", "list", "line", "end")
KINDS = stories.IMAGINE_KINDS
PREFIXES = {n[:k] for n in DRAWABLE_THINGS for k in range(len(n) + 1)}     # every start of a name
WATCHED = sorted((DRAWABLE_THINGS - EVERYDAY_WORDS) | set(NOT_DRAWABLE), key=len, reverse=True)
TAIL = re.compile(r"\b(" + "|".join(re.escape(w.lower()) for w in WATCHED) + r")s?$")   # a watched word just ended


def problems(row: dict[str, Any] | None) -> set[str]:
    """Every rule a scene breaks, by kind (stories.imagine_kinds), or "unreadable"."""
    return {"unreadable"} if row is None else stories.imagine_kinds(row)


def offending(picked: list[str], says: str) -> set[str]:
    """The words of a line that break the "name only the things picked" rules (unpicked drawable, NOT_DRAWABLE)."""
    unpicked = {n for n in _mentions(DRAWABLE_THINGS - EVERYDAY_WORDS, says)
                if not any(re.search(rf"\b{re.escape(n)}\b", t) for t in picked)}
    return unpicked | {w for w in _mentions(NOT_DRAWABLE, says) if not any(w in t for t in picked)}


def list_ok(text: str) -> bool:
    """Can the list part still become "name, name, name |" with every name on the drawing GPT's list?"""
    head, bar, _ = text.partition("|")
    items = head.split(",")
    done, partial = [i.strip() for i in items[:-1]], items[-1].lstrip()
    if len(items) > MAX_THINGS or any(d not in DRAWABLE_THINGS for d in done):
        return False
    if bar:
        return len(items) == MAX_THINGS and partial.strip() in DRAWABLE_THINGS
    return partial in PREFIXES


class Sampler:
    """The game's sampling (temperature, top-k) with a ban applied to every candidate before the top-k cut."""

    def __init__(self, model: Any, tok: Any, variant: str):
        self.model, self.tok, self.variant = model, tok, variant
        self.pieces = [self._piece(i) for i in range(model.cfg.vocab_size)]

    def _piece(self, i: int) -> str:
        if i in self.tok.extra_text:
            return self.tok.extra_text[i]
        if i >= self.tok.sp_size:
            return "\0"                                    # an id this tokenizer does not know: never a name
        p = self.tok.sp.id_to_piece(i)
        return "" if p.startswith("<0x") else p.replace("▁", " ")

    def allowed(self, text: str, i: int) -> bool:
        """May token i come next after the completion `text`?"""
        v, cand = self.variant, text + self.pieces[i]
        if "|" not in text:                                    # the list part
            return i != self.tok.eos_id and list_ok(cand)
        picked = [t.strip() for t in text.partition("|")[0].split(",")]
        says = text.partition("|")[2]
        if i == self.tok.eos_id:
            return v != "end" or all(_named(t, says) for t in picked)
        if v in ("line", "end") and TAIL.search(cand.partition("|")[2].lower()):
            return not (offending(picked, cand.partition("|")[2]) - offending(picked, says))
        return True

    @torch.no_grad()
    def complete(self, ids: torch.Tensor, max_new: int) -> tuple[str, float]:
        """A completion, and the probability the unbiased sampler would have put on banned tokens, added over the
        steps (the expected number of banned picks: how much the ban had to change)."""
        cache: list[dict[str, Any]] = [{} for _ in self.model.blocks]
        logits, _ = self.model(ids, cache=cache)
        pos, out, text, banned = ids.size(1), [], "", 0.0
        for _ in range(max_new):
            scores = (logits[0, -1].float() / TEMPERATURE).cpu()
            order = scores.argsort(descending=True).tolist()
            top = scores[order[:TOP_K]].softmax(-1).tolist()          # what the game's sampler would draw from
            keep = []
            for rank, i in enumerate(order):
                if self.allowed(text, i):
                    keep.append(i)
                    if len(keep) == TOP_K:
                        break
                elif rank < TOP_K:
                    banned += top[rank]
            if not keep:
                break
            nxt = keep[int(torch.multinomial(scores[keep].softmax(-1), 1))]
            out.append(nxt)
            text += self.pieces[nxt]
            if nxt == self.tok.eos_id:
                break
            logits, _ = self.model(torch.tensor([[nxt]], device=ids.device), cache=cache, start_pos=pos)
            pos += 1
        return self.tok.decode(out).replace(END, "").strip(), banned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="prefix: <out>_summary.json and <out>_outputs.jsonl")
    ap.add_argument("--n", type=int, default=0, help="at most this many imagine prompts (0 = all)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--device")
    a = ap.parse_args()
    device = torch.device(a.device) if a.device else pick_device()
    model, tok = load_model(a.checkpoint, device)
    rows = [json.loads(line) for line in a.data.read_text().splitlines()]
    picked = [(i, ex) for i, ex in enumerate(rows) if ex["task"] == "imagine"]   # i: the evaluation's seed
    if a.n:
        picked = picked[:a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"checkpoint": str(a.checkpoint), "data": str(a.data), "prompts": len(picked),
                               "device": str(device), "variants": {}}
    with open(f"{a.out}_outputs.jsonl", "w") as log:
        for variant in a.variants.split(","):
            sampler = None if variant.startswith("base") else Sampler(model, tok, variant)
            kinds: Counter[str] = Counter()
            ok, banned = 0, 0.0
            start = time.time()
            for i, ex in picked:
                torch.manual_seed(i + (1 if variant == "base2" else 0))
                ids = torch.tensor([tok.encode(ex["prompt"])], device=device)
                if sampler is None:
                    gen = model.generate(ids, MAX_NEW["imagine"], temperature=TEMPERATURE, top_k=TOP_K,
                                         stop_id=tok.eos_id)
                    out, b = tok.decode(gen[0, ids.size(1):].tolist()).replace(END, "").strip(), 0
                else:
                    out, b = sampler.complete(ids, MAX_NEW["imagine"])
                row = parse("imagine", out)
                found = problems(row)
                rule = imagine_problem(row) if row else "unreadable"
                assert (rule is None) == (not found), (out, rule, found)    # same verdict as the rule
                ok += not found
                banned += b
                kinds.update(found)
                log.write(json.dumps({"variant": variant, "index": i, "seed": ex["seed"], "name": ex["name"],
                                      "output": out, "problems": sorted(found), "banned_mass": round(b, 3)}) + "\n")
            n = len(picked)
            summary["variants"][variant] = {"rules_ok": round(ok / n, 3), "passed": ok,
                                            "kinds": {k: kinds[k] for k in (*KINDS, "unreadable")},
                                            "banned_mass_per_scene": round(banned / n, 2),
                                            "seconds": round(time.time() - start)}
            print(variant, json.dumps(summary["variants"][variant]), flush=True)
    Path(f"{a.out}_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
