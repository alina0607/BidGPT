"""Write a single-file, dependency-free checker for cat sales written by another model (owner, 2026-10-12: Muse runs in
the cloud, so it gets the whole checker and fixes its own sales before sending them).

    PYTHONPATH=. python tools/make_sale_checker.py 2100 2199 ../data/external_sales/muse/check_sales.py

The rule functions are copied from sellanything_ml/buyers/stories.py with inspect, so the checker applies exactly the
rules this repository applies (tools/write_sales.py); the seeds' buyer names are written into it. On top of those it
checks what tools/eval_buyer_b1.py measures: reasonable moves, questions new in meaning, and notes.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

from sellanything_ml.buyers import stories as S

FUNCTIONS = (S._words, S._spoken_problem, S._offer_problem, S._amount_problem, S.reaction_problem, S._mentions,
             S._name_at, S._named, S.imagine_problem, S._scene_problem)

DRIVER = '''

MAX_MOVE, MAX_PEAK, SAME_QUESTION, NOTE_WORDS = 0.40, 1.6, 0.5, 12
# Quality gates (owner, 2026-10-12, after Muse's first 100 sales): v9's corrected sales meet them almost always
# (3% of reactions under 8 words), telegraphic or copied lines do not.
MIN_REACTION_WORDS, MIN_QUESTION_WORDS, MIN_SCENE_WORDS, SAME_OPENING = 8, 5, 12, 3
FIRST_PERSON = re.compile(r"\\b(i|i'd|i'll|i'm|i've|my|me|we|our|us)\\b", re.I)


def norm(text):
    return re.sub(r"\\$\\d+", "$N", " ".join(str(text).lower().split()))


def repeats(question, asked):
    """At least half of the question's content words were in one earlier question."""
    words = _words(question)
    return bool(words) and any(len(words & _words(q)) >= SAME_QUESTION * len(words) for q in asked)


def check_sale(seed, spec):
    """Every rule the sale breaks, as readable strings (empty list = the sale passes)."""
    p = []
    names = BUYERS.get(seed)
    if names is None:
        return [f"seed {seed} is not one of the seeds this checker knows"]
    for part in ("cards", "open", "rounds", "imagine"):
        if part not in spec:
            return [f"missing \\"{part}\\""]
    for part in ("cards", "open"):
        if set(spec[part]) != set(names):
            p.append(f"{part}: buyers must be exactly {sorted(names)}, got {sorted(spec[part])}")
    offers, opening = {}, {}
    for n in names:
        try:
            offer, says = spec["open"][n]
            offers[n] = opening[n] = int(offer)
        except (KeyError, TypeError, ValueError):
            p.append(f"open {n}: must be [offer, greeting]")
            continue
        if offer <= 0:
            p.append(f"open {n}: the opening offer must be above $0")
        problem = _spoken_problem(str(says), n)
        if problem:
            p.append(f"open {n}: {problem}")
        cares = spec["cards"].get(n, {}).get("cares_about")
        if not isinstance(cares, list) or not 2 <= len(cares) <= 3:
            p.append(f"cards {n}: cares_about must be a list of two or three phrases")
    asked_by, questions = set(), []
    for k, rnd in enumerate(spec["rounds"], 1):
        where = f"round {k}"
        asker, q, a, note = rnd.get("name"), str(rnd.get("q", "")), str(rnd.get("a", "")), str(rnd.get("note", ""))
        present = [n for n in names if offers.get(n, 0) > 0]
        if asker not in present:
            p.append(f"{where}: {asker} is not a buyer still bidding and cannot ask")
        if asker in asked_by:
            p.append(f"{where}: {asker} already asked once")
        asked_by.add(asker)
        if len(q.split()) < MIN_QUESTION_WORDS:
            p.append(f"{where}: the question has under {MIN_QUESTION_WORDS} words; ask it the way a person would: {q!r}")
        if repeats(q, questions):
            p.append(f"{where}: the question repeats an earlier one in meaning: {q!r}")
        questions.append(q)
        for label, text in (("question", q), ("answer", a), ("note", note)):
            if any(d in text for d in DASHES):
                p.append(f"{where}: the {label} uses a dash")
        if not note or len(note.split()) > NOTE_WORDS:
            p.append(f"{where}: the note must be 1 to {NOTE_WORDS} words")
        react = rnd.get("react", {})
        openings = {}
        for n, r in react.items():
            if isinstance(r, list) and len(r) == 3:
                openings.setdefault(" ".join(str(r[2]).lower().split()[:3]), []).append(n)
        for start, who in openings.items():
            if len(who) >= SAME_OPENING:
                p.append(f"{where}: {len(who)} buyers start the same way ({start!r}: {', '.join(who)}); vary them")
        missing, extra = set(present) - set(react), set(react) - set(present)
        if missing or extra:
            p.append(f"{where}: reactions missing for {sorted(missing)}, from buyers not bidding {sorted(extra)}")
        for n in [x for x in present if x in react]:
            try:
                sign, offer, says = react[n]
                offer = int(offer)
            except (TypeError, ValueError):
                p.append(f"{where} {n}: a reaction must be [sign, offer, says]")
                continue
            problem = reaction_problem({"sign": sign, "offer": offer, "says": says}, {}, offers[n], n, a)
            if not problem and len(str(says).split()) < MIN_REACTION_WORDS:
                problem = (f"the line has under {MIN_REACTION_WORDS} words; say what the seller said and why it "
                           "matters for your own life")
            if problem:
                p.append(f"{where} {n}: {problem}")
            elif offer and (abs(offer - offers[n]) > MAX_MOVE * offers[n] or offer > MAX_PEAK * opening[n]):
                p.append(f"{where} {n}: ${offers[n]} to ${offer} is not reasonable (at most {MAX_MOVE:.0%} of the "
                         f"offer in one move, at most {MAX_PEAK} times the opening ${opening[n]})")
            offers[n] = offer
    present = [n for n in names if offers.get(n, 0) > 0]
    if set(spec["imagine"]) != set(present):
        p.append(f"imagine: scenes must be from exactly the buyers still bidding {sorted(present)}")
    taken = []
    for n in [x for x in present if x in spec["imagine"]]:
        try:
            things, says = spec["imagine"][n]
        except (TypeError, ValueError):
            p.append(f"imagine {n}: must be [[three things], says]")
            continue
        problem = _scene_problem({"things": things, "says": says}, taken)
        if problem:
            p.append(f"imagine {n}: {problem}")
        else:
            taken.append(set(map(str, things)))
        if any(d in str(says) for d in DASHES):
            p.append(f"imagine {n}: the line uses a dash")
        if not FIRST_PERSON.search(str(says)) or len(str(says).split()) < MIN_SCENE_WORDS:
            p.append(f"imagine {n}: say it in the first person, at least {MIN_SCENE_WORDS} words, naming what each "
                     "thing is for in your life")
    return p


def copied(specs, against):
    """Lines that also appear in another sale: (seed, place) -> problem. `against` holds sales already accepted."""
    seen = {}
    for seed, spec in list(against.items()) + list(specs.items()):
        for n, v in spec.get("open", {}).items():
            seen.setdefault(("greeting", norm(v[1] if isinstance(v, list) and len(v) == 2 else v)), set()).add(seed)
        for rnd in spec.get("rounds", []):
            seen.setdefault(("question", norm(rnd.get("q", ""))), set()).add(seed)
            for n, r in rnd.get("react", {}).items():
                if isinstance(r, list) and len(r) == 3:
                    seen.setdefault(("reaction", norm(r[2])), set()).add(seed)
    out = {}
    for (kind, text), seeds in seen.items():
        if len(seeds) > 1:
            for seed in seeds & set(specs):
                out.setdefault(seed, []).append(f"this {kind} also appears in sale(s) "
                                                f"{sorted(seeds - {seed})}: {text!r}; write it in your own words")
    return out


def load(path):
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out, broken = {}, {}
    for f in files:
        try:
            out[int(f.stem)] = json.loads(f.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            broken[f.name] = str(e)
    return out, broken


def main():
    """python check_sales.py <file.json or folder> [--against <folder of accepted sales>]"""
    args = sys.argv[1:]
    against = {}
    if "--against" in args:
        i = args.index("--against")
        against, _ = load(Path(args[i + 1]))
        args = args[:i] + args[i + 2:]
    specs, broken = load(Path(args[0]))
    for name, error in broken.items():
        print(f"{name}: not valid JSON ({error})")
    dup = copied(specs, {k: v for k, v in against.items() if k not in specs})
    bad = len(broken)
    for seed in sorted(specs):
        problems = check_sale(seed, specs[seed]) + dup.get(seed, [])
        bad += bool(problems)
        print(f"{seed}.json: {'OK' if not problems else str(len(problems)) + ' problems'}")
        for problem in problems:
            print(f"  - {problem}")
    print(f"{len(specs) - bad + len(broken)} of {len(specs) + len(broken)} sales pass")
    sys.exit(1 if bad else 0)


def _old_main():
    target = Path(sys.argv[1])
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    bad = 0
    for f in files:
        try:
            spec = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"{f.name}: not valid JSON ({e})")
            bad += 1
            continue
        problems = check_sale(int(f.stem), spec)
        bad += bool(problems)
        print(f"{f.name}: {'OK' if not problems else str(len(problems)) + ' problems'}")
        for problem in problems:
            print(f"  - {problem}")
    print(f"{len(files) - bad} of {len(files)} sales pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
'''


def checker(first: int, last: int) -> str:
    buyers = {seed: [b["name"] for b in S.sale_for_seed(seed)["buyers"]] for seed in range(first, last + 1)}
    head = f'''"""Checks cat sales for the Sell Anything buyer model: every rule the training sales are held to.

    python check_sales.py <seed>.json          # one sale
    python check_sales.py <folder>             # every <seed>.json in a folder

A sale for seed N is the JSON object asked for in the prompt for seed N, saved as N.json. Every problem is printed
with its round and buyer; fix them and run again until every sale passes. Needs only Python 3.9+.
Generated from the Sell Anything repository (ml/tools/make_sale_checker.py); do not edit by hand.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

BUYERS = {buyers!r}

SIGNS = {S.SIGNS!r}
DASHES = {S.DASHES!r}
DOLLARS = re.compile({S.DOLLARS.pattern!r})
VAGUE = {S.VAGUE!r}
BROKEN_OPENER = re.compile({S.BROKEN_OPENER.pattern!r}, re.I)
STOP = frozenset({sorted(S.STOP)!r})
MAX_THINGS = {S.MAX_THINGS!r}
DRAWABLE_THINGS = frozenset({sorted(S.DRAWABLE_THINGS)!r})
EVERYDAY_WORDS = frozenset({sorted(S.EVERYDAY_WORDS)!r})
NOT_DRAWABLE = {S.NOT_DRAWABLE!r}


'''
    body = "\n\n".join(inspect.getsource(f) for f in FUNCTIONS)
    return head + body + DRIVER


def main() -> None:
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    out.write_text(checker(first, last))
    print(f"checker for seeds {first}-{last} written to {out}")


if __name__ == "__main__":
    main()
