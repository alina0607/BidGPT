"""Does the model read negation and contrast words? Minimal pairs: two contexts that differ only by a negation (or
"but", or a quantity word), each with the conclusion that fits it and the one that fits the other. A model that
ignores the small word picks the same conclusion for both contexts and scores 50%. Scored on the conclusion's tokens
given the context, for the pre-trained model (before any sales) and for B5 (after fine-tuning on sales); pass other
checkpoints to score them instead (owner, 2026-10-28: does the model read "doesn't", "no", "but"?).

The trap test (data/written_sales/prompts_3000_plan.md section 5.2) scores groups of seller answers read by one buyer
(ml/data/eval/trap_test/items.jsonl, one JSON object per line: group, part, persona, cares_about, greeting, question,
answer, want "+"/"0"/"-", cue). A group is right only when every answer in it gets the right direction; also reported:
answers right, pairs of answers with different directions both right, and those where the small word moves the
probabilities the right way. Three baselines that read no small word (or only a negation word) are printed with it:
a model that does not beat them has learned a shortcut.

    cd ml && python3 tools/negation_probe.py                         # the pre-trained model and B5
    cd ml && python3 tools/negation_probe.py runs/sft_b6/best.pt     # any checkpoints
    cd ml && python3 tools/negation_probe.py --baselines             # the trap test's baselines only, no model
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, ".")

ITEMS_FILE = Path("data/eval/trap_test/items.jsonl")
PARTS = ("T", "H", "C", "U", "P", "A", "N9", "O16")
FIELDS = {"group", "part", "persona", "cares_about", "greeting", "question", "answer", "want", "cue"}
SIGNS = "+0-"

ITEMS = [  # (kind, context that fits A, context that fits B, conclusion A, conclusion B)
    ("quantity", "She sheds a lot,", "She sheds very little,", " so she is a bad cat for someone with allergies.", " so she is a good cat for someone with allergies."),
    ("doesn't", "She sheds a lot,", "She doesn't shed much,", " so she is a bad cat for someone with allergies.", " so she is a good cat for someone with allergies."),
    ("not", "It is raining,", "It is not raining,", " so take an umbrella.", " so leave the umbrella at home."),
    ("not", "The shop is open today,", "The shop is not open today,", " so we can buy bread there.", " so we cannot buy bread there."),
    ("doesn't", "He likes children,", "He doesn't like children,", " so he is a good pet for a family.", " so he is a poor pet for a family."),
    ("never", "She has been vaccinated,", "She has never been vaccinated,", " so she does not need her shots.", " so she still needs her shots."),
    ("not", "The kitten is already spayed,", "The kitten is not spayed yet,", " so the new owner will not pay for the operation.", " so the new owner will pay for the operation."),
    ("barely", "He eats well,", "He barely eats,", " so the vet is pleased.", " so the vet is worried."),
    ("don't", "I have plenty of money,", "I don't have much money,", " so I can buy an expensive cat.", " so I want a cheap cat."),
    ("isn't", "The water is hot,", "The water isn't hot,", " so it is dangerous to touch.", " so it is safe to touch."),
    ("ain't", "She has papers,", "She ain't got no papers,", " so she is a registered pedigree.", " so she is not a registered pedigree."),
    ("not", "The dog is friendly,", "The dog is not friendly,", " so the children can play with it.", " so the children should stay away from it."),
    ("never", "She gets along with other cats,", "She has never met another cat,", " so she will be fine with your old cat.", " so nobody knows how she will treat your old cat."),
    ("not", "The room is dark,", "The room is not dark,", " so we need a lamp.", " so we do not need a lamp."),
    ("hardly", "He scratches all the time,", "He hardly ever scratches,", " so your sofa is in danger.", " so your sofa is safe."),
    ("does not", "She makes noise at night,", "She does not make noise at night,", " so you will lose sleep.", " so you will sleep well."),
    ("not", "He is used to being alone,", "He is not used to being alone,", " so he will be fine while you work.", " so he may cry while you work."),
    ("no", "There is milk in the fridge,", "There is no milk in the fridge,", " so we can make tea with milk.", " so we must buy milk."),
    ("don't", "I know her age,", "I don't know her age,", " so I can tell you how old she is.", " so I cannot tell you how old she is."),
    ("but", "The car is old and unreliable,", "The car is old but reliable,", " so it often breaks down.", " so it rarely breaks down."),
    ("but", "He is rich and generous,", "He is rich but stingy,", " so he often gives money away.", " so he rarely gives money away."),
    ("but", "The cat is young and healthy,", "The cat is young but sick,", " so she needs no vet visits soon.", " so she needs vet visits soon."),
    ("word", "The test was positive,", "The test was negative,", " so she has the virus.", " so she does not have the virus."),
    ("word", "She is spayed,", "She is intact,", " so the operation is already done.", " so the operation is still to be done."),
]


# The same test in a sale: one buyer's real prompt, the seller's answer changed by one small word, and the direction
# the review rules call for (+ up, 0 keep, - down). Scored on the first token of the reaction (the direction sign).
SALE = [  # (persona, cares about, greeting, question, answer A, sign A, answer B, sign B)
    ("is allergic and worried about fur", ["gentle on allergies", "little fur on clothes"],
     "My eyes stream around most cats, so I need one that leaves very little fur about.", "How much does she shed?",
     "She sheds a lot.", "-", "She doesn't shed much.", "+"),
    ("is allergic and worried about fur", ["gentle on allergies", "little fur on clothes"],
     "My eyes stream around most cats, so I need one that leaves very little fur about.", "How much does she shed?",
     "She sheds a lot.", "-", "She barely sheds at all.", "+"),
    ("is allergic and worried about fur", ["gentle on allergies", "little fur on clothes"],
     "My eyes stream around most cats, so I need one that leaves very little fur about.", "How much does she shed?",
     "She has long fur and sheds everywhere.", "-", "She has long fur but hardly sheds.", "+"),
    ("is buying for a family with young kids", ["gentle with young kids", "patient", "healthy"],
     "My two boys are four and six, so I need a cat who stays calm when they get loud.", "Is she good with children?",
     "Yes, she loves children.", "+", "No, she doesn't like children.", "-"),
    ("is buying for a family with young kids", ["gentle with young kids", "patient", "healthy"],
     "My two boys are four and six, so I need a cat who stays calm when they get loud.", "Is she good with children?",
     "She's great with kids.", "+", "She isn't good with kids.", "-"),
    ("already has two dogs", ["fine with dogs", "confident", "easy-going"],
     "Comet and Rook, my two greyhounds, run the house, so I need a cat who won't mind them.", "Has she lived with dogs?",
     "Yes, she has lived with two dogs.", "+", "No, she has never lived with a dog.", "-"),
    ("already has an older cat at home", ["gets along with my old cat", "calm", "healthy"],
     "Bramble is thirteen and sleeps by the stove all day, so I need a cat who will leave him in peace.",
     "Has she lived with another cat?", "Yes, she lived with another cat for years.", "+",
     "Not really, she has never lived with another cat.", "-"),
    ("wants things in excellent shape", ["excellent health", "fully vaccinated", "no hidden problems"],
     "I repair clocks and one loose screw means the job isn't done, so I want a cat in top condition.",
     "Has she had her shots?", "Yes, all her shots are done.", "+", "No, she has never had any shots.", "-"),
    ("wants things in excellent shape", ["excellent health", "fully vaccinated", "no hidden problems"],
     "I repair clocks and one loose screw means the job isn't done, so I want a cat in top condition.",
     "Has she been tested for FIV?", "She tested negative for FIV.", "+", "She tested positive for FIV.", "-"),
    ("wants things in excellent shape", ["excellent health", "fully vaccinated", "no hidden problems"],
     "I repair clocks and one loose screw means the job isn't done, so I want a cat in top condition.",
     "How is her health?", "She is healthy and has no problems.", "+", "She is healthy but has a heart murmur.", "-"),
    ("wants things in excellent shape", ["excellent health", "fully vaccinated", "no hidden problems"],
     "I repair clocks and one loose screw means the job isn't done, so I want a cat in top condition.",
     "Is she spayed?", "Yes, she's already spayed.", "0", "No, she isn't spayed yet.", "-"),
    ("loves rare, premium and pedigree things", ["a recognised pedigree", "papers to prove it"],
     "I collect antique swords with papers back to their makers, so I want a pedigree cat with papers.",
     "Does she have papers?", "Yes, she has her pedigree papers.", "+", "No, she ain't got no papers.", "-"),
    ("works long hours and has little free time", ["fine on her own all day", "few vet errands"],
     "I'm out from seven till seven most days, so I need a cat who copes alone.", "Is she fine on her own?",
     "Yes, she's fine on her own all day.", "+", "No, she isn't okay alone; she cries when left.", "-"),
    ("lives on a busy road", ["stays indoors", "calm", "not a runner"],
     "Lorries thunder past my door all night, so I need a cat who never wants to go out.", "Does she go outside?",
     "She's an indoor cat and never goes out.", "+", "She goes out all the time.", "-"),
    ("cares a lot about how things look", ["looks right in a styled home", "gentle on furniture"],
     "Everything in my flat is white and gold, so I want a cat who won't wreck it.", "Does she scratch furniture?",
     "She scratches the sofa a lot.", "-", "She doesn't scratch furniture at all.", "+"),
    ("lives in a small apartment", ["quiet", "happy in a small space"],
     "My flat is one room above a bakery, so I need a quiet cat who is happy indoors.", "Is she noisy?",
     "She meows a lot, day and night.", "-", "She's quiet; she doesn't meow much.", "+"),
]

# The situation of an answer (notebook 10 section 4, most specific first): the topic the keyword-only baseline reads.
TOPICS = [
    ("fiv", r"\bfiv\b|\bfelv\b"), ("spay", r"spay|neuter|\bfixed\b|\bintact\b|\bentire\b"),
    ("papers", r"papers?\b|pedigree|registered|lineage|purebred"), ("children", r"\bkids?\b|child"),
    ("dogs", r"(?<!like a )\bdogs?\b"), ("other cats", r"another cat|other cats?|\bcats?\b.*\blived with"),
    ("vaccines/vet", r"vaccin|shots?\b|\bvet\b|check-?up|bloodwork"),
    ("illness", r"heart|diabet|thyroid|kidney|cold\b|blind|deaf|three legs|teeth|stiff|arthrit|pills|insulin"),
    ("age", r"\b\d+\s*(years?|yrs?|months?)\b|\bkitten\b|\bsenior\b|\bold\b"),
    ("indoor/outdoor", r"indoor|outdoor|outside|catio|roam|farm"),
    ("origin", r"stray|rehom|litter of|previous owner|moved abroad|care home|allergic\b"),
    ("food", r"\bfood\b|\beats?\b|chicken|\bwet\b|\bdry\b|fussy"), ("handling", r"picked up|carried|\bhold\b|wriggle"),
    ("fleas/mites", r"flea|mite"), ("microchip", r"microchip|\bchip"), ("litter", r"\blitter\b"),
    ("coat/shedding", r"\bcoat\b|\bfur\b|shed|groom|fluffy|long-?haired|short-?haired"),
    ("temperament", (r"calm|lazy|playful|shy|cuddly|independent|affection|friendly|energetic|nervous|follows people|"
                     r"noisy|vocal|meows")),
    ("habits", r"vacuum|\btap\b|scratch|sofa|post\b|claw"), ("price", r"\$\d+|price|offer|cheap"),
]
# Any negation word, the held-out ones included: the negation-only baseline knows them all.
NEGATION = re.compile(r"n't\b|\b(not|no|never|none|nothing|nobody|without|nope|nah|barely|hardly|rarely|seldom|"
                      r"ain'?t|dont|doesnt|isnt|hasnt|wont|cant|didnt)\b", re.IGNORECASE)
# The held-out words (plan 5.1) and how often B5's training answers had them: barely, hardly and no longer a few
# times ("almost never seen"), the rest never.
HELD_OUT = {"barely": "H almost never seen", "hardly": "H almost never seen", "no longer": "H almost never seen",
            "ain't": "H never seen", "rarely": "H never seen", "seldom": "H never seen", "even though": "H never seen",
            "there's no denying": "H never seen"}
MIN_SEEN = 20               # training reactions a persona needs on a topic before its own majority is used


def situation(text: str) -> str:
    """The topic of a question and answer (TOPICS, first match), "other" when none matches."""
    low = text.lower().replace("’", "'")
    return next((name for name, pat in TOPICS if re.search(pat, low)), "other")


def load_items(path: Path = ITEMS_FILE) -> list[dict]:
    """The trap-test items, checked: every field, a known part, a direction sign, and one prompt per group."""
    items, prompts = [], {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if set(item) != FIELDS:
            raise ValueError(f"{path}:{n}: fields {sorted(set(item) ^ FIELDS)} differ from {sorted(FIELDS)}")
        if item["part"] not in PARTS or item["want"] not in SIGNS:
            raise ValueError(f"{path}:{n}: part {item['part']!r} or want {item['want']!r} is not allowed")
        prompt = (item["persona"], tuple(item["cares_about"]), item["greeting"], item["question"], item["part"])
        if prompts.setdefault(item["group"], prompt) != prompt:
            raise ValueError(f"{path}:{n}: group {item['group']} has another buyer, question or part than its first line")
        items.append(item)
    return items


def training_majority(sales: list[dict]) -> dict[tuple[str, str], str]:
    """The most common reaction sign in training sales per (persona, topic) and per ("*", topic): what a model that
    reads only the topic's keyword (and who the buyer is), no small word, would answer."""
    counts: dict[tuple[str, str], Counter] = {}
    for sale in sales:
        cards, topic = sale.get("cards", {}), None
        for e in sale.get("events", []):
            if e["kind"] == "answer":
                topic = situation(f"{e.get('q', '')} {e['says']}")
            elif e["kind"] == "react" and topic and str(e.get("sign")) in SIGNS:
                persona = cards.get(e["name"], {}).get("persona", "")
                for key in ((persona, topic), ("*", topic)):
                    counts.setdefault(key, Counter())[str(e["sign"])] += 1
    return {k: (c.most_common(1)[0][0] if k[0] == "*" or sum(c.values()) >= MIN_SEEN else "")
            for k, c in counts.items()}


def keyword_sign(item: dict, majority: dict[tuple[str, str], str]) -> str:
    """Baseline 2: the training majority for this buyer type on this topic (else for the topic, else keep)."""
    topic = situation(f"{item['question']} {item['answer']}")
    return majority.get((item["persona"], topic)) or majority.get(("*", topic)) or "0"


def negation_sign(item: dict, majority: dict[tuple[str, str], str]) -> str:
    """Baseline 1: lower when the answer has a negation word, else the keyword baseline."""
    return "-" if NEGATION.search(item["answer"].replace("’", "'")) else keyword_sign(item, majority)


def score(items: list[dict], probs: list[dict[str, float]]) -> dict[str, dict[str, int]]:
    """Per part and over all ("all"; "not all 0": the groups whose answers do not all want 0): groups with every answer
    right, answers right, pairs in a group with different wants both right, and pairs whose small word moves the
    probabilities the right way (the answer wanting x has the higher P(x) of the two, both ways). `probs` holds each
    item's sign probabilities, or a 1.0 on the sign a baseline picks."""
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item["group"], []).append(i)
    out: dict[str, dict[str, int]] = {}
    for members in groups.values():
        part = items[members[0]]["part"]
        right = [max(probs[i], key=probs[i].get) == items[i]["want"] for i in members]
        pairs = [(a, b) for a, b in combinations(members, 2) if items[a]["want"] != items[b]["want"]]
        keys = ["all", part] + ([] if all(items[i]["want"] == "0" for i in members) else ["not all 0"])
        if part == "H":
            words = [c for c in dict.fromkeys(items[i]["cue"] for i in members) if c in HELD_OUT]
            keys += [f"H {c}" for c in words] + list(dict.fromkeys(HELD_OUT[c] for c in words))
        for key in keys:
            s = out.setdefault(key, Counter())
            s["groups"] += 1
            s["groups right"] += all(right)
            s["answers"] += len(members)
            s["answers right"] += sum(right)
            s["pairs"] += len(pairs)
            s["pairs both right"] += sum(right[members.index(a)] and right[members.index(b)] for a, b in pairs)
            s["moved right"] += sum(probs[b][items[b]["want"]] > probs[a][items[b]["want"]]
                                    and probs[a][items[a]["want"]] > probs[b][items[a]["want"]] for a, b in pairs)
    return out


def picked(signs: list[str]) -> list[dict[str, float]]:
    """A baseline's picks as probabilities (1.0 on the sign picked)."""
    return [{x: float(x == s) for x in SIGNS} for s in signs]


def report(label: str, scores: dict[str, dict[str, int]], moved: bool = True) -> None:
    """One line per part: groups right, answers right, pairs both right (and moved the right way)."""
    print(label)
    order = ["all", "not all 0", *PARTS, "H almost never seen", "H never seen", *(f"H {w}" for w in HELD_OUT)]
    for key in [k for k in order if k in scores]:
        s = scores[key]
        line = (f"   {key:>16}: groups {s['groups right']}/{s['groups']}, answers {s['answers right']}/{s['answers']}"
                f", pairs both right {s['pairs both right']}/{s['pairs']}")
        print(line + (f", moved the right way {s['moved right']}/{s['pairs']}" if moved else ""))


def baselines(items: list[dict]) -> None:
    """The trap test's three baselines, which need no model."""
    from sellanything_ml.buyers.sales_sft import SOURCES_B5
    sales = [json.loads(line) for path in SOURCES_B5 for line in Path(path).read_text().splitlines() if line.strip()]
    majority = training_majority(sales)
    report("baseline: always keep", score(items, picked(["0"] * len(items))), moved=False)
    report(f"baseline: keyword only (the B5 training majority for this buyer type on this topic; {len(sales)} sales)",
           score(items, picked([keyword_sign(i, majority) for i in items])), moved=False)
    report("baseline: negation only (lower on any negation word, else keyword only)",
           score(items, picked([negation_sign(i, majority) for i in items])), moved=False)


def text_logp(net, tok, context, conclusion):
    """Log-probability of the conclusion's tokens given the context."""
    import torch
    c, full = tok.encode(context), tok.encode(context + conclusion)
    with torch.no_grad():
        logits, _ = net(torch.tensor([full]))
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    return sum(logp[i - 1, full[i]].item() for i in range(len(c), len(full)))


def sale_signs(net, tok, persona, cares, greeting, question, answer):
    """The probabilities of the three direction signs for this buyer's reaction to the answer."""
    import torch

    from sellanything_ml.buyers.format import REACT
    from sellanything_ml.buyers.sales_sft import Story, animal, buyer_prompt, card
    s = Story("Robin")
    s.opening = f"Your opening ($70): {greeting}"
    s.asked("Sam", question)
    s.rounds[-1].append(f"Seller: {answer}")
    p = buyer_prompt(card("Robin", {"persona": persona, "cares_about": cares}, 70, net.prompt_format), s.text(live=True),
                     REACT, animal(net.prompt_format))
    with torch.no_grad():
        logits, _ = net(torch.tensor([tok.encode(p)]))
    ids = {sign: tok.encode(sign)[0] for sign in SIGNS}
    probs = torch.softmax(torch.stack([logits[0, -1, ids[sign]] for sign in SIGNS]), -1)
    return dict(zip(SIGNS, probs.tolist(), strict=True))


def general(net, tok, label):
    """The 24 general-English minimal pairs."""
    right, both, by = 0, 0, {}
    for kind, ca, cb, a, b in ITEMS:
        ok_a = text_logp(net, tok, ca, a) > text_logp(net, tok, ca, b)
        ok_b = text_logp(net, tok, cb, b) > text_logp(net, tok, cb, a)
        right += ok_a + ok_b
        both += ok_a and ok_b
        by.setdefault(kind, []).append(ok_a and ok_b)
    n = len(ITEMS)
    print(f"{label}: right {right}/{2 * n} ({right / (2 * n):.0%}); pairs with both right {both}/{n} ({both / n:.0%})")
    print("   both right by kind:", ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by.items()))


def sale_pairs(net, tok, label):
    """The original 16 sale pairs (B5: 4/16 both right, 15/16 moved the right way)."""
    both = moved = 0
    for persona, cares, greeting, q, a, sa, b, sb in SALE:
        pa, pb = (sale_signs(net, tok, persona, cares, greeting, q, x) for x in (a, b))
        both += max(pa, key=pa.get) == sa and max(pb, key=pb.get) == sb
        moved += pb[sb] > pa[sb] and pa[sa] > pb[sa]
    print(f"{label} in a sale: both directions right {both}/{len(SALE)}; the small word moves both directions the "
          f"right way {moved}/{len(SALE)}")


def trap(net, tok, label, items):
    """The trap test on one checkpoint."""
    probs = [sale_signs(net, tok, i["persona"], i["cares_about"], i["greeting"], i["question"], i["answer"])
             for i in items]
    report(f"{label} on the trap test ({len(items)} answers)", score(items, probs))


def main() -> None:
    args = sys.argv[1:]
    items_path = Path(args[args.index("--items") + 1]) if "--items" in args else ITEMS_FILE
    items = load_items(items_path) if items_path.exists() else []
    if "--baselines" in args:
        baselines(items)
        return
    import torch

    from sellanything_ml.llm.evaluate import load_model
    flags = {"--items", str(items_path)}
    checkpoints = [a for a in args if a not in flags] or ["runs/pretrain_large/final.pt", "runs/sft_b5/best.pt"]
    for path in checkpoints:
        net, tok = load_model(path, torch.device("cpu"))
        general(net, tok, path)
        if net.prompt_format == 1:          # a pre-trained model has never seen a sale prompt
            continue
        sale_pairs(net, tok, path)
        if items:
            trap(net, tok, path, items)
    if items:
        baselines(items)
    else:
        print(f"no trap test: {items_path} is not there")


if __name__ == "__main__":
    main()
