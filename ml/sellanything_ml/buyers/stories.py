"""Cat sales written turn by turn by a teacher model, TinyStories-style (Eldan & Li, 2023).

The program only decides whose turn it is; the teacher writes every line. A sale is:

- an opening: the teacher writes each buyer's card (what they care about in a cat, the most they can pay), an
  opening line and a starting offer;
- rounds: one buyer who has not asked yet asks what matters to them, the seller answers that buyer from the cat's
  true facts, and every buyer reacts: first WHY (which item on their own card the answer touches, good or bad,
  with what it means for a cat), then what they say, then their new offer;
- after the last round, the seller takes the best offer.

`render` lays a sale out with explicit markers (each round, who asked, whom the answer is to, every reaction with
its old and new offer), so a model trained on it can see which answer each reaction responds to. `check` counts
reactions whose offer moves against their own WHY, or past the buyer's budget.

Facts about the cat and the buyers come from random seeds, so every sale differs. The teacher is
Qwen2.5-32B-Instruct (Apache-2.0, so its text may train other models): on this Mac through MLX for checking the
format one sale at a time, and on Kaggle's GPUs through vLLM (tools/kaggle_dialogues/) for volume.

    python -m sellanything_ml.buyers.stories --seed 1 --rounds 1      # opening + one question, on the Mac
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, TypeVar, cast

from .cat_knowledge import HELD_OUT, TOPICS, knowledge_prompt
from .drawable import DRAWABLE

ML_DIR = Path(__file__).resolve().parents[2]
LOCAL_TEACHER = ML_DIR / "data" / "raw" / "teacher" / "qwen2.5-32b-instruct-4bit"
OUT_DIR = ML_DIR / "runs" / "buyer_dialogues"
N_BUYERS = 10
JSON_TRIES = 3

AGES_MONTHS = [2, 4, 7, 12, 24, 36, 60, 72, 96, 120, 156, 180]
BREEDS = ["a mixed-breed tabby", "a black domestic shorthair", "a Ragdoll", "a Maine Coon", "a Siamese",
          "a British Shorthair", "a Persian", "a Bengal", "a Sphynx", "a calico", "a Scottish Fold",
          "an orange tabby of unknown breed"]
HOME = ["indoor-only", "indoor with a catio", "indoor-outdoor", "used to roaming a farm"]
HEALTH_KITTEN = ["no vaccines yet", "had the first of three kitten vaccines", "had two of three kitten vaccines",
                 "finished the kitten vaccine series", "dewormed but not vaccinated"]
HEALTH_ADULT = ["fully vaccinated, vet check last month", "vaccinated but no vet visit in two years",
                "no vaccination records", "recovering from a cold", "microchipped and healthy"]
HEALTH_SENIOR = ["bloodwork last month was normal", "bloodwork shows early kidney disease, on a kidney diet",
                 "no bloodwork in years", "on medication for an overactive thyroid", "has some stiff joints"]
VIRUS = ["tested negative for FIV and FeLV", "never tested for FIV or FeLV", "tested FIV positive"]
COMPANY = {"dogs": ["has lived with a dog", "has never met a dog"],          # one answer per kind of company
           "cats": ["has lived with another cat", "has never met another cat"],
           "children": ["is fine with children", "has never met children"]}
TEMPER = ["very cuddly", "independent", "shy with strangers", "playful and noisy at night", "calm and lazy",
          "a bit feisty, scratches when annoyed", "follows people around like a dog"]
QUIRKS = ["only drinks from a running tap", "uses a scratching post, never the sofa", "sheds a lot",
          "meows loudly for breakfast", "picky about litter", "afraid of the vacuum"]   # company is in COMPANY
PERSONAS = ["lives in a small apartment", "wants things in excellent shape", "works long hours and has little free time",
            "is buying for a family with young kids", "loves rare, premium and pedigree things", "mostly wants a good deal",
            "spends weekends hiking and wants an adventure cat", "cares a lot about how things look",
            "is a student on a tight budget", "already has two dogs", "is allergic and worried about fur",
            "is retired and home all day", "lost an old cat last year", "lives on a busy road", "breeds show cats",
            "is a nurse working night shifts", "already has an older cat at home"]
NAMES = ["Maya", "Owen", "Felix", "Elena", "Aisha", "Diego", "Andre", "Chloe", "Hana", "Ben", "Jonah", "Ruby",
         "Leo", "Mei", "Sam", "Priya", "Tomas", "Ines"]
STYLES = ["short and sometimes vague", "friendly and detailed", "makes typos and uses abbreviations",
          "honest even about the cat's flaws", "tries to oversell the cat"]
SETTING = ("A secondhand marketplace. A seller is selling a cat; a typical price for a cat here is about $75. "
           "Buyers speak naturally and in character, in one or two sentences, and know real things about cats. "
           "Everything anyone says is plain spoken English, the way people talk face to face: never use a dash "
           "(— or –) in a spoken line.")
DASHES = ("—", "–")

Ask = Callable[[str], str]           # prompt → the teacher's reply
T = TypeVar("T")
Steps = Generator[Any, Any, T]       # yields a prompt (or a list of prompts, one per buyer) for the teacher, is
                                     # sent its reply (or the list of replies), returns T
Sale = dict[str, Any]                # cat facts, buyers, seller style, typical price (seed_sale)
Event = dict[str, Any]               # one step of a sale: open, ask, answer, react or sold
JSON_SPAN = re.compile(r"[\[{].*[\]}]", re.S)
SIGNS = {"+": 1, "-": -1, "0": 0}


def _age(months: int) -> str:
    """The age as a seller says it: "4 months", "1 year", "13 years"."""
    return f"{months} months" if months < 12 else f"{months // 12} year{'s' if months >= 24 else ''}"


def seed_sale(rng: random.Random) -> Sale:
    """Random but possible facts: kittens under 6 months are not fixed yet; a Sphynx has no fur to shed."""
    months, breed = rng.choice(AGES_MONTHS), rng.choice(BREEDS)
    fixed = "not yet spayed/neutered" if months < 6 else rng.choice(["spayed/neutered", "not yet spayed/neutered"])
    quirks = [q for q in QUIRKS if not (breed == "a Sphynx" and q == "sheds a lot")]
    health = HEALTH_KITTEN if months < 12 else HEALTH_SENIOR if months > 120 else HEALTH_ADULT
    cat = {"age": _age(months), "months": months, "breed": breed, "home": rng.choice(HOME), "fixed": fixed,
           "health": rng.choice(health), "virus": rng.choice(VIRUS),
           "company": [rng.choice(COMPANY[k]) for k in rng.sample(sorted(COMPANY), 2)],
           "temperament": rng.choice(TEMPER), "quirks": rng.sample(quirks, 2)}
    buyers = [{"name": n, "persona": p}
              for n, p in zip(rng.sample(NAMES, N_BUYERS), rng.sample(PERSONAS, N_BUYERS), strict=True)]
    return {"cat": cat, "buyers": buyers, "seller_style": rng.choice(STYLES), "typical_price": 75}


# Sales written to the "more human seller" standard (owner, 2026-10-13, NEXT_SESSION_PROMPT §4): the buyer GPT misread
# a player's "heart attack", "20 years old" and "pure breed but no paper", facts no sheet had. These seeds get two of
# the facts below on top of seed_sale's, drawn so they never contradict the rest of the sheet.
HUMAN_SELLER_SEEDS = range(2650, 2700)
PEDIGREE = ("a Ragdoll", "a Maine Coon", "a Siamese", "a British Shorthair", "a Persian", "a Bengal", "a Sphynx",
            "a Scottish Fold")
WHITE_POSSIBLE = ("a Maine Coon", "a British Shorthair", "a Persian", "a Scottish Fold")
CLEAN_BILL = ("microchipped and healthy", "bloodwork last month was normal")      # a sickness would contradict these
EXTRA_FACTS = {   # name: (weight, the fact as the seller is told it, or one per draw)
    "very old": (3, None),
    "heart": (3, ("has a heart murmur the vet keeps an eye on",
                  "has hypertrophic cardiomyopathy (HCM), on heart pills every day",
                  "had one heart episode last year, the vet says stable now")),
    "no papers": (3, None),
    "diabetes": (1, ("diabetic, needs an insulin shot every day",)),
    "thyroid": (1, ("has an overactive thyroid, on pills every day",)),
    "one eye": (0.6, ("blind in the left eye", "blind in the right eye")),
    "three legs": (0.6, ("has three legs, a back leg was amputated", "has three legs, a front leg was amputated")),
    "deaf": (1, ("all white and born deaf",)),
    "teeth": (1, ("had four teeth pulled last year", "had three teeth pulled last year")),
}


# Facts that say a vet saw the cat lately (last year, or ongoing pills, shots or check-ups). They contradict "no vet
# visit in two years" (found 2026-10-14 on seed 2686: "no vet visit in two years" and "had one heart episode last year,
# the vet says stable now"). Only seeds from VET_RULE_FROM on get this rule, so every sheet written before keeps its
# facts. Seed 2654 had the clash in its written sale and is rewritten with the rule (owner, 2026-10-14); the written
# 2664 and 2679 keep their sheets (their sellers never said both things).
NO_VET = "vaccinated but no vet visit in two years"
# Found 2026-10-26 on seed 2718: a one year old cat "with no vet visit in two years". From YOUNG_NO_VET_FROM on such a
# cat is three years old instead (the same asking price step); of the earlier seeds only YOUNG_NO_VET_ALSO change.
YOUNG_NO_VET_FROM = 2700
YOUNG_NO_VET_ALSO = (2098, 2328, 2347, 2526, 2532, 2643,   # written before the rule; fixed with their lines (owner, 2026-10-26)
                     2301, 2310, 2505, 2622, 2628, 2637)   # exactly two years old (owner, 2026-10-27): the kitten shots fall inside the two years
# Found 2026-10-26 on seed 2793: a two month old kitten that "finished the kitten vaccine series", though boosters run to
# about 16 to 18 weeks. From YOUNG_NO_VET_FROM on such a kitten is four months old instead (the same kitten price step);
# of the earlier seeds only FULL_SERIES_ALSO change.
FULL_SERIES = "finished the kitten vaccine series"
FULL_SERIES_ALSO = (2251, 2609)     # written before the rule; fixed with their lines (2026-10-26)
VET_LATELY = ("heart", "diabetes", "thyroid", "teeth")
VET_RULE_FROM = 2686
VET_RULE_ALSO = (2654,)


def _extra_fits(name: str, c: dict[str, Any], vet_rule: bool = False) -> bool:
    """Whether an extra fact agrees with the rest of the cat's sheet."""
    if vet_rule and c["health"] == NO_VET and name in VET_LATELY:
        return False
    sick = c["health"] not in CLEAN_BILL
    return {"very old": c["months"] >= 12,       # an adult's health line fits an old cat too
            "heart": c["months"] >= 12 and sick,
            "no papers": c["breed"] in PEDIGREE,
            "diabetes": c["months"] >= 84 and sick,
            "thyroid": c["months"] > 120 and sick and "thyroid" not in c["health"],
            "one eye": True, "three legs": True,
            "deaf": c["breed"] in WHITE_POSSIBLE,
            "teeth": c["months"] >= 84}[name]


def sale_for_seed(seed: int) -> Sale:
    """seed_sale(seed), with two extra facts (cat["extra"]) on the human-seller seeds (a very old cat's age is 16 to
    22 years), one or two LIFE_FACTS on the LIFE_SEEDS, and the target facts on the PAIRED_SEEDS. Every tool that writes or checks a sale on a seed uses
    this, so the sheet and the stored sale agree."""
    s = seed_sale(random.Random(seed))
    if (seed >= YOUNG_NO_VET_FROM or seed in YOUNG_NO_VET_ALSO) and s["cat"]["health"] == NO_VET and s["cat"]["months"] <= 24:
        s["cat"]["months"] = 36                 # "no vet visit in two years" cannot fit a cat of two or under
        s["cat"]["age"] = _age(36)
    if (seed >= YOUNG_NO_VET_FROM or seed in FULL_SERIES_ALSO) and s["cat"]["health"] == FULL_SERIES and s["cat"]["months"] < 4:
        s["cat"]["months"] = 4                  # a finished kitten vaccine series cannot fit a two month old
        s["cat"]["age"] = _age(4)
    if seed in PAIRED_SEEDS:
        apply_paired(seed, s)
        return s
    if seed in LIFE_SEEDS and seed not in HUMAN_SELLER_SEEDS:
        add_life_facts(seed, s["cat"])
    if seed not in HUMAN_SELLER_SEEDS:
        return s
    rng, c = random.Random(seed * 1009 + 7), s["cat"]
    for _ in range(2):          # "very old" first changes the age, so the second draw sees the new one
        names = [n for n in EXTRA_FACTS
                 if _extra_fits(n, c, seed >= VET_RULE_FROM or seed in VET_RULE_ALSO) and n not in c.get("extra_kinds", [])]
        name = rng.choices(names, weights=[EXTRA_FACTS[n][0] for n in names])[0]
        if name == "very old":
            c["months"] = 12 * rng.randint(16, 22)
            c["age"] = _age(c["months"])
            fact = f"{c['age']} old"
        elif name == "no papers":
            fact = f"a purebred {c['breed'][2:] if c['breed'].startswith('a ') else c['breed']}, but never had papers"
        else:
            fact = rng.choice(cast(tuple[str, ...], EXTRA_FACTS[name][1]))     # only the two above have None
        c.setdefault("extra_kinds", []).append(name)
        c.setdefault("extra", []).append(fact)
    if seed in LIFE_SEEDS:
        add_life_facts(seed, c)
    return s


# Facts a real adopter asks about that no sheet had (owner, 2026-10-14, from a shelter's list of questions to ask:
# how the cat came to be rehomed, being picked up, food, fleas and mites, microchip; litter and coat from PetFinder below). Only seeds not yet written get
# them, so every stored sale keeps its sheet; one or two per cat, drawn so they never contradict the rest of it.
LIFE_SEEDS = (*range(2593, 2650), *range(2670, 3000))     # 2700-2899 added 2026-10-26, 2900-2999 on 10-27, before any was written
LIFE_FACTS = {   # kind: facts for a kitten (under a year), facts for an older cat
    "origin": (("found as a stray kitten", "came from a neighbour's unplanned litter"),
               ("the previous owner moved abroad and could not take the cat", "rehomed because a family member became "
                "allergic", "the previous owner went into a care home", "found as a stray and never claimed")),
    "handling": (("loves being picked up and carried", "hates being picked up but sits close by",
                  "lets you hold them for a minute, then wriggles off"),) * 2,
    "food": (("eats one brand of dry food and nothing else", "eats wet food twice a day, any brand",
              "a fussy eater, only eats chicken wet food"),) * 2,
    "pests": (("treated for fleas and ear mites last month", "on a monthly flea treatment",
               "has never had a flea treatment"),) * 2,
    "microchip": (("microchipped", "not microchipped"),) * 2,
    "litter": (("fully litter trained", "still learning to use the litter box"),
               ("fully litter trained", "litter trained but kicks litter everywhere")),
    "coat": (("a long fluffy coat", "a short coat"),) * 2,      # only where the breed does not already say it
}
UNSET_COAT = ("a mixed-breed tabby", "a calico", "an orange tabby of unknown breed")
# How often each kind is drawn: how often real people write it when rehoming a cat (PetFinder.my Adoption Prediction,
# Kaggle 2019, 6,570 English cat descriptions, counted 2026-10-14 for analysis only): rescued, stray or a reason for
# rehoming 24% + 12%, food 21%, litter 21%, fleas or mites 3.6%, being picked up or carried 2.8%, microchip 0.4%.
# Coat: grooming or fur 8.4% of the descriptions; a personal taste, so buyers can take it either way (owner, 2026-10-14).
# A kitten from a litter has met its mother and littermates, so it cannot have "never met another cat" (found
# 2026-10-26 on seed 2762); such a cat was found as a stray kitten instead. Five written sales were fixed with their lines.
LITTER = "came from a neighbour's unplanned litter"
# Fleas and ear mites are treated with products from the vet, so a cat with "no vet visit in two years" cannot have
# been "treated for fleas and ear mites last month" (found 2026-10-27 on seed 2973); such a cat has never had a flea
# treatment instead (owner, 2026-10-28). Four written sales were fixed with their lines.
TREATED = "treated for fleas and ear mites last month"
LIFE_WEIGHTS = {"origin": 3, "food": 2, "litter": 2, "handling": 1, "pests": 1, "microchip": 0.5, "coat": 1}


def _life_options(kind: str, c: dict[str, Any]) -> list[str]:
    """The facts of one kind that agree with the cat's sheet."""
    options = list(LIFE_FACTS[kind][c["months"] >= 12])
    sheet = " ".join([c["health"], *c.get("extra", [])])
    if kind == "food" and any(w in sheet for w in ("diet", "insulin", "diabetic")):
        return []                                           # the sheet already says what this cat eats
    if kind == "pests" and "dewormed" in sheet:
        options.remove("has never had a flea treatment")
    if kind == "microchip" and "microchipped" in sheet:
        return []
    if kind == "litter" and "picky about litter" in c["quirks"]:
        return []
    if kind == "coat" and c["breed"] not in UNSET_COAT:
        return []
    return options


def add_life_facts(seed: int, c: dict[str, Any]) -> None:
    """One or two LIFE_FACTS on the cat's sheet (c["life_kinds"], c["life"]), from their own random draw, so the
    facts sale_for_seed drew before stay exactly as they were."""
    rng = random.Random(seed * 2003 + 11)
    kinds = [k for k in LIFE_FACTS if _life_options(k, c)]
    picked: list[str] = []
    for _ in range(rng.choice((1, 2))):
        left = [k for k in kinds if k not in picked]
        picked.append(rng.choices(left, weights=[LIFE_WEIGHTS[k] for k in left])[0])
    for kind in picked:
        c.setdefault("life_kinds", []).append(kind)
        c.setdefault("life", []).append(rng.choice(_life_options(kind, c)))
    if LITTER in c["life"] and "has never met another cat" in c["company"]:
        c["life"][c["life"].index(LITTER)] = "found as a stray kitten"     # after the draw, so no other fact moves
    if TREATED in c["life"] and c["health"] == NO_VET:
        c["life"][c["life"].index(TREATED)] = "has never had a flea treatment"


# Sales that teach the buyer GPT the small words ("not", "never", "but", data/written_sales/prompts_3000_plan.md,
# approved by the owner 2026-10-29). Each of these seeds is given two or three target facts, the topics kept as they
# were: only the values below are new, the other side of an existing topic (decision A), and they are never drawn at
# random, so no other sheet changes. A target is (topic, side, value, group): side A and B are the two sides of a
# topic and U is "not known" (value None: the sheet says nothing about it, so the seller does not know); the members
# of one group are the paired sales (one topic, opposite meanings, the buyer the small word turns in every one of
# them, plan 2.3). Everything else on the sheet is drawn as before and kept consistent with the targets.
SHEDS_LITTLE = "sheds very little"
AGAINST = {"dogs": "doesn't get on with dogs", "children": "doesn't like children",
           "cats": "doesn't get on with other cats"}
# (topic, side, value, group); seeds 3050-3149 add (heard, asker) and seeds 3150-3234 also a shape (see below)
PairedTarget = (tuple[str, str, str | None, str]
                | tuple[str, str, str | None, str, str, str | None]
                | tuple[str, str, str | None, str, str, str | None, str])
PAIRED_SEEDS: dict[int, tuple[PairedTarget, ...]] = {   # allocated once to the quotas of plan section 3, then frozen
    # (seeds 3000-3004 are the pilot)
    3000: (("shed", "B", "sheds very little", "shed1"), ("home", "B", "indoor-outdoor", "home3"),
            ("spay", "B", "not yet spayed/neutered", "spay3")),
    3001: (("vacc", "U", "recovering from a cold", "vacc1"), ("dogs", "B", "doesn't get on with dogs", "dogs1")),
    3002: (("kids", "B", "doesn't like children", "kids1"), ("age", "B", "2 years", "age1")),
    3003: (("cats", "B", "doesn't get on with other cats", "cats1"), ("clingy", "A", "independent", "clingy2"),
            ("litter", "A", "fully litter trained", "litter2")),
    3004: (("age", "A", "2 months", "age1"), ("papers", "U", None, "papers1")),
    3005: (("noise", "B", "meows loudly for breakfast", "noise2"), ("spay", "A", "spayed/neutered", "spay3")),
    3006: (("vacc", "B", "no vaccines yet", "vacc2"),
            ("scratch", "B", "a bit feisty, scratches when annoyed", "scratch1"),
            ("microchip", "A", "microchipped", "microchip2")),
    3007: (("cats", "B", "doesn't get on with other cats", "cats2"), ("spay", "B", "not yet spayed/neutered", "spay1")),
    3008: (("noise", "B", "playful and noisy at night", "noise3"), ("spay", "A", "spayed/neutered", "spay4")),
    3009: (("shed", "B", "sheds very little", "shed5"), ("vacc", "B", "no vaccination records", "vacc1")),
    3010: (("kids", "U", "has never met children", "kids2"), ("age", "B", "2 years", "age4"),
            ("papers", "B", "no papers", "papers2")),
    3011: (("dogs", "A", "has lived with a dog", "dogs2"),
            ("scratch", "B", "a bit feisty, scratches when annoyed", "scratch3")),
    3012: (("shed", "U", None, "shed1"), ("dogs", "B", "doesn't get on with dogs", "dogs2")),
    3013: (("cats", "U", "has never met another cat", "cats2"), ("senior", "B", "no bloodwork in years", "senior2"),
            ("microchip", "B", "not microchipped", "microchip1")),
    3014: (("kids", "B", "doesn't like children", "kids2"), ("noise", "B", "playful and noisy at night", "noise1")),
    3015: (("clingy", "A", "independent", "clingy3"), ("spay", "A", "spayed/neutered", "spay1")),
    3016: (("vacc", "B", "no vaccination records", "vacc3"), ("home", "B", "used to roaming a farm", "home2"),
            ("pests", "A", "on a monthly flea treatment", "pests2")),
    3017: (("kids", "B", "doesn't like children", "kids3"), ("spay", "B", "not yet spayed/neutered", "spay1"),
            ("pests", "B", "has never had a flea treatment", "pests1")),
    3018: (("scratch", "B", "a bit feisty, scratches when annoyed", "scratch2"),
            ("pests", "B", "has never had a flea treatment", "pests2")),
    3019: (("senior", "A", "bloodwork last month was normal", "senior2"),
            ("spay", "B", "not yet spayed/neutered", "spay4")),
    3020: (("fiv", "A", "tested negative for FIV and FeLV", "fiv3"), ("dogs", "U", "has never met a dog", "dogs1"),
            ("scratch", "A", "uses a scratching post, never the sofa", "scratch3")),
    3021: (("kids", "A", "is fine with children", "kids3"),
            ("clingy", "B", "follows people around like a dog", "clingy2"),
            ("microchip", "A", "microchipped", "microchip1")),
    3022: (("vacc", "A", "fully vaccinated, vet check last month", "vacc3"),
            ("fiv", "A", "tested negative for FIV and FeLV", "fiv2"),
            ("handling", "A", "loves being picked up and carried", "handling2")),
    3023: (("vacc", "U", "has some stiff joints", "vacc2"),
            ("handling", "A", "loves being picked up and carried", "handling1")),
    3024: (("dogs", "A", "has lived with a dog", "dogs1"),
            ("food", "A", "eats wet food twice a day, any brand", "food2"),
            ("origin", "C", "found as a stray and never claimed", "origin2")),
    3025: (("shed", "A", "sheds a lot", "shed5"),
            ("pests", "A", "treated for fleas and ear mites last month", "pests1"),
            ("origin", "C", "found as a stray and never claimed", "origin1")),
    3026: (("vacc", "A", "finished the kitten vaccine series", "vacc2"),
            ("spay", "B", "not yet spayed/neutered", "spay2")),
    3027: (("age", "B", "2 years", "age2"), ("handling", "B", "hates being picked up but sits close by", "handling1")),
    3028: (("shed", "B", "sheds very little", "shed2"), ("fiv", "A", "tested negative for FIV and FeLV", "fiv1"),
            ("kids", "A", "is fine with children", "kids2")),
    3029: (("fiv", "B", "tested FIV positive", "fiv1"), ("home", "B", "indoor-outdoor", "home1"),
            ("litter", "B", "still learning to use the litter box", "litter1")),
    3030: (("senior", "A", "bloodwork last month was normal", "senior1"), ("papers", "B", "no papers", "papers1")),
    3031: (("cats", "A", "has lived with another cat", "cats2"), ("microchip", "B", "not microchipped", "microchip2")),
    3032: (("dogs", "B", "doesn't get on with dogs", "dogs3"), ("clingy", "B", "very cuddly", "clingy1")),
    3033: (("fiv", "U", "never tested for FIV or FeLV", "fiv1"), ("age", "B", "2 years", "age3")),
    3034: (("noise", "B", "meows loudly for breakfast", "noise4"), ("papers", "U", None, "papers2")),
    3035: (("cats", "U", "has never met another cat", "cats1"),
            ("handling", "B", "hates being picked up but sits close by", "handling2")),
    3036: (("shed", "A", "sheds a lot", "shed2"), ("home", "A", "indoor-only", "home3")),
    3037: (("dogs", "A", "has lived with a dog", "dogs3"), ("clingy", "A", "independent", "clingy1"),
            ("age", "A", "2 months", "age2")),
    3038: (("shed", "A", "sheds a lot", "shed3"), ("fiv", "U", "never tested for FIV or FeLV", "fiv2"),
            ("spay", "B", "not yet spayed/neutered", "spay4")),
    3039: (("vacc", "A", "fully vaccinated, vet check last month", "vacc1"),
            ("food", "B", "a fussy eater, only eats chicken wet food", "food1")),
    3040: (("shed", "B", "sheds very little", "shed3"), ("dogs", "U", "has never met a dog", "dogs2")),
    3041: (("shed", "A", "sheds a lot", "shed4"), ("food", "B", "eats one brand of dry food and nothing else", "food2"),
            ("litter", "A", "fully litter trained", "litter1")),
    3042: (("clingy", "B", "very cuddly", "clingy3"),
            ("senior", "B", "bloodwork shows early kidney disease, on a kidney diet", "senior1"),
            ("spay", "B", "not yet spayed/neutered", "spay3")),
    3043: (("kids", "U", "has never met children", "kids1"),
            ("scratch", "A", "uses a scratching post, never the sofa", "scratch1"),
            ("spay", "B", "not yet spayed/neutered", "spay2")),
    3044: (("kids", "A", "is fine with children", "kids1"), ("cats", "A", "has lived with another cat", "cats1")),
    3045: (("shed", "B", "sheds very little", "shed4"), ("fiv", "B", "tested FIV positive", "fiv3"),
            ("scratch", "A", "uses a scratching post, never the sofa", "scratch2")),
    3046: (("shed", "U", None, "shed2"), ("home", "A", "indoor-only", "home1"),
            ("litter", "B", "litter trained but kicks litter everywhere", "litter2")),
    3047: (("shed", "A", "sheds a lot", "shed1"), ("spay", "A", "spayed/neutered", "spay2")),
    3048: (("age", "A", "2 months", "age4"), ("food", "A", "eats wet food twice a day, any brand", "food1")),
    3049: (("fiv", "B", "tested FIV positive", "fiv2"), ("home", "A", "indoor-only", "home2"),
            ("age", "A", "2 months", "age3")),
}
PAIRED_BUYERS = {   # topic: the buyers whose direction the small word turns (plan 3.1, last column); always present
    "shed": ("is allergic and worried about fur",), "vacc": ("wants things in excellent shape",),
    "fiv": ("already has an older cat at home",), "kids": ("is buying for a family with young kids",),
    "dogs": ("already has two dogs",), "cats": ("already has an older cat at home",),
    "clingy": ("works long hours and has little free time", "is retired and home all day"),
    "scratch": ("cares a lot about how things look", "is buying for a family with young kids"),
    "home": ("lives on a busy road", "spends weekends hiking and wants an adventure cat"),
    "noise": ("is a nurse working night shifts",),
    "papers": ("breeds show cats", "loves rare, premium and pedigree things"),
    "food": ("is a student on a tight budget",), "pests": ("already has two dogs",),
    "litter": ("cares a lot about how things look",), "handling": ("is buying for a family with young kids",),
    "microchip": ("lives on a busy road", "is a student on a tight budget"),
}
PAIRED_GROUP_BUYERS = {"spay1": ("breeds show cats",), "spay2": ("breeds show cats",)}   # 6 of the 12 spay sales
# A breeder only breeds from a young cat, so a breeder never bids on an intact cat over five years old (owner,
# 2026-10-30: the pilot's breeder raised for "intact" on a 13 year old): the spay sales with a breeder are cats of
# breeding age on both sides, a sale whose papers need a breeder gets an intact cat of breeding age, and a breeder
# drawn at random for an older intact cat is given another persona.
BREEDER = "breeds show cats"
BREEDING_MONTHS = 60
COMPANY_TOPIC = {"kids": "children", "dogs": "dogs", "cats": "cats"}
LIFE_TOPICS = ("food", "pests", "litter", "handling", "microchip", "origin")
PAIRED_LIFE = 2                      # every paired cat has exactly two LIFE_FACTS (plan 2.2)
# the kinds plan 3.3 found shortest of good reaction lines are drawn first for the life facts not set by a target
PAIRED_LIFE_WEIGHTS = {"food": 2, "pests": 2, "litter": 2, "handling": 2, "microchip": 2, "origin": 1}
LONG_HAIR = ("a Maine Coon", "a Persian", "a Ragdoll")

# Seeds 3050-3149 (data/written_sales/prompts_3050_plan.md, approved by the owner 2026-10-31): buyers B5 left on keep
# when another buyer asked their topic, so a target there is (topic, side, value, group, heard, asker). heard says how
# the buyer it turns hears it: "self" (that buyer asks; asker is which of them, or None where any buyer it moves may
# ask), "other" (the buyer asker asks, one whose card or greeting makes the question natural) or "inside" (said in the
# answer to a question about something else; asker None). Buyers are named by their BUYER key. Each sale also has one
# control (PAIRED_CONTROLS: kind, facts) and two "but" answers (PAIRED_BUTS: kind, connector, heard, asker, the buyer
# the good half raises, the buyer the bad half lowers, facts); PAIRED_PRESENT adds the personas plan 3.5 wants in at
# least 70 sales. All of it was allocated once and is frozen.
HEARD_FROM = 3050
BUYER = {"apartment": PERSONAS[0], "excellent": PERSONAS[1], "long hours": PERSONAS[2], "kids": PERSONAS[3],
         "pedigree": PERSONAS[4], "deal": PERSONAS[5], "hiker": PERSONAS[6], "looks": PERSONAS[7],
         "student": PERSONAS[8], "dogs": PERSONAS[9], "allergic": PERSONAS[10], "retired": PERSONAS[11],
         "lost cat": PERSONAS[12], "busy road": PERSONAS[13], "breeder": PERSONAS[14], "nurse": PERSONAS[15],
         "old cat": PERSONAS[16]}
PAIRED_PRESENT: dict[int, tuple[str, ...]] = {}
PAIRED_CONTROLS: dict[int, tuple[str, tuple[str, ...]]] = {}
PAIRED_BUTS: dict[int, tuple[tuple[str, str, str, str | None, str | None, str | None, tuple[str, ...]], ...]] = {}
PAIRED_GROUP_BUYERS.update({g: (BREEDER,) for g in ("spay11", "spay12", "spay15", "spay16")})   # 10 of 20
PAIRED_SEEDS.update({   # 3050-3149 (prompts_3050_plan.md 3.1 and 2), allocated once, then frozen
    3050: (("age", "B", "2 years", "age18", "self", None),
           ("home", "A", "indoor-only", "home15", "other", "apartment"),
           ("fiv", "U", "never tested for FIV or FeLV", "fiv13", "self", "old cat")),
    3051: (("vacc", "U", "has some stiff joints", "vacc15", "self", "excellent"),
           ("kids", "B", "doesn't like children", "kids17", "other", "retired")),
    3052: (("spay", "B", "not yet spayed/neutered", "spay14", "other", "lost cat"),
           ("dogs", "U", "has never met a dog", "dogs15", "other", "apartment")),
    3053: (("age", "B", "2 years", "age13", "other", "pedigree"),
           ("litter", "A", "fully litter trained", "litter13", "inside", None)),
    3054: (("vacc", "U", "has some stiff joints", "vacc14", "inside", None),
           ("noise", "B", "playful and noisy at night", "noise14", "self", None),
           ("dogs", "U", "has never met a dog", "dogs16", "inside", None)),
    3055: (("senior", "B", "bloodwork shows early kidney disease, on a kidney diet", "senior11", "self", None),
           ("cats", "A", "has lived with another cat", "cats12", "other", "student"),
           ("spay", "B", "not yet spayed/neutered", "spay14", "other", "deal")),
    3056: (("fiv", "B", "tested FIV positive", "fiv14", "other", "hiker"),
           ("cats", "B", "doesn't get on with other cats", "cats13", "self", "old cat"),
           ("food", "A", "eats wet food twice a day, any brand", "food14", "self", "student")),
    3057: (("vacc", "A", "fully vaccinated, vet check last month", "vacc13", "other", "old cat"),
           ("fiv", "B", "tested FIV positive", "fiv12", "other", "excellent"),
           ("papers", "U", None, "papers13", "other", "retired")),
    3058: (("spay", "B", "not yet spayed/neutered", "spay16", "self", "breeder"),
           ("cats", "B", "doesn't get on with other cats", "cats14", "self", "old cat"),
           ("kids", "A", "is fine with children", "kids12", "inside", None)),
    3059: (("senior", "A", "bloodwork last month was normal", "senior12", "inside", None),
           ("handling", "A", "loves being picked up and carried", "handling11", "self", "kids"),
           ("shed", "U", None, "shed13", "other", "nurse")),
    3060: (("food", "B", "eats one brand of dry food and nothing else", "food14", "inside", None),
           ("shed", "A", "sheds a lot", "shed13", "other", "excellent")),
    3061: (("cats", "B", "doesn't get on with other cats", "cats12", "inside", None),
           ("origin", "C", "found as a stray and never claimed", "origin14", "self", None),
           ("scratch", "A", "uses a scratching post, never the sofa", "scratch11", "inside", None)),
    3062: (("clingy", "B", "follows people around like a dog", "clingy13", "self", "retired"),
           ("handling", "A", "loves being picked up and carried", "handling11", "self", "kids"),
           ("cats", "U", "has never met another cat", "cats15", "other", "apartment")),
    3063: (("age", "B", "2 years", "age14", "self", None),
           ("pests", "B", "has never had a flea treatment", "pests14", "self", "dogs"),
           ("kids", "U", "has never met children", "kids17", "self", "kids")),
    3064: (("litter", "B", "still learning to use the litter box", "litter12", "other", "retired"),
           ("spay", "A", "spayed/neutered", "spay18", "other", "student"),
           ("fiv", "A", "tested negative for FIV and FeLV", "fiv11", "self", "old cat")),
    3065: (("clingy", "B", "very cuddly", "clingy12", "other", "lost cat"),
           ("kids", "A", "is fine with children", "kids17", "inside", None),
           ("shed", "B", "sheds very little", "shed11", "inside", None)),
    3066: (("age", "A", "2 months", "age14", "inside", None),
           ("dogs", "U", "has never met a dog", "dogs17", "self", "dogs")),
    3067: (("cats", "A", "has lived with another cat", "cats16", "inside", None),
           ("fiv", "U", "never tested for FIV or FeLV", "fiv11", "inside", None),
           ("scratch", "B", "a bit feisty, scratches when annoyed", "scratch13", "other", "apartment")),
    3068: (("handling", "B", "hates being picked up but sits close by", "handling13", "self", "kids"),
           ("dogs", "B", "doesn't get on with dogs", "dogs11", "inside", None),
           ("shed", "B", "sheds very little", "shed16", "inside", None)),
    3069: (("vacc", "B", "no vaccination records", "vacc12", "self", "excellent"),
           ("handling", "B", "hates being picked up but sits close by", "handling12", "inside", None)),
    3070: (("handling", "A", "loves being picked up and carried", "handling15", "inside", None),
           ("shed", "A", "sheds a lot", "shed16", "self", "allergic")),
    3071: (("home", "B", "indoor-outdoor", "home15", "other", "retired"),
           ("food", "B", "eats one brand of dry food and nothing else", "food12", "self", "student")),
    3072: (("spay", "B", "not yet spayed/neutered", "spay11", "inside", None),
           ("fiv", "U", "never tested for FIV or FeLV", "fiv14", "self", "old cat")),
    3073: (("age", "B", "2 years", "age15", "self", None),
           ("home", "B", "used to roaming a farm", "home11", "self", "busy road")),
    3074: (("vacc", "U", "has some stiff joints", "vacc16", "inside", None),
           ("kids", "B", "doesn't like children", "kids14", "inside", None)),
    3075: (("kids", "B", "doesn't like children", "kids15", "self", "kids"),
           ("home", "B", "used to roaming a farm", "home14", "inside", None)),
    3076: (("litter", "A", "fully litter trained", "litter12", "other", "retired"),
           ("spay", "B", "not yet spayed/neutered", "spay13", "inside", None),
           ("clingy", "A", "independent", "clingy13", "self", "long hours")),
    3077: (("home", "A", "indoor-only", "home11", "other", "apartment"),
           ("scratch", "B", "a bit feisty, scratches when annoyed", "scratch11", "self", "looks")),
    3078: (("kids", "A", "is fine with children", "kids16", "self", "kids"),
           ("cats", "A", "has lived with another cat", "cats11", "self", "old cat")),
    3079: (("kids", "U", "has never met children", "kids15", "other", "lost cat"),
           ("food", "A", "eats wet food twice a day, any brand", "food11", "inside", None),
           ("spay", "A", "spayed/neutered", "spay12", "self", "breeder")),
    3080: (("fiv", "A", "tested negative for FIV and FeLV", "fiv12", "inside", None),
           ("litter", "B", "litter trained but kicks litter everywhere", "litter14", "inside", None)),
    3081: (("kids", "U", "has never met children", "kids12", "other", "apartment"),
           ("home", "A", "indoor-only", "home16", "self", "busy road"),
           ("microchip", "B", "not microchipped", "microchip11", "inside", None)),
    3082: (("cats", "B", "doesn't get on with other cats", "cats13", "inside", None),
           ("clingy", "A", "independent", "clingy14", "other", "student")),
    3083: (("shed", "A", "sheds a lot", "shed14", "self", "allergic"),
           ("litter", "B", "litter trained but kicks litter everywhere", "litter13", "self", "looks")),
    3084: (("cats", "B", "doesn't get on with other cats", "cats17", "other", "lost cat"),
           ("dogs", "A", "has lived with a dog", "dogs15", "other", "deal"),
           ("spay", "A", "spayed/neutered", "spay17", "other", "student")),
    3085: (("spay", "A", "spayed/neutered", "spay11", "other", "student"),
           ("shed", "A", "sheds a lot", "shed12", "inside", None),
           ("kids", "B", "doesn't like children", "kids11", "self", "kids")),
    3086: (("dogs", "B", "doesn't get on with dogs", "dogs16", "other", "student"),
           ("cats", "U", "has never met another cat", "cats14", "inside", None)),
    3087: (("clingy", "A", "independent", "clingy12", "inside", None),
           ("handling", "A", "loves being picked up and carried", "handling12", "other", "retired"),
           ("fiv", "A", "tested negative for FIV and FeLV", "fiv13", "other", "excellent")),
    3088: (("vacc", "B", "no vaccines yet", "vacc15", "self", "excellent"),
           ("pests", "A", "on a monthly flea treatment", "pests14", "inside", None)),
    3089: (("dogs", "B", "doesn't get on with dogs", "dogs17", "other", "student"),
           ("shed", "B", "sheds very little", "shed13", "inside", None),
           ("home", "B", "indoor-outdoor", "home13", "self", "busy road")),
    3090: (("age", "A", "2 months", "age15", "self", None),
           ("fiv", "B", "tested FIV positive", "fiv14", "other", "lost cat"),
           ("dogs", "A", "has lived with a dog", "dogs16", "other", "hiker")),
    3091: (("microchip", "A", "microchipped", "microchip12", "self", "student"),
           ("food", "B", "a fussy eater, only eats chicken wet food", "food13", "other", "retired"),
           ("fiv", "A", "tested negative for FIV and FeLV", "fiv12", "self", "old cat")),
    3092: (("papers", "U", None, "papers14", "self", "pedigree"),
           ("kids", "B", "doesn't like children", "kids12", "other", "lost cat"),
           ("fiv", "U", "never tested for FIV or FeLV", "fiv12", "self", "old cat")),
    3093: (("age", "A", "2 months", "age18", "other", "lost cat"),
           ("microchip", "B", "not microchipped", "microchip12", "inside", None),
           ("pests", "A", "on a monthly flea treatment", "pests13", "inside", None)),
    3094: (("shed", "U", None, "shed16", "self", "allergic"),
           ("clingy", "B", "follows people around like a dog", "clingy14", "self", "retired"),
           ("pests", "B", "has never had a flea treatment", "pests12", "other", "old cat")),
    3095: (("kids", "U", "has never met children", "kids16", "other", "lost cat"),
           ("shed", "B", "sheds very little", "shed14", "other", "nurse"),
           ("dogs", "B", "doesn't get on with dogs", "dogs12", "inside", None)),
    3096: (("age", "A", "2 months", "age17", "self", None),
           ("cats", "A", "has lived with another cat", "cats15", "other", "student"),
           ("fiv", "A", "tested negative for FIV and FeLV", "fiv14", "self", "old cat")),
    3097: (("microchip", "A", "microchipped", "microchip13", "self", "busy road"),
           ("scratch", "B", "a bit feisty, scratches when annoyed", "scratch14", "other", "excellent")),
    3098: (("age", "A", "2 months", "age13", "inside", None),
           ("microchip", "B", "not microchipped", "microchip14", "other", "retired")),
    3099: (("vacc", "A", "fully vaccinated, vet check last month", "vacc11", "self", "excellent"),
           ("shed", "B", "sheds very little", "shed11", "other", "looks")),
    3100: (("vacc", "A", "fully vaccinated, vet check last month", "vacc14", "self", "excellent"),
           ("shed", "U", None, "shed12", "other", "nurse"),
           ("noise", "B", "meows loudly for breakfast", "noise11", "inside", None)),
    3101: (("spay", "A", "spayed/neutered", "spay16", "self", "breeder"),
           ("clingy", "B", "very cuddly", "clingy11", "other", "lost cat"),
           ("cats", "A", "has lived with another cat", "cats13", "inside", None)),
    3102: (("age", "A", "2 months", "age11", "inside", None),
           ("spay", "B", "not yet spayed/neutered", "spay13", "other", "lost cat")),
    3103: (("pests", "A", "treated for fleas and ear mites last month", "pests11", "self", "dogs"),
           ("food", "A", "eats wet food twice a day, any brand", "food13", "other", "long hours"),
           ("spay", "A", "spayed/neutered", "spay13", "other", "student")),
    3104: (("spay", "A", "spayed/neutered", "spay15", "self", "breeder"),
           ("shed", "B", "sheds very little", "shed15", "self", "allergic"),
           ("dogs", "U", "has never met a dog", "dogs14", "other", "apartment")),
    3105: (("spay", "B", "not yet spayed/neutered", "spay12", "self", "breeder"),
           ("shed", "U", None, "shed14", "self", "allergic"),
           ("clingy", "A", "independent", "clingy11", "inside", None)),
    3106: (("vacc", "U", "recovering from a cold", "vacc11", "self", "excellent"),
           ("handling", "A", "loves being picked up and carried", "handling13", "other", "retired"),
           ("scratch", "A", "uses a scratching post, never the sofa", "scratch12", "self", "kids")),
    3107: (("vacc", "B", "no vaccination records", "vacc16", "self", "excellent"),
           ("kids", "B", "doesn't like children", "kids16", "self", "kids"),
           ("pests", "B", "has never had a flea treatment", "pests11", "self", "dogs")),
    3108: (("vacc", "B", "no vaccines yet", "vacc14", "inside", None),
           ("dogs", "A", "has lived with a dog", "dogs11", "self", "dogs"),
           ("noise", "B", "playful and noisy at night", "noise13", "inside", None)),
    3109: (("vacc", "A", "finished the kitten vaccine series", "vacc15", "other", "old cat"),
           ("clingy", "A", "independent", "clingy11", "self", "retired"),
           ("fiv", "B", "tested FIV positive", "fiv11", "inside", None)),
    3110: (("age", "B", "2 years", "age11", "other", "pedigree"),
           ("shed", "A", "sheds a lot", "shed15", "other", "apartment"),
           ("cats", "B", "doesn't get on with other cats", "cats16", "other", "student")),
    3111: (("dogs", "A", "has lived with a dog", "dogs14", "inside", None),
           ("home", "A", "indoor-only", "home13", "other", "retired"),
           ("scratch", "A", "uses a scratching post, never the sofa", "scratch14", "self", "looks")),
    3112: (("papers", "B", "no papers", "papers11", "inside", None),
           ("pests", "B", "has never had a flea treatment", "pests13", "other", "old cat"),
           ("kids", "A", "is fine with children", "kids13", "inside", None)),
    3113: (("scratch", "A", "uses a scratching post, never the sofa", "scratch11", "inside", None),
           ("handling", "A", "loves being picked up and carried", "handling12", "other", "lost cat")),
    3114: (("age", "B", "2 years", "age16", "other", "hiker"),
           ("home", "B", "used to roaming a farm", "home12", "inside", None),
           ("shed", "B", "sheds very little", "shed12", "self", "allergic")),
    3115: (("dogs", "B", "doesn't get on with dogs", "dogs13", "other", "hiker"),
           ("spay", "B", "not yet spayed/neutered", "spay17", "other", "lost cat"),
           ("kids", "A", "is fine with children", "kids11", "other", "retired")),
    3116: (("papers", "U", None, "papers12", "other", "excellent"),
           ("vacc", "A", "finished the kitten vaccine series", "vacc12", "other", "lost cat")),
    3117: (("kids", "U", "has never met children", "kids11", "other", "lost cat"),
           ("cats", "U", "has never met another cat", "cats11", "inside", None),
           ("home", "A", "indoor-only", "home12", "self", "busy road")),
    3118: (("origin", "C", "found as a stray and never claimed", "origin12", "inside", None),
           ("kids", "U", "has never met children", "kids14", "inside", None),
           ("cats", "A", "has lived with another cat", "cats14", "self", "old cat")),
    3119: (("senior", "B", "bloodwork shows early kidney disease, on a kidney diet", "senior13", "self", None),
           ("kids", "A", "is fine with children", "kids14", "inside", None)),
    3120: (("age", "B", "2 years", "age12", "other", "retired"),
           ("dogs", "A", "has lived with a dog", "dogs13", "inside", None),
           ("shed", "U", None, "shed11", "inside", None)),
    3121: (("vacc", "B", "no vaccines yet", "vacc13", "self", "excellent"),
           ("dogs", "U", "has never met a dog", "dogs11", "self", "dogs"),
           ("spay", "B", "not yet spayed/neutered", "spay18", "inside", None)),
    3122: (("vacc", "B", "no vaccination records", "vacc11", "other", "old cat"),
           ("scratch", "A", "uses a scratching post, never the sofa", "scratch13", "other", "excellent"),
           ("cats", "B", "doesn't get on with other cats", "cats11", "other", "apartment")),
    3123: (("papers", "B", "no papers", "papers12", "self", "breeder"),
           ("kids", "A", "is fine with children", "kids11", "other", "old cat"),
           ("food", "A", "eats wet food twice a day, any brand", "food12", "other", "long hours")),
    3124: (("spay", "A", "spayed/neutered", "spay14", "inside", None),
           ("papers", "U", None, "papers11", "other", "excellent"),
           ("kids", "A", "is fine with children", "kids15", "self", "kids")),
    3125: (("spay", "B", "not yet spayed/neutered", "spay11", "self", "breeder"),
           ("kids", "B", "doesn't like children", "kids13", "self", "kids")),
    3126: (("dogs", "A", "has lived with a dog", "dogs12", "other", "deal"),
           ("handling", "B", "hates being picked up but sits close by", "handling15", "other", "retired"),
           ("kids", "A", "is fine with children", "kids12", "self", "kids")),
    3127: (("cats", "A", "has lived with another cat", "cats11", "other", "student"),
           ("shed", "B", "sheds very little", "shed14", "self", "allergic"),
           ("litter", "A", "fully litter trained", "litter14", "self", "looks")),
    3128: (("dogs", "B", "doesn't get on with dogs", "dogs14", "self", "dogs"),
           ("shed", "B", "sheds very little", "shed16", "other", "excellent"),
           ("food", "B", "a fussy eater, only eats chicken wet food", "food11", "self", "student")),
    3129: (("litter", "A", "fully litter trained", "litter11", "other", "nurse"),
           ("noise", "B", "meows loudly for breakfast", "noise12", "self", None),
           ("origin", "C", "found as a stray and never claimed", "origin13", "inside", None)),
    3130: (("vacc", "U", "recovering from a cold", "vacc13", "other", "retired"),
           ("home", "A", "indoor-only", "home14", "self", "hiker")),
    3131: (("origin", "C", "found as a stray and never claimed", "origin11", "self", None),
           ("senior", "A", "bloodwork last month was normal", "senior13", "self", None),
           ("shed", "A", "sheds a lot", "shed11", "other", "apartment")),
    3132: (("clingy", "A", "independent", "clingy12", "other", "apartment"),
           ("cats", "U", "has never met another cat", "cats12", "self", "old cat")),
    3133: (("senior", "B", "no bloodwork in years", "senior14", "inside", None),
           ("handling", "B", "hates being picked up but sits close by", "handling11", "inside", None)),
    3134: (("scratch", "A", "uses a scratching post, never the sofa", "scratch12", "self", "looks"),
           ("handling", "A", "loves being picked up and carried", "handling14", "self", "kids"),
           ("cats", "B", "doesn't get on with other cats", "cats15", "other", "retired")),
    3135: (("dogs", "A", "has lived with a dog", "dogs17", "other", "apartment"),
           ("kids", "U", "has never met children", "kids13", "other", "retired"),
           ("pests", "A", "treated for fleas and ear mites last month", "pests12", "other", "kids")),
    3136: (("senior", "B", "no bloodwork in years", "senior12", "inside", None),
           ("handling", "B", "hates being picked up but sits close by", "handling14", "inside", None),
           ("microchip", "A", "microchipped", "microchip11", "other", "nurse")),
    3137: (("vacc", "A", "finished the kitten vaccine series", "vacc16", "other", "old cat"),
           ("cats", "U", "has never met another cat", "cats17", "other", "student"),
           ("dogs", "U", "has never met a dog", "dogs13", "self", "dogs")),
    3138: (("age", "A", "2 months", "age12", "self", None),
           ("microchip", "A", "microchipped", "microchip14", "other", "retired")),
    3139: (("shed", "U", None, "shed15", "other", "excellent"),
           ("dogs", "U", "has never met a dog", "dogs12", "inside", None)),
    3140: (("cats", "U", "has never met another cat", "cats16", "inside", None),
           ("kids", "B", "doesn't like children", "kids17", "inside", None),
           ("senior", "A", "bloodwork last month was normal", "senior11", "inside", None)),
    3141: (("scratch", "B", "a bit feisty, scratches when annoyed", "scratch12", "other", "excellent"),
           ("dogs", "B", "doesn't get on with dogs", "dogs15", "inside", None)),
    3142: (("age", "B", "2 years", "age17", "self", None),
           ("fiv", "B", "tested FIV positive", "fiv13", "other", "retired"),
           ("cats", "A", "has lived with another cat", "cats12", "self", "old cat")),
    3143: (("papers", "B", "no papers", "papers14", "self", "breeder"),
           ("litter", "B", "still learning to use the litter box", "litter11", "self", "looks"),
           ("fiv", "A", "tested negative for FIV and FeLV", "fiv11", "other", "excellent")),
    3144: (("spay", "B", "not yet spayed/neutered", "spay12", "self", "breeder"),
           ("cats", "U", "has never met another cat", "cats13", "self", "old cat"),
           ("age", "A", "2 months", "age16", "other", "lost cat")),
    3145: (("spay", "B", "not yet spayed/neutered", "spay15", "self", "breeder"),
           ("cats", "A", "has lived with another cat", "cats17", "self", "old cat"),
           ("shed", "B", "sheds very little", "shed13", "self", "allergic")),
    3146: (("shed", "B", "sheds very little", "shed15", "self", "allergic"),
           ("dogs", "A", "has lived with a dog", "dogs11", "self", "dogs")),
    3147: (("senior", "A", "bloodwork last month was normal", "senior14", "self", None),
           ("microchip", "B", "not microchipped", "microchip13", "self", "student"),
           ("shed", "B", "sheds very little", "shed12", "self", "allergic")),
    3148: (("dogs", "B", "doesn't get on with dogs", "dogs13", "self", "dogs"),
           ("vacc", "U", "recovering from a cold", "vacc12", "other", "old cat")),
    3149: (("papers", "B", "no papers", "papers13", "inside", None),
           ("dogs", "A", "has lived with a dog", "dogs12", "self", "dogs"),
           ("home", "B", "indoor-outdoor", "home16", "other", "retired")),
})
PAIRED_CONTROLS.update({   # plan 3.3
    3050: ("not_only", ("eats wet food twice a day, any brand", "fully vaccinated, vet check last month",)),
    3051: ("neg_question", ("spayed/neutered",)),
    3052: ("double_neg", ("picky about litter",)),
    3053: ("neg_question", ("tested negative for FIV and FeLV",)),
    3054: ("double_neg", ("meows loudly for breakfast",)),
    3055: ("double_neg", ("a bit feisty, scratches when annoyed",)),
    3056: ("not_only", ("microchipped and healthy", "spayed/neutered",)),
    3057: ("no_doubt", ("not yet spayed/neutered",)),
    3058: ("unrelated", ("uses a scratching post, never the sofa",)),
    3059: ("no_doubt", ("not yet spayed/neutered",)),
    3060: ("unrelated", ("used to roaming a farm",)),
    3061: ("double_neg", ("picky about litter",)),
    3062: ("no_doubt", ("not yet spayed/neutered",)),
    3063: ("neg_question", ("tested negative for FIV and FeLV",)),
    3064: ("not_only", ("indoor with a catio", "is fine with children",)),
    3065: ("not_only", ("indoor with a catio", "spayed/neutered",)),
    3066: ("neg_question", ("is fine with children",)),
    3067: ("unrelated", ("meows loudly for breakfast",)),
    3068: ("unrelated", ("shy with strangers",)),
    3069: ("double_neg", ("picky about litter",)),
    3070: ("double_neg", ("shy with strangers",)),
    3071: ("no_doubt", ("not yet spayed/neutered",)),
    3072: ("unrelated", ("meows loudly for breakfast",)),
    3073: ("not_only", ("fully vaccinated, vet check last month", "spayed/neutered",)),
    3074: ("neg_question", ("spayed/neutered",)),
    3075: ("not_only", ("eats wet food twice a day, any brand", "very cuddly",)),
    3076: ("unrelated", ("has lived with another cat",)),
    3077: ("no_doubt", ("not yet spayed/neutered",)),
    3078: ("unrelated", ("not yet spayed/neutered",)),
    3079: ("double_neg", ("sheds a lot",)),
    3080: ("unrelated", ("loves being picked up and carried",)),
    3081: ("unrelated", ("has lived with a dog",)),
    3082: ("not_only", ("uses a scratching post, never the sofa", "indoor-only",)),
    3083: ("double_neg", ("eats one brand of dry food and nothing else",)),
    3084: ("no_doubt", ("meows loudly for breakfast",)),
    3085: ("unrelated", ("microchipped and healthy",)),
    3086: ("unrelated", ("not yet spayed/neutered",)),
    3087: ("not_only", ("spayed/neutered", "indoor with a catio",)),
    3088: ("unrelated", ("used to roaming a farm",)),
    3089: ("unrelated", ("meows loudly for breakfast",)),
    3090: ("neg_question", ("fully litter trained",)),
    3091: ("double_neg", ("picky about litter",)),
    3092: ("neg_question", ("has lived with a dog",)),
    3093: ("not_only", ("is fine with children", "has lived with another cat",)),
    3094: ("unrelated", ("picky about litter",)),
    3095: ("unrelated", ("not yet spayed/neutered",)),
    3096: ("neg_question", ("is fine with children",)),
    3097: ("unrelated", ("sheds a lot",)),
    3098: ("double_neg", ("meows loudly for breakfast",)),
    3099: ("not_only", ("fully litter trained", "is fine with children",)),
    3100: ("double_neg", ("calm and lazy",)),
    3101: ("unrelated", ("no vaccination records",)),
    3102: ("neg_question", ("has lived with another cat",)),
    3103: ("unrelated", ("no vaccination records",)),
    3104: ("unrelated", ("found as a stray and never claimed",)),
    3105: ("double_neg", ("a fussy eater, only eats chicken wet food",)),
    3106: ("not_only", ("tested negative for FIV and FeLV", "has lived with a dog",)),
    3107: ("unrelated", ("a fussy eater, only eats chicken wet food",)),
    3108: ("unrelated", ("picky about litter",)),
    3109: ("unrelated", ("picky about litter",)),
    3110: ("not_only", ("loves being picked up and carried", "calm and lazy",)),
    3111: ("not_only", ("spayed/neutered", "fully vaccinated, vet check last month",)),
    3112: ("double_neg", ("eats one brand of dry food and nothing else",)),
    3113: ("double_neg", ("independent",)),
    3114: ("no_doubt", ("lets you hold them for a minute, then wriggles off",)),
    3115: ("no_doubt", ("used to roaming a farm",)),
    3116: ("unrelated", ("not yet spayed/neutered",)),
    3117: ("no_doubt", ("uses a scratching post, never the sofa",)),
    3118: ("no_doubt", ("afraid of the vacuum",)),
    3119: ("not_only", ("has lived with another cat", "spayed/neutered",)),
    3120: ("no_doubt", ("uses a scratching post, never the sofa",)),
    3121: ("no_doubt", ("meows loudly for breakfast",)),
    3122: ("double_neg", ("calm and lazy",)),
    3123: ("no_doubt", ("microchipped",)),
    3124: ("double_neg", ("meows loudly for breakfast",)),
    3125: ("double_neg", ("eats one brand of dry food and nothing else",)),
    3126: ("double_neg", ("afraid of the vacuum",)),
    3127: ("unrelated", ("shy with strangers",)),
    3128: ("neg_question", ("is fine with children",)),
    3129: ("not_only", ("has lived with another cat", "calm and lazy",)),
    3130: ("not_only", ("has lived with a dog", "eats wet food twice a day, any brand",)),
    3131: ("no_doubt", ("indoor with a catio",)),
    3132: ("not_only", ("spayed/neutered", "microchipped",)),
    3133: ("double_neg", ("on a monthly flea treatment",)),
    3134: ("double_neg", ("meows loudly for breakfast",)),
    3135: ("double_neg", ("picky about litter",)),
    3136: ("unrelated", ("meows loudly for breakfast",)),
    3137: ("not_only", ("spayed/neutered", "indoor with a catio",)),
    3138: ("double_neg", ("eats wet food twice a day, any brand",)),
    3139: ("unrelated", ("spayed/neutered",)),
    3140: ("double_neg", ("picky about litter",)),
    3141: ("unrelated", ("rehomed because a family member became allergic",)),
    3142: ("unrelated", ("fully litter trained",)),
    3143: ("not_only", ("calm and lazy", "indoor with a catio",)),
    3144: ("double_neg", ("eats wet food twice a day, any brand",)),
    3145: ("no_doubt", ("follows people around like a dog",)),
    3146: ("no_doubt", ("dewormed but not vaccinated",)),
    3147: ("not_only", ("loves being picked up and carried", "tested negative for FIV and FeLV",)),
    3148: ("unrelated", ("only drinks from a running tap",)),
    3149: ("unrelated", ("only drinks from a running tap",)),
})
PAIRED_BUTS.update({   # plan 3.4: (kind, connector, heard, asker, raised, lowered, facts)
    3050: (("bad but good", "but", "other", "hiker", "old cat", "dogs",
            ("has never met a dog", "has lived with another cat",)),
           ("fragment", "but", "other", "nurse", "busy road", "long hours",
            ("microchipped", "only drinks from a running tap",))),
    3051: (("good but bad", "but", "other", "nurse", "busy road", "kids",
            ("indoor-only", "hates being picked up but sits close by",)),
           ("bad but good", "but", "self", "long hours", "old cat", "long hours",
            ("follows people around like a dog", "has lived with another cat",))),
    3052: (("good but bad", "but", "other", "kids", "apartment", "allergic",
            ("calm and lazy", "sheds a lot",)),
           ("fragment", "tho", "other", "retired", "dogs", "kids",
            ("hates being picked up but sits close by", "on a monthly flea treatment",))),
    3053: (("bad but good", "though", "other", "deal", "busy road", "long hours",
            ("only drinks from a running tap", "microchipped",)),
           ("fragment", "but", "other", "student", "busy road", "old cat",
            ("has never met another cat", "indoor-only",))),
    3054: (("good but bad", "but", "other", "nurse", "kids", "allergic",
            ("loves being picked up and carried", "sheds a lot",)),
           ("bad but good", "but", "other", "hiker", "old cat", "student",
            ("not microchipped", "tested negative for FIV and FeLV",))),
    3055: (("good but bad", "but", "self", "kids", "kids", "long hours",
            ("is fine with children", "litter trained but kicks litter everywhere",)),
           ("bad but good", "but", "self", "old cat", "old cat", "long hours",
            ("only drinks from a running tap", "on a monthly flea treatment",))),
    3056: (("bad but good", "but", "self", "busy road", "apartment", "busy road",
            ("used to roaming a farm", "calm and lazy",)),
           ("fragment", "tho", "other", "apartment", "looks", "kids",
            ("uses a scratching post, never the sofa", "has never met children",))),
    3057: (("good but bad", "but", "self", "busy road", "old cat", "busy road",
            ("treated for fleas and ear mites last month", "used to roaming a farm",)),
           ("bad but good", "but", "other", "apartment", "dogs", "old cat",
            ("has never met another cat", "has lived with a dog",))),
    3058: (("bad but good", "but", "self", "kids", "retired", "kids",
            ("hates being picked up but sits close by", "very cuddly",)),
           ("fragment", "tho", "other", "retired", "old cat", "busy road",
            ("tested negative for FIV and FeLV", "indoor-outdoor",))),
    3059: (("good but bad", "but", "self", "long hours", "dogs", "long hours",
            ("has lived with a dog", "only drinks from a running tap",)),
           ("fragment", "tho", "other", "lost cat", "old cat", "student",
            ("has lived with another cat", "a fussy eater, only eats chicken wet food",))),
    3060: (("fragment", "tho", "self", "old cat", "old cat", "kids",
            ("a bit feisty, scratches when annoyed", "has lived with another cat",)),
           ("caveat", "but", "other", "nurse", None, "excellent",
            ("vaccinated but no vet visit in two years",))),
    3061: (("bad but good", "but", "other", "excellent", "long hours", "old cat",
            ("never tested for FIV or FeLV", "independent",)),
           ("fragment", "but", "other", "nurse", "busy road", "kids",
            ("microchipped", "has never met children",))),
    3062: (("bad but good", "though", "other", "long hours", "old cat", "dogs",
            ("has never met a dog", "tested negative for FIV and FeLV",)),
           ("fragment", "tho", "other", "dogs", "busy road", "old cat",
            ("has never had a flea treatment", "indoor with a catio",))),
    3063: (("bad but good", "but", "self", "long hours", "dogs", "long hours",
            ("only drinks from a running tap", "has lived with a dog",)),
           ("caveat", "though", "other", "lost cat", None, "kids",
            ("lets you hold them for a minute, then wriggles off",))),
    3064: (("good but bad", "but", "other", "excellent", "looks", "student",
            ("uses a scratching post, never the sofa", "not microchipped",)),
           ("caveat", "but", "other", "lost cat", None, "excellent",
            ("had the first of three kitten vaccines",))),
    3065: (("bad but good", "but", "self", "long hours", "busy road", "long hours",
            ("only drinks from a running tap", "microchipped",)),
           ("caveat", "but", "self", "excellent", None, "excellent",
            ("had two of three kitten vaccines",))),
    3066: (("bad but good", "but", "other", "nurse", "kids", "looks",
            ("still learning to use the litter box", "loves being picked up and carried",)),
           ("caveat", "tho", "other", "old cat", None, "excellent",
            ("had two of three kitten vaccines",))),
    3067: (("bad but good", "though", "self", "kids", "kids", "dogs",
            ("has never met a dog", "loves being picked up and carried",)),
           ("fragment", "tho", "other", "looks", "busy road", "long hours",
            ("only drinks from a running tap", "indoor-only",))),
    3068: (("good but bad", "but", "other", "lost cat", "kids", "old cat",
            ("is fine with children", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "self", "busy road", "dogs", "busy road",
            ("used to roaming a farm", "on a monthly flea treatment",))),
    3069: (("good but bad", "but", "other", "apartment", "long hours", "old cat",
            ("independent", "has never met another cat",)),
           ("bad but good", "but", "other", "nurse", "busy road", "kids",
            ("has never met children", "indoor with a catio",))),
    3070: (("good but bad", "but", "other", "looks", "busy road", "old cat",
            ("indoor-only", "has never met another cat",)),
           ("fragment", "but", "self", "kids", "looks", "kids",
            ("uses a scratching post, never the sofa", "has never met children",))),
    3071: (("good but bad", "but", "other", "hiker", "dogs", "allergic",
            ("has lived with a dog", "sheds a lot",)),
           ("bad but good", "but", "other", "allergic", "retired", "old cat",
            ("never tested for FIV or FeLV", "very cuddly",))),
    3072: (("good but bad", "but", "self", "excellent", "excellent", "old cat",
            ("finished the kitten vaccine series", "has never met another cat",)),
           ("caveat", "though", "self", "kids", None, "kids",
            ("lets you hold them for a minute, then wriggles off",))),
    3073: (("good but bad", "but", "other", "retired", "kids", "dogs",
            ("is fine with children", "has never met a dog",)),
           ("fragment", "but", "self", "old cat", "retired", "old cat",
            ("very cuddly", "never tested for FIV or FeLV",))),
    3074: (("good but bad", "but", "self", "looks", "looks", "busy road",
            ("uses a scratching post, never the sofa", "indoor-outdoor",)),
           ("bad but good", "but", "other", "deal", "old cat", "student",
            ("a fussy eater, only eats chicken wet food", "has lived with another cat",))),
    3075: (("good but bad", "but", "self", "allergic", "kids", "allergic",
            ("loves being picked up and carried", "sheds a lot",)),
           ("bad but good", "but", "other", "lost cat", "excellent", "old cat",
            ("has never met another cat", "fully vaccinated, vet check last month",))),
    3076: (("good but bad", "but", "self", "old cat", "busy road", "old cat",
            ("indoor-only", "has never had a flea treatment",)),
           ("bad but good", "but", "other", "lost cat", "looks", "excellent",
            ("vaccinated but no vet visit in two years", "uses a scratching post, never the sofa",))),
    3077: (("good but bad", "but", "self", "old cat", "old cat", "excellent",
            ("has lived with another cat", "vaccinated but no vet visit in two years",)),
           ("fragment", "but", "self", "kids", "kids", "old cat",
            ("never tested for FIV or FeLV", "is fine with children",))),
    3078: (("good but bad", "but", "other", "retired", "old cat", "kids",
            ("tested negative for FIV and FeLV", "hates being picked up but sits close by",)),
           ("bad but good", "but", "self", "busy road", "retired", "busy road",
            ("indoor-outdoor", "very cuddly",))),
    3079: (("bad but good", "but", "self", "busy road", "excellent", "busy road",
            ("used to roaming a farm", "fully vaccinated, vet check last month",)),
           ("fragment", "tho", "self", "kids", "busy road", "kids",
            ("a bit feisty, scratches when annoyed", "microchipped",))),
    3080: (("good but bad", "but", "other", "student", "dogs", "long hours",
            ("has lived with a dog", "only drinks from a running tap",)),
           ("bad but good", "but", "self", "looks", "looks", "kids",
            ("has never met children", "uses a scratching post, never the sofa",))),
    3081: (("good but bad", "tho", "other", "hiker", "apartment", "old cat",
            ("calm and lazy", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "self", "dogs", "dogs", "long hours",
            ("only drinks from a running tap", "on a monthly flea treatment",))),
    3082: (("good but bad", "but", "self", "kids", "old cat", "kids",
            ("tested negative for FIV and FeLV", "has never met children",)),
           ("bad but good", "but", "self", "long hours", "old cat", "long hours",
            ("litter trained but kicks litter everywhere", "bloodwork last month was normal",))),
    3083: (("good but bad", "but", "other", "old cat", "excellent", "dogs",
            ("fully vaccinated, vet check last month", "has never met a dog",)),
           ("bad but good", "but", "other", "excellent", "old cat", "kids",
            ("a bit feisty, scratches when annoyed", "has lived with another cat",))),
    3084: (("fragment", "tho", "other", "looks", "old cat", "kids",
            ("tested negative for FIV and FeLV", "a bit feisty, scratches when annoyed",)),
           ("caveat", "though", "other", "kids", None, "looks",
            ("litter trained but kicks litter everywhere",))),
    3085: (("good but bad", "but", "self", "dogs", "dogs", "kids",
            ("has lived with a dog", "a bit feisty, scratches when annoyed",)),
           ("bad but good", "but", "self", "long hours", "kids", "long hours",
            ("only drinks from a running tap", "loves being picked up and carried",))),
    3086: (("good but bad", "but", "other", "old cat", "dogs", "kids",
            ("treated for fleas and ear mites last month", "a bit feisty, scratches when annoyed",)),
           ("bad but good", "but", "other", "long hours", "excellent", "student",
            ("a fussy eater, only eats chicken wet food", "fully vaccinated, vet check last month",))),
    3087: (("good but bad", "but", "other", "apartment", "dogs", "long hours",
            ("has lived with a dog", "litter trained but kicks litter everywhere",)),
           ("bad but good", "but", "self", "old cat", "old cat", "excellent",
            ("recovering from a cold", "has lived with another cat",))),
    3088: (("good but bad", "but", "self", "kids", "dogs", "kids",
            ("has lived with a dog", "has never met children",)),
           ("bad but good", "but", "self", "long hours", "old cat", "long hours",
            ("only drinks from a running tap", "tested negative for FIV and FeLV",))),
    3089: (("bad but good", "but", "self", "dogs", "dogs", "kids",
            ("has never met children", "on a monthly flea treatment",)),
           ("fragment", "but", "other", "hiker", "old cat", "student",
            ("not microchipped", "tested negative for FIV and FeLV",))),
    3090: (("good but bad", "but", "other", "dogs", "looks", "long hours",
            ("uses a scratching post, never the sofa", "follows people around like a dog",)),
           ("bad but good", "though", "self", "busy road", "busy road", "old cat",
            ("has never met another cat", "indoor with a catio",))),
    3091: (("bad but good", "but", "self", "dogs", "retired", "dogs",
            ("has never met a dog", "very cuddly",)),
           ("fragment", "tho", "self", "busy road", "looks", "busy road",
            ("uses a scratching post, never the sofa", "used to roaming a farm",))),
    3092: (("good but bad", "though", "self", "kids", "old cat", "kids",
            ("on a monthly flea treatment", "a bit feisty, scratches when annoyed",)),
           ("bad but good", "but", "other", "nurse", "busy road", "long hours",
            ("litter trained but kicks litter everywhere", "indoor with a catio",))),
    3093: (("bad but good", "but", "self", "busy road", "busy road", "allergic",
            ("sheds a lot", "indoor-only",)),
           ("caveat", "but", "other", "nurse", None, "excellent",
            ("dewormed but not vaccinated",))),
    3094: (("good but bad", "but", "other", "apartment", "busy road", "kids",
            ("microchipped and healthy", "has never met children",)),
           ("fragment", "tho", "other", "long hours", "kids", "old cat",
            ("loves being picked up and carried", "has never met another cat",))),
    3095: (("good but bad", "though", "other", "kids", "dogs", "busy road",
            ("treated for fleas and ear mites last month", "used to roaming a farm",)),
           ("caveat", "but", "other", "retired", None, "kids",
            ("lets you hold them for a minute, then wriggles off",))),
    3096: (("good but bad", "but", "other", "kids", "old cat", "student",
            ("on a monthly flea treatment", "a fussy eater, only eats chicken wet food",)),
           ("caveat", "but", "self", "excellent", None, "excellent",
            ("dewormed but not vaccinated",))),
    3097: (("bad but good", "though", "other", "retired", "dogs", "kids",
            ("has never met children", "has lived with a dog",)),
           ("fragment", "tho", "self", "looks", "busy road", "looks",
            ("litter trained but kicks litter everywhere", "indoor with a catio",))),
    3098: (("bad but good", "but", "other", "apartment", "retired", "allergic",
            ("sheds a lot", "very cuddly",)),
           ("fragment", "but", "other", "student", "busy road", "dogs",
            ("has never met a dog", "indoor with a catio",))),
    3099: (("good but bad", "though", "other", "lost cat", "old cat", "kids",
            ("tested negative for FIV and FeLV", "a bit feisty, scratches when annoyed",)),
           ("bad but good", "but", "self", "old cat", "old cat", "busy road",
            ("used to roaming a farm", "has lived with another cat",))),
    3100: (("good but bad", "tho", "self", "kids", "old cat", "kids",
            ("has lived with another cat", "has never met children",)),
           ("caveat", "tho", "self", "looks", None, "looks",
            ("litter trained but kicks litter everywhere",))),
    3101: (("good but bad", "but", "other", "student", "dogs", "old cat",
            ("has lived with a dog", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "other", "retired", "busy road", "old cat",
            ("has never had a flea treatment", "indoor with a catio",))),
    3102: (("caveat", "though", "self", "excellent", None, "excellent",
            ("had two of three kitten vaccines",)),
           ("caveat", "but", "other", "nurse", None, "kids",
            ("lets you hold them for a minute, then wriggles off",))),
    3103: (("good but bad", "but", "self", "looks", "looks", "dogs",
            ("uses a scratching post, never the sofa", "has never met a dog",)),
           ("fragment", "but", "self", "busy road", "busy road", "long hours",
            ("indoor-only", "only drinks from a running tap",))),
    3104: (("good but bad", "but", "self", "busy road", "long hours", "busy road",
            ("independent", "used to roaming a farm",)),
           ("bad but good", "but", "other", "old cat", "excellent", "dogs",
            ("has never had a flea treatment", "fully vaccinated, vet check last month",))),
    3105: (("bad but good", "but", "other", "hiker", "old cat", "dogs",
            ("has never met a dog", "tested negative for FIV and FeLV",)),
           ("caveat", "but", "other", "old cat", None, "excellent",
            ("had the first of three kitten vaccines",))),
    3106: (("good but bad", "though", "other", "old cat", "retired", "allergic",
            ("very cuddly", "sheds a lot",)),
           ("bad but good", "but", "other", "looks", "old cat", "busy road",
            ("used to roaming a farm", "has lived with another cat",))),
    3107: (("good but bad", "but", "other", "retired", "busy road", "allergic",
            ("indoor with a catio", "sheds a lot",)),
           ("bad but good", "though", "other", "deal", "retired", "dogs",
            ("has never met a dog", "very cuddly",))),
    3108: (("good but bad", "though", "self", "kids", "kids", "student",
            ("is fine with children", "not microchipped",)),
           ("caveat", "but", "other", "nurse", None, "kids",
            ("lets you hold them for a minute, then wriggles off",))),
    3109: (("bad but good", "though", "other", "looks", "dogs", "busy road",
            ("used to roaming a farm", "has lived with a dog",)),
           ("fragment", "but", "self", "kids", "looks", "kids",
            ("has never met children", "uses a scratching post, never the sofa",))),
    3110: (("bad but good", "but", "other", "kids", "busy road", "student",
            ("not microchipped", "indoor with a catio",)),
           ("fragment", "tho", "self", "old cat", "old cat", "kids",
            ("tested negative for FIV and FeLV", "has never met children",))),
    3111: (("good but bad", "though", "self", "dogs", "dogs", "allergic",
            ("treated for fleas and ear mites last month", "sheds a lot",)),
           ("fragment", "tho", "self", "old cat", "retired", "old cat",
            ("never tested for FIV or FeLV", "very cuddly",))),
    3112: (("good but bad", "but", "self", "excellent", "excellent", "old cat",
            ("finished the kitten vaccine series", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "other", "nurse", "dogs", "allergic",
            ("sheds a lot", "has lived with a dog",))),
    3113: (("good but bad", "but", "self", "excellent", "excellent", "kids",
            ("finished the kitten vaccine series", "has never met children",)),
           ("bad but good", "though", "self", "busy road", "old cat", "busy road",
            ("indoor-outdoor", "tested negative for FIV and FeLV",))),
    3114: (("good but bad", "but", "other", "student", "long hours", "old cat",
            ("independent", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "self", "old cat", "old cat", "kids",
            ("has never met children", "on a monthly flea treatment",))),
    3115: (("good but bad", "but", "other", "busy road", "old cat", "allergic",
            ("tested negative for FIV and FeLV", "sheds a lot",)),
           ("fragment", "tho", "other", "kids", "busy road", "hiker",
            ("has some stiff joints", "microchipped",))),
    3116: (("good but bad", "though", "self", "kids", "busy road", "kids",
            ("microchipped", "has never met children",)),
           ("bad but good", "though", "other", "nurse", "dogs", "busy road",
            ("used to roaming a farm", "has lived with a dog",))),
    3117: (("good but bad", "but", "self", "old cat", "busy road", "old cat",
            ("microchipped", "never tested for FIV or FeLV",)),
           ("fragment", "tho", "other", "nurse", "old cat", "excellent",
            ("recovering from a cold", "treated for fleas and ear mites last month",))),
    3118: (("good but bad", "but", "self", "busy road", "busy road", "old cat",
            ("indoor with a catio", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "other", "deal", "old cat", "long hours",
            ("only drinks from a running tap", "treated for fleas and ear mites last month",))),
    3119: (("good but bad", "but", "self", "long hours", "long hours", "busy road",
            ("independent", "indoor-outdoor",)),
           ("fragment", "but", "self", "old cat", "old cat", "allergic",
            ("on a monthly flea treatment", "sheds a lot",))),
    3120: (("good but bad", "but", "other", "nurse", "retired", "kids",
            ("very cuddly", "has never met children",)),
           ("bad but good", "but", "other", "apartment", "kids", "busy road",
            ("indoor-outdoor", "loves being picked up and carried",))),
    3121: (("good but bad", "but", "self", "kids", "busy road", "kids",
            ("indoor with a catio", "has never met children",)),
           ("fragment", "but", "other", "old cat", "dogs", "kids",
            ("treated for fleas and ear mites last month", "a bit feisty, scratches when annoyed",))),
    3122: (("good but bad", "but", "other", "lost cat", "old cat", "kids",
            ("tested negative for FIV and FeLV", "hates being picked up but sits close by",)),
           ("bad but good", "but", "self", "kids", "kids", "allergic",
            ("sheds a lot", "is fine with children",))),
    3123: (("good but bad", "but", "self", "excellent", "excellent", "old cat",
            ("fully vaccinated, vet check last month", "has never met another cat",)),
           ("bad but good", "but", "self", "busy road", "busy road", "long hours",
            ("follows people around like a dog", "indoor with a catio",))),
    3124: (("good but bad", "but", "other", "busy road", "old cat", "allergic",
            ("has lived with another cat", "sheds a lot",)),
           ("fragment", "but", "other", "allergic", "long hours", "busy road",
            ("indoor-outdoor", "independent",))),
    3125: (("good but bad", "but", "self", "excellent", "kids", "excellent",
            ("loves being picked up and carried", "recovering from a cold",)),
           ("fragment", "but", "self", "dogs", "long hours", "dogs",
            ("independent", "has never met a dog",))),
    3126: (("good but bad", "but", "other", "dogs", "retired", "old cat",
            ("very cuddly", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "self", "busy road", "busy road", "long hours",
            ("still learning to use the litter box", "indoor-only",))),
    3127: (("good but bad", "but", "other", "retired", "old cat", "dogs",
            ("bloodwork last month was normal", "has never met a dog",)),
           ("bad but good", "but", "other", "nurse", "old cat", "student",
            ("not microchipped", "tested negative for FIV and FeLV",))),
    3128: (("good but bad", "but", "other", "allergic", "long hours", "busy road",
            ("independent", "indoor-outdoor",)),
           ("bad but good", "but", "self", "kids", "excellent", "kids",
            ("hates being picked up but sits close by", "fully vaccinated, vet check last month",))),
    3129: (("fragment", "but", "self", "old cat", "looks", "old cat",
            ("never tested for FIV or FeLV", "uses a scratching post, never the sofa",)),
           ("caveat", "but", "other", "retired", None, "excellent",
            ("vaccinated but no vet visit in two years",))),
    3130: (("good but bad", "but", "other", "nurse", "busy road", "allergic",
            ("microchipped", "sheds a lot",)),
           ("bad but good", "though", "other", "excellent", "retired", "long hours",
            ("only drinks from a running tap", "very cuddly",))),
    3131: (("bad but good", "though", "other", "retired", "dogs", "student",
            ("not microchipped", "has lived with a dog",)),
           ("fragment", "but", "other", "deal", "apartment", "long hours",
            ("only drinks from a running tap", "calm and lazy",))),
    3132: (("bad but good", "though", "self", "kids", "old cat", "kids",
            ("has never met children", "on a monthly flea treatment",)),
           ("fragment", "but", "self", "long hours", "old cat", "long hours",
            ("tested negative for FIV and FeLV", "only drinks from a running tap",))),
    3133: (("bad but good", "tho", "self", "long hours", "looks", "long hours",
            ("follows people around like a dog", "uses a scratching post, never the sofa",)),
           ("fragment", "but", "self", "dogs", "dogs", "kids",
            ("has never met children", "has lived with a dog",))),
    3134: (("good but bad", "but", "other", "apartment", "old cat", "long hours",
            ("on a monthly flea treatment", "follows people around like a dog",)),
           ("fragment", "but", "other", "deal", "dogs", "old cat",
            ("has lived with a dog", "never tested for FIV or FeLV",))),
    3135: (("good but bad", "but", "self", "busy road", "apartment", "busy road",
            ("calm and lazy", "used to roaming a farm",)),
           ("bad but good", "but", "self", "allergic", "busy road", "allergic",
            ("sheds a lot", "microchipped and healthy",))),
    3136: (("bad but good", "but", "other", "student", "old cat", "long hours",
            ("follows people around like a dog", "has lived with another cat",)),
           ("fragment", "tho", "other", "retired", "kids", "busy road",
            ("is fine with children", "used to roaming a farm",))),
    3137: (("good but bad", "but", "self", "looks", "looks", "student",
            ("uses a scratching post, never the sofa", "not microchipped",)),
           ("bad but good", "but", "other", "nurse", "old cat", "long hours",
            ("still learning to use the litter box", "tested negative for FIV and FeLV",))),
    3138: (("fragment", "tho", "other", "apartment", "looks", "busy road",
            ("indoor-outdoor", "uses a scratching post, never the sofa",)),
           ("caveat", "though", "other", "lost cat", None, "excellent",
            ("had two of three kitten vaccines",))),
    3139: (("good but bad", "but", "self", "old cat", "old cat", "excellent",
            ("tested negative for FIV and FeLV", "recovering from a cold",)),
           ("bad but good", "but", "self", "kids", "old cat", "kids",
            ("a bit feisty, scratches when annoyed", "has lived with another cat",))),
    3140: (("bad but good", "though", "self", "busy road", "busy road", "old cat",
            ("never tested for FIV or FeLV", "indoor-only",)),
           ("caveat", "but", "other", "apartment", None, "kids",
            ("hates being picked up but sits close by",))),
    3141: (("good but bad", "but", "other", "nurse", "excellent", "busy road",
            ("fully vaccinated, vet check last month", "used to roaming a farm",)),
           ("caveat", "but", "self", "looks", None, "looks",
            ("litter trained but kicks litter everywhere",))),
    3142: (("good but bad", "but", "other", "student", "dogs", "busy road",
            ("has lived with a dog", "used to roaming a farm",)),
           ("bad but good", "but", "other", "deal", "old cat", "long hours",
            ("only drinks from a running tap", "treated for fleas and ear mites last month",))),
    3143: (("good but bad", "tho", "other", "long hours", "dogs", "allergic",
            ("has lived with a dog", "sheds a lot",)),
           ("fragment", "but", "other", "old cat", "excellent", "long hours",
            ("finished the kitten vaccine series", "only drinks from a running tap",))),
    3144: (("good but bad", "but", "other", "retired", "old cat", "long hours",
            ("treated for fleas and ear mites last month", "only drinks from a running tap",)),
           ("caveat", "but", "other", "nurse", None, "excellent",
            ("had the first of three kitten vaccines",))),
    3145: (("good but bad", "but", "other", "apartment", "dogs", "busy road",
            ("has lived with a dog", "indoor-outdoor",)),
           ("bad but good", "but", "self", "looks", "looks", "old cat",
            ("has never had a flea treatment", "uses a scratching post, never the sofa",))),
    3146: (("bad but good", "though", "other", "looks", "old cat", "busy road",
            ("indoor-outdoor", "tested negative for FIV and FeLV",)),
           ("fragment", "tho", "other", "old cat", "kids", "long hours",
            ("loves being picked up and carried", "only drinks from a running tap",))),
    3147: (("good but bad", "but", "self", "kids", "retired", "kids",
            ("very cuddly", "has never met children",)),
           ("bad but good", "but", "other", "hiker", "busy road", "dogs",
            ("has never met a dog", "indoor with a catio",))),
    3148: (("good but bad", "but", "other", "apartment", "old cat", "kids",
            ("has lived with another cat", "a bit feisty, scratches when annoyed",)),
           ("fragment", "tho", "other", "looks", "busy road", "dogs",
            ("indoor-only", "has never had a flea treatment",))),
    3149: (("good but bad", "but", "other", "hiker", "busy road", "old cat",
            ("microchipped and healthy", "never tested for FIV or FeLV",)),
           ("bad but good", "but", "self", "student", "retired", "student",
            ("a fussy eater, only eats chicken wet food", "very cuddly",))),
})
PAIRED_PRESENT.update({   # plan 3.5
    3050: ("breeder",), 3051: ("allergic",), 3052: ("breeder",), 3053: ("breeder", "allergic",),
    3056: ("breeder", "dogs", "allergic",), 3060: ("breeder",), 3061: ("breeder",), 3062: ("breeder", "allergic",),
    3063: ("breeder",), 3064: ("breeder",), 3065: ("breeder",), 3066: ("breeder",), 3067: ("breeder", "allergic",),
    3068: ("breeder",), 3069: ("breeder",), 3071: ("breeder",), 3073: ("breeder",), 3075: ("breeder",),
    3077: ("breeder",), 3078: ("breeder", "allergic",), 3081: ("breeder",), 3083: ("breeder",),
    3084: ("breeder", "allergic",), 3086: ("breeder",), 3088: ("breeder",), 3090: ("breeder", "allergic",),
    3091: ("breeder",), 3093: ("breeder",), 3095: ("breeder",), 3096: ("breeder",), 3098: ("breeder",),
    3099: ("breeder",), 3101: ("allergic",), 3102: ("breeder", "dogs",), 3103: ("breeder",), 3106: ("breeder",),
    3107: ("breeder",), 3108: ("breeder",), 3109: ("breeder",), 3110: ("breeder",), 3111: ("breeder",),
    3113: ("breeder",), 3114: ("breeder",), 3118: ("allergic",), 3120: ("breeder",), 3121: ("breeder",),
    3122: ("dogs",), 3126: ("breeder",), 3128: ("breeder",), 3130: ("breeder", "dogs",),
    3132: ("breeder", "allergic",), 3135: ("breeder",), 3136: ("dogs",), 3137: ("breeder",),
    3138: ("breeder", "allergic",), 3139: ("breeder",), 3141: ("breeder", "allergic",), 3142: ("breeder",),
    3144: ("allergic",), 3146: ("breeder",), 3148: ("breeder",), 3149: ("allergic",),
})


# Seeds 3150-3234 (data/written_sales/prompts_3150_plan.md, approved by the owner 2026-11-10): sayings buyer GPT B6 had
# seen too few times (notebook 14 sections 18-22). A target may carry a seventh field, its shape: how the seller says
# it (ml/tools/external_prompt.py SHAPES) and which buyers it needs in the sale (SHAPE_BUYERS, by BUYER key). The
# targets, controls and "but" answers were allocated once by data/written_sales/review_tools/allocate_3150.py, then
# frozen below.
SAYINGS_FROM = 3150
GOOD_HEALTH = ("microchipped and healthy", "fully vaccinated, vet check last month")
SHAPE_BUYERS = {
    "player": ("old cat", "long hours", "excellent"),       # "she has FIV" / "got FIV" / "fiv+"
    "player_good": ("old cat", "long hours", "excellent"),  # the same, then "but shes healthy"
    "healthy": ("excellent", "long hours"),                 # "She's healthy." and nothing else
    "dog_only": ("old cat", "dogs"),                        # "Not really, but she has lived with a dog."
    "dog_said": ("old cat", "dogs"),                        # "She has lived with a dog." (no word on cats)
    "cat_lived": ("old cat", "dogs"),                       # "She has lived with another cat."
    "age_only": ("long hours",),                            # "She's 2 years old." and nothing else
    "intact_months": ("long hours",),                       # "shes intact, only 2 months"
    "unsure": ("excellent", "kids", "old cat"),             # "I'm not sure about her shots."
    "shed": ("allergic", "looks"),                          # doesn't shed much / not much / barely / hardly
    "shed_intact": ("allergic",),                           # "barely sheds, but she isn't spayed yet"
}
PAIRED_SEEDS.update({   # 3150-3234 (prompts_3150_plan.md 3.1): (topic, side, value, group, heard, asker, shape)
    3150: (('vacc', 'A', 'microchipped and healthy', 'fivp1', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw1', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly1', 'self', None, 'age_only')),
    3151: (('fiv', 'B', 'tested FIV positive', 'fivp1', 'self', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure1', 'self', 'excellent', 'unsure'),
           ('cats', 'U', 'has never met another cat', 'dogonly1', 'self', 'old cat', 'dog_only')),
    3152: (('dogs', 'A', 'has lived with a dog', 'dogonly1', 'self', 'dogs', 'dog_said'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly1', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw1', 'other', 'apartment', 'shed')),
    3153: (('cats', 'A', 'has lived with another cat', 'dogonly1', 'self', 'old cat', 'cat_lived'),
           ('age', 'A', '2 months', 'ageonly1', 'other', 'pedigree', 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw1', 'self', 'allergic', 'shed_intact')),
    3154: (('fiv', 'B', 'tested FIV positive', 'fivp1', 'self', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure1', 'other', 'nurse', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw2', 'self', 'allergic', 'shed')),
    3155: (('cats', 'U', 'has never met another cat', 'dogonly2', 'other', 'retired', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly2', 'other', 'deal', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw2', 'inside', None, 'shed')),
    3156: (('fiv', 'B', 'tested FIV positive', 'fivp2', 'other', 'nurse', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure1', 'self', 'kids', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw2', 'other', 'excellent', 'shed')),
    3157: (('cats', 'U', 'has never met another cat', 'dogonly2', 'self', 'old cat', 'dog_only'),
           ('age', 'A', '2 months', 'ageonly2', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw3', 'other', 'retired', 'shed_intact')),
    3158: (('vacc', 'A', 'microchipped and healthy', 'fivp2', 'other', 'lost cat', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw3', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly2', 'inside', None, 'age_only')),
    3159: (('fiv', 'B', 'tested FIV positive', 'fivp2', 'other', 'lost cat', 'player'),
           ('shed', 'B', 'sheds very little', 'shedw3', 'other', 'kids', 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly2', 'inside', None, 'dog_only')),
    3160: (('fiv', 'B', 'tested FIV positive', 'fivp3', 'self', None, 'player_good'),
           ('shed', 'B', 'sheds very little', 'shedw4', 'inside', None, 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly3', 'self', 'old cat', 'dog_only')),
    3161: (('cats', 'U', 'has never met another cat', 'dogonly3', 'other', 'apartment', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly3', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw4', 'self', 'allergic', 'shed')),
    3162: (('fiv', 'B', 'tested FIV positive', 'fivp3', 'self', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure2', 'inside', None, 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw4', 'other', 'nurse', 'shed')),
    3163: (('fiv', 'B', 'tested FIV positive', 'fivp3', 'inside', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure2', 'self', 'old cat', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw5', 'self', 'allergic', 'shed')),
    3164: (('fiv', 'B', 'tested FIV positive', 'fivp4', 'inside', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure2', 'other', 'retired', 'unsure'),
           ('age', 'B', '2 years', 'ageonly3', 'other', 'lost cat', 'age_only')),
    3165: (('vacc', 'A', 'microchipped and healthy', 'fivp4', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw5', 'inside', None, 'shed'),
           ('age', 'B', '2 years', 'ageonly3', 'self', None, 'age_only')),
    3166: (('cats', 'U', 'has never met another cat', 'dogonly3', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly4', 'inside', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw5', 'other', 'apartment', 'shed')),
    3167: (('vacc', 'A', 'microchipped and healthy', 'fivp4', 'other', 'deal', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw6', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly4', 'other', 'lost cat', 'age_only')),
    3168: (('fiv', 'B', 'tested FIV positive', 'fivp5', 'other', 'retired', 'player'),
           ('vacc', 'U', 'on medication for an overactive thyroid', 'unsure3', 'self', 'excellent', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw6', 'other', 'apartment', 'shed')),
    3169: (('cats', 'U', 'has never met another cat', 'dogonly4', 'inside', None, 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly4', 'other', 'student', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw6', 'inside', None, 'shed')),
    3170: (('fiv', 'B', 'tested FIV positive', 'fivp5', 'other', 'kids', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure3', 'inside', None, 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw7', 'self', 'allergic', 'shed')),
    3171: (('fiv', 'B', 'tested FIV positive', 'fivp5', 'self', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure3', 'self', 'kids', 'unsure'),
           ('age', 'B', '2 years', 'ageonly5', 'inside', None, 'age_only')),
    3172: (('dogs', 'A', 'has lived with a dog', 'dogonly4', 'other', 'retired', 'dog_said'),
           ('age', 'A', '2 months', 'ageonly5', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw7', 'self', 'allergic', 'shed_intact')),
    3173: (('fiv', 'B', 'tested FIV positive', 'fivp6', 'self', None, 'player'),
           ('vacc', 'U', 'bloodwork last month was normal', 'unsure4', 'other', 'apartment', 'unsure'),
           ('cats', 'A', 'has lived with another cat', 'dogonly4', 'other', 'nurse', 'cat_lived')),
    3174: (('cats', 'U', 'has never met another cat', 'dogonly5', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly5', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw7', 'other', 'retired', 'shed')),
    3175: (('vacc', 'A', 'microchipped and healthy', 'fivp6', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw8', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly6', 'other', 'student', 'age_only')),
    3176: (('cats', 'U', 'has never met another cat', 'dogonly5', 'other', 'nurse', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly6', 'other', 'lost cat', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw8', 'inside', None, 'shed')),
    3177: (('fiv', 'B', 'tested FIV positive', 'fivp6', 'other', 'apartment', 'player'),
           ('vacc', 'U', 'bloodwork last month was normal', 'unsure4', 'self', 'old cat', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw8', 'self', 'allergic', 'shed')),
    3178: (('fiv', 'B', 'tested FIV positive', 'fivp7', 'other', 'nurse', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure4', 'other', 'lost cat', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw9', 'other', 'apartment', 'shed')),
    3179: (('cats', 'U', 'has never met another cat', 'dogonly5', 'self', 'old cat', 'dog_only'),
           ('age', 'A', '2 months', 'ageonly6', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw9', 'inside', None, 'shed_intact')),
    3180: (('vacc', 'A', 'microchipped and healthy', 'fivp7', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw9', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly7', 'inside', None, 'age_only')),
    3181: (('fiv', 'B', 'tested FIV positive', 'fivp7', 'inside', None, 'player'),
           ('shed', 'B', 'sheds very little', 'shedw10', 'inside', None, 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly6', 'other', 'kids', 'dog_only')),
    3182: (('dogs', 'A', 'has lived with a dog', 'dogonly6', 'self', 'dogs', 'dog_said'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly7', 'inside', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw10', 'other', 'nurse', 'shed')),
    3183: (('fiv', 'B', 'tested FIV positive', 'fivp8', 'inside', None, 'player_good'),
           ('shed', 'B', 'sheds very little', 'shedw10', 'self', 'allergic', 'shed'),
           ('cats', 'A', 'has lived with another cat', 'dogonly6', 'self', 'old cat', 'cat_lived')),
    3184: (('vacc', 'A', 'microchipped and healthy', 'fivp8', 'other', 'nurse', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw11', 'other', 'retired', 'shed'),
           ('age', 'B', '2 years', 'ageonly7', 'other', 'pedigree', 'age_only')),
    3185: (('cats', 'U', 'has never met another cat', 'dogonly7', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly8', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw11', 'inside', None, 'shed')),
    3186: (('fiv', 'B', 'tested FIV positive', 'fivp8', 'self', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure5', 'self', 'excellent', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw11', 'self', 'allergic', 'shed')),
    3187: (('fiv', 'B', 'tested FIV positive', 'fivp9', 'self', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure5', 'inside', None, 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw12', 'other', 'excellent', 'shed')),
    3188: (('cats', 'A', 'has lived with another cat', 'dogonly7', 'inside', None, 'cat_lived'),
           ('age', 'A', '2 months', 'ageonly8', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw12', 'other', 'retired', 'shed_intact')),
    3189: (('fiv', 'B', 'tested FIV positive', 'fivp9', 'other', 'lost cat', 'player'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure5', 'self', 'kids', 'unsure'),
           ('age', 'B', '2 years', 'ageonly8', 'other', 'student', 'age_only')),
    3190: (('fiv', 'B', 'tested FIV positive', 'fivp9', 'other', 'retired', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure6', 'other', 'deal', 'unsure'),
           ('dogs', 'A', 'has lived with a dog', 'dogonly7', 'inside', None, 'dog_said')),
    3191: (('cats', 'U', 'has never met another cat', 'dogonly8', 'inside', None, 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly9', 'other', 'apartment', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw12', 'self', 'allergic', 'shed')),
    3192: (('vacc', 'A', 'microchipped and healthy', 'fivp10', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw13', 'inside', None, 'shed'),
           ('age', 'B', '2 years', 'ageonly9', 'inside', None, 'age_only')),
    3193: (('cats', 'U', 'has never met another cat', 'dogonly8', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly9', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw13', 'other', 'kids', 'shed')),
    3194: (('fiv', 'B', 'tested FIV positive', 'fivp10', 'self', None, 'player_good'),
           ('shed', 'B', 'sheds very little', 'shedw13', 'self', 'allergic', 'shed'),
           ('dogs', 'A', 'has lived with a dog', 'dogonly8', 'other', 'apartment', 'dog_said')),
    3195: (('fiv', 'B', 'tested FIV positive', 'fivp10', 'self', None, 'player'),
           ('shed', 'B', 'sheds very little', 'shedw14', 'other', 'nurse', 'shed'),
           ('dogs', 'A', 'has lived with a dog', 'dogonly9', 'self', 'dogs', 'dog_said')),
    3196: (('dogs', 'A', 'has lived with a dog', 'dogonly9', 'other', 'lost cat', 'dog_said'),
           ('age', 'A', '2 months', 'ageonly10', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw14', 'self', 'allergic', 'shed_intact')),
    3197: (('fiv', 'B', 'tested FIV positive', 'fivp11', 'inside', None, 'player'),
           ('vacc', 'U', 'bloodwork last month was normal', 'unsure6', 'self', 'old cat', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw14', 'inside', None, 'shed')),
    3198: (('fiv', 'B', 'tested FIV positive', 'fivp11', 'inside', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure6', 'inside', None, 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw15', 'self', 'allergic', 'shed')),
    3199: (('cats', 'U', 'has never met another cat', 'dogonly9', 'other', 'retired', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly10', 'inside', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw15', 'other', 'nurse', 'shed')),
    3200: (('vacc', 'A', 'microchipped and healthy', 'fivp11', 'other', 'retired', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw15', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly10', 'other', 'student', 'age_only')),
    3201: (('fiv', 'B', 'tested FIV positive', 'fivp12', 'other', 'kids', 'player'),
           ('shed', 'B', 'sheds very little', 'shedw16', 'inside', None, 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly10', 'self', 'old cat', 'dog_only')),
    3202: (('cats', 'U', 'has never met another cat', 'dogonly10', 'inside', None, 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly11', 'other', 'deal', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw16', 'self', 'allergic', 'shed')),
    3203: (('fiv', 'B', 'tested FIV positive', 'fivp12', 'other', 'apartment', 'player_good'),
           ('shed', 'B', 'sheds very little', 'shedw16', 'other', 'excellent', 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly10', 'self', 'old cat', 'dog_only')),
    3204: (('vacc', 'A', 'microchipped and healthy', 'fivp12', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw17', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly11', 'self', None, 'age_only')),
    3205: (('cats', 'U', 'has never met another cat', 'dogonly11', 'other', 'apartment', 'dog_only'),
           ('age', 'A', '2 months', 'ageonly11', 'inside', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw17', 'other', 'retired', 'shed_intact')),
    3206: (('fiv', 'B', 'tested FIV positive', 'fivp13', 'self', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure7', 'self', 'excellent', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw17', 'inside', None, 'shed')),
    3207: (('fiv', 'B', 'tested FIV positive', 'fivp13', 'self', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure7', 'other', 'deal', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw18', 'other', 'kids', 'shed')),
    3208: (('dogs', 'A', 'has lived with a dog', 'dogonly11', 'inside', None, 'dog_said'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly12', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw18', 'self', 'allergic', 'shed')),
    3209: (('vacc', 'A', 'microchipped and healthy', 'fivp13', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw18', 'other', 'nurse', 'shed'),
           ('age', 'B', '2 years', 'ageonly12', 'self', None, 'age_only')),
    3210: (('cats', 'U', 'has never met another cat', 'dogonly11', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly12', 'other', 'student', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw19', 'inside', None, 'shed')),
    3211: (('fiv', 'B', 'tested FIV positive', 'fivp14', 'other', 'nurse', 'player'),
           ('vacc', 'U', 'has some stiff joints', 'unsure7', 'self', 'kids', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw19', 'self', 'allergic', 'shed')),
    3212: (('fiv', 'B', 'tested FIV positive', 'fivp14', 'other', 'lost cat', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure8', 'other', 'nurse', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw19', 'other', 'apartment', 'shed')),
    3213: (('cats', 'A', 'has lived with another cat', 'dogonly12', 'other', 'lost cat', 'cat_lived'),
           ('age', 'A', '2 months', 'ageonly13', 'other', 'kids', 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw20', 'inside', None, 'shed_intact')),
    3214: (('fiv', 'B', 'tested FIV positive', 'fivp14', 'inside', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure8', 'self', 'old cat', 'unsure'),
           ('age', 'B', '2 years', 'ageonly13', 'self', None, 'age_only')),
    3215: (('fiv', 'B', 'tested FIV positive', 'fivp15', 'self', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure8', 'inside', None, 'unsure'),
           ('cats', 'U', 'has never met another cat', 'dogonly12', 'other', 'nurse', 'dog_only')),
    3216: (('cats', 'U', 'has never met another cat', 'dogonly12', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly13', 'inside', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw20', 'self', 'allergic', 'shed')),
    3217: (('fiv', 'B', 'tested FIV positive', 'fivp15', 'inside', None, 'player'),
           ('vacc', 'U', 'bloodwork last month was normal', 'unsure9', 'self', 'excellent', 'unsure'),
           ('cats', 'U', 'has never met another cat', 'dogonly13', 'inside', None, 'dog_only')),
    3218: (('vacc', 'A', 'microchipped and healthy', 'fivp15', 'other', 'student', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw20', 'inside', None, 'shed'),
           ('age', 'B', '2 years', 'ageonly14', 'inside', None, 'age_only')),
    3219: (('cats', 'U', 'has never met another cat', 'dogonly13', 'self', 'old cat', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly14', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw21', 'other', 'excellent', 'shed')),
    3220: (('fiv', 'B', 'tested FIV positive', 'fivp16', 'self', None, 'player'),
           ('vacc', 'U', 'recovering from a cold', 'unsure9', 'other', 'nurse', 'unsure'),
           ('age', 'B', '2 years', 'ageonly14', 'other', 'student', 'age_only')),
    3221: (('vacc', 'A', 'microchipped and healthy', 'fivp16', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw21', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly15', 'self', None, 'age_only')),
    3222: (('dogs', 'A', 'has lived with a dog', 'dogonly13', 'self', 'dogs', 'dog_said'),
           ('age', 'A', '2 months', 'ageonly15', 'other', 'kids', 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw21', 'self', 'allergic', 'shed_intact')),
    3223: (('fiv', 'B', 'tested FIV positive', 'fivp16', 'other', 'retired', 'player'),
           ('vacc', 'U', 'has some stiff joints', 'unsure9', 'self', 'kids', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw22', 'other', 'excellent', 'shed')),
    3224: (('fiv', 'B', 'tested FIV positive', 'fivp17', 'other', 'kids', 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure10', 'inside', None, 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw22', 'inside', None, 'shed')),
    3225: (('cats', 'U', 'has never met another cat', 'dogonly14', 'other', 'kids', 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly15', 'other', 'lost cat', 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw22', 'self', 'allergic', 'shed')),
    3226: (('fiv', 'B', 'tested FIV positive', 'fivp17', 'self', None, 'player'),
           ('shed', 'B', 'sheds very little', 'shedw23', 'other', 'kids', 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly14', 'self', 'old cat', 'dog_only')),
    3227: (('cats', 'U', 'has never met another cat', 'dogonly14', 'inside', None, 'dog_only'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly16', 'self', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw23', 'self', 'allergic', 'shed')),
    3228: (('fiv', 'B', 'tested FIV positive', 'fivp17', 'self', None, 'player_good'),
           ('shed', 'B', 'sheds very little', 'shedw23', 'inside', None, 'shed'),
           ('cats', 'U', 'has never met another cat', 'dogonly15', 'self', 'old cat', 'dog_only')),
    3229: (('vacc', 'A', 'microchipped and healthy', 'fivp18', 'other', 'lost cat', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw24', 'self', 'allergic', 'shed'),
           ('age', 'B', '2 years', 'ageonly16', 'inside', None, 'age_only')),
    3230: (('cats', 'U', 'has never met another cat', 'dogonly15', 'other', 'retired', 'dog_only'),
           ('age', 'A', '2 months', 'ageonly16', 'self', None, 'age_only'),
           ('shed', 'B', 'sheds very little', 'shedw24', 'other', 'apartment', 'shed_intact')),
    3231: (('fiv', 'B', 'tested FIV positive', 'fivp18', 'inside', None, 'player'),
           ('vacc', 'U', 'has some stiff joints', 'unsure10', 'self', 'old cat', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw24', 'other', 'apartment', 'shed')),
    3232: (('fiv', 'B', 'tested FIV positive', 'fivp18', 'inside', None, 'player_good'),
           ('vacc', 'U', 'microchipped and healthy', 'unsure10', 'other', 'retired', 'unsure'),
           ('shed', 'B', 'sheds very little', 'shedw25', 'self', 'allergic', 'shed')),
    3233: (('dogs', 'A', 'has lived with a dog', 'dogonly15', 'other', 'nurse', 'dog_said'),
           ('spay', 'B', 'not yet spayed/neutered', 'ageonly17', 'inside', None, 'intact_months'),
           ('shed', 'B', 'sheds very little', 'shedw25', 'inside', None, 'shed')),
    3234: (('vacc', 'A', 'microchipped and healthy', 'fivp19', 'self', 'excellent', 'healthy'),
           ('shed', 'B', 'sheds very little', 'shedw25', 'other', 'kids', 'shed'),
           ('age', 'B', '2 years', 'ageonly17', 'other', 'lost cat', 'age_only')),
})
PAIRED_CONTROLS.update({   # prompts_3150_plan.md 3.3
    3150: ('no_doubt', ('calm and lazy',)),
    3151: ('unrelated', ('playful and noisy at night',)),
    3152: ('unrelated', ('indoor-outdoor',)),
    3153: ('not_only', ('loves being picked up and carried', 'eats wet food twice a day, any brand')),
    3154: ('double_neg', ('independent',)),
    3155: ('no_doubt', ('used to roaming a farm',)),
    3156: ('unrelated', ('found as a stray and never claimed',)),
    3157: ('double_neg', ('lets you hold them for a minute, then wriggles off',)),
    3158: ('not_only', ('very cuddly', 'indoor-only')),
    3159: ('unrelated', ('eats wet food twice a day, any brand',)),
    3160: ('neg_question', ('spayed/neutered',)),
    3161: ('double_neg', ('loves being picked up and carried',)),
    3162: ('no_doubt', ('indoor-outdoor',)),
    3163: ('unrelated', ('meows loudly for breakfast',)),
    3164: ('not_only', ('spayed/neutered', 'indoor-only')),
    3165: ('double_neg', ('eats wet food twice a day, any brand',)),
    3166: ('unrelated', ('indoor-only',)),
    3167: ('not_only', ('treated for fleas and ear mites last month', 'tested negative for FIV and FeLV')),
    3168: ('neg_question', ('indoor-only',)),
    3169: ('no_doubt', ('loves being picked up and carried',)),
    3170: ('unrelated', ('the previous owner went into a care home',)),
    3171: ('double_neg', ('picky about litter',)),
    3172: ('unrelated', ('dewormed but not vaccinated',)),
    3173: ('not_only', ('microchipped', 'uses a scratching post, never the sofa')),
    3174: ('double_neg', ('afraid of the vacuum',)),
    3175: ('no_doubt', ('tested negative for FIV and FeLV',)),
    3176: ('unrelated', ('found as a stray kitten',)),
    3177: ('double_neg', ('picky about litter',)),
    3178: ('not_only', ('spayed/neutered', 'loves being picked up and carried')),
    3179: ('neg_question', ('indoor-only',)),
    3180: ('unrelated', ('not yet spayed/neutered',)),
    3181: ('no_doubt', ('spayed/neutered',)),
    3182: ('not_only', ('fully litter trained', 'loves being picked up and carried')),
    3183: ('double_neg', ('eats one brand of dry food and nothing else',)),
    3184: ('unrelated', ('indoor-only',)),
    3185: ('unrelated', ('eats one brand of dry food and nothing else',)),
    3186: ('double_neg', ('playful and noisy at night',)),
    3187: ('no_doubt', ('not yet spayed/neutered',)),
    3188: ('unrelated', ('no vaccines yet',)),
    3189: ('not_only', ('spayed/neutered', 'indoor-only')),
    3190: ('double_neg', ('very cuddly',)),
    3191: ('neg_question', ('tested negative for FIV and FeLV',)),
    3192: ('not_only', ('tested negative for FIV and FeLV', 'loves being picked up and carried')),
    3193: ('no_doubt', ('no vaccines yet',)),
    3194: ('unrelated', ('microchipped',)),
    3195: ('double_neg', ('calm and lazy',)),
    3196: ('unrelated', ('had two of three kitten vaccines',)),
    3197: ('not_only', ('calm and lazy', 'is fine with children')),
    3198: ('neg_question', ('is fine with children',)),
    3199: ('double_neg', ('on a monthly flea treatment',)),
    3200: ('no_doubt', ('has lived with another cat',)),
    3201: ('unrelated', ('not yet spayed/neutered',)),
    3202: ('unrelated', ('loves being picked up and carried',)),
    3203: ('not_only', ('fully litter trained', 'spayed/neutered')),
    3204: ('double_neg', ('a fussy eater, only eats chicken wet food',)),
    3205: ('no_doubt', ('picky about litter',)),
    3206: ('unrelated', ('spayed/neutered',)),
    3207: ('not_only', ('has lived with a dog', 'treated for fleas and ear mites last month')),
    3208: ('double_neg', ('playful and noisy at night',)),
    3209: ('neg_question', ('spayed/neutered',)),
    3210: ('unrelated', ('fully litter trained',)),
    3211: ('double_neg', ('picky about litter',)),
    3212: ('not_only', ('very cuddly', 'spayed/neutered')),
    3213: ('no_doubt', ('picky about litter',)),
    3214: ('unrelated', ('indoor-only',)),
    3215: ('double_neg', ('afraid of the vacuum',)),
    3216: ('unrelated', ('had two of three kitten vaccines',)),
    3217: ('neg_question', ('spayed/neutered',)),
    3218: ('not_only', ('eats wet food twice a day, any brand', 'has lived with another cat')),
    3219: ('no_doubt', ('microchipped',)),
    3220: ('unrelated', ('spayed/neutered',)),
    3221: ('double_neg', ('playful and noisy at night',)),
    3222: ('unrelated', ('had two of three kitten vaccines',)),
    3223: ('not_only', ('has lived with another cat', 'uses a scratching post, never the sofa')),
    3224: ('double_neg', ('afraid of the vacuum',)),
    3225: ('no_doubt', ('had two of three kitten vaccines',)),
    3226: ('unrelated', ('not microchipped',)),
    3227: ('not_only', ('tested negative for FIV and FeLV', 'indoor-only')),
    3228: ('neg_question', ('fully litter trained',)),
    3229: ('double_neg', ('picky about litter',)),
    3230: ('unrelated', ('loves being picked up and carried',)),
    3231: ('no_doubt', ('is fine with children',)),
    3232: ('not_only', ('uses a scratching post, never the sofa', 'indoor-only')),
    3233: ('double_neg', ('uses a scratching post, never the sofa',)),
    3234: ('unrelated', ('a fussy eater, only eats chicken wet food',)),
})
PAIRED_BUTS.update({   # prompts_3150_plan.md 3.3: one per sale (kind, connector, heard, asker, raised, lowered, facts)
    3150: (('good but bad', 'but', 'other', 'nurse', 'kids', 'long hours', ('is fine with children', 'only drinks from a running tap')),),
    3151: (('bad but good', 'though', 'other', 'retired', 'dogs', 'long hours', ('only drinks from a running tap', 'on a monthly flea treatment')),),
    3152: (('fragment', 'tho', 'self', 'old cat', 'busy road', 'old cat', ('never tested for FIV or FeLV', 'microchipped')),),
    3153: (('good but bad', 'but', 'other', 'apartment', 'long hours', 'old cat', ('independent', 'never tested for FIV or FeLV')),),
    3154: (('bad but good', 'but', 'other', 'lost cat', 'looks', 'kids', ('has never met children', 'uses a scratching post, never the sofa')),),
    3155: (('fragment', 'though', 'self', 'looks', 'apartment', 'looks', ('picky about litter', 'calm and lazy')),),
    3156: (('good but bad', 'tho', 'other', 'deal', 'long hours', 'hiker', ('independent', 'indoor with a catio')),),
    3157: (('bad but good', 'but', 'other', 'hiker', 'busy road', 'dogs', ('has never had a flea treatment', 'indoor with a catio')),),
    3158: (('fragment', 'but', 'self', 'looks', 'dogs', 'looks', ('picky about litter', 'has lived with a dog')),),
    3159: (('good but bad', 'though', 'other', 'nurse', 'dogs', 'long hours', ('on a monthly flea treatment', 'follows people around like a dog')),),
    3160: (('bad but good', 'tho', 'other', 'retired', 'busy road', 'student', ('a fussy eater, only eats chicken wet food', 'indoor-only')),),
    3161: (('fragment', 'but', 'other', 'student', 'looks', 'long hours', ('only drinks from a running tap', 'fully litter trained')),),
    3162: (('good but bad', 'but', 'self', 'kids', 'old cat', 'kids', ('has lived with another cat', 'a bit feisty, scratches when annoyed')),),
    3163: (('bad but good', 'though', 'other', 'lost cat', 'hiker', 'kids', ('has never met children', 'used to roaming a farm')),),
    3164: (('fragment', 'tho', 'other', 'pedigree', 'kids', 'long hours', ('only drinks from a running tap', 'is fine with children')),),
    3165: (('good but bad', 'but', 'self', 'long hours', 'dogs', 'long hours', ('has lived with a dog', 'follows people around like a dog')),),
    3166: (('bad but good', 'but', 'other', 'hiker', 'old cat', 'apartment', ('playful and noisy at night', 'tested negative for FIV and FeLV')),),
    3167: (('fragment', 'though', 'other', 'nurse', 'apartment', 'looks', ('litter trained but kicks litter everywhere', 'calm and lazy')),),
    3168: (('good but bad', 'tho', 'self', 'old cat', 'long hours', 'old cat', ('independent', 'has never met another cat')),),
    3169: (('bad but good', 'but', 'other', 'retired', 'long hours', 'hiker', ('indoor-only', 'independent')),),
    3170: (('fragment', 'but', 'other', 'student', 'old cat', 'long hours', ('only drinks from a running tap', 'has lived with another cat')),),
    3171: (('good but bad', 'though', 'other', 'apartment', 'kids', 'long hours', ('is fine with children', 'follows people around like a dog')),),
    3172: (('bad but good', 'tho', 'self', 'kids', 'dogs', 'kids', ('a bit feisty, scratches when annoyed', 'treated for fleas and ear mites last month')),),
    3173: (('fragment', 'but', 'other', 'pedigree', 'dogs', 'hiker', ('indoor-only', 'on a monthly flea treatment')),),
    3174: (('good but bad', 'but', 'other', 'deal', 'long hours', 'busy road', ('independent', 'indoor-outdoor')),),
    3175: (('bad but good', 'though', 'other', 'hiker', 'dogs', 'student', ('eats one brand of dry food and nothing else', 'has lived with a dog')),),
    3176: (('fragment', 'tho', 'other', 'looks', 'dogs', 'hiker', ('indoor with a catio', 'on a monthly flea treatment')),),
    3177: (('good but bad', 'but', 'other', 'nurse', 'apartment', 'old cat', ('calm and lazy', 'has never met another cat')),),
    3178: (('bad but good', 'but', 'self', 'kids', 'student', 'kids', ('has never met children', 'eats wet food twice a day, any brand')),),
    3179: (('fragment', 'though', 'other', 'student', 'dogs', 'long hours', ('very cuddly', 'on a monthly flea treatment')),),
    3180: (('good but bad', 'tho', 'other', 'apartment', 'looks', 'hiker', ('uses a scratching post, never the sofa', 'indoor-only')),),
    3181: (('bad but good', 'but', 'other', 'lost cat', 'hiker', 'long hours', ('only drinks from a running tap', 'indoor-outdoor')),),
    3182: (('fragment', 'but', 'self', 'apartment', 'old cat', 'apartment', ('playful and noisy at night', 'tested negative for FIV and FeLV')),),
    3183: (('good but bad', 'though', 'other', 'hiker', 'deal', 'looks', ('spayed/neutered', 'litter trained but kicks litter everywhere')),),
    3184: (('bad but good', 'tho', 'other', 'hiker', 'looks', 'old cat', ('has never met another cat', 'uses a scratching post, never the sofa')),),
    3185: (('fragment', 'but', 'self', 'long hours', 'looks', 'long hours', ('very cuddly', 'uses a scratching post, never the sofa')),),
    3186: (('good but bad', 'but', 'other', 'nurse', 'looks', 'kids', ('uses a scratching post, never the sofa', 'has never met children')),),
    3187: (('bad but good', 'though', 'other', 'retired', 'kids', 'long hours', ('follows people around like a dog', 'is fine with children')),),
    3188: (('fragment', 'tho', 'self', 'hiker', 'retired', 'hiker', ('indoor with a catio', 'very cuddly')),),
    3189: (('good but bad', 'but', 'other', 'apartment', 'looks', 'student', ('uses a scratching post, never the sofa', 'eats one brand of dry food and nothing else')),),
    3190: (('bad but good', 'but', 'other', 'lost cat', 'deal', 'kids', ('hates being picked up but sits close by', 'spayed/neutered')),),
    3191: (('fragment', 'though', 'other', 'pedigree', 'dogs', 'hiker', ('indoor with a catio', 'treated for fleas and ear mites last month')),),
    3192: (('good but bad', 'tho', 'self', 'long hours', 'apartment', 'long hours', ('calm and lazy', 'only drinks from a running tap')),),
    3193: (('bad but good', 'but', 'other', 'hiker', 'busy road', 'long hours', ('very cuddly', 'microchipped')),),
    3194: (('fragment', 'but', 'other', 'nurse', 'looks', 'apartment', ('playful and noisy at night', 'uses a scratching post, never the sofa')),),
    3195: (('good but bad', 'though', 'self', 'hiker', 'looks', 'hiker', ('uses a scratching post, never the sofa', 'indoor with a catio')),),
    3196: (('bad but good', 'tho', 'other', 'retired', 'hiker', 'long hours', ('very cuddly', 'indoor-outdoor')),),
    3197: (('fragment', 'but', 'other', 'student', 'old cat', 'looks', ('litter trained but kicks litter everywhere', 'has lived with another cat')),),
    3198: (('good but bad', 'but', 'self', 'long hours', 'kids', 'long hours', ('loves being picked up and carried', 'very cuddly')),),
    3199: (('bad but good', 'though', 'other', 'lost cat', 'looks', 'apartment', ('playful and noisy at night', 'uses a scratching post, never the sofa')),),
    3200: (('fragment', 'tho', 'other', 'pedigree', 'looks', 'busy road', ('used to roaming a farm', 'uses a scratching post, never the sofa')),),
    3201: (('good but bad', 'but', 'other', 'deal', 'apartment', 'long hours', ('calm and lazy', 'only drinks from a running tap')),),
    3202: (('bad but good', 'but', 'self', 'looks', 'hiker', 'looks', ('still learning to use the litter box', 'indoor-outdoor')),),
    3203: (('fragment', 'though', 'other', 'looks', 'hiker', 'long hours', ('very cuddly', 'indoor-outdoor')),),
    3204: (('good but bad', 'tho', 'other', 'nurse', 'hiker', 'long hours', ('used to roaming a farm', 'very cuddly')),),
    3205: (('bad but good', 'but', 'self', 'dogs', 'retired', 'dogs', ('has never had a flea treatment', 'very cuddly')),),
    3206: (('fragment', 'but', 'other', 'student', 'looks', 'old cat', ('has never met another cat', 'uses a scratching post, never the sofa')),),
    3207: (('good but bad', 'though', 'other', 'apartment', 'hiker', 'old cat', ('used to roaming a farm', 'has never met another cat')),),
    3208: (('bad but good', 'tho', 'self', 'looks', 'excellent', 'looks', ('still learning to use the litter box', 'finished the kitten vaccine series')),),
    3209: (('fragment', 'but', 'other', 'pedigree', 'hiker', 'long hours', ('only drinks from a running tap', 'used to roaming a farm')),),
    3210: (('good but bad', 'but', 'other', 'deal', 'dogs', 'busy road', ('treated for fleas and ear mites last month', 'indoor-outdoor')),),
    3211: (('bad but good', 'though', 'other', 'hiker', 'busy road', 'long hours', ('follows people around like a dog', 'microchipped')),),
    3212: (('fragment', 'tho', 'self', 'old cat', 'hiker', 'old cat', ('has never met another cat', 'indoor-outdoor')),),
    3213: (('good but bad', 'but', 'other', 'nurse', 'long hours', 'hiker', ('independent', 'indoor-only')),),
    3214: (('bad but good', 'but', 'other', 'retired', 'deal', 'kids', ('hates being picked up but sits close by', 'spayed/neutered')),),
    3215: (('fragment', 'though', 'self', 'long hours', 'deal', 'long hours', ('very cuddly', 'spayed/neutered')),),
    3216: (('good but bad', 'tho', 'other', 'apartment', 'long hours', 'hiker', ('independent', 'indoor with a catio')),),
    3217: (('bad but good', 'but', 'other', 'lost cat', 'busy road', 'allergic', ('sheds a lot', 'indoor with a catio')),),
    3218: (('fragment', 'but', 'self', 'kids', 'dogs', 'kids', ('a bit feisty, scratches when annoyed', 'on a monthly flea treatment')),),
    3219: (('good but bad', 'though', 'other', 'deal', 'long hours', 'dogs', ('independent', 'has never had a flea treatment')),),
    3220: (('bad but good', 'tho', 'other', 'hiker', 'old cat', 'kids', ('hates being picked up but sits close by', 'has lived with another cat')),),
    3221: (('fragment', 'but', 'other', 'looks', 'dogs', 'old cat', ('has never met another cat', 'treated for fleas and ear mites last month')),),
    3222: (('good but bad', 'but', 'self', 'busy road', 'long hours', 'busy road', ('independent', 'indoor-outdoor')),),
    3223: (('bad but good', 'though', 'other', 'student', 'kids', 'dogs', ('has never met a dog', 'loves being picked up and carried')),),
    3224: (('fragment', 'tho', 'other', 'student', 'kids', 'looks', ('litter trained but kicks litter everywhere', 'loves being picked up and carried')),),
    3225: (('good but bad', 'but', 'self', 'student', 'old cat', 'student', ('tested negative for FIV and FeLV', 'not microchipped')),),
    3226: (('bad but good', 'but', 'other', 'lost cat', 'excellent', 'busy road', ('used to roaming a farm', 'fully vaccinated, vet check last month')),),
    3227: (('fragment', 'though', 'other', 'pedigree', 'dogs', 'long hours', ('only drinks from a running tap', 'on a monthly flea treatment')),),
    3228: (('good but bad', 'tho', 'self', 'long hours', 'dogs', 'long hours', ('treated for fleas and ear mites last month', 'only drinks from a running tap')),),
    3229: (('bad but good', 'but', 'other', 'hiker', 'long hours', 'busy road', ('indoor-outdoor', 'independent')),),
    3230: (('fragment', 'but', 'other', 'looks', 'busy road', 'apartment', ('meows loudly for breakfast', 'indoor-only')),),
    3231: (('good but bad', 'though', 'other', 'nurse', 'old cat', 'apartment', ('has lived with another cat', 'meows loudly for breakfast')),),
    3232: (('bad but good', 'tho', 'self', 'old cat', 'kids', 'old cat', ('has never met another cat', 'is fine with children')),),
    3233: (('fragment', 'but', 'other', 'student', 'long hours', 'busy road', ('used to roaming a farm', 'independent')),),
    3234: (('good but bad', 'but', 'other', 'apartment', 'hiker', 'long hours', ('used to roaming a farm', 'very cuddly')),),
})
PAIRED_PRESENT.update({   # breeders in half of the intact_months sales
    3152: ('breeder',),
    3161: ('breeder',),
    3169: ('breeder',),
    3176: ('breeder',),
    3185: ('breeder',),
    3193: ('breeder',),
    3202: ('breeder',),
    3210: ('breeder',),
    3219: ('breeder',),
    3227: ('breeder',),
})



def paired_buyers(seed: int) -> list[str]:
    """The personas that must bid in a paired sale: the buyers its targets turn, and from HEARD_FROM on also the
    buyers who ask its targets, the buyers its "but" answers move or who ask them, and its PAIRED_PRESENT; from
    SAYINGS_FROM on also the buyers a target's shape needs (SHAPE_BUYERS)."""
    need = [p for t in PAIRED_SEEDS[seed] for p in PAIRED_BUYERS.get(t[0], ()) + PAIRED_GROUP_BUYERS.get(t[3], ())]
    need += [BUYER[k] for t in PAIRED_SEEDS[seed] if len(t) > 6 for k in SHAPE_BUYERS[t[6]]]
    need += [BUYER[t[5]] for t in PAIRED_SEEDS[seed] if len(t) > 5 and t[5]]
    need += [BUYER[k] for b in PAIRED_BUTS.get(seed, ()) for k in b[3:6] if k]
    need += [BUYER[k] for k in PAIRED_PRESENT.get(seed, ())]
    return list(dict.fromkeys(need))


def paired_months(target: PairedTarget) -> list[int]:
    """The ages (AGES_MONTHS) a target fact fits: a kitten's or an adult's health line, an adult's company, ..."""
    topic, side, value, _ = target[:4]
    shape = target[6] if len(target) > 6 else None
    if shape == "intact_months":            # "shes intact, only 2 months": a kitten too young for the operation
        return [2, 4]
    if shape in ("player_good", "healthy"):     # "has FIV but shes healthy" is said of a grown cat
        return [m for m in AGES_MONTHS if 12 <= m <= 120]
    if PAIRED_GROUP_BUYERS.get(target[3]) == (BREEDER,):
        return [m for m in paired_months((topic, side, value, "")) if m <= BREEDING_MONTHS]
    if topic == "age":
        return [2] if side == "A" else [24]
    if topic == "senior" or value in HEALTH_SENIOR:
        return [m for m in AGES_MONTHS if m > 120]
    if value in HEALTH_KITTEN:
        return [m for m in AGES_MONTHS if (4 if value == FULL_SERIES else 2) <= m < 12]
    if value in HEALTH_ADULT:
        return [m for m in AGES_MONTHS if 12 <= m <= 120]
    if value in LIFE_FACTS["litter"][0] and value not in LIFE_FACTS["litter"][1]:
        return [m for m in AGES_MONTHS if m < 12]
    if value in LIFE_FACTS["origin"][1] or (value in LIFE_FACTS["litter"][1] and value not in LIFE_FACTS["litter"][0]):
        return [m for m in AGES_MONTHS if m >= 12]
    if value in AGAINST.values():           # not getting on with others is said of a grown cat
        return [m for m in AGES_MONTHS if m >= 12]
    if topic == "spay" and side == "A":
        return [m for m in AGES_MONTHS if m >= 6]
    return list(AGES_MONTHS)


def _kind_of(entry: str) -> str:
    """The kind of company ("dogs", "cats", "children") a company entry is about."""
    return next(k for k in COMPANY if entry in COMPANY[k] or entry == AGAINST[k])


def apply_paired(seed: int, s: Sale) -> None:
    """Put a paired seed's targets on the sheet seed_sale drew and make the rest agree with them: an age, breed,
    health line, quirks, company and two LIFE_FACTS that fit (from their own random draw), and the buyers the targets
    turn (they replace other personas; every name stays)."""
    rng, c = random.Random(seed * 3011 + 13), s["cat"]
    targets = PAIRED_SEEDS[seed]
    by = {t[0]: t[2] for t in targets}
    months = [m for m in AGES_MONTHS if all(m in paired_months(t) for t in targets)]
    if c["months"] not in months:
        c["months"] = rng.choice(months)
    c["age"] = _age(c["months"])
    breeds = [b for b in BREEDS if not ("shed" in by and (b == "a Sphynx" or (by["shed"] == SHEDS_LITTLE and b in LONG_HAIR)))
              and ("papers" not in by or b in PEDIGREE)]
    if c["breed"] not in breeds:
        c["breed"] = rng.choice(breeds)
    shapes = {t[6] for t in targets if len(t) > 6}
    if "spay" in by:
        c["fixed"] = by["spay"]
    elif "shed_intact" in shapes:           # "sheds very little, but not spayed yet" in one answer
        c["fixed"] = "not yet spayed/neutered"
    elif c["months"] < 6:
        c["fixed"] = "not yet spayed/neutered"
    young = c["fixed"] != "spayed/neutered" or seed >= HEARD_FROM      # from HEARD_FROM on, a breeder bids on no
    if BREEDER in paired_buyers(seed) and young and c["months"] > BREEDING_MONTHS:   # cat over five at all
        c["months"] = rng.choice([m for m in months if m <= BREEDING_MONTHS     # a breeder the papers need
                                  and (c["fixed"] != "spayed/neutered" or m >= 6)])   # (spayed: six months on)
        c["age"] = _age(c["months"])
    life = {t[0]: t[2] for t in targets if t[0] in LIFE_TOPICS}
    if "vacc" in by or "senior" in by:
        c["health"] = by.get("vacc") or by["senior"]
    else:
        stage = HEALTH_KITTEN if c["months"] < 12 else HEALTH_SENIOR if c["months"] > 120 else HEALTH_ADULT
        fits = [h for h in stage if not (h == NO_VET and (c["months"] <= 24 or life.get("pests") == TREATED))
                and not (h == FULL_SERIES and c["months"] < 4) and not ("microchip" in life and "microchipped" in h)
                and not ("food" in life and "diet" in h) and not (life.get("pests") == "has never had a flea treatment"
                                                                 and "dewormed" in h)]
        if c["health"] not in fits:
            c["health"] = rng.choice(fits)
    if "player_good" in shapes and "vacc" not in by:    # the good half of "has FIV but shes healthy" is on the sheet
        good = [h for h in GOOD_HEALTH if not ("microchip" in life and "microchipped" in h)]
        if c["health"] not in good:
            c["health"] = rng.choice(good)
    if "fiv" in by:
        c["virus"] = by["fiv"]
    if "home" in by:
        c["home"] = by["home"]
    company = [cast(str, by[t]) for t in COMPANY_TOPIC if t in by]     # a company target always has a value
    taken = {_kind_of(e) for e in company}
    company += [e for e in c["company"] if _kind_of(e) not in taken][:2 - len(company)]
    if len(company) < 2:
        kind = rng.choice([k for k in sorted(COMPANY) if k not in {_kind_of(e) for e in company}])
        company.append(rng.choice(COMPANY[kind]))
    if "dog_only" in shapes:                # "Not really, but she has lived with a dog": both are on the sheet
        company = [cast(str, by["cats"]), "has lived with a dog"]
    c["company"] = company
    temper = [t[2] for t in targets if t[2] in TEMPER]
    if temper:
        c["temperament"] = temper[0]
    elif "scratch" in by and c["temperament"] == TEMPER[5]:  # one that only uses its post doesn't scratch when annoyed
        c["temperament"] = rng.choice([t for t in TEMPER if t != TEMPER[5]])
    quirks = [t[2] for t in targets if t[2] in QUIRKS or t[2] == SHEDS_LITTLE]
    banned = set(quirks)
    if "shed" in by or c["breed"] == "a Sphynx":
        banned.add("sheds a lot")
    if "litter" in life:
        banned.add("picky about litter")
    if c["temperament"] == TEMPER[5]:
        banned.add(QUIRKS[1])
    kept = [q for q in c["quirks"] if q not in banned]       # the drawn quirks stay where they can
    rest = [q for q in QUIRKS if q not in banned and q not in kept]
    while len(quirks) < 2:
        quirks.append(kept.pop(0) if kept else rest.pop(rng.randrange(len(rest))))
    c["quirks"] = quirks
    for key in ("extra", "extra_kinds", "life", "life_kinds"):
        c.pop(key, None)
    if by.get("papers") == "no papers":
        c["extra_kinds"], c["extra"] = ["no papers"], [f"a purebred {c['breed'][2:]}, but never had papers"]
    c["life_kinds"], c["life"] = list(life), list(life.values())
    while len(c["life"]) < PAIRED_LIFE:
        kinds = [k for k in PAIRED_LIFE_WEIGHTS if k not in c["life_kinds"] and _life_options(k, c)]
        kind = rng.choices(kinds, weights=[PAIRED_LIFE_WEIGHTS[k] for k in kinds])[0]
        c["life_kinds"].append(kind)
        c["life"].append(rng.choice(_life_options(kind, c)))
    if LITTER in c["life"] and "has never met another cat" in c["company"]:
        c["life"][c["life"].index(LITTER)] = "found as a stray kitten"
    if TREATED in c["life"] and c["health"] == NO_VET:
        c["life"][c["life"].index(TREATED)] = "has never had a flea treatment"
    need = paired_buyers(seed)
    free = [b for b in reversed(s["buyers"]) if b["persona"] not in need]
    for persona in need:
        if persona not in {b["persona"] for b in s["buyers"]}:
            free.pop(0)["persona"] = persona
    if young and c["months"] > BREEDING_MONTHS:
        for b in s["buyers"]:
            if b["persona"] == BREEDER:
                b["persona"] = rng.choice([p for p in PERSONAS if p not in {x["persona"] for x in s["buyers"]}])


def facts(s: Sale) -> str:
    """The cat's true facts in one line, as the seller is told them."""
    c = s["cat"]
    extra = [f for f in c.get("extra", []) if not f.endswith(" old")]         # a new age is already c["age"]
    return (f"{c['age']} old, {c['breed']}, {c['home']}, {c['fixed']}, {c['health']}, {c['virus']}, "
            f"{c['company'][0]}, {c['company'][1]}, temperament: {c['temperament']}; also: {c['quirks'][0]}; "
            f"{c['quirks'][1]}" + "".join(f"; {f}" for f in extra + c.get("life", [])))


BREED_TOPICS = {"a Ragdoll": "breed_hcm", "a Maine Coon": "breed_hcm", "a Persian": "breed_pkd",
                "a Scottish Fold": "breed_fold"}
PERSONA_TOPICS = {"dogs": "dogs", "kids": "children", "allergic": "allergies", "busy road": "outdoors"}


def _stage(months: int) -> str:
    """The life-stage topic for an age in months (cat_knowledge.TOPICS)."""
    return "kitten" if months < 12 else "young_adult" if months < 84 else "mature_adult" if months <= 120 else "senior"


def topics_for(s: Sale) -> list[str]:
    """The knowledge topics this sale touches (by the cat's age and breed, its home, and who the buyers are);
    topics held out of dialogues (cat_knowledge.HELD_OUT) never are."""
    c = s["cat"]
    personas = " ".join(b["persona"] for b in s["buyers"])
    found = [_stage(c["months"]), "other_cats", "care", BREED_TOPICS.get(c["breed"]),
             "outdoors" if c["home"] != "indoor-only" else None]
    found += [topic for word, topic in PERSONA_TOPICS.items() if word in personas]
    held = set(HELD_OUT)
    return [t for t in dict.fromkeys(found) if t and t not in held]


def knowledge(s: Sale) -> str:
    """The general cat facts for the topics this sale touches, one per line."""
    return "\n".join(f"- {f}" for t in topics_for(s) for f in TOPICS[t]["facts"])


def _json(prompt: str) -> Steps[object]:
    """The teacher's reply parsed as JSON (the outermost list or object in it), retried JSON_TRIES times."""
    for _ in range(JSON_TRIES):
        m = JSON_SPAN.search((yield prompt + "\n\nReply with JSON only, no other text."))
        if m is None:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _json_each(prompts: list[str]) -> Steps[list[object]]:
    """Several prompts sent together (one per buyer), each reply parsed as JSON; a reply without JSON is asked again,
    up to JSON_TRIES times, and is None if it never has any."""
    out: list[object] = [None] * len(prompts)
    todo = list(range(len(prompts)))
    for _ in range(JSON_TRIES):
        if not todo:
            break
        replies = yield [prompts[i] + "\n\nReply with JSON only, no other text." for i in todo]
        again = []
        for i, reply in zip(todo, replies, strict=True):
            m = JSON_SPAN.search(reply)
            try:
                out[i] = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                out[i] = None
            if out[i] is None:
                again.append(i)
        todo = again
    return out


def _offer_move(e: Event) -> str:
    """"$70 -> $80" for a reaction, "$70" for an opening or a question."""
    return f"${e['old']} -> ${e['offer']}" if "old" in e else f"${e['offer']}"


class Market:
    """The buyers' cards and offers, and every step of the sale so far."""

    def __init__(self, s: Sale):
        self.s = s
        self.cards: dict[str, dict[str, Any]] = {}
        self.offers: dict[str, int] = {}
        self.events: list[Event] = []
        self.react_retried = 0           # rewrites asked (buyer × rewrite) for reactions that broke a rule
        self.react_dropped = 0           # reactions still breaking a rule after the rewrites, left out
        self.imagine_retried = 0         # rewrites asked (buyer × rewrite) for scenes that broke a rule
        self.imagine_dropped = 0         # scenes still breaking a rule after the rewrites, left out

    def present(self) -> list[str]:
        """Buyers still bidding: a buyer who withdrew (offer 0) no longer asks or reacts."""
        return [n for n in self.cards if self.offers[n] > 0]

    def card_text(self) -> str:
        """Every bidding buyer's card and current offer, one per line (the prompt that picks the next question)."""
        return "\n".join(f"- {n}, who {self.cards[n]['persona']}; cares about: "
                         f"{', '.join(self.cards[n]['cares_about'])}; can pay at most ${self.cards[n]['budget']}; "
                         f"current offer ${self.offers[n]}" for n in self.present())

    def questions_and_answers(self, upto: int | None = None) -> str:
        """Every question asked so far and the seller's answer, oldest first: what every buyer heard."""
        return "\n".join(f"{e['to']} asked: {e['q']}\nSELLER: {e['says']}" for e in self.events[:upto]
                         if e["kind"] == "answer") or "(no questions yet)"

    def own_lines(self, name: str) -> str:
        """What one buyer has said so far (opening, question, reactions), with their offers: their own side of the sale,
        without the other buyers' lines (a buyer bids from their own card; owner, 2026-10-11)."""
        label = {"open": "Opening", "ask": "Your question", "react": "Your reaction"}
        mine = [e for e in self.events if e.get("name") == name and e["kind"] in label]     # not answers or scenes
        return "\n".join(f"{label[e['kind']]} ({_offer_move(e)}): {e['says']}" for e in mine) or "(nothing yet)"

    def card_line(self, name: str) -> str:
        """One buyer's own card and offer, addressed to them ("You are Maya, who ...")."""
        c = self.cards[name]
        return (f"You are {name}, who {c['persona']}. You care about: {', '.join(c['cares_about'])}. You can pay at "
                f"most ${c['budget']}. Your offer now: ${self.offers[name]}.")

    def transcript(self, upto: int | None = None) -> str:
        """The sale as the buyers heard it, up to (not including) event `upto`."""
        out = []
        for e in self.events[:upto]:
            if e["kind"] in ("open", "ask"):
                out.append(f"{e['name']} (${e['offer']}): {e['says']}")
            elif e["kind"] == "answer":
                out.append(f"SELLER to {e['to']}: {e['says']}")
            elif e["kind"] == "react":
                new = f"${e['offer']}" if e["offer"] else "withdraws"
                out.append(f"{e['name']} (${e['old']} -> {new}): {e['says']}")
            elif e["kind"] == "sold":
                out.append(f"SOLD to {e['name']} for ${e['price']}")
            elif e["kind"] == "imagine":
                out.append(f"{e['name']} pictures it with {', '.join(e['things'])}: {e['says']}")
            else:
                raise ValueError(f"unknown event kind {e['kind']!r}")
        return "\n".join(out)


def _opening(m: Market) -> Steps[None]:
    """The teacher writes every buyer's card, greeting and starting offer in one reply; buyers it leaves out or
    writes without a budget, offer or greeting are not in the sale."""
    rows = yield from _json("The buyers:\n" + "\n".join(f"- {b['name']}, who {b['persona']}" for b in m.s["buyers"]) +
                 "\n\nFor each buyer, write their card and their opening: what they care about in a cat (two or three "
                 "short phrases that follow from who they are), the most they can pay, a greeting, and a starting "
                 "offer no higher than that. The greeting tells one concrete thing about their own life that explains "
                 "what they want (not the formula 'I am X so I want Y'), and the ten greetings differ in length and "
                 'tone. Return a JSON list, one object per buyer in this order: '
                 '{"name": ..., "cares_about": [...], "budget": <dollars>, "says": ..., "offer": <dollars>}.')
    personas = {b["name"]: b["persona"] for b in m.s["buyers"]}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("name") in personas and {"budget", "offer", "says"} <= row.keys():
            name = row["name"]
            m.cards[name] = {"persona": personas[name], "cares_about": list(row.get("cares_about", [])),
                             "budget": int(row["budget"])}
            m.offers[name] = int(row["offer"])
            m.events.append({"kind": "open", "name": name, "offer": m.offers[name], "says": row["says"]})


def _answer(m: Market, name: str, question: str) -> Steps[str]:
    """The seller sees the cat's facts, what is true about cats in general, its own earlier answers and only the
    buyer now asking. What the seller says about this cat is not judged (owner); what it says about cats in general
    must agree with the facts (owner, 2026-10-11: no "FIV can pass to dogs")."""
    earlier = "\n".join(f"Q ({e['to']}): {e['q']}\nA: {e['says']}" for e in m.events if e["kind"] == "answer")
    return (yield f"You are the seller. True facts about your cat: {facts(m.s)}.\nYour way of answering: "
               f"{m.s['seller_style']}.\n\nTrue facts about cats in general:\n{knowledge(m.s)}\n\n"
               f"Your earlier answers:\n{earlier or '(none yet)'}\n\n{name} asks: "
               f"{question}\n\nAnswer {name} in one to three sentences. Stay true to the facts (you may be vague or "
               "play them up if that is your style, but never state something false as a fact, and describe the "
               "cat's temperament and habits only as the facts do). Anything you say about cats in general must "
               "agree with the general facts above, and never hint at a risk those facts rule out (for example, "
               "telling someone with kids or dogs to be careful because the cat is FIV positive). Write only the "
               "answer.")


# Example lines in the owner's style (2026-10-11 review). Two are shown per call, drawn at random, because a fixed
# example was copied: 2,260 of 4,261 reactions in Kaggle v9 began with its "Good to know, but I'm still waiting".
REACTION_EXAMPLES = (
    "It's nice that she gets along with other cats, my old tabby could use a calm friend. I can go up to $95.",
    "Oh, kidney disease means a lot of vet visits on my budget. I'll have to come down to $60.",
    "I'm retired and on my own all day, so a cat who follows me around and sits on my lap would be wonderful "
    "company. I'll go up to $140.",
    "A cat who wants to be followed and petted needs more attention than I can give after twelve-hour shifts. "
    "I'll drop to $30.",
    "Getting along with other cats is nice, but I don't have any other cats and I don't need a clingy one, so I'll "
    "stay at $65.",
    "If she's FIV positive she should stay indoors, and I wanted a cat to take hiking with me, so I'll come down to "
    "$65.",
    "I don't have kids, so how she is with children doesn't change anything for me. I'll stay at $75, but I'd like "
    "to know how much she sheds.",
)
# Lines that give no reason (owner: 嚴禁), and the copied opener. The last four were found in the Modal run of
# 2026-10-11 (data/kaggle_out/dialogues_modal/).
VAGUE = ("sounds perfect for me", "sounds even better", "exactly what i'm looking for", "exactly what i want",
         "the kind of cat i'm looking for", "not what i'm looking for", "good to know, but i'm still waiting",
         "perfect fit", "what i was looking for", "what i care about", "fits my lifestyle")
# "She being a short-haired tabby doesn't matter": not English, and 162 of 1,413 lines in that run began this way.
BROKEN_OPENER = re.compile(r"(?:^|[.!?]\s+)(?:she|he|it|the cat|this cat)\s+being\b", re.I)
STOP = frozenset("about after again also because been before being could does from have just like more much need "
                 "only really said says should some than that their them then there these they this those very "
                 "want were what when which while will with would your she's he's it's i'm i'll i'd that's good "
                 "know great nice sounds pretty".split())
REACT_TRIES = 3


def _words(text: str) -> set[str]:
    """The first four letters of each word of four or more letters, stop words left out ("vaccines" -> "vacc"):
    two lines share a word when these sets meet."""
    return {w[:4] for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 4 and w not in STOP}


def reaction_problem(row: dict[str, Any], card: dict[str, Any], old: int, name: str, answer: str) -> str | None:
    """Why a reaction breaks a rule a program can check (owner, 2026-10-11), or None."""
    sign, says = str(row.get("sign", "")), str(row.get("says", ""))
    try:
        offer = int(row["offer"])
    except (KeyError, TypeError, ValueError):
        return "offer must be a whole number of dollars"
    problem = _offer_problem(sign, offer, old, card) or _amount_problem(says, offer) or _spoken_problem(says, name)
    if problem is None and sign == "0" and not _words(says) & _words(answer):
        problem = "an unchanged offer must name what the seller just said"
    return problem


def _offer_problem(sign: str, offer: int, old: int, card: dict[str, Any]) -> str | None:
    """The sign must agree with the move from `old` to `offer`, and the offer stay within a budget if the card has one."""
    if sign not in SIGNS:
        return 'sign must be "+", "-" or "0"'
    if (sign == "+" and offer <= old) or (sign == "-" and offer >= old) or (sign == "0" and offer != old):
        return f'sign {sign} does not match the offer going from ${old} to ${offer}'
    if "budget" in card and offer > card["budget"]:          # no budget: buyers without a limit (2026-10-12)
        return f"${offer} is above the budget of ${card['budget']}"
    return None


def _amount_problem(says: str, offer: int) -> str | None:
    """A line that says one dollar amount must say the new offer (a withdrawal, offer 0, is not checked)."""
    amounts = {int(x) for x in DOLLARS.findall(says)}
    if offer and len(amounts) == 1 and offer not in amounts:
        return f"the line says ${amounts.pop()} but the offer is ${offer}"
    return None


def _spoken_problem(says: str, name: str) -> str | None:
    """Why a buyer's spoken line breaks a rule on how it is said (dash, own name, broken opener, no reason), or None."""
    if any(dash in says for dash in DASHES):
        return "the line uses a dash"
    if re.search(rf"\b{re.escape(name)}\b", says):
        return "the buyer names themselves"
    if BROKEN_OPENER.search(says):
        return 'a sentence starts "She being ..." (or "The cat being ..."), which is not English; write a full sentence'
    vague = [v for v in VAGUE if v in says.lower().replace("\u2019", "'")]
    if vague:
        return f'"{vague[0]}" gives no reason; say what was just said and what it means for your own life'
    return None


def _reactions(m: Market, asker: str) -> Steps[None]:
    """Every buyer still bidding reacts to the last answer: sign, one or two spoken sentences that give the reason
    for the new offer, and the new offer (0 = withdraws). The reason is part of what the buyer says, in the first
    person (owner, 2026-10-11). Each buyer is asked on their own, seeing only their own card, the questions and
    answers, and their own earlier lines: when all buyers shared one prompt, reasons were taken from other buyers'
    cards (Modal run, 2026-10-11). A reaction that breaks a rule a program can check (`reaction_problem`) is asked
    again, told what was wrong, up to twice; one that still breaks it is left out and the buyer keeps their offer."""
    last = next(e for e in reversed(m.events) if e["kind"] == "answer")
    rng = random.Random(f"{facts(m.s)}|{len(m.events)}")
    examples = " and ".join(f'"{x}"' for x in rng.sample(REACTION_EXAMPLES, 2))
    shared = (f"What a knowledgeable buyer knows about cats (true; use what fits):\n{knowledge(m.s)}\n\n"
              f"Questions and answers before this one:\n{m.questions_and_answers(len(m.events) - 1)}\n\n"
              f'The seller has just answered {asker}\'s question "{last["q"]}":\n"{last["says"]}"\n\n'
              "React to this latest answer only, as the buyer described below. First decide from your own card "
              "whether it touches something you care about and whether that is good or bad for you (use real cat "
              "knowledge, e.g. what an age, a health record or a habit means): sign = \"+\" (higher offer), \"-\" "
              "(lower offer, or 0 to withdraw) or \"0\" (same offer); offer = your new offer, never above your "
              "budget. Then says = what you say out loud to the seller: one or two fluent, grammatical sentences in "
              "the first person, the way a real person talks at a market, reacting to what was just said and giving "
              f"your own reason, with the new amount if it changed. Two examples of the style: {examples} Use your "
              "own words; never copy an example or start the way it starts, and never start a sentence with \"She "
              "being\", \"He being\" or \"The cat being\". Never speak about yourself by name. Use only what the "
              "seller actually said: never react to a fact the seller has not said. Every reason agrees with the "
              "facts about cats above: never fear or hope for something those facts rule out (for example, FIV "
              "cannot pass to people or dogs). The reason names what the seller just said and how it meets or "
              "misses your own situation, in concrete words from your own life as your card describes it and "
              "nothing from anyone else's life: if your card does not mention dogs, children or another cat, you "
              "have none. Saying only that the cat is or is not what you want (\"a perfect fit\", \"exactly what I "
              "was looking for\") is not a reason and is not allowed. Facts from earlier answers are already in your "
              "offer, so they are never the reason again, even when the seller repeats them. When this answer "
              "touches nothing you care about, keep the offer and say what was just said and why it does not matter "
              "for your own situation; you may add what you would still like to know, which must be something no "
              "answer has covered yet. When it does touch your card but you keep the offer, say why. Withdrawing is "
              "final: withdraw only when something the seller actually said makes this cat impossible for you (for "
              "example a breeder hearing she cannot be bred), not merely less attractive or not yet known (that is "
              "a lower or unchanged offer); then sign \"-\", offer 0, and name what was said and why it rules the "
              "cat out for you.\n\n")
    todo = list(m.present())
    notes: dict[str, str] = {}
    for _ in range(REACT_TRIES):
        prompts = [shared + f"{m.card_line(n)}\nWhat you have said so far:\n{m.own_lines(n)}{notes.get(n, '')}\n\n"
                   'Return {"sign": ..., "says": ..., "offer": <dollars>}.' for n in todo]
        rows = yield from _json_each(prompts)
        notes, still = {}, []
        for name, row in zip(todo, rows, strict=True):
            problem = (reaction_problem(row, m.cards[name], m.offers[name], name, last["says"])
                       if isinstance(row, dict) else "no reaction was given")
            if problem is None:             # so the reply is a dict
                reply = cast(dict[str, Any], row)
                m.events.append({"kind": "react", "name": name, "old": m.offers[name], "offer": int(reply["offer"]),
                                 "sign": str(reply["sign"]), "says": str(reply["says"])})
                m.offers[name] = int(reply["offer"])
            else:
                still.append(name)
                notes[name] = f"\n\nYour last reaction broke a rule ({problem}); write it again."
        todo = still
        if not todo:
            return
        m.react_retried += len(todo)
    m.react_retried -= len(todo)
    m.react_dropped += len(todo)


def _round(m: Market, k: int, asked: set[str]) -> Steps[bool]:
    """Round k: one buyer who has not asked asks, the seller answers them, everyone reacts. False if no one asked."""
    left = [n for n in m.present() if n not in asked]
    q = yield from _json(f"Conversation so far:\n{m.transcript()}\n\nBuyers and their cards:\n{m.card_text()}\n\n"
              f"What a knowledgeable buyer knows about cats (true; use what fits):\n{knowledge(m.s)}\n\n"
              f"Choose ONE buyer who has not asked a question yet ({', '.join(left)}) and write the question that "
              "buyer most needs answered, given their card, what they know about cats and what has already been "
              'said. Do not repeat a question already asked. Return {"name": ..., "question": ...}.')
    if not isinstance(q, dict) or q.get("name") not in set(left):
        return False
    name = q["name"]
    asked.add(name)
    m.events.append({"kind": "ask", "round": k, "name": name, "offer": m.offers[name], "says": q["question"]})
    m.events.append({"kind": "answer", "round": k, "to": name, "q": q["question"],
                     "says": (yield from _answer(m, name, q["question"]))})
    yield from _reactions(m, name)
    return True


MAX_THINGS = 3                       # exactly this many things in every scene (owner, 2026-10-11)


# The owner's examples (2026-10-11), two shown per sale at random: a fixed pair was copied (in Kaggle v9, 312 of 392
# scenes used two or more of its things; bed and couch 241 times each).
IMAGINE_EXAMPLES = (
    ("works long hours", "couch, alarm clock and bed", "With my long hours she'll need a good couch to nap on, an "
     "alarm clock so I never get so busy that I forget to feed her, and my bed to curl up on when I finally get home."),
    ("lost an old cat", "picture frame, couch and pillow", "I'd show her my old cat's photo in this frame, and let her "
     "have the couch he loved most and his old pillow."),
    ("cares how things look", "flower, vase and pillow", "I'll buy fresh flowers every day and keep them in a vase "
     "next to her to set off how pretty she is, and give her a lovely pillow of her own."),
    ("hikes every weekend", "backpack, map and mountain", "I want to take her up the mountain with me, riding in my "
     "backpack right next to the map."),
    ("lives in a small apartment", "skyscraper, house plant and bed", "From my little apartment she can watch the "
     "skyscraper across the street, nap by my house plant and share the one bed I have room for."),
)
IMAGINE_TRIES = 4                    # the first answer and up to three rewrites (with 3, 77 of 142 scenes were dropped)
DRAWABLE_THINGS = frozenset(DRAWABLE) - {"cat"}


# Class names that are also everyday words ("keep an eye on", "the key thing", "it matches", "not a fan", "my house"):
# a line using them is not taken to mention that object. Body parts and shapes are left out for the same reason.
EVERYDAY_WORDS = frozenset({"eye", "key", "matches", "fan", "line", "face", "house", "hand", "foot", "leg", "arm", "ear",
                            "nose", "mouth", "tooth", "finger", "toe", "knee", "beard", "moustache", "smiley face",
                            "circle", "square", "triangle", "hexagon", "octagon", "zigzag", "squiggle"})
# Things buyers name that the drawing GPT cannot draw (none of these is one of its classes; tested). Not complete:
# anything new found while spot-checking is added here.
NOT_DRAWABLE = ("window", "blanket", "rug", "carpet", "sofa", "shelf", "bookshelf", "bookcase", "bowl", "toy",
                "litter box", "litter", "scratching post", "cat tree", "catio", "cushion", "curtain", "desk", "lamp",
                "plant", "yard", "backyard", "balcony", "food", "treat", "collar", "leash", "carrier", "box", "mat",
                "boots", "trail", "porch")


def _mentions(names: frozenset[str] | tuple[str, ...], line: str) -> list[str]:
    """Names (singular or with an s) that appear in a line as whole words."""
    low = line.lower()
    return [n for n in names if re.search(rf"\b{re.escape(n)}s?\b", low)]


def _name_at(thing: str, line: str) -> int | None:
    """Where a line first names a thing (by its last word: "alarm clock" by "clock", "picture frame" by "frame")."""
    found = re.search(rf"\b{re.escape(thing.split()[-1].lower())}", line.lower())
    return found.start() if found else None


def _named(thing: str, line: str) -> bool:
    """Whether a line names a thing (by its last word: "alarm clock" by "clock", "picture frame" by "frame")."""
    return _name_at(thing, line) is not None


def mention_order(things: list[str], line: str) -> list[str]:
    """The things in the order the line first names them (prompt format 9 writes its list in this order); a thing
    the line never names keeps its place after the named ones."""
    at = {t: _name_at(t, line) for t in things}
    return sorted(things, key=lambda t: (at[t] is None, at[t] or 0))


def imagine_problem(row: dict[str, Any]) -> str | None:
    """Why an imagined scene breaks the owner's rules (2026-10-11), or None: every thing must be on the drawing GPT's
    list and named in the line, and the line names no other object: no other drawable class (everyday words aside)
    and nothing on NOT_DRAWABLE."""
    things = row.get("things")
    if not isinstance(things, list) or len(things) != MAX_THINGS:
        return f"things must be a list of exactly {MAX_THINGS} items"
    off = [str(t) for t in things if str(t) not in DRAWABLE_THINGS]
    if off:
        return f"{', '.join(off)} {'is' if len(off) == 1 else 'are'} not on the list of things that can be drawn"
    says = str(row.get("says", ""))
    unsaid = [str(t) for t in things if not _named(str(t), says)]
    if unsaid:
        return f"the line never names {', '.join(unsaid)} or says what it is for"
    picked = [str(t) for t in things]
    unpicked = sorted(n for n in _mentions(DRAWABLE_THINGS - EVERYDAY_WORDS, says)      # "clock" inside "alarm clock"
                      if not any(re.search(rf"\b{re.escape(n)}\b", t) for t in picked))
    if unpicked:
        return f"the line mentions {', '.join(unpicked)} without picking it; name only the things picked"
    other = [w for w in _mentions(NOT_DRAWABLE, says) if not any(w in t for t in things)]
    if other:
        return f"the line mentions {', '.join(other)}, which cannot be drawn; mention only the things picked"
    return None


IMAGINE_KINDS = ("count", "off_list", "unnamed", "unpicked", "not_drawable")


def imagine_kinds(row: dict[str, Any]) -> set[str]:
    """Every rule of imagine_problem a scene breaks, by kind (imagine_problem stops at the first): for reports that
    count which kinds of mistake a model makes. Empty exactly when imagine_problem is None."""
    things, says = row.get("things"), str(row.get("says", ""))
    if not isinstance(things, list) or len(things) != MAX_THINGS:
        return {"count"}
    picked = [str(t) for t in things]
    kinds = set()
    if any(t not in DRAWABLE_THINGS for t in picked):
        kinds.add("off_list")
    if any(not _named(t, says) for t in picked):
        kinds.add("unnamed")
    if any(not any(re.search(rf"\b{re.escape(n)}\b", t) for t in picked)
           for n in _mentions(DRAWABLE_THINGS - EVERYDAY_WORDS, says)):
        kinds.add("unpicked")
    if any(not any(w in t for t in picked) for w in _mentions(NOT_DRAWABLE, says)):
        kinds.add("not_drawable")
    return kinds


def _scene_problem(row: object, taken: list[set[str]]) -> str | None:
    """Why a scene cannot be kept: it breaks a rule (`imagine_problem`) or shares two things with a scene already kept."""
    if not isinstance(row, dict):
        return "no answer was given"
    problem = imagine_problem(row)
    if problem is None:
        same = next((t for t in taken if len(t & {str(x) for x in row["things"]}) >= 2), None)
        if same:
            problem = (f"another buyer already pictured {', '.join(sorted(same & set(map(str, row['things']))))}; "
                       "pick things from your own life")
    return problem


def _imagine(m: Market) -> Steps[None]:
    """Every buyer still bidding pictures the cat at their own place: three drawable things around it, chosen from
    their own card and what was said, and one or two spoken sentences naming each thing and what it is for. This
    replaces the rule-based scenes (scenes.py) as training data: the buyer model learns what to picture (owner,
    2026-10-11). Each buyer is asked on their own (their card, the answers, their own lines) and is told the words
    that cannot be drawn. A scene that breaks a rule (`imagine_problem`), or shares two things with a scene already
    kept, is asked again, told what was wrong, up to three times; one that still breaks it is dropped, never kept."""
    things = ", ".join(sorted(DRAWABLE_THINGS))
    rng = random.Random(f"{facts(m.s)}|imagine")
    examples = " and ".join(f'a buyer who {who} picks {picked}: "{line}"'
                            for who, picked, line in rng.sample(IMAGINE_EXAMPLES, 2))
    shared = (f"Things that can be drawn: {things}.\n\nThe questions and answers in this sale:\n"
              f"{m.questions_and_answers()}\n\nThe questions are over. Picture this cat at your own place, as your "
              "card, your life and what you heard suggest. things = exactly three things that would be around the cat "
              "in that picture, each written exactly as it appears in the list above (not the cat itself; nothing "
              "that is not on the list). Pick things from your own life (your work, your hobbies, your family, your "
              "home) as your card describes it, not furniture anyone has. says = one or two sentences in the first "
              "person, the way a real person talks, that name every thing you picked and what it is for in your life "
              "with this cat. Mention no other object at all: no other thing from the list, and none of these, which "
              f"cannot be drawn: {', '.join(NOT_DRAWABLE)}. Two examples: {examples}. Use your own things and words; "
              "do not reuse the examples or their things unless they truly belong to your life.\n\n")
    todo = list(m.present())
    notes: dict[str, str] = {}
    taken: list[set[str]] = []
    for _ in range(IMAGINE_TRIES):
        prompts = [shared + f"{m.card_line(n)}\nWhat you said during the sale:\n{m.own_lines(n)}{notes.get(n, '')}"
                   '\n\nReturn {"things": [...], "says": ...}.' for n in todo]
        rows = yield from _json_each(prompts)
        notes, still = {}, []
        for name, row in zip(todo, rows, strict=True):
            problem = _scene_problem(row, taken)
            if problem is None:             # so the reply is a dict
                reply = cast(dict[str, Any], row)
                taken.append({str(x) for x in reply["things"]})
                m.events.append({"kind": "imagine", "name": name, "things": [str(t) for t in reply["things"]],
                                 "says": str(reply.get("says", ""))})
            else:
                still.append(name)
                notes[name] = f"\n\nYour last answer broke a rule ({problem}); write it again."
        todo = still
        if not todo:
            return
        m.imagine_retried += len(todo)
    m.imagine_retried -= len(todo)
    m.imagine_dropped += len(todo)


def sale_steps(s: Sale, rounds: int = N_BUYERS) -> Steps[Market]:
    """One sale as a generator: it yields each prompt for the teacher and is sent back the reply."""
    m = Market(s)
    yield from _opening(m)
    asked: set[str] = set()
    for k in range(1, rounds + 1):
        if not (yield from _round(m, k, asked)):
            break
    if rounds == N_BUYERS and m.present():
        yield from _imagine(m)
        best = max(m.present(), key=m.offers.__getitem__)
        m.events.append({"kind": "sold", "name": best, "price": m.offers[best]})
    return m


def _render_react(e: Event) -> str:
    """A reaction with its sign, what was said and the offer change ("out" for a withdrawal)."""
    new = f"${e['offer']}" if e["offer"] else "out"
    why = f" | why: {e['why']}" if e.get("why") else ""          # only sales made before 2026-10-11 have a WHY
    return f"<|react|> {e['name']} ({e['sign']}){why} | says: {e['says']} | ${e['old']}->{new}"


RENDER: dict[str, Callable[[Event], list[str]]] = {
    "open": lambda e: [f"<|open|> {e['name']} ${e['offer']} | {e['says']}"],
    "ask": lambda e: [f"<|round {e['round']}|>", f"<|ask|> {e['name']} ${e['offer']} | {e['says']}"],
    "answer": lambda e: [f"<|answer to {e['to']}|> {e['says']}"],
    "react": lambda e: [_render_react(e)],
    "sold": lambda e: [f"<|sold|> {e['name']} ${e['price']}"],
    "imagine": lambda e: [f"<|imagine|> {e['name']} | things: {', '.join(e['things'])} | says: {e['says']}"],
    # the seller's decision on a buyer's final offer (owner 2026-10-13); a counter carries the buyer's reply
    "decide": lambda e: [f"<|decide|> {e['name']} ${e['offer']} | {e['action']}"
                         + (f" ${e['price']}" if e.get("price") else "") + f" | says: {e['says']}"]
                        + ([f"<|reply|> {e['name']} | {e['reply'][0]} | says: {e['reply'][1]}"] if e.get("reply") else []),
}


def render(m: Market) -> str:
    """The training layout: cards, opening, then each round between markers, every reaction with its sign, what was
    said and the offer change (sales made before 2026-10-11 also carry a separate WHY)."""
    out = ["<|sale|> cat"]
    out += [f"<|buyer|> {n} | {c['persona']} | cares about: {', '.join(c['cares_about'])}"
            + (f" | budget ${c['budget']}" if "budget" in c else "")
            for n, c in m.cards.items()]
    for e in m.events:
        if e["kind"] not in RENDER:
            raise ValueError(f"unknown event kind {e['kind']!r}")
        out += RENDER[e["kind"]](e)
    return "\n".join(out)


DOLLARS = re.compile(r"\$\s?(\d+)")


def check(m: Market) -> dict[str, int]:
    """Rule counts over the reactions: offer moving against its own sign; "+" or "-" with the offer unchanged; above
    the buyer's budget; a dollar amount said that is not the new offer (when only one amount is said); a buyer
    naming themselves; over every spoken line, dashes; imagined things that the drawing GPT cannot draw, and imagined
    things the buyer's line never names (matched on the last word: "alarm clock" by "clock")."""
    reacts = [e for e in m.events if e["kind"] == "react"]
    against = sum(1 for e in reacts if SIGNS.get(e["sign"], 0) * (e["offer"] - e["old"]) < 0
                  or (e["sign"] == "0" and e["offer"] != e["old"]))
    still = sum(1 for e in reacts if e["sign"] in ("+", "-") and e["offer"] == e["old"])
    over = sum(1 for e in reacts if e["offer"] > m.cards[e["name"]].get("budget", float("inf")))   # none: no limit
    said = [[int(x) for x in DOLLARS.findall(e["says"])] for e in reacts]
    wrong_amount = sum(1 for e, amounts in zip(reacts, said, strict=True)
                       if len(set(amounts)) == 1 and e["offer"] and amounts[0] != e["offer"])
    own_name = sum(1 for e in reacts if re.search(rf"\b{re.escape(e['name'])}\b", e["says"]))
    dashes = sum(1 for e in m.events if any(d in e.get("says", "") for d in DASHES))
    pictured = [t for e in m.events if e["kind"] == "imagine" for t in e["things"]]
    unsaid = sum(1 for e in m.events if e["kind"] == "imagine" for t in e["things"] if not _named(t, e["says"]))
    return {"imagined": sum(e["kind"] == "imagine" for e in m.events), "things_pictured": len(pictured),
            "things_not_drawable": sum(t not in DRAWABLE_THINGS for t in pictured), "things_not_said": unsaid,
            "imagine_retried": getattr(m, "imagine_retried", 0), "imagine_dropped": getattr(m, "imagine_dropped", 0),
            "react_retried": getattr(m, "react_retried", 0), "react_dropped": getattr(m, "react_dropped", 0),
            "reactions": len(reacts), "offer_against_sign": against, "sign_without_move": still,
            "over_budget": over, "amount_said_differs": wrong_amount, "names_self": own_name,
            "lines_with_dash": dashes}


def _items(m: Market, start: int, end: int) -> list[str]:
    """The answer at `start`, unnumbered, and the reactions after it, up to `end`, numbered for the reviewer."""
    answer = m.events[start]
    out = [f"SELLER to {answer['to']} (who asked: {answer['q']}): {answer['says']}"]
    for i in range(start + 1, end):
        e = m.events[i]
        card = m.cards[e["name"]]
        out.append(f"[{i}] {e['name']} (who {card['persona']}; cares about {', '.join(card['cares_about'])}) "
                   f"says: {e['says']} | sign {e['sign']} | ${e['old']} -> ${e['offer']}")
    return out


def judge_steps(m: Market) -> Steps[list[dict[str, Any]]]:
    """The teacher as a reviewer of buyers, one round at a time: each reaction against only what had been said by
    then, the facts about cats (all topics) and the buyer's own card. Sellers may exaggerate or lie, so their answers
    are not judged and the cat's true facts are kept from the reviewer. Returns the reactions it finds wrong or
    doubtful."""
    every_fact = "\n".join(f"- {f}" for t in TOPICS.values() for f in t["facts"])
    problems: list[dict[str, Any]] = []
    for start in (i for i, e in enumerate(m.events) if e["kind"] == "answer"):
        end = next((i for i in range(start + 1, len(m.events)) if m.events[i]["kind"] != "react"), len(m.events))
        rows = yield from _json(
            f"You are checking buyers' reactions in training data for mistakes of logic.\nTrue facts about cats:\n"
            f"{every_fact}\n\nConversation before this round:\n{m.transcript(start)}\n\nThis round:\n"
            + "\n".join(_items(m, start, end)) + "\n\n"
            "The seller may exaggerate or lie about this cat and is not being checked; a buyer who takes the seller at their word "
            "is reasoning correctly. A reaction may use only what has been said so far, this round's answer included, "
            "and never something the seller has not said. It is wrong if the reason it gives misreads what was said, "
            "contradicts the facts about cats, points the wrong way for that buyer's card, or withdraws ($0) over a "
            "drawback that does not rule the cat out for that buyer. Sign 0 with no reason is right when the answer "
            "touches nothing on the buyer's card. Flag a reaction only if no sensible buyer with that card could decide that "
            "way; another reasonable decision existing is not a mistake. Return a JSON list of only the numbered "
            'reactions that are wrong or doubtful (an empty list if every one is fine): {"id": <number>, '
            '"verdict": "wrong" | "doubtful", "problem": "<short>"}.')
        if not isinstance(rows, list):
            rows = [{"id": start, "verdict": "doubtful", "problem": "the reviewer gave no list"}]
        problems += [r for r in rows if isinstance(r, dict) and r.get("verdict") in ("wrong", "doubtful")]
    return problems


def knowledge_steps(topic: str, form: str) -> Steps[dict[str, Any]]:
    """One knowledge text written by the teacher, then checked by it against the topic's facts."""
    text = yield knowledge_prompt(topic, form)
    facts_list = "\n".join(f"- {f}" for f in TOPICS[topic]["facts"])
    rows = yield from _json(f"You are checking training data for mistakes.\nTrue facts:\n{facts_list}\n\nText:\n{text}\n\n"
                            "List every sentence of the text that contradicts the facts, adds a medical claim or number "
                            "that is not in them, or contradicts itself. Return a JSON list (empty if there are none): "
                            '{"sentence": "...", "problem": "<short>"}.')
    problems = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else \
        [{"sentence": None, "problem": "the reviewer gave no list"}]
    return {"topic": topic, "form": form, "text": text, "problems": problems}


def drive(steps: Steps[T], ask: Ask) -> T:
    """Run a generator of prompts to the end, answering each prompt (or each prompt of a list) with ask."""
    try:
        prompt = next(steps)
        while True:
            prompt = steps.send([ask(p) for p in prompt] if isinstance(prompt, list) else ask(prompt))
    except StopIteration as done:
        return done.value


def run_many(all_steps: list[Steps[T]], ask_many: Callable[[list[str]], list[str]]) -> list[T]:
    """Run many generators side by side: each turn, every waiting prompt (a list of prompts counts as its prompts)
    goes to ask_many in one batch (how a GPU server such as vLLM is kept busy), and each reply is sent back to the
    generator that asked."""
    results: list[Any] = [None] * len(all_steps)
    waiting: dict[int, Any] = {}
    for i, steps in enumerate(all_steps):
        try:
            waiting[i] = next(steps)
        except StopIteration as done:
            results[i] = done.value
    while waiting:
        ids = list(waiting)
        batch = [p for i in ids for p in (waiting[i] if isinstance(waiting[i], list) else [waiting[i]])]
        replies = iter(ask_many(batch))
        for i in ids:
            if isinstance(waiting[i], list):
                reply: Any = [next(replies) for _ in waiting[i]]
            else:
                reply = next(replies)
            try:
                waiting[i] = all_steps[i].send(reply)
            except StopIteration as done:
                results[i] = done.value
                del waiting[i]
    return results


def sale(ask: Ask, s: Sale, rounds: int = N_BUYERS) -> Market:
    """One sale written by `ask` from start to end (see sale_steps)."""
    return drive(sale_steps(s, rounds), ask)


def judge(ask: Ask, m: Market) -> list[dict[str, Any]]:
    """The teacher as a reviewer (see judge_steps)."""
    return drive(judge_steps(m), ask)


def sale_and_review(s: Sale, rounds: int = N_BUYERS) -> Steps[tuple[Market, list[dict[str, Any]]]]:
    """A sale, then the teacher's review of its reactions, as one generator (for run_many)."""
    m = yield from sale_steps(s, rounds)
    return m, (yield from judge_steps(m))


def mlx_teacher(path: Path = LOCAL_TEACHER, max_tokens: int = 2500) -> Ask:
    """The teacher on this Mac's GPU through MLX: temperature 0.8, top-p 0.95, as on the cloud."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(path))
    sampler = make_sampler(temp=0.8, top_p=0.95)

    def ask(prompt: str) -> str:
        msgs = [{"role": "system", "content": SETTING}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return generate(model, tok, text, max_tokens=max_tokens, sampler=sampler).strip()
    return ask


def main() -> None:
    """Command line: one sale on this Mac, printed and saved under runs/buyer_dialogues/."""
    parser = argparse.ArgumentParser(description="One cat sale written by the teacher model on this Mac.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=N_BUYERS)
    args = parser.parse_args()
    s = seed_sale(random.Random(args.seed))
    t0 = time.time()
    m = sale(mlx_teacher(), s, args.rounds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"cat_seed{args.seed}_rounds{args.rounds}_mac.json"
    report = {"seed": s, "events": m.events, "cards": m.cards, "check": check(m), "seconds": round(time.time() - t0)}
    path.write_text(json.dumps(report, indent=1))
    print(f"cat: {facts(s)}\nseller: {s['seller_style']}\n\n{render(m)}\n\ncheck: {report['check']}, "
          f"{report['seconds']} s\nsaved {path}")


if __name__ == "__main__":
    main()
