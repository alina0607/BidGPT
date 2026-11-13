"""The batch after the small words (data/written_sales/prompts_3050_plan.md): seeds 3050-3149 in stories.PAIRED_SEEDS
with who hears each target, their controls, "but" answers and buyers, the prompts and cue_plan.json, the batch's
check_cues.py, and the trap test's own-question scoring in tools/negation_probe.py."""

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sellanything_ml.buyers import stories as S

ML = Path(__file__).resolve().parents[1]
sys.path.append(str(ML / "tools"))
import external_prompt as E  # noqa: E402
import negation_probe as N  # noqa: E402

FOLDER = ML.parent / "data" / "written_sales" / "prompts_3050_3149"
SEEDS = range(3050, 3150)
FLIPS = {topic: {S.BUYER[k] for k in S.BUYER if S.BUYER[k] in ps} for topic, ps in S.PAIRED_BUYERS.items()}


def _digest(seeds) -> str:
    h = hashlib.sha256()
    for seed in seeds:
        h.update(json.dumps(S.sale_for_seed(seed), sort_keys=True).encode())
    return h.hexdigest()


def _flips(t) -> set[str]:
    return FLIPS.get(t[0], set()) | set(S.PAIRED_GROUP_BUYERS.get(t[3], ()))


def test_the_sheets_before_3050_and_after_3149_are_what_they_were():
    # recorded 2026-10-31 from the committed stories.py, before 3050-3149 were added
    assert _digest(range(3000)) == "7d65b084c29b9ef5ed149c59843a8b1fade823d8b54701c469855f4e7bbd699f"
    assert _digest(range(3000, 3050)) == "7056a60e839a10c5c15b1b4705162788ac72bf37ed4e00b58be137f4a0a6bd0e"
    assert _digest(range(3150, 3300)) == "d6846b0aa164e4e7cf1a0150a571be2ea63a3dda204531462db8248aa9677a22"


def test_the_targets_follow_the_quotas_of_section_3_1():
    batch = {seed: ts for seed, ts in S.PAIRED_SEEDS.items() if seed >= S.HEARD_FROM}
    assert sorted(batch) == list(SEEDS)
    targets = [t for ts in batch.values() for t in ts]
    assert len(targets) == 265 and all(len(t) == 6 for t in targets)
    assert Counter(len(ts) for ts in batch.values()) == {3: 65, 2: 35}
    sides = Counter((t[0], t[1]) for t in targets)
    # owner 2026-10-31 ("可以改配額"): more good-news sides, so the buyers a target turns raise about as often as
    # they lower (plan 3.1, within 10%); the pilot 3050-3054 kept as it was
    plan = {"kids": (9, 8, 7), "dogs": (9, 8, 7), "cats": (9, 8, 7), "shed": (6, 12, 6), "spay": (8, 12, 0),
            "age": (8, 8, 0), "vacc": (6, 6, 6), "fiv": (6, 5, 4), "handling": (7, 5, 0), "home": (6, 6, 0),
            "clingy": (6, 4, 0), "scratch": (6, 4, 0), "microchip": (4, 4, 0), "pests": (4, 4, 0), "litter": (4, 4, 0),
            "food": (4, 4, 0), "papers": (0, 4, 4), "senior": (4, 4, 0), "noise": (0, 4, 0)}
    for topic, (a, b, u) in plan.items():
        assert (sides[(topic, "A")], sides[(topic, "B")], sides[(topic, "U")]) == (a, b, u), topic
    assert sides[("origin", "C")] == 4
    groups: dict[str, list[int]] = {}
    for seed, ts in batch.items():
        assert len({t[0] for t in ts}) == len(ts), seed
        for t in ts:
            groups.setdefault(t[3], []).append(seed)
    assert all(len(set(v)) == len(v) for v in groups.values())
    assert not set(groups) & {t[3] for seed, ts in S.PAIRED_SEEDS.items() if seed < S.HEARD_FROM for t in ts}
    breeder = [s for s, ts in batch.items() if any(S.PAIRED_GROUP_BUYERS.get(t[3]) == (S.BREEDER,) for t in ts)]
    assert len(breeder) == 10                                  # half the spay sales have a breeder
    signs = Counter()
    for seed, ts in batch.items():
        for t in ts:
            for persona, sign in E.FLIP_SIGNS.get((t[0], t[1]), {}).items():
                if persona in S.paired_buyers(seed) and (persona != S.BREEDER or S.PAIRED_GROUP_BUYERS.get(t[3])
                                                         or t[0] == "papers"):
                    signs["-" if sign == "out" else sign] += 1
    assert abs(signs["+"] - signs["-"]) <= 0.1 * (signs["+"] + signs["-"])     # plan 3.1: raises about as many


def test_heard_and_askers_follow_section_2():
    targets = [(seed, t) for seed in SEEDS for t in S.PAIRED_SEEDS[seed]]
    heard = Counter(t[4] for _, t in targets)
    assert heard == {"self": 103, "other": 95, "inside": 67}           # about 40 / 35 / 25%
    for topic in ("kids", "dogs", "cats", "handling"):
        mine = [t for _, t in targets if t[0] == topic]
        assert sum(t[4] != "self" for t in mine) >= 0.6 * len(mine), topic
    for seed in SEEDS:
        personas = {b["persona"] for b in S.sale_for_seed(seed)["buyers"]}
        askers = []
        for t in S.PAIRED_SEEDS[seed]:
            if t[4] == "inside":
                assert t[5] is None
            elif t[4] == "other":
                assert S.BUYER[t[5]] not in _flips(t) and S.BUYER[t[5]] in personas, (seed, t)
            elif t[5]:
                assert S.BUYER[t[5]] in _flips(t), (seed, t)
            askers += [t[5]] if t[5] else []
        askers += [b[3] for b in S.PAIRED_BUTS[seed]]
        assert len(askers) == len(set(askers)), seed                  # a buyer asks once in a sale


def test_controls_buts_and_buyers_follow_sections_3_3_to_3_5():
    controls = Counter(S.PAIRED_CONTROLS[s][0] for s in SEEDS)
    assert controls == {"unrelated": 30, "double_neg": 24, "not_only": 20, "no_doubt": 16, "neg_question": 10}
    buts = [b for s in SEEDS for b in S.PAIRED_BUTS[s]]
    assert all(len(S.PAIRED_BUTS[s]) == 2 for s in SEEDS)
    assert Counter(b[0] for b in buts) == {"good but bad": 70, "bad but good": 70, "fragment": 40, "caveat": 20}
    assert sum(b[1] in ("though", "tho") for b in buts) >= 40
    assert sum(b[2] == "other" for b in buts) >= 100
    for seed in SEEDS:
        s = S.sale_for_seed(seed)
        sheet = E.sheet_facts(s["cat"])
        said = {t[2] for t in S.PAIRED_SEEDS[seed]}
        personas = {b["persona"] for b in s["buyers"]}
        _, facts = S.PAIRED_CONTROLS[seed]
        used = list(facts)
        for b_kind, connector, heard, asker, up, down, b_facts in S.PAIRED_BUTS[seed]:
            assert connector in ("but", "though", "tho") and len(b_facts) == (1 if b_kind == "caveat" else 2)
            assert (up is None) == (b_kind == "caveat") and down and up != down
            assert (asker in (up, down)) == (heard == "self"), (seed, b_facts)
            assert {S.BUYER[k] for k in (asker, up, down) if k} <= personas
            used += list(b_facts)
        assert set(used) <= set(sheet) and not set(used) & said, seed     # sheet facts, never a target's
        assert len(used) == len(set(used)), seed                           # each answer has its own facts
        assert set(S.paired_buyers(seed)) <= personas and len(S.paired_buyers(seed)) <= S.N_BUYERS
        if S.BREEDER in personas:
            assert s["cat"]["months"] <= S.BREEDING_MONTHS, seed          # a breeder bids on no cat over five
    for persona in ("kids", "dogs", "old cat", "allergic", "breeder"):
        assert sum(S.BUYER[persona] in {b["persona"] for b in S.sale_for_seed(s)["buyers"]} for s in SEEDS) >= 70


def test_the_cue_plan_writes_heard_and_askers_and_keeps_the_balance():
    plan = E.cue_plan(list(SEEDS))["seeds"]
    targets = [t for e in plan.values() for t in e["targets"]]
    assert all(t["heard"] in ("self", "other", "inside") for t in targets)
    assert all(t["asker_name"] for t in targets if t["heard"] == "other")
    negated = [t for t in targets if t["cue"] in E.NEGATION_FAMILIES]
    news = Counter(t["news"] for t in negated)
    assert abs(news["+"] - news["-"]) <= 0.1 * len(negated)                     # plan 3000 4.3
    for topic in {t["topic"] for t in targets} - {"age"}:
        cues = Counter(t["cue"] for t in targets if t["topic"] == topic and t["cue"] != "plain")
        assert max(cues.values(), default=0) <= max(2, 0.4 * sum(t["topic"] == topic for t in targets)), topic
    for seed, e in plan.items():
        assert e["control"]["kind"] == S.PAIRED_CONTROLS[int(seed)][0] and len(e["buts"]) == 2
        assert "contrast" not in e and all(b["fact_words"] for b in e["buts"])


def test_the_prompt_folder_is_up_to_date_and_names_who_asks():
    assert json.loads((FOLDER / "cue_plan.json").read_text()) == json.loads(json.dumps(E.cue_plan(list(SEEDS))))
    plan = E.cue_plan(list(SEEDS))["seeds"]
    for seed in SEEDS:
        text = (FOLDER / f"seed_{seed}.md").read_text()
        assert "## This sale's small words and who hears them" in text and S.facts(S.sale_for_seed(seed)) in text
        for t in plan[str(seed)]["targets"]:
            if t["heard"] == "other":
                assert f"Heard: asked by {t['asker_name']} ({t['asker']}), not by a buyer it turns" in text
            if t["heard"] == "inside":
                assert "Heard: said inside the answer to a question about something else" in text
        for b in plan[str(seed)]["buts"]:
            assert f'joined with "{b["connector"]}"; asked by {b["asker_name"]}' in text
        ban = next(line for line in text.splitlines() if line.startswith("kept out of training"))
        assert not re.search(E.HELD_OUT_PATTERN, text.replace(ban, "").lower()), seed
    facts = json.loads((FOLDER / "life_facts.json").read_text())
    assert facts == {str(s): {"kinds": c["life_kinds"], "facts": c["life"]}
                     for s in SEEDS for c in [S.sale_for_seed(s)["cat"]] if "life" in c}
    for name in ("check_sales.py", "check_lessons.py", "check_life.py", "check_cues.py", "scene_lists.json",
                 "AGENT_INSTRUCTIONS.md", "WRITER_BRIEF.txt", "REVIEW_BRIEF.txt"):
        assert (FOLDER / name).exists(), name
    assert "42. **Heard from another buyer" in (FOLDER / "AGENT_INSTRUCTIONS.md").read_text()


def _check_cues():
    spec = importlib.util.spec_from_file_location("check_cues_3050", FOLDER / "check_cues.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _but_text(b: dict) -> str:
    """A "but" answer of the plan: its facts joined with its connector, a fragment one starting with "nah"."""
    if b["kind"] == "caveat":
        return f"{b['facts'][0].capitalize()}, {b['connector']} that's all."
    start = "nah, " if b["kind"] == "fragment" else ""
    return f"{start}{b['facts'][0]}, {b['connector']} {b['facts'][1]}."


def _sale_3050():
    """A sale for seed 3050 that follows its plan: 2 years (self, anyone), indoor-only (never, other: the planned
    asker), never tested for FIV (never, self: the buyer with an older cat), a not-only control, its two "but"
    answers and two fragments."""
    e = E.cue_plan([3050])["seeds"]["3050"]
    names = list(e["names"])
    t = {x["topic"]: x for x in e["targets"]}
    b1, b2 = e["buts"]
    taken = {t["home"]["asker_name"], t["fiv"]["asker_name"], b1["asker_name"], b2["asker_name"]}
    free = [n for n in names if n not in taken]
    rounds = [
        (t["fiv"]["asker_name"], "What about the feline virus test?", "Never been tested for FIV, sorry.", "never tested"),
        (free[0], "How old is she?", "She's 2 years old.", "2 years old"),
        (t["home"]["asker_name"], "Does she go outside?", "She never goes outside, indoor only.", "never goes out"),
        (free[1], "What does she eat?", "Not only does she eat wet food of any brand, but she's fully vaccinated too.",
         "any wet food; fully vaccinated"),
        *((b["asker_name"], f"Anything else I should know, part {k}?", _but_text(b), "; ".join(b["facts"]))
          for k, b in enumerate((b1, b2), 1)),
        (free[2], "Is she shy?", "Nope, she's calm with people.", "not shy"),
        (free[3], "Is she loud?", "Not really, she's quiet most days.", "quiet"),
    ]
    return {"cards": {n: {"cares_about": ["calm"]} for n in names}, "open": {n: [70, f"Hello from {n}."] for n in names},
            "rounds": [{"name": who, "q": q, "a": a, "note": note,
                        "react": {n: ["0", 70, "That keeps my offer as it was for now."] for n in names}}
                       for who, q, a, note in rounds], "imagine": {}}


def test_check_cues_passes_a_sale_that_follows_its_plan_and_flags_who_asked():
    cc = _check_cues()
    sale = _sale_3050()
    found, summary = cc.problems("3050", sale)
    assert not [x for x in found if x.startswith("problem")], found
    assert sorted(summary["buts"]) == sorted(b["kind"] for b in E.cue_plan([3050])["seeds"]["3050"]["buts"])
    e = E.cue_plan([3050])["seeds"]["3050"]
    other = next(x for x in e["targets"] if x["heard"] == "other")
    # every buyer the other-asked target turns is listed to read, keeps included
    assert sum(x.startswith("read: round 3 (home A, heard other)") for x in found) == len(other["flip"])
    wrong = _sale_3050()
    turned = next(n for n, p in e["names"].items() if p in other["flip"])
    wrong["rounds"][2]["name"], wrong["rounds"][1]["name"] = turned, wrong["rounds"][2]["name"]
    assert any("asked by " + turned in x and x.startswith("problem") for x in cc.problems("3050", wrong)[0])
    other_word = _sale_3050()
    b = e["buts"][0]
    other_word["rounds"][4]["a"] = _but_text({**b, "connector": "and"})
    assert any(f"no \"but\" answer ({b['kind']}, {b['connector']}" in x for x in cc.problems("3050", other_word)[0])


def test_check_cues_reads_a_target_inside_another_answer_by_its_own_sentence():
    cc = _check_cues()
    t = {"topic": "kids", "side": "B", "value": "doesn't like children", "cue": "nt", "heard": "inside",
         "topic_words": E.TOPIC_WORDS["kids"], "side_words": E.SIDE_WORDS[("kids", "B")][0],
         "not_words": None, "cue_words": E.CUE_FAMILIES["nt"][0]}
    sale = {"rounds": [{"name": "Ann", "q": "What's her temperament like?",
                        "a": "Calm and lazy most of the day. She doesn't like children.", "note": "calm; doesn't like children",
                        "react": {}}]}
    assert cc.target(sale, t) == ([], 1) and cc.heard(sale, t, 1) == []
    sale["rounds"][0]["q"] = "Is she good with kids?"
    assert cc.heard(sale, t, 1)[0].startswith("problem")


def test_own_question_scoring_lets_only_the_buyers_a_topic_moves_ask():
    items = [{"group": "T01a", "want": "-"}, {"group": "T01a", "want": "0"}, {"group": "N02a", "want": "0"}]
    assert N.own_askers(items) == [N.TESTED, N.TESTED, N.OTHER]
