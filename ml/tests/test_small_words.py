"""The small-words batch (data/written_sales/prompts_3000_plan.md): the paired sheets of seeds 3000-3049, their
prompts and cue_plan.json, the batch's check_cues.py, and the trap test in tools/negation_probe.py."""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from sellanything_ml.buyers import stories as S

ML = Path(__file__).resolve().parents[1]
sys.path.append(str(ML / "tools"))
import external_prompt as E
import negation_probe as N

FOLDER = ML.parent / "data" / "written_sales" / "prompts_3000_3049"
PAIRED = range(3000, 3050)
PILOT = range(3000, 3005)


def _digest(seeds) -> str:
    h = hashlib.sha256()
    for seed in seeds:
        h.update(json.dumps(S.sale_for_seed(seed), sort_keys=True).encode())
    return h.hexdigest()


def test_every_other_seed_draws_exactly_what_it_drew_before():
    # recorded 2026-10-29 before PAIRED_SEEDS was added
    assert _digest(range(3000)) == "7d65b084c29b9ef5ed149c59843a8b1fade823d8b54701c469855f4e7bbd699f"
    assert _digest(range(3050, 3300)) == "a9740bd988e7c142bc06d46262eee1b7efb4deb3d02c9c51007945ddee05e2e6"


def test_the_new_values_are_never_drawn_at_random():
    new = {S.SHEDS_LITTLE, *S.AGAINST.values()}
    for seed in [*range(2000, 3000), *range(3050, 3300)]:
        assert not new & set(E.sheet_facts(S.sale_for_seed(seed)["cat"])), seed


def test_paired_seeds_are_the_fifty_of_the_plan_with_the_quotas_of_section_3():
    assert sorted(S.PAIRED_SEEDS) == list(PAIRED)
    targets = [t for ts in S.PAIRED_SEEDS.values() for t in ts]
    assert len(targets) == 122
    assert all(2 <= len(ts) <= 3 for ts in S.PAIRED_SEEDS.values())
    sides = Counter((t[0], t[1]) for t in targets)
    plan = {"shed": (5, 5, 2), "spay": (4, 8, 0), "age": (4, 4, 0), "vacc": (3, 3, 2), "fiv": (3, 3, 2),
            "kids": (3, 3, 2), "dogs": (3, 3, 2), "cats": (2, 2, 2), "clingy": (3, 3, 0), "scratch": (3, 3, 0),
            "home": (3, 3, 0), "noise": (0, 4, 0), "senior": (2, 2, 0), "papers": (0, 2, 2), "food": (2, 2, 0),
            "pests": (2, 2, 0), "litter": (2, 2, 0), "handling": (2, 2, 0), "microchip": (2, 2, 0)}
    for topic, (a, b, u) in plan.items():
        assert (sides[(topic, "A")], sides[(topic, "B")], sides[(topic, "U")]) == (a, b, u), topic
    assert sides[("origin", "C")] == 2
    by_group: dict[str, list[int]] = {}
    for seed, ts in S.PAIRED_SEEDS.items():
        assert len({t[0] for t in ts}) == len(ts), seed                 # one target per topic in a sale
        for t in ts:
            by_group.setdefault(t[3], []).append(seed)
    assert all(len(set(v)) == len(v) for v in by_group.values())      # the paired sales are different sales
    breeder = by_group["spay1"] + by_group["spay2"]                     # 6 of the 12 spay sales have a breeder
    assert len(breeder) == 6 and all("breeds show cats" in S.paired_buyers(s) for s in breeder)


def _field(c, topic, side, value):
    if value is None:
        return {"shed": not any("shed" in q for q in c["quirks"]), "papers": "no papers" not in c.get("extra_kinds", [])}[topic]
    if topic == "age":
        return c["age"] == value
    if topic == "papers":
        return c.get("extra_kinds") == ["no papers"] and c["extra"][0].endswith("but never had papers")
    return value in E.sheet_facts(c)


@pytest.mark.parametrize("seed", PAIRED)
def test_each_paired_sheet_has_its_targets_and_agrees_with_itself(seed):
    s = S.sale_for_seed(seed)
    c = s["cat"]
    for topic, side, value, _ in S.PAIRED_SEEDS[seed]:
        assert _field(c, topic, side, value), (topic, side, value)
    stage = S.HEALTH_KITTEN if c["months"] < 12 else S.HEALTH_SENIOR if c["months"] > 120 else S.HEALTH_ADULT
    assert c["health"] in stage
    assert c["age"] == S._age(c["months"]) and c["months"] in S.AGES_MONTHS
    assert not (c["months"] < 6 and c["fixed"] == "spayed/neutered")
    assert not (c["health"] == S.NO_VET and c["months"] <= 24)
    assert not (c["health"] == S.FULL_SERIES and c["months"] < 4)
    assert not (c["breed"] == "a Sphynx" and any("shed" in q for q in c["quirks"]))
    assert len(set(c["quirks"])) == 2 and len({S._kind_of(e) for e in c["company"]}) == 2
    assert not (S.LITTER in c["life"] and "has never met another cat" in c["company"])
    assert not (S.TREATED in c["life"] and c["health"] == S.NO_VET)
    assert len(c["life"]) == S.PAIRED_LIFE == len(set(c["life_kinds"]))
    for kind, fact in zip(c["life_kinds"], c["life"], strict=True):
        assert fact in S.LIFE_FACTS[kind][c["months"] >= 12] or fact == "found as a stray kitten", (kind, fact)
    if any(v in c["company"] for v in S.AGAINST.values()):
        assert c["months"] >= 12                       # not getting on with others is said of a grown cat
    assert not ({"picky about litter"} & set(c["quirks"]) and "litter" in c["life_kinds"])
    personas = [b["persona"] for b in s["buyers"]]
    assert len(set(personas)) == len({b["name"] for b in s["buyers"]}) == S.N_BUYERS
    assert set(S.paired_buyers(seed)) <= set(personas)


def test_the_cue_plan_follows_the_plan():
    plan = E.cue_plan(list(PAIRED))
    seeds = plan["seeds"]
    controls = Counter(e["control"]["kind"] for e in seeds.values())
    assert controls == {k: v[0] for k, v in E.CONTROLS.items()}                       # plan 4: 15/12/10/8/5
    assert Counter(e["contrast"]["order"] for e in seeds.values()) == {"good but bad": 25, "bad but good": 25}
    targets = [t for e in seeds.values() for t in e["targets"]]
    negated = [t for t in targets if t["cue"] in E.NEGATION_FAMILIES]
    news = Counter(t["news"] for t in negated)
    assert abs(news["+"] - news["-"]) <= 0.1 * len(negated)                           # plan 4.3: half good, half bad
    for topic in {t["topic"] for t in targets} - {"age"}:
        cues = Counter(t["cue"] for t in targets if t["topic"] == topic and t["cue"] != "plain")
        n = sum(t["topic"] == topic for t in targets)
        assert max(cues.values(), default=0) <= max(2, 0.4 * n), topic                # plan 3.2: at most 40%
    pilot = [t for s in PILOT for t in seeds[str(s)]["targets"]]
    values = {t["value"] for t in pilot}
    assert {S.SHEDS_LITTLE, *S.AGAINST.values(), "2 months", "2 years"} <= values
    assert any(t["cue"] == "intact" for t in pilot) and any(t["topic"] == "vacc" and t["side"] == "U" for t in pilot)
    assert len({seeds[str(s)]["control"]["kind"] for s in PILOT}) >= 3
    for e in seeds.values():                       # the control and "but" facts are on the sheet and findable
        for f in e["control"]["facts"] + e["contrast"]["facts"]:
            assert E.fact_words(f)


def test_the_prompt_folder_is_up_to_date_and_names_no_held_out_word_outside_its_ban():
    assert json.loads((FOLDER / "cue_plan.json").read_text()) == json.loads(json.dumps(E.cue_plan(list(PAIRED))))
    for seed in PAIRED:
        text = (FOLDER / f"seed_{seed}.md").read_text()
        assert "## This sale's small words" in text and S.facts(S.sale_for_seed(seed)) in text
        ban = next(line for line in text.splitlines() if line.startswith("kept out of training"))
        rest = text.replace(ban, "")
        assert not __import__("re").search(E.HELD_OUT_PATTERN, rest.lower()), seed
    for name in ("check_sales.py", "check_lessons.py", "check_life.py", "check_cues.py", "life_facts.json",
                 "scene_lists.json", "AGENT_INSTRUCTIONS.md", "WRITER_BRIEF.txt"):
        assert (FOLDER / name).exists(), name


def _check_cues():
    spec = importlib.util.spec_from_file_location("check_cues", FOLDER / "check_cues.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sale(seed, answers):
    """A sale whose rounds give these answers (question, answer, note), with every buyer keeping $70."""
    names = list(E.cue_plan([seed])["seeds"][str(seed)]["names"])
    return {"cards": {n: {"cares_about": ["calm"]} for n in names},
            "open": {n: [70, f"Hello from {n}."] for n in names},
            "rounds": [{"name": names[k], "q": q, "a": a, "note": note,
                        "react": {n: ["0", 70, "That keeps my offer as it was for now."] for n in names}}
                       for k, (q, a, note) in enumerate(answers)],
            "imagine": {}}


GOOD_3000 = [   # seed 3000: sheds very little (n't), indoor-outdoor (plain), not yet spayed (intact); control: unrelated
    ("How much does she shed?", "She doesn't shed much at all.", "doesn't shed much"),
    ("Does she go outside?", "She's indoor-outdoor, out most afternoons.", "indoor-outdoor"),
    ("Is she spayed?", "She's intact, not done yet.", "intact, not spayed"),
    ("Is she used to dogs?", "No clue what her first name was, but she has lived with a dog.", "lived with a dog"),
    ("Any habits?", "Uses a scratching post, never the sofa, but no bloodwork in years.", "post; no bloodwork in years"),
    ("Is she friendly?", "nope, she's shy with strangers", "shy with strangers"),
    ("Is she friendly with kids?", "Not really sure, she has never met children.", "never met children"),
]


def test_check_cues_passes_a_sale_that_follows_its_plan_and_flags_each_fault():
    cc = _check_cues()
    found, _ = cc.problems("3000", _sale(3000, GOOD_3000))
    assert not [x for x in found if x.startswith("problem")], found
    late = list(GOOD_3000)
    late[0] = ("How much does she shed?", "Her coat is short and soft, and she doesn't shed much.", "doesn't shed much")
    assert any("after the first 6 words" in x for x in cc.problems("3000", _sale(3000, late))[0])
    dropped = list(GOOD_3000)
    dropped[2] = ("Is she spayed?", "She's intact, not done yet.", "spay still to do")
    assert any("note of round 3 drops" in x for x in cc.problems("3000", _sale(3000, dropped))[0])
    held = list(GOOD_3000)
    held[5] = ("Is she friendly?", "nope, she's barely friendly with strangers", "shy")
    assert any("held-out word 'barely'" in x for x in cc.problems("3000", _sale(3000, held))[0])
    no_control = [a for a in GOOD_3000 if "first name" not in a[1]]
    assert any("no control answer" in x for x in cc.problems("3000", _sale(3000, no_control))[0])
    plain = [a for a in GOOD_3000 if not a[1].lower().startswith(("nope", "not really"))]
    assert any("player's fragment" in x for x in cc.problems("3000", _sale(3000, plain))[0])


def test_check_cues_lists_a_turned_buyer_moving_the_wrong_way_for_reading():
    cc = _check_cues()
    sale = _sale(3000, GOOD_3000)
    names = E.cue_plan([3000])["seeds"]["3000"]["names"]
    allergic = next(n for n, p in names.items() if p == "is allergic and worried about fur")
    sale["rounds"][0]["react"][allergic] = ["-", 60, "Less fur still means fur in my sinuses. Down to $60."]
    found, _ = cc.problems("3000", sale)
    assert any(x.startswith("read: round 1") and allergic in x and "asks +" in x for x in found)


# ---- the trap test ----

def _item(group, answer, want, part="T", persona="is allergic and worried about fur", cue="plain"):
    return {"group": group, "part": part, "persona": persona, "cares_about": ["little fur"], "greeting": "Hi.",
            "question": "Does she shed?", "answer": answer, "want": want, "cue": cue}


def test_load_items_reads_the_format_and_refuses_a_broken_line(tmp_path):
    good = tmp_path / "items.jsonl"
    good.write_text("\n".join(json.dumps(_item("T01a", a, w)) for a, w in (("She sheds a lot.", "-"),
                                                                           ("She doesn't shed.", "+"))))
    assert [i["want"] for i in N.load_items(good)] == ["-", "+"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**_item("T01a", "x", "+"), "want": "up"}))
    with pytest.raises(ValueError):
        N.load_items(bad)
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(json.dumps(x) for x in (_item("T01a", "a", "+"), {**_item("T01a", "b", "-"), "question": "?"})))
    with pytest.raises(ValueError):
        N.load_items(other)


def test_score_counts_whole_groups_pairs_and_moves():
    items = [_item("T01a", "She sheds a lot.", "-"), _item("T01a", "She doesn't shed.", "+"),
             _item("T01a", "Not sure.", "0"), _item("C01a", "No doubt she sheds.", "0", part="C"),
             _item("C01a", "She sheds.", "0", part="C")]
    probs = [{"+": .1, "0": .2, "-": .7}, {"+": .6, "0": .2, "-": .2}, {"+": .5, "0": .4, "-": .1},
             {"+": .1, "0": .8, "-": .1}, {"+": .1, "0": .8, "-": .1}]
    s = N.score(items, probs)
    assert (s["all"]["groups"], s["all"]["groups right"]) == (2, 1)        # T01a misses "Not sure."
    assert (s["all"]["answers right"], s["T"]["pairs"], s["T"]["pairs both right"]) == (4, 3, 1)
    assert s["T"]["moved right"] == 3        # every pair moved the right way, though one answer is still wrong
    assert s["not all 0"]["groups"] == 1 and "not all 0" in s and s["C"]["pairs"] == 0


def test_the_baselines_read_only_the_topic_or_a_negation_word():
    sales = [{"cards": {"Ann": {"persona": "is allergic and worried about fur"}, "Bo": {"persona": "x"}},
              "events": [{"kind": "answer", "q": "Does she shed?", "says": "She sheds a lot."}]
              + [{"kind": "react", "name": "Ann", "sign": "-"}] * N.MIN_SEEN + [{"kind": "react", "name": "Bo", "sign": "0"}] * 30}]
    majority = N.training_majority(sales)
    assert majority[("is allergic and worried about fur", "coat/shedding")] == "-"
    assert majority[("*", "coat/shedding")] == "0"
    lot, little = _item("T", "She sheds a lot.", "-"), _item("T", "She doesn't shed.", "+")
    assert N.keyword_sign(lot, majority) == N.keyword_sign(little, majority) == "-"
    assert N.negation_sign(little, majority) == "-" and N.negation_sign({**lot, "persona": "y"}, majority) == "0"


def test_the_written_trap_test_loads_if_it_is_there():
    if not (ML / N.ITEMS_FILE).exists():
        pytest.skip("the trap test items are not written yet")
    items = N.load_items(ML / N.ITEMS_FILE)
    assert {i["part"] for i in items} <= set(N.PARTS) and len({i["group"] for i in items}) >= 50
