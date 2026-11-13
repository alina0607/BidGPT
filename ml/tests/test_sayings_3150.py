"""The batch of sayings buyer GPT B6 had seen too few times (data/written_sales/prompts_3150_plan.md): seeds 3150-3234
in stories.PAIRED_SEEDS with a shape on every target, their sheets, the prompts and cue_plan.json."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from sellanything_ml.buyers import stories as S

ML = Path(__file__).resolve().parents[1]
sys.path.append(str(ML / "tools"))
import external_prompt as E  # noqa: E402

FOLDER = ML.parent / "data" / "written_sales" / "prompts_3150_3234"
SEEDS = range(3150, 3235)


def test_the_sheets_before_3150_are_what_they_were():
    # recorded 2026-11-10 before 3150-3234 were added (the same as with the stories.py before that change)
    h = hashlib.sha256()
    for seed in range(3050, 3150):
        h.update(json.dumps(S.sale_for_seed(seed), sort_keys=True).encode())
    assert h.hexdigest() == "12c408432dd419aeb945df8d4a9e6cf3e058a694a2eab4c90e01f97beb439d8a"


def test_the_targets_follow_the_quotas_of_section_3_1():
    batch = {seed: ts for seed, ts in S.PAIRED_SEEDS.items() if seed >= S.SAYINGS_FROM}
    assert sorted(batch) == list(SEEDS)
    targets = [t for ts in batch.values() for t in ts]
    assert all(len(ts) == 3 and len({t[0] for t in ts}) == 3 for ts in batch.values())
    assert all(len(t) == 7 and t[6] in S.SHAPE_BUYERS and t[6] in E.SHAPES for t in targets)
    assert Counter(t[6] for t in targets) == {
        "player": 20, "player_good": 20, "healthy": 15, "dog_only": 30, "dog_said": 10, "cat_lived": 5,
        "age_only": 30, "intact_months": 20, "unsure": 30, "shed": 65, "shed_intact": 10}
    assert Counter(t[2] for t in targets if t[6] == "age_only") == {"2 years": 20, "2 months": 10}
    assert all(t[4] != "inside" for t in targets if t[6] == "healthy")       # item 49: someone asks if she is healthy
    assert all(len(S.PAIRED_BUTS[s]) == 1 for s in SEEDS) and set(S.PAIRED_CONTROLS) >= set(SEEDS)


def test_each_sheet_says_what_its_targets_need():
    for seed in SEEDS:
        s = S.sale_for_seed(seed)
        c, personas = s["cat"], {b["persona"] for b in s["buyers"]}
        need = S.paired_buyers(seed)
        assert len(need) <= S.N_BUYERS and set(need) <= personas, seed
        if S.BREEDER in personas:
            assert c["months"] <= S.BREEDING_MONTHS, seed
        for t in S.PAIRED_SEEDS[seed]:
            shape = t[6]
            if shape in ("player", "player_good"):
                assert c["virus"] == "tested FIV positive", seed
            if shape == "player_good":
                assert c["health"] in S.GOOD_HEALTH and 12 <= c["months"] <= 120, seed
            if shape == "healthy":
                assert c["health"] == "microchipped and healthy", seed
            if shape == "unsure":
                assert c["health"] == t[2] and "vacc" not in c["health"], seed
            if shape == "dog_only":
                assert c["company"] == ["has never met another cat", "has lived with a dog"], seed
            if shape == "intact_months":
                assert c["months"] in (2, 4) and c["fixed"] == "not yet spayed/neutered", seed
            if shape == "shed_intact":
                assert c["fixed"] == "not yet spayed/neutered" and "sheds very little" in c["quirks"], seed


def test_barely_and_hardly_are_taught_from_3150_only():
    assert E.held_out(3149)[0] == E.HELD_OUT_WORDS
    assert set(E.held_out(3150)[0]) == set(E.HELD_OUT_WORDS) - {"barely", "hardly"}
    plan = E.cue_plan(list(SEEDS))
    assert "barely" not in plan["held_out"] and "barely" not in plan["held_out_pattern"]
    shed = Counter(t["cue"] for e in plan["seeds"].values() for t in e["targets"] if t.get("shape") == "shed")
    assert shed == {"much": 17, "barely": 16, "notmuch": 16, "hardly": 16}


def test_the_prompts_and_cue_plan_on_disk_are_current():
    plan = E.cue_plan(list(SEEDS))
    assert json.loads((FOLDER / "cue_plan.json").read_text()) == json.loads(json.dumps(plan))
    for seed in SEEDS:
        text = (FOLDER / f"seed_{seed}.md").read_text()
        assert E.cues(seed, plan["seeds"][str(seed)]).strip() in text, seed
        for t in plan["seeds"][str(seed)]["targets"]:
            assert t["expect"] and t["shape"], seed
