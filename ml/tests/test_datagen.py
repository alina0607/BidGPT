"""Prompt format and training-data generation (docs/buyer-ai.md §7–8)."""

import random

from sellanything_ml.buyers import datagen, load_items, load_labels, load_traits
from sellanything_ml.buyers import format as fmt


def test_react_target_round_trips():
    for line in ("Oh nice, that's perfect.", None):
        for delta in (-2, -1, 0, 1, 2):
            target = fmt.react_target("Maya", line, delta)
            name, parsed_line, parsed_delta = fmt.parse_react(target)
            assert parsed_delta == delta
            assert parsed_line == line
            assert name == ("Maya" if line else None)


def test_parse_rejects_malformed_output():
    assert fmt.parse_react("Maya: sure<|interest|>+3<|end|>") is None     # out of range
    assert fmt.parse_react("Maya: sure<|end|>") is None                   # no Δ
    assert fmt.parse_line("no speaker here<|end|>") is None


def test_context_omits_attrs_when_not_given():
    with_attrs = fmt.BuyerView("dog", ["age", "size"], "Maya, who is retired.", []).render()
    without = fmt.BuyerView("dog", None, "Maya, who is retired.", []).render()
    assert fmt.ATTRS in with_attrs and fmt.ATTRS not in without


def test_scenes_are_consistent_with_the_engine():
    items, traits = load_items(), load_traits()
    rng = random.Random(0)
    for key in ("dog", "bicycle", "house", "lipstick", "wristwatch"):
        for _ in range(5):
            scene, examples = datagen.simulate_scene(items[key], traits, rng)
            assert examples
            for ex in examples:
                datagen.check(ex, items)            # parses, Δ in text == label
            # every engine Δ that produced a react example matches the scene's event record
            deltas = {(e["attr"], b): d for e in scene["events"] for b, d in e["deltas"].items()}
            for ex in examples:
                if ex["task"] == "react":
                    assert deltas[(ex["attr"], ex["buyer"])] == ex["delta"]


def test_reason_tag_matches_largest_contribution():
    assert datagen.reason_tag({"space": 2, "active": 1}, {"space": -2, "active": 1}) == "space"
    assert datagen.reason_tag({"style": 2}, {"space": -2}) is None


def test_item_splits_cover_everything_once():
    labels = load_labels()
    splits = datagen.item_splits(labels)
    buy = splits["buy_train"] + splits["buy_heldout"]
    refuse = splits["refuse_train"] + splits["refuse_heldout"]
    # "buy_*" holds every sellable item: plain 'buy' and 'risky' (sellable, some buyers refuse)
    assert sorted(buy) == sorted(k for k, v in labels.items() if v in ("buy", "risky"))
    assert sorted(refuse) == sorted(k for k, v in labels.items() if v == "refuse")
    assert not set(splits["buy_train"]) & set(splits["buy_heldout"])


def test_prompt_format_2_puts_the_engine_question_in_ask_prompts():
    items, traits = load_items(), load_traits()
    _, v1 = datagen.simulate_scene(items["dog"], traits, random.Random(5), prompt_format=1)
    _, v2 = datagen.simulate_scene(items["dog"], traits, random.Random(5), prompt_format=2)
    asks = [(a, b) for a, b in zip(v1, v2) if a["task"] == "ask"]
    assert asks and all(a["target"] == b["target"] for a, b in zip(v1, v2))      # same scene
    for a, b in asks:
        question = items["dog"].attributes[a["attr"]].question
        assert a["prompt"].endswith(f"<|ask|>{a['attr']}<|end|>")
        assert b["prompt"].endswith(f"<|ask|>{a['attr']}: {question}<|end|>")


def test_datasets_without_meta_are_format_1(tmp_path):
    assert datagen.prompt_format_of(tmp_path) == 1
    (tmp_path / "meta.json").write_text('{"prompt_format": 2}')
    assert datagen.prompt_format_of(tmp_path) == 2


def test_is_current_catches_changed_data_but_not_renamed_rows(tmp_path):
    import json
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "sellanything_ml.buyers.datagen", "--scenes-per-item", "10",
                    "--out", str(tmp_path)], check=True, capture_output=True)
    assert datagen.is_current(tmp_path)

    train = (tmp_path / "train.jsonl").read_text().splitlines()
    first = json.loads(train[0])
    renamed = {**first, "prompt": first["prompt"].replace("<|item|>", "<|item|>zorvel "), "swapped_to": "zorvel"}
    (tmp_path / "train.jsonl").write_text("\n".join([json.dumps(renamed), *train[1:]]) + "\n")
    assert datagen.is_current(tmp_path)                 # augment.py's renames are expected

    edited = {**first, "target": first["target"].replace(":", ": Well,", 1)}
    (tmp_path / "train.jsonl").write_text("\n".join([json.dumps(edited), *train[1:]]) + "\n")
    assert not datagen.is_current(tmp_path)             # anything else that differs is stale


def test_format_3_targets_never_spell_the_item():
    items, traits = load_items(), load_traits()
    item = items["hedgehog"]
    _, examples = datagen.simulate_scene(item, traits, random.Random(2), prompt_format=3)
    names = fmt.item_names(item.key, item.noun)
    for ex in examples:
        datagen.check(ex, items)
        assert not any(n.lower() in ex["target"].lower() for n in names), ex["target"]
        if ex["task"] == "ask":
            assert fmt.ITEM_SLOT in ex["prompt"].split("<|ask|>")[1] or item.noun not in ex["prompt"].split("<|ask|>")[1]
    assert any(fmt.ITEM_SLOT in ex["target"] for ex in examples)


def test_format_3_refusals_use_the_slot_too():
    # The item's name has to become the slot in refusals as well ("A {item}? You can't be serious.").
    rows = datagen.refusal_examples("rainbow", load_traits(), random.Random(0), 10, prompt_format=3)
    assert all("rainbow" not in r["target"].lower() for r in rows)
    assert sum(fmt.ITEM_SLOT in r["target"] for r in rows) >= 5


def test_blind_collect_maps_letters_back_and_needs_most_probes(tmp_path, monkeypatch):
    import json

    from sellanything_ml.datagen import blind
    monkeypatch.setattr(blind, "ROUND_DIR", tmp_path / "r2")
    keys = [{"key": "k0", "chunk": "chunk_001", "answer_id": "req#0", "order": ["mixed", "pedigree"], "probe": False}]
    keys += [{"key": f"p{i}", "chunk": "chunk_001", "answer_id": f"x#{i}", "order": ["a", "b"], "probe": True}
             for i in range(blind.PROBES)]
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "keys.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keys))
    out = tmp_path / "out"
    out.mkdir()
    lines = ["k0\tB\thigh\tRagdoll is a pedigree breed"] + [f"p{i}\tnone\thigh\tabout age not size" for i in range(blind.PROBES)]
    (out / "chunk_001.out.tsv").write_text("\n".join(lines) + "\n")
    blind.collect(out, tmp_path / "keys", "m", "d")
    (row,) = [json.loads(x) for x in (tmp_path / "r2" / "batch_001.jsonl").read_text().splitlines()]
    assert row["answer_id"] == "req#0" and row["maps_to"] == "pedigree" and row["confidence"] == "high"
    # too many probes answered with a letter: the whole chunk is not trusted
    (tmp_path / "r2" / "batch_001.jsonl").unlink()
    for i in range(blind.PROBES - blind.PROBE_PASS + 1):
        lines[1 + i] = f"p{i}\tA\thigh\tlooks like option A"
    (out / "chunk_001.out.tsv").write_text("\n".join(lines) + "\n")
    stats = blind.collect(out, tmp_path / "keys", "m", "d")
    assert not (tmp_path / "r2" / "batch_001.jsonl").exists() and "chunk_001" in stats["rejected chunks"]
