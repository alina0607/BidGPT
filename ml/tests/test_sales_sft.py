"""B-plan SFT examples from cat sales (buyers/sales_sft.py)."""

from sellanything_ml.buyers import format as fmt
from sellanything_ml.buyers import sales_sft as S

CARDS = {"Ann": {"persona": "has two dogs", "cares_about": ["used to dogs"], "budget": 90},
         "Bo": {"persona": "is a student", "cares_about": ["cheap"], "budget": 40}}


def sale(rounds: int = 3) -> dict:
    events = [{"kind": "open", "name": "Ann", "offer": 60, "says": "Hi, I have dogs."},
              {"kind": "open", "name": "Bo", "offer": 30, "says": "Hey, money is tight."}]
    offers = {"Ann": 60, "Bo": 30}
    for k in range(1, rounds + 1):
        asker = "Ann" if k % 2 else "Bo"
        events += [{"kind": "ask", "round": k, "name": asker, "offer": 1, "says": f"Question {k}?"},
                   {"kind": "answer", "round": k, "to": asker, "q": f"Question {k}?", "says": f"Answer {k}."}]
        for n in ("Ann", "Bo"):
            new = offers[n] + 5
            events.append({"kind": "react", "name": n, "old": offers[n], "offer": new, "sign": "+",
                           "says": f"{n} likes answer {k}. Up to ${new}."})
            offers[n] = new
    events += [{"kind": "imagine", "name": "Ann", "things": ["dog", "bench", "sun"], "says": "My dog, a bench, sun."},
               {"kind": "sold", "name": "Ann", "price": offers["Ann"]}]
    return {"seed": 7, "cards": CARDS, "events": events}


def test_one_example_per_decision_with_the_offer_from_the_reactions():
    ex = S.examples(sale())
    assert [e["task"] for e in ex].count("react") == 6 and [e["task"] for e in ex].count("ask") == 3
    ask3 = [e for e in ex if e["task"] == "ask"][2]
    assert "offer now $70" in ask3["prompt"]                 # 60 + 5 + 5, not the stale offer in the ask event
    assert ex[0]["target"] == "Hi, I have dogs. | $60" + fmt.END and fmt.OPEN in ex[0]["prompt"]


def test_a_buyer_sees_only_their_own_reactions_and_the_last_k_rounds():
    ex = S.examples(sale(4))
    last_ann = [e for e in ex if e["task"] == "react" and e["name"] == "Ann"][-1]["prompt"]
    assert "Bo likes" not in last_ann and "Bo asked: Question 4?" in last_ann
    assert "Asked earlier: Question 1?\n" in last_ann                # rounds 2 and 3 in full, then round 4 so far
    assert "Answer 1." not in last_ann and "Answer 2." in last_ann and "Answer 4." in last_ann
    assert "You ($70 -> $75): Ann likes answer 3." in last_ann
    assert last_ann.endswith(fmt.REACT)


def test_react_and_imagine_targets():
    ex = S.examples(sale())
    react = [e for e in ex if e["task"] == "react"][0]
    assert react["target"] == "+ | Ann likes answer 1. Up to $65. | $65" + fmt.END
    imagine = [e for e in ex if e["task"] == "imagine"][0]
    assert imagine["target"].startswith("dog, bench, sun | ") and imagine["prompt"].endswith(fmt.IMAGINE)


def test_withdrawal_is_out_and_the_split_is_stable():
    assert S.money(0) == "out" and S.money(45) == "$45"
    assert [S.is_val(s) for s in range(2000, 2100)] == [S.is_val(s) for s in range(2000, 2100)]
    assert 3 <= sum(S.is_val(s) for s in range(2000, 2100)) <= 20


def test_only_format_4_shows_a_budget():
    c = CARDS["Ann"]
    assert "budget $90" in S.card("Ann", c, 60, prompt_format=4)
    assert "budget" not in S.card("Ann", c, 60) and S.card("Ann", c, 60).endswith("offer now $60")


def test_new_task_tokens_keep_their_ids():
    assert fmt.EXTRA_TOKENS.index(fmt.OPEN) == 6 and fmt.EXTRA_TOKENS.index(fmt.IMAGINE) == 7
    assert len(fmt.EXTRA_TOKENS[:fmt.N_BASE_EXTRA]) == 14


def test_format_6_puts_the_animal_the_persona_and_the_things_in_tokens():
    card = {"persona": "lives on a busy road", "cares_about": ["calm"]}
    sale6 = {"seed": 7, "cards": {"Ann": card}, "events": [
        {"kind": "open", "name": "Ann", "offer": 60, "says": "Hi."},
        {"kind": "imagine", "name": "Ann", "things": ["alarm clock", "pillow", "door"], "says": "My alarm clock."}]}
    opening, scene = S.examples(sale6, prompt_format=6)
    assert opening["prompt"].startswith(f"{fmt.ITEM}<|cat|>{fmt.END}{fmt.PERSONA}Ann, <|buyer:lives on a busy road|> |")
    assert scene["target"] == "<|thing:alarm clock|><|thing:pillow|><|thing:door|> | My alarm clock." + fmt.END
    assert S.examples(sale6)[1]["target"].startswith("alarm clock, pillow, door | ")     # format 5 unchanged


def test_format_6_tokens_come_after_the_older_ones():
    assert fmt.EXTRA_TOKENS[:fmt.N_BASE_EXTRA] == fmt.EXTRA_TOKENS[:14] and fmt.N_BASE_EXTRA == 14
    assert len(fmt.THING_TOKENS) == 344 and len(fmt.BUYER_TOKENS) == 17 and "cat" not in fmt.THING_TOKENS
    assert len(set(fmt.EXTRA_TOKENS)) == len(fmt.EXTRA_TOKENS)


def noted_sale(rounds: int = 4) -> dict:
    s = sale(rounds)
    for e in s["events"]:
        if e["kind"] == "answer":
            e["note"] = f"note {e['round']}"
    return s


def test_format_7_keeps_a_note_of_older_answers_and_learns_to_write_it():
    ex = S.examples(noted_sale(), prompt_format=7)
    notes = [e for e in ex if e["task"] == "note"]
    assert [n["target"] for n in notes] == [f"note {k}{fmt.END}" for k in range(1, 5)]
    assert all(n["prompt"].endswith(fmt.NOTE) and n["name"] == ("Ann" if k % 2 else "Bo")
               for k, n in enumerate(notes, 1))                            # the asker writes the note
    assert "Seller: Answer 2." in notes[1]["prompt"] and "note 2" not in notes[1]["prompt"]
    last_ann = [e for e in ex if e["task"] == "react" and e["name"] == "Ann"][-1]["prompt"]
    assert "Asked earlier: Question 1? -> note 1\n" in last_ann
    plain = S.examples(noted_sale())                                       # format 5: no notes, no note task
    assert "note" not in {e["task"] for e in plain} and "Asked earlier: Question 1?\n" in \
        [e for e in plain if e["task"] == "react" and e["name"] == "Ann"][-1]["prompt"]


def test_the_note_token_took_the_first_spare_slot():
    assert fmt.EXTRA_TOKENS.index(fmt.NOTE) == 8 and fmt.N_BASE_EXTRA == 14
    assert 7 not in fmt.TOKEN_FORMATS and 7 in fmt.NOTE_FORMATS


def test_format_8_adds_the_seller_tasks_only_in_format_8():
    import json
    import pytest
    from sellanything_ml.buyers.sales_sft import WRITTEN_SALES, examples
    from sellanything_ml.buyers.format import SELLER, DECIDE, ANSWER
    path = WRITTEN_SALES / "sales_b3_decide.jsonl"
    if not path.exists():
        pytest.skip("sales_b3_decide.jsonl not built")
    sale = json.loads(path.read_text().splitlines()[0])
    plain = {e["task"] for e in examples(sale, 5)}
    seller = examples(sale, 8)
    tasks = {e["task"] for e in seller}
    assert not plain & {"answer", "decide", "reply"}
    assert {"answer", "decide"} <= tasks
    n_decide = sum(e["kind"] == "decide" for e in sale["events"])
    assert sum(e["task"] == "decide" for e in seller) == n_decide
    for e in seller:
        if e["task"] in ("answer", "decide"):
            assert SELLER in e["prompt"] and (ANSWER if e["task"] == "answer" else DECIDE) in e["prompt"]
            assert f"asking ${sale['sale']['asking']}" in e["prompt"]


def test_format_9_writes_the_scene_line_first_in_the_order_the_line_names_things():
    ex = S.examples({**sale(), "events": [*sale()["events"][:-2],
                                          {"kind": "imagine", "name": "Ann", "things": ["sun", "alarm clock", "dog"],
                                           "says": "My dog wakes me before the clock, then naps in the sun."},
                                          sale()["events"][-1]]}, 9)
    imagine = [e for e in ex if e["task"] == "imagine"][0]
    assert imagine["target"] == "My dog wakes me before the clock, then naps in the sun. | dog, alarm clock, sun" + \
        fmt.END
    five = [e for e in S.examples(sale(), 5) if e["task"] == "imagine"][0]
    nine = [e for e in S.examples(sale(), 9) if e["task"] == "imagine"][0]
    assert nine["prompt"] == five["prompt"]                       # only the target changes
    assert [e for e in S.examples(sale(), 9) if e["task"] != "imagine"] == \
        [e for e in S.examples(sale(), 5) if e["task"] != "imagine"]


def test_mention_order_uses_the_rules_own_naming_test():
    from sellanything_ml.buyers.stories import mention_order
    assert mention_order(["picture frame", "vase", "sock"], "A vase, a frame, no sock") == ["vase", "picture frame",
                                                                                           "sock"]
    assert mention_order(["bed", "map", "key"], "My map and my bed") == ["map", "bed", "key"]   # unnamed: last


def test_format_10_keeps_the_seller_answers_of_older_rounds(monkeypatch):
    monkeypatch.setattr(S, "STORY_WORDS", 10**6)
    cat = {"age": "2 years", "breed": "mixed", "home": "indoor", "fixed": "spayed", "health": "healthy",
           "virus": "FIV negative", "company": [], "temperament": "calm", "quirks": []}
    ex = S.examples({**sale(4), "sale": {"cat": cat, "seller_style": "plain", "asking": 80}}, 10)
    last_ann = [e for e in ex if e["task"] == "react" and e["name"] == "Ann"][-1]["prompt"]
    assert "Asked earlier: Question 1? -> Answer 1.\n" in last_ann
    assert "Ann likes answer 1." not in last_ann           # the buyer's own old reaction stays out
    assert "You ($70 -> $75): Ann likes answer 3." in last_ann    # the last two rounds are as in format 8
    assert 10 in fmt.SELLER_FORMATS and 10 in fmt.ANSWER_FORMATS and 8 not in fmt.ANSWER_FORMATS


def test_format_10_shares_the_story_budget_and_drops_the_oldest_answers_first(monkeypatch):
    story = S.Story("Ann", answers=True)
    for k in range(1, 6):
        story.asked("Bo", f"Q{k}?")
        story.answered(" ".join(f"w{k}.{i}" for i in range(10)))
    full = story.text()
    assert "Q1? -> w1.0 w1.1" in full and full.count("w1.") == 10
    monkeypatch.setattr(S, "ANSWER_WORDS", 4)
    assert "Q1? -> w1.0 w1.1 w1.2 w1.3... / Q2?" in story.text()
    monkeypatch.setattr(S, "STORY_WORDS", S._words(story.text(k=2).splitlines()[1:]) + 16)
    monkeypatch.setattr(S, "MIN_ANSWER_WORDS", 2)
    older = story.text().splitlines()[0]
    assert older.startswith("Asked earlier: Q1? / Q2? -> w2.0") and "Q3? -> w3.0" in older   # Q1 lost its answer
    monkeypatch.setattr(S, "STORY_WORDS", 0)
    assert story.text().splitlines()[0] == "Asked earlier: Q1? / Q2? / Q3?"             # no room: as format 8


def test_formats_before_10_ignore_the_older_answers():
    story = S.Story("Ann")
    for k in range(1, 4):
        story.asked("Ann", f"Q{k}?")
        story.answered(f"A{k}.")
    assert story.text() == "Asked earlier: Q1?\nYou asked: Q2?\nSeller: A2.\nYou asked: Q3?\nSeller: A3."
