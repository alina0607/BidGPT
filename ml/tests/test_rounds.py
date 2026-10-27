"""The game's selling round for items other than cats (buyers/rounds.py): typed answers and risky items."""

import pytest

from sellanything_ml.buyers import phrases
from sellanything_ml.buyers.rounds import RoundSession
from sellanything_ml.buyers.schema import load_traits


def test_a_typed_answer_is_shown_as_typed_but_heard_as_its_option():
    rnd = RoundSession.start("cat", 4, seed=3)
    while rnd.question[1].name != "age":
        rnd.answer(rnd.question[1].options[0].id)
    lines, heard = rnd.answer_text("shes like 12")
    assert lines[0] == "You: shes like 12"
    assert heard == {"option": "senior", "answer": "She's eleven.", "rule": "number"}
    # the buyer model hears the option text it was trained on
    assert "You: She's eleven." in rnd.heard and "You: shes like 12" not in rnd.heard
    assert len(rnd.log) == len(rnd.sources) == len(rnd.heard)


def test_an_unclear_answer_gets_the_question_again():
    rnd = RoundSession.start("cat", 4, seed=3)
    asker, attr = rnd.question
    heard_before = list(rnd.heard)
    lines, heard = rnd.answer_text("asdf")
    assert heard is None and rnd.question == (asker, attr)
    assert lines[0] == "You: asdf" and lines[1].startswith(asker.name + ":") and attr.question in lines[1]
    assert rnd.heard == heard_before            # the model never sees the confusion
    assert len(rnd.log) == len(rnd.sources)
    with pytest.raises(ValueError):
        rnd.answer_text("   ")


def test_risky_items_lose_the_wary_buyers_only():
    traits = load_traits()
    wary = {k for k, t in traits.traits.items() if t.refuses_risky}
    assert wary == {"parent", "retiree", "apartment_dweller"}
    left_somewhere = stayed_somewhere = False
    for seed in range(30):
        s = RoundSession.start("tiger", 4, seed=seed)
        assert s.label == "risky"
        for b in s.round.buyers:
            refuses = bool(wary & set(b.traits))
            assert b.present != refuses, (seed, b.name, b.traits)
            left_somewhere |= refuses
            stayed_somewhere |= not refuses
            if refuses:      # says why, in terms of their own trait, before any question
                line = next(line for line in s.log if line.startswith(b.name + ":"))
                assert any(line.endswith(p) for t in wary & set(b.traits) for p in phrases.RISKY_REFUSE[t])
    assert left_somewhere and stayed_somewhere
