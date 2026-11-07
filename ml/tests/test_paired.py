"""Two buyer models playing one cat sale side by side (buyers/paired.py), with scripted stand-ins for the models."""

import json

from sellanything_ml.buyers import cat_market as C
from sellanything_ml.buyers.paired import PairedSale

CARDS = {p: [{"cares_about": ["calm"]}] for p in C.stories.PERSONAS}


class Scripted:
    """A model that answers every task with a fixed line and records its prompts."""

    prompt_format = 5

    def __init__(self, name: str, react: str, ask: str):
        self.name, self.react, self.ask, self.prompts = name, react, ask, []

    def say(self, prompt: str, task: str, seed: int) -> str:
        self.prompts.append(prompt)
        return {"open": "Hi there. | $50", "ask": self.ask, "react": self.react}[task]


def models():
    return (Scripted("buyer_gpt_a", "+ | Calm is good for me. Up to $60. | $60", "Is she calm?"),
            Scripted("buyer_gpt_b", "- | Calm is not for me. Down to $40. | $40", "Is she loud?"))


def test_both_models_hear_the_same_question_and_answer_and_keep_their_own_offers(tmp_path):
    a, b = models()
    sale = PairedSale.start([a, b], 3, seed=5, out_dir=tmp_path, cards=CARDS)
    st = sale.state()
    assert st["compare"] == ["a", "b"] and len(st["buyers"]) == 3
    assert st["question"]["by"] == "a" and st["question"]["text"] == "Is she calm?"
    asker = st["question"]["asker"]
    # the question model a wrote is in model b's prompts too, in the same buyer's name
    lines, heard = sale.answer_text("she is very calm")
    assert heard["answer"] == "she is very calm" and "You: she is very calm" in lines
    reacts_b = [p for p in b.prompts if p.endswith("<|react|>")]
    assert all("asked: Is she calm?" in p for p in reacts_b)          # the asker reads "You asked", others the name
    assert sum(f"{asker} asked: Is she calm?" in p for p in reacts_b) == 2
    assert all("Seller: she is very calm" in p for p in a.prompts + b.prompts if p.endswith("<|react|>"))
    turn = sale.state()["turns"][0]
    assert {r["offer"] for r in turn["reactions"]["a"].values()} == {60}
    assert {r["offer"] for r in turn["reactions"]["b"].values()} == {40}
    assert sale.state()["question"]["by"] == "b"                      # the models take turns writing questions
    assert {b_["offers"]["a"] for b_ in sale.state()["buyers"]} == {60}


def test_the_sale_is_saved_after_every_turn_and_ends_when_everyone_asked(tmp_path):
    a, b = models()
    sale = PairedSale.start([a, b], 2, seed=6, out_dir=tmp_path, cards=CARDS)
    sale.answer_text("yes")
    sale.answer_text("no")
    st = sale.state()
    assert st["finished"] and st["question"] is None and len(st["turns"]) == 2
    saved = json.loads(sale._path(".json").read_text())
    assert len(saved["turns"]) == 2
    md = sale._path(".md").read_text()
    assert "## Turn 2" in md and "More logical" in md and "Down to $40" in md


def test_a_paired_sale_needs_two_different_models():
    a, _ = models()
    try:
        PairedSale.start([a, a], 2, seed=1, cards=CARDS, save=False)
    except ValueError as e:
        assert "two different models" in str(e)
    else:
        raise AssertionError("a model compared with itself")
