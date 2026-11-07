"""One buyer model playing one cat sale in several ways of picking words (buyers/variants.py), with a scripted
stand-in whose line depends on the way."""

import json

from sellanything_ml.buyers import cat_market as C
from sellanything_ml.buyers.variants import WaysSale

CARDS = {p: [{"cares_about": ["calm"]}] for p in C.stories.PERSONAS}
WAYS = ["greedy", "topp", "topk", "topk2"]


class ByWay:
    """Answers each task with a line that names the way it was asked for, and records every call."""

    name, prompt_format = "buyer_gpt_b5", 5

    def __init__(self):
        self.calls = []

    @staticmethod
    def way(temperature, top_k, top_p, seed):
        if temperature == 0:
            return "greedy"
        if top_p:
            return "topp"
        return "topk2" if seed % 2 else "topk"

    def say(self, prompt, task, seed, temperature=None, top_k=None, top_p=None):
        self.calls.append((prompt, task, seed, temperature, top_k, top_p))
        way = self.way(temperature, top_k, top_p, seed)
        offer = {"greedy": 40, "topp": 50, "topk": 60, "topk2": 70}[way]
        return {"open": f"Hi from {way}. | $55", "ask": f"Is she calm ({way})?",
                "react": f"{'-' if offer < 55 else '+'} | I heard you ({way}). | ${offer}"}[task]


def test_every_way_answers_the_same_prompt_and_the_stories_follow_the_game_way(tmp_path, monkeypatch):
    gpt = ByWay()
    monkeypatch.setattr(C.random.Random, "randrange", lambda self, n: 1000)   # an even seed: topk2 is seed + 1
    sale = WaysSale.start(gpt, WAYS, 2, seed=3, out_dir=tmp_path, cards=CARDS)
    st = sale.state()
    assert st["compare"] == [f"b5:{w}" for w in WAYS] and st["lead"] == "b5:topk"
    assert st["question"]["text"] == "Is she calm (topk)?" and st["question"]["by"] == "b5:topk"
    assert st["question"]["ways"]["b5:greedy"] == "Is she calm (greedy)?"
    name = st["buyers"][0]["name"]
    assert {st["openings"][f"b5:{w}"][name]["says"] for w in WAYS} == {f"Hi from {w}." for w in WAYS}
    sale.answer_text("she is calm")
    turn = sale.state()["turns"][0]
    assert {turn["reactions"][f"b5:{w}"][name]["offer"] for w in WAYS} == {40, 50, 60, 70}
    assert {turn["reactions"][f"b5:{w}"][name]["old"] for w in WAYS} == {55}      # one shared offer before
    # the four ways of one line had the same prompt; only the game's way entered the stories
    reacts = [c for c in gpt.calls if c[1] == "react"]
    assert len({c[0] for c in reacts[:4]}) == 1 and [c[2] for c in reacts[:4]] == [1000, 1000, 1000, 1001]
    later = [c[0] for c in gpt.calls if c[1] == "ask"][-1]
    assert "I heard you (topk)." in later and "(greedy)" not in later
    assert sale.market.offers[name] == 60


def test_the_ways_sale_is_saved_with_the_other_questions(tmp_path):
    sale = WaysSale.start(ByWay(), WAYS, 2, seed=8, out_dir=tmp_path, cards=CARDS)
    sale.answer_text("yes")
    saved = json.loads(sale._path(".json").read_text())
    assert saved["model_stats"]["b5:greedy"]["unreadable"] == 0 and len(saved["turns"]) == 1
    md = sale._path(".md").read_text()
    assert "_b5_ways" in sale._path(".md").name and "same prompt" in md
    assert "b5:greedy: Is she calm (greedy)?" in md


def test_the_ways_must_include_the_game_way():
    for ways in (["greedy", "topp"], ["topk", "topk"], ["topk", "nope"]):
        try:
            WaysSale.start(ByWay(), ways, 2, seed=1, save=False, cards=CARDS)
        except ValueError:
            continue
        raise AssertionError(ways)
