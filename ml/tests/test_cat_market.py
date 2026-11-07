"""The game's cat sale run by the buyer GPT (buyers/cat_market.py), with a scripted stand-in for the model."""

from sellanything_ml.buyers import cat_market as C
from sellanything_ml.buyers.format import ASK, IMAGINE, OPEN, REACT

CARDS = {p: [{"cares_about": ["calm"], "budget": 80}] for p in C.stories.PERSONAS}


class ScriptedGPT:
    """Answers each task with a fixed line in the trained shape, and records the prompts it was given."""

    prompt_format = 5
    name = "buyer_gpt_scripted"

    def __init__(self, react: str = "+ | Calm is good for me. Up to $60. | $60"):
        self.prompts: list[str] = []
        self.react = react

    def say(self, prompt: str, task: str, seed: int) -> str:
        self.prompts.append(prompt)
        return {"open": "Hi there. | $50", "ask": "Is she calm?", "react": self.react,
                "imagine": "chair, sun, book | She'd nap on my chair in the sun while I read my book."}[task]


def test_parse_reads_the_trained_shapes_and_nothing_else():
    assert C.parse("open", "Hello. | $55") == {"says": "Hello.", "offer": 55}
    assert C.parse("react", "- | No papers rules her out. | out") == {"sign": "-", "says": "No papers rules her out.",
                                                                     "offer": 0}
    assert C.parse("react", "+ | Nice.") is None and C.parse("open", "Hello") is None
    assert C.parse("imagine", "dog, sun, bench | text")["things"] == ["dog", "sun", "bench"]
    assert C.parse("imagine", "<|thing:dog|><|thing:sun|><|thing:alarm clock|> | text")["things"] == \
        ["dog", "sun", "alarm clock"]                                           # format 6


def test_format_9_reads_the_list_after_the_line():
    assert C.parse("imagine", "My dog | my sun | sun, dog, bench", 9)["says"] == "My dog | my sun"   # list is last
    assert C.parse("imagine", "Nap in the sun by my dog. | sun, dog, bench", 9) == \
        {"things": ["sun", "dog", "bench"], "says": "Nap in the sun by my dog."}
    assert C.parse("imagine", "dog, sun, bench | text", 5) == C.parse("imagine", "dog, sun, bench | text")
    assert C.parse("imagine", "Nap in the sun.", 9) is None                   # no list at all
    assert C.max_new("imagine", 9) == C.LINE_FIRST_IMAGINE and C.max_new("imagine", 5) == C.MAX_NEW["imagine"]
    assert C.max_new("react", 9) == C.MAX_NEW["react"]


def test_a_sale_runs_open_ask_react_and_sell():
    gpt = ScriptedGPT()
    m = C.CatMarket.start(gpt, 3, seed=1, cards=CARDS)
    assert m.question is not None and m.state()["question"]["options"] == []
    assert all(OPEN in p for p in gpt.prompts[:3]) and ASK in gpt.prompts[3]
    for _ in range(3):
        lines, heard = m.answer_text("She is very calm.")
        assert heard["answer"] == "She is very calm." and lines[0] == "You: She is very calm."
    assert m.question is None and m.state()["finished"]
    reacts = [p for p in gpt.prompts if p.endswith(REACT)]
    assert len(reacts) == 9 and "Seller: She is very calm." in reacts[0]
    best = max(m.present(), key=m.offers.get)
    assert m.sell(best)["price"] == 60 and m.moods[best] == "won"
    objects = m.scene(best)
    assert [o["label"] for o in objects] == ["cat", "chair", "sun", "book"] and gpt.prompts[-1].endswith(IMAGINE)
    assert m.scene(best) is objects                      # written once


def test_offers_are_kept_as_written_and_unreadable_lines_are_counted_not_replaced():
    gpt = ScriptedGPT(react="+ | Up to $500. | $500")
    m = C.CatMarket.start(gpt, 2, seed=2, cards=CARDS)
    m.answer_text("Yes.")
    assert all(m.offers[n] == 500 for n in m.present())          # no spending limit (owner, 2026-10-12)
    assert "budget" not in gpt.prompts[0]
    m = C.CatMarket.start(ScriptedGPT(react="I like her"), 2, seed=2, cards=CARDS)
    m.answer_text("Yes.")
    assert all(m.offers[n] == 50 for n in m.cards) and m.stats["unreadable"] == 2
    assert m.sources.count("model_unreadable") == 2


def test_a_format_4_model_only_gets_cards_with_a_budget():
    gpt = ScriptedGPT()
    gpt.prompt_format = 4
    pool = {p: [{"cares_about": ["calm"]}, {"cares_about": ["quiet"], "budget": 70}] for p in C.stories.PERSONAS}
    m = C.CatMarket.start(gpt, 3, seed=4, cards=pool)
    assert all(c.get("budget") == 70 for c in m.cards.values()) and "budget $70" in gpt.prompts[0]


class ScriptedSeller(ScriptedGPT):
    """A format-8 stand-in: the buyer lines above, plus the seller's answer and decision and a buyer's reply."""

    prompt_format = 8

    def __init__(self, decide: str = "counter | $70 | $50 is under my $75. Could you do $70?",
                 reply: str = "accept | $70 is fair; I'll pay it."):
        super().__init__()
        self.decide, self.reply = decide, reply

    def say(self, prompt: str, task: str, seed: int) -> str:
        if task in ("answer", "decide", "reply"):
            self.prompts.append(prompt)
            return {"answer": "She's very calm, yes.", "decide": self.decide, "reply": self.reply}[task]
        return super().say(prompt, task, seed)


def test_parse_reads_the_seller_lines():
    assert C.parse("decide", "counter | $70 | Could you do $70?") == {"action": "counter", "price": 70,
                                                                      "says": "Could you do $70?"}
    assert C.parse("decide", "accept | - | Deal.") == {"action": "accept", "price": None, "says": "Deal."}
    assert C.parse("decide", "counter | - | No price.") is None and C.parse("decide", "accept | $70 | Hm.") is None
    assert C.parse("reply", "walk | Too much for me.") == {"action": "walk", "says": "Too much for me."}
    assert C.parse("answer", "She's calm.") == {"says": "She's calm."} and C.parse("answer", "a | b") is None
    assert C.decide_problem({"action": "accept"}, 50, 75) and not C.decide_problem({"action": "accept"}, 65, 75)
    assert C.decide_problem({"action": "counter", "price": 72}, 50, 75)
    assert not C.decide_problem({"action": "counter", "price": 70}, 50, 75)


def test_the_model_sells_and_buys_by_itself():
    gpt = ScriptedSeller()
    sheet = C.seller_sheet(2500)
    m = C.CatMarket.start(gpt, 3, seed=1, cards=CARDS, seller=sheet)
    assert m.sold is not None and m.state()["finished"] and m.waiting is None
    answers = [p for p in gpt.prompts if p.endswith(C.TASK_TOKENS["answer"])]
    assert len(answers) == 3 and f"asking ${sheet['asking']}" in answers[0]
    # the seller hears what the training sales gave it: openings, questions, its own answers, reactions
    assert "asked: Is she calm?" in answers[0] and "You: She's very calm, yes." in answers[1]
    assert "Seller: She's very calm, yes." in [p for p in gpt.prompts if p.endswith(REACT)][0]
    decides = [p for p in gpt.prompts if p.endswith(C.TASK_TOKENS["decide"])]
    assert len(decides) == 3 and "'s final offer: $60" in decides[0]
    assert all(d["action"] == "counter" and d["reply"]["action"] == "accept" for d in m.decisions.values())
    assert m.sold["price"] == 70 and m.sold["buyer"] == next(iter(m.cards))     # equal prices: the first seat


def test_a_refusing_seller_sells_to_nobody_and_the_human_seller_path_is_closed():
    m = C.CatMarket.start(ScriptedSeller(decide="refuse | - | Not this time."), 2, seed=3, cards=CARDS,
                          seller=C.seller_sheet(2501))
    assert m.sold["buyer"] is None and m.sold["deals"] == {}
    for act in (lambda: m.answer_text("Yes."), lambda: m.sell("Ben")):
        try:
            act()
            raise AssertionError("a model seller's sale took a human seller's step")
        except ValueError:
            pass


def test_the_player_buys_against_the_model():
    gpt = ScriptedSeller()
    m = C.CatMarket.start(gpt, 2, seed=4, cards=CARDS, seller=C.seller_sheet(2502), player="Alina")
    assert m.waiting == "open" and "Alina" not in [p.split("<|persona|>")[1][:5] for p in gpt.prompts]
    m.player_open("Hello, I'd like a calm cat.", 40)
    assert m.waiting == "react"                       # the first model buyer asked, the seller answered
    for _ in range(2):                                # after both model buyers' questions
        m.player_react("Good to hear.", 45)
    assert m.waiting == "ask"
    m.player_ask("Does she scratch?")
    assert "Alina asked: Does she scratch?" in m.heard() and m.waiting == "react"
    m.player_react("Fine.", 50)
    assert m.waiting == "reply" and m.decisions["Alina"]["action"] == "counter"
    m.player_reply(True, "Deal at $70.")
    assert m.sold["deals"]["Alina"] == 70 and m.state()["waiting"] is None
    try:
        m.player_react("Again.", 60)
        raise AssertionError("the player reacted after the sale closed")
    except ValueError:
        pass


def test_only_a_format_8_model_can_sell():
    try:
        C.CatMarket.start(ScriptedGPT(), 2, seed=5, cards=CARDS, seller=C.seller_sheet(2503))
        raise AssertionError("a format-5 model was made the seller")
    except ValueError:
        pass


def test_a_format_10_model_hears_the_older_answers():
    for prompt_format, kept in ((5, False), (10, True)):
        gpt = ScriptedGPT()
        gpt.prompt_format = prompt_format
        m = C.CatMarket.start(gpt, 4, seed=1, cards=CARDS)
        for k in range(4):
            m.answer_text(f"Answer {k}.")
        last = [p for p in gpt.prompts if p.endswith(REACT)][-1]
        assert ("-> Answer 0." in last) is kept and "Seller: Answer 3." in last
