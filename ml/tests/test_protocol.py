from sellanything_ml.server import PROTOCOL_VERSION, handle_message


class FakeClassifier:
    def predict(self, strokes, k=3):
        return [{"label": "apple", "prob": 0.9}, {"label": "pear", "prob": 0.05}][:k]


class FakeModels:
    classifier = FakeClassifier()


class NoModels:
    classifier = None


def test_ping():
    assert handle_message({"type": "ping"}) == {"type": "pong", "protocol": PROTOCOL_VERSION}


def test_classify_returns_top_k():
    reply = handle_message({"type": "classify", "strokes": [[[0, 10], [0, 10]]]}, FakeModels())
    assert reply["type"] == "classify_result"
    assert reply["top_k"][0]["label"] == "apple"


def test_classify_echoes_partial_only_when_asked():
    msg = {"type": "classify", "strokes": [[[0, 10], [0, 10]]]}
    assert "partial" not in handle_message(msg, FakeModels())
    assert handle_message({**msg, "partial": True}, FakeModels())["partial"] is True
    assert handle_message({**msg, "partial": True}, NoModels())["partial"] is True   # errors too
    assert handle_message({"type": "classify", "strokes": [], "partial": True}, FakeModels())["type"] == "error"


def test_classify_rejects_bad_strokes():
    for strokes in ([], [[[0, 1], [0]]], [[["a"], [1]]], "nope"):
        assert handle_message({"type": "classify", "strokes": strokes}, FakeModels())["type"] == "error"


def test_classify_without_a_model_says_how_to_get_one():
    reply = handle_message({"type": "classify", "strokes": [[[0, 10], [0, 10]]]}, NoModels())
    assert reply["type"] == "error" and "train" in reply["message"]


def test_chat_reply_shape():
    reply = handle_message({"type": "chat", "session": "s1", "text": "I want to sell a bike"})
    assert reply["type"] == "chat_reply"
    assert "offer" in reply


def test_unknown_type():
    assert handle_message({"type": "nope"})["type"] == "error"


class Sessions:
    classifier = None

    def __init__(self):
        self.rounds = {}


def test_round_plays_start_to_sale():
    models = Sessions()
    reply = handle_message({"type": "round_start", "session": "s", "item": "bicycle", "seed": 3}, models)
    assert reply["type"] == "round_state"
    state = reply["state"]
    assert len(state["buyers"]) == 4 and state["question"] and not state["finished"]
    assert reply["new_lines"][-1].startswith(state["question"]["asker"] + ":")
    while not state["finished"]:
        option = state["question"]["options"][0]["id"]
        reply = handle_message({"type": "round_answer", "session": "s", "option": option}, models)
        assert reply["type"] == "round_state" and reply["new_lines"][0].startswith("You:")
        state = reply["state"]
    present = [b for b in state["buyers"] if b["present"]]
    if present:
        sold = handle_message({"type": "round_sell", "session": "s", "buyer": present[0]["name"]}, models)
        assert sold["type"] == "round_result" and sold["price"] == present[0]["offer"]


def test_refuse_item_round_is_over_immediately():
    reply = handle_message({"type": "round_start", "session": "t", "item": "sun", "seed": 1}, Sessions())
    assert reply["state"]["finished"] and reply["state"]["label"] == "refuse"
    assert all("sun" in line.lower() or "pass" in line for line in reply["new_lines"])


def test_round_errors_are_messages_not_crashes():
    models = Sessions()
    assert handle_message({"type": "round_answer", "session": "x", "option": "a"}, models)["type"] == "error"
    assert handle_message({"type": "round_start", "session": "x", "item": "unicorn"}, models)["type"] == "error"
    handle_message({"type": "round_start", "session": "x", "item": "dog", "seed": 1}, models)
    bad = handle_message({"type": "round_answer", "session": "x", "option": "nope"}, models)
    assert bad["type"] == "error" and "nope" in bad["message"]
    early = handle_message({"type": "round_sell", "session": "x", "buyer": "Maya"}, models)
    assert early["type"] == "error"


def test_old_rounds_are_dropped_past_the_limit():
    from sellanything_ml.server import MAX_ROUNDS
    models = Sessions()
    for i in range(MAX_ROUNDS + 3):
        handle_message({"type": "round_start", "session": f"s{i}", "item": "sun", "seed": i}, models)
    assert len(models.rounds) == MAX_ROUNDS
    assert "s0" not in models.rounds and f"s{MAX_ROUNDS + 2}" in models.rounds


def test_a_bug_in_a_handler_becomes_an_error_reply(monkeypatch, capsys):
    from sellanything_ml import server

    def broken(msg, models):
        raise RuntimeError("boom")

    monkeypatch.setitem(server.HANDLERS, "ping", broken)
    reply = server._reply_to(b'{"type": "ping"}\n', server.Models())
    assert reply["type"] == "error" and "RuntimeError: boom" in reply["message"]
    assert "Traceback" in capsys.readouterr().err


def test_fixed_sketch_answers_without_a_model():
    from sellanything_ml.server import Models
    reply = handle_message({"type": "classify", "strokes": [[[0, 1], [0, 1]]]}, Models(sketch_label="bicycle"))
    assert reply == {"type": "classify_result", "top_k": [{"label": "bicycle", "prob": 1.0}]}


def test_moods_follow_the_round():
    models = Sessions()
    state = handle_message({"type": "round_start", "session": "m", "item": "bicycle", "seed": 3}, models)["state"]
    moods = {b["name"]: b["mood"] for b in state["buyers"]}
    assert moods[state["question"]["asker"]] == "talk"
    assert set(moods.values()) <= {"idle", "talk"}
    seen = set()
    while not state["finished"]:
        option = state["question"]["options"][0]["id"]
        state = handle_message({"type": "round_answer", "session": "m", "option": option}, models)["state"]
        seen |= {b["mood"] for b in state["buyers"]}
        assert all(b["mood"] == "left" for b in state["buyers"] if not b["present"])
    assert seen <= {"idle", "talk", "happy", "unhappy", "left"} and seen & {"happy", "unhappy"}
    buyer = next(b["name"] for b in state["buyers"] if b["present"])
    sold = handle_message({"type": "round_sell", "session": "m", "buyer": buyer}, models)
    # The game only sees the reply, so the winner's mood has to be in it.
    assert {b["name"]: b["mood"] for b in sold["state"]["buyers"]}[buyer] == "won"


def test_play_log_records_requests_and_replies(tmp_path):
    import json

    from sellanything_ml.server import PlayLog
    log = PlayLog(tmp_path)
    log.write(b'{"type": "ping"}\n', {"type": "pong", "protocol": 1})
    log.write(b"not json\n", {"type": "error", "message": "invalid JSON"})
    (day,) = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in day.read_text().splitlines()]
    assert rows[0]["request"] == {"type": "ping"} and rows[0]["reply"]["type"] == "pong"
    assert rows[1]["request"] == "not json\n" and "t" in rows[1]
    PlayLog(None).write(b"{}", {})            # disabled: writes nothing, no error


def test_cards_explain_offers_with_traits_and_reactions():
    models = Sessions()
    state = handle_message({"type": "round_start", "session": "c", "item": "cat", "seed": 4}, models)["state"]
    assert all(b["traits"] and all(isinstance(t, str) for t in b["traits"]) for b in state["buyers"])
    seen_reaction = False
    while not state["finished"]:
        option = state["question"]["options"][0]["id"]
        before = {b["name"]: b["offer"] for b in state["buyers"]}
        state = handle_message({"type": "round_answer", "session": "c", "option": option}, models)["state"]
        for b in state["buyers"]:
            change = b["change"]
            assert change is not None and abs(change["amount"] - ((b["offer"] or 0) - before[b["name"]])) < 1e-6
            said = change.get("reaction")
            if said and said not in ("outbid the others", "walked away"):
                seen_reaction = True
                # the card and the sprite tell the same story
                assert b["mood"] == ("happy" if change["liked"] else "unhappy"), (b["name"], change, b["mood"])
    assert seen_reaction


def test_round_answer_takes_typed_text():
    models = Sessions()
    state = handle_message({"type": "round_start", "session": "t", "item": "cat", "seed": 3}, models)["state"]
    unclear = handle_message({"type": "round_answer", "session": "t", "text": "asdf"}, models)
    assert unclear["type"] == "round_state" and unclear["heard"] is None
    assert unclear["state"]["question"] == state["question"]
    attribute = state["question"]["attribute"]
    typed = {"breed": "just a regular mixed cat", "age": "shes like 12", "litter": "yep",
             "temperament": "very shy", "health": "fully vaccinated"}[attribute]
    reply = handle_message({"type": "round_answer", "session": "t", "text": typed}, models)
    assert reply["new_lines"][0] == f"You: {typed}"
    assert reply["heard"]["option"] in {o["id"] for o in state["question"]["options"]}
    assert handle_message({"type": "round_answer", "session": "t", "text": " "}, models)["type"] == "error"


def test_draw_takes_names_typos_plurals_and_says_when_it_cannot():
    from sellanything_ml.server import Models
    models = Models(sketch_label="bicycle")          # test drawer: real name matching, fixed drawing
    for text, label in [("bicycle", "bicycle"), ("a bicyle", "bicycle"), ("Dogs", "dog"),
                        ("eiffel tower", "The Eiffel Tower")]:
        reply = handle_message({"type": "draw", "text": text}, models)
        assert reply["ok"] and reply["label"] == label, text
        assert reply["strokes"] and all(len(xs) == len(ys) for xs, ys in reply["strokes"])
    reply = handle_message({"type": "draw", "text": "unicorn spaceship"}, models)
    assert reply["type"] == "draw_result" and not reply["ok"] and "don't know" in reply["message"]


def test_draw_with_no_text_picks_a_random_class_reproducibly():
    from sellanything_ml.server import Models
    models = Models(sketch_label="bicycle")
    a = handle_message({"type": "draw", "text": "", "seed": 3}, models)
    b = handle_message({"type": "draw", "random": True, "seed": 3}, models)
    assert a["ok"] and a["label"] == b["label"]
    assert handle_message({"type": "draw", "text": 5}, models)["type"] == "error"


def test_live_guesses_use_the_partial_model_and_the_final_guess_never_does():
    class Fixed:
        def __init__(self, label):
            self.label = label

        def predict(self, strokes, k=3):
            return [{"label": self.label, "prob": 1.0}]

    class Both:
        classifier, partial, rounds = Fixed("final"), Fixed("live"), {}

    msg = {"type": "classify", "strokes": [[[0, 10], [0, 10]]]}
    assert handle_message({**msg, "partial": True}, Both())["top_k"][0]["label"] == "live"
    assert handle_message(msg, Both())["top_k"][0]["label"] == "final"
    Both.partial = None                                     # nothing exported: live guesses fall back
    assert handle_message({**msg, "partial": True}, Both())["top_k"][0]["label"] == "final"


def test_the_server_routes_cat_sales_to_the_buyer_gpt(monkeypatch):
    from sellanything_ml.buyers import cat_market

    class ScriptedGPT:
        prompt_format = 5
        name = "buyer_gpt_scripted"

        def say(self, prompt, task, seed):
            return {"open": "Hi there. | $50", "ask": "Is she calm?", "react": "+ | Calm is good for me. Up to $60. | $60",
                    "imagine": "chair, sun, book | She'd nap on my chair in the sun while I read my book."}[task]

    class Models:
        rounds: dict = {}
        cat_buyers = ScriptedGPT()
        voice = None

    cards = {p: [{"cares_about": ["calm"]}] for p in cat_market.stories.PERSONAS}
    monkeypatch.setattr(cat_market, "load_cards", lambda: cards)
    reply = handle_message({"type": "round_start", "session": "s", "item": "cat", "buyers": 2, "seed": 3}, Models())
    assert reply["type"] == "round_state" and reply["state"]["voice"] == "buyer_gpt_scripted"
    reply = handle_message({"type": "round_answer", "session": "s", "text": "She is calm."}, Models())
    assert reply["heard"]["answer"] == "She is calm."
    assert handle_message({"type": "round_answer", "session": "s", "option": "x"}, Models())["type"] == "error"


def test_a_compare_sale_plays_two_named_models_side_by_side(monkeypatch, tmp_path):
    from test_paired import CARDS, models

    from sellanything_ml.buyers import cat_market, paired

    a, b = models()

    class Models:
        rounds: dict = {}
        cat_buyers = a
        voice = None

        def cat_model(self, name):
            return {"a": a, "b": b}.get(name)

    monkeypatch.setattr(cat_market, "load_cards", lambda: CARDS)
    monkeypatch.setattr(paired, "COMPARE_DIR", tmp_path)
    start = {"type": "round_start", "session": "c", "item": "cat", "buyers": 2, "seed": 4}
    reply = handle_message({**start, "compare": ["a", "b"]}, Models())
    assert reply["type"] == "round_state" and reply["state"]["compare"] == ["a", "b"]
    reply = handle_message({"type": "round_answer", "session": "c", "text": "yes"}, Models())
    assert len(reply["state"]["turns"]) == 1 and reply["heard"]["answer"] == "yes"
    assert handle_message({"type": "round_sell", "session": "c", "buyer": "x"}, Models())["type"] == "error"
    assert handle_message({**start, "compare": ["a", "zz"]}, Models())["type"] == "error"
    assert handle_message({**start, "compare": "a"}, Models())["type"] == "error"
    assert list(tmp_path.glob("*_seed4.md"))                                 # saved for study


def test_a_compare_sale_plays_one_model_in_several_ways_of_picking_words(monkeypatch, tmp_path):
    from test_variants import CARDS, ByWay

    from sellanything_ml.buyers import cat_market, paired

    class Models:
        rounds: dict = {}
        voice = None

        def cat_model(self, name):
            return ByWay() if name == "b5" else None

    monkeypatch.setattr(cat_market, "load_cards", lambda: CARDS)
    monkeypatch.setattr(paired, "COMPARE_DIR", tmp_path)
    start = {"type": "round_start", "session": "w", "item": "cat", "buyers": 2, "seed": 4}
    ways = ["b5:greedy", "b5:topp", "b5:topk", "b5:topk2"]
    reply = handle_message({**start, "compare": ways}, Models())
    assert reply["type"] == "round_state" and reply["state"]["compare"] == ways
    assert reply["state"]["labels"]["b5:topk"] == "top-k 50 (the game)"
    reply = handle_message({"type": "round_answer", "session": "w", "text": "yes"}, Models())
    assert set(reply["state"]["turns"][0]["reactions"]) == set(ways)
    for bad in (["b5:greedy", "b4:topk"], ["b5:topk", "b5"], ["b5:greedy", "b5:topp"], ["zz:topk"]):
        assert handle_message({**start, "compare": bad}, Models())["type"] == "error"
    assert list(tmp_path.glob("*_seed4_b5_ways.md"))


def test_the_player_buys_a_cat_from_the_model_seller():
    from test_cat_market import CARDS, ScriptedSeller

    from sellanything_ml.buyers import cat_market

    class SellerModels:
        rounds: dict = {}
        cat_seller = ScriptedSeller()

    real = cat_market.load_cards
    cat_market.load_cards = lambda: CARDS
    try:
        m = SellerModels()
        start = {"type": "round_start", "session": "b", "item": "cat", "buyers": 2, "seed": 9, "role": "buyer",
                 "name": "Alina"}
        state = handle_message(start, m)["state"]
        assert state["waiting"] == "open" and state["player"] == "Alina" and state["asking"] > 0
        step = {"type": "round_buy", "session": "b"}
        assert handle_message({**step, "step": "ask", "text": "Hi?"}, m)["type"] == "error"       # not yet
        state = handle_message({**step, "step": "open", "text": "Hello!", "offer": 40}, m)["state"]
        while state["waiting"] in ("react", "ask"):
            msg = {"step": "react", "text": "OK.", "offer": 50} if state["waiting"] == "react" else \
                {"step": "ask", "text": "Is she healthy?"}
            state = handle_message({**step, **msg}, m)["state"]
        assert state["waiting"] == "reply"
        reply = handle_message({**step, "step": "reply", "text": "Deal.", "accept": True}, m)
        assert reply["state"]["finished"] and reply["state"]["sold"]["deals"]["Alina"] == 70
        assert handle_message({**start, "role": "watcher"}, m)["type"] == "error"
    finally:
        cat_market.load_cards = real
