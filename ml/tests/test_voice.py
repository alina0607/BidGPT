"""The guardrail between the buyer model and the engine (buyers/voice.py), with canned outputs."""

from sellanything_ml.buyers.rounds import RoundSession
from sellanything_ml.buyers.format import BuyerView
from sellanything_ml.buyers.voice import ModelVoice, VoiceStats


class CannedVoice(ModelVoice):
    """ModelVoice with the model replaced by a fixed reply."""

    def __init__(self, reply: str):          # skip loading a checkpoint
        self.reply = reply
        self.stats = VoiceStats()

    def _complete(self, prompt: str, temperature: float = 0.7) -> str:
        return self.reply


def test_react_used_when_sign_agrees():
    v = CannedVoice("Maya: Love that.<|interest|>+1<|end|>")
    assert v.react(BuyerView("dog", ["size"], "Maya, who is retired.", []), "Maya", engine_delta=2) == "Love that."


def test_react_rejected_when_sign_disagrees_or_wrong_speaker():
    v = CannedVoice("Maya: Love that.<|interest|>+2<|end|>")
    assert v.react(BuyerView("dog", [], "p", []), "Maya", engine_delta=-1) is None
    assert v.react(BuyerView("dog", [], "p", []), "Leo", engine_delta=2) is None
    assert v.stats.fallback == {"react_disagrees": 1, "react_format": 1}


def test_ask_must_stay_on_topic():
    on = CannedVoice("Maya: So, is the dog potty trained yet?<|end|>")
    off = CannedVoice("Maya: Do you like pizza?<|end|>")
    q = "Is the dog potty trained?"
    assert on.ask(BuyerView("dog", [], "p", []), "Maya", "potty", q)
    assert off.ask(BuyerView("dog", [], "p", []), "Maya", "potty", q) is None


def test_round_falls_back_to_phrases_on_garbage():
    rnd = RoundSession.start("dog", 3, seed=1, voice=CannedVoice("no format at all"))
    assert rnd.question is not None and rnd.log            # still asked, via phrases
    while rnd.question:
        rnd.answer(rnd.question[1].options[0].id)
    assert rnd.state()["voice"] == "model" and rnd.voice.stats.used == 0


def test_ask_must_keep_the_item_name():
    # Seen on a held-out item: shares "what kind" with the real question, but the noun is garbled.
    garbled = CannedVoice("Felix: Quick question: what kind of kittery is it?<|end|>")
    right = CannedVoice("Felix: Quick question: what kind of knife is it?<|end|>")
    q = "What kind of knife is it?"
    assert garbled.ask(BuyerView("knife", [], "p", []), "Felix", "kind", q) is None
    assert right.ask(BuyerView("knife", [], "p", []), "Felix", "kind", q) == "Quick question: what kind of knife is it?"


def test_short_questions_still_need_their_topic_word():
    # "How old is it?" has no word longer than three letters; it is still about "old".
    v = CannedVoice("Sam: Mind if I ask? How many glass does it have?<|end|>")
    assert v.ask(BuyerView("wine bottle", [], "p", []), "Sam", "age", "How old is it?") is None
    ok = CannedVoice("Sam: Mind if I ask, how old is it?<|end|>")
    assert ok.ask(BuyerView("wine bottle", [], "p", []), "Sam", "age", "How old is it?")


class PromptRecorder(CannedVoice):
    def _complete(self, prompt: str, temperature: float = 0.7) -> str:
        self.prompt = prompt
        return self.reply


def test_format_2_checkpoints_get_the_question_in_the_prompt():
    v = PromptRecorder("Maya: Is the dog potty trained?<|end|>")
    v.ask(BuyerView("dog", [], "p", []), "Maya", "potty", "Is the dog potty trained?")
    assert v.prompt.endswith("<|ask|>potty<|end|>")                       # format 1 (v1, v2)
    v.prompt_format = 2
    v.ask(BuyerView("dog", [], "p", []), "Maya", "potty", "Is the dog potty trained?")
    assert v.prompt.endswith("<|ask|>potty: Is the dog potty trained?<|end|>")


def test_on_topic_rule():
    from sellanything_ml.buyers.format import on_topic
    assert on_topic("Before I decide, how many are there?", "How many are there?", "bag of peanuts")
    assert not on_topic("Before I decide, how old is it?", "How many are there?", "bag of peanuts")
    assert not on_topic("Does the saxmalone come with anything else?",
                        "Does the saxophone come with anything else?", "saxophone")
    assert on_topic("Can you tell me, does the saxophone come with anything else?",
                    "Does the saxophone come with anything else?", "saxophone")


def test_fill_slot_agrees_articles_and_capitals():
    from sellanything_ml.buyers.format import fill_slot
    assert fill_slot("Is the {item} healthy?", "horse") == "Is the horse healthy?"
    assert fill_slot("I'd buy an {item}.", "bicycle") == "I'd buy a bicycle."
    assert fill_slot("A {item}? Sure.", "apple") == "An apple? Sure."
    assert fill_slot("{item}? No. {item} is not for me.", "tiger") == "Tiger? No. Tiger is not for me."
    assert fill_slot("No slot here.", "tiger") == "No slot here."


def test_format_3_sends_the_question_with_a_slot_and_fills_the_answer():
    v = PromptRecorder("Felix: Quick question: what kind of {item} is it?<|end|>")
    v.prompt_format = 3
    view = BuyerView("knife", [], "p", [], ("knife",))
    assert v.ask(view, "Felix", "kind", "What kind of knife is it?") == "Quick question: what kind of knife is it?"
    assert v.prompt.endswith("<|ask|>kind: What kind of {item} is it?<|end|>")
    r = CannedVoice("Felix: That {item} sounds great.<|interest|>+1<|end|>")
    assert r.react(view, "Felix", engine_delta=1) == "That knife sounds great."


class ReadingVoice(CannedVoice):
    """A model trained on `parse`: reads every typed answer as option B, says nothing usable otherwise."""

    understands = True

    def _complete(self, prompt: str, temperature: float = 0.7) -> str:
        return "B<|end|>" if "<|parse|>" in prompt else "no format at all"


def test_typed_answers_go_to_the_model_when_it_can_read_them():
    rnd = RoundSession.start("dog", 3, seed=1, voice=ReadingVoice(""))
    _, attr = rnd.question
    lines, heard = rnd.answer_text("whatever the player typed")
    assert heard == {"option": attr.options[1].id, "answer": attr.options[1].answer, "rule": "model"}
    assert lines[0] == "You: whatever the player typed"


def test_parse_none_and_garbage_mean_ask_again():
    from sellanything_ml.buyers.schema import load_items
    attr = load_items()["cat"].attributes["breed"]
    assert CannedVoice("none<|end|>").parse("cat", attr, "she's 3 months old") is None
    garbage = CannedVoice("Z<|end|>")
    assert garbage.parse("cat", attr, "ragdoll") is None and garbage.stats.fallback == {"parse_format": 1}
    assert CannedVoice("A<|end|>").parse("cat", attr, "ragdoll") == attr.options[0].id
