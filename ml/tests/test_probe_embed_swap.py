"""probe_embed_swap: the hook swaps only the input vectors, and every condition reads the same prompt."""

import sys
from pathlib import Path

import pytest
import torch

from sellanything_ml.llm.model import GPT, GPTConfig
from sellanything_ml.llm.tokenizer import TOKENIZER_DIR, Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
import probe_embed_swap as P  # noqa: E402

MODEL_FILE = TOKENIZER_DIR / "sp8192.model"
PROMPT = ("<|item|>cat<|end|><|persona|>Chloe, who is a student | offer now $55<|end|>"
          "<|log|>Your opening ($55): Hi!\nOwen asked: Is she a pedigree?\n"
          "Seller: She is a long haired tabby and very calm.<|end|><|react|>")
pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="no tokenizer file")


def setup():
    tok = Tokenizer(MODEL_FILE)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=256)
    return GPT(cfg).eval(), tok


def test_find_pieces_points_at_the_word():
    _, tok = setup()
    ids = tok.encode(PROMPT)
    places = P.find_pieces(ids, P.spoken(tok, "long haired"))
    assert [ids[p] for p in places] == P.spoken(tok, "long haired")
    assert P.pieces(tok, [ids[p] for p in places]) == ["▁long", "▁ha", "ired"]


def test_missing_word_is_an_error():
    _, tok = setup()
    with pytest.raises(ValueError):
        P.find_pieces(tok.encode(PROMPT), P.spoken(tok, "zebra"))


def test_swap_changes_only_later_positions_and_leaves_the_table_alone():
    model, tok = setup()
    ids = torch.tensor([tok.encode(PROMPT)])
    table = model.embed.weight.detach().clone()
    place = 20
    with torch.no_grad():
        plain, _ = model(ids)
        with P.swapped_inputs(model, {place: torch.randn(64)}):
            swapped, _ = model(ids)
        again, _ = model(ids)
    assert torch.allclose(plain[0, :place], swapped[0, :place], atol=1e-6)
    assert not torch.allclose(plain[0, place:], swapped[0, place:], atol=1e-4)
    assert torch.equal(table, model.embed.weight) and torch.equal(plain, again)    # hook gone after the block


def test_swapping_in_the_own_vector_changes_nothing():
    model, tok = setup()
    ids = tok.encode(PROMPT)
    own = {p: model.embed.weight[ids[p]].detach() for p in (5, 6)}
    signs = P.sign_ids(tok)
    assert P.sign_probs(model, ids, signs) == pytest.approx(P.sign_probs(model, ids, signs, own))


def test_answer_is_replaced_and_read_back():
    new = P.with_answer(PROMPT, "He is intact.")
    assert P.last_answer(new) == "He is intact." and new.endswith("<|end|><|react|>")
    assert new.count("Seller: ") == 1 and "Owen asked" in new


def test_every_condition_gives_probabilities_that_add_to_one():
    model, tok = setup()
    case = {"name": "hair", "prompt": PROMPT, "word": "long haired", "want": "0", "donor": "fluffy"}
    rows = P.run_case(model, tok, case)
    assert [r["condition"] for r in rows] == list(P.CONDITIONS)
    for r in rows:
        assert r["p+"] + r["p0"] + r["p-"] == pytest.approx(1.0)
        assert 0 < r["mass"] <= 1
    base = next(r for r in rows if r["condition"] == "base")
    direct = P.sign_probs(model, tok.encode(PROMPT), P.sign_ids(tok))
    assert base["p0"] == pytest.approx(direct["0"])
    swap = next(r for r in rows if r["condition"] == "text_swap")
    assert swap["p0"] != base["p0"]                 # a different text, a different prompt


def test_summary_has_a_line_for_every_condition():
    model, tok = setup()
    rows = P.run_case(model, tok, {"name": "a", "prompt": PROMPT, "word": "tabby", "want": "+", "donor": "cat"})
    assert len(P.summarise(rows).splitlines()) == 1 + len(P.CONDITIONS)


def test_question_before_the_answer_can_be_replaced():
    new = P.with_answer(PROMPT, "He is intact.", asked="Is he intact?")
    assert "Owen asked: Is he intact?\nSeller: He is intact.<|end|><|react|>" in new
    assert "Is she a pedigree?" not in new


def test_blank_want_stops_the_run_and_names_the_cases(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"name": "x", "prompt": "p", "word": "w", "donor": "d", "want": ""}\n')
    with pytest.raises(SystemExit, match="fill it in for: x"):
        P.load_cases(f, tmp_path)


def test_pair_summary_counts_both_right_same_pick_and_order():
    def row(case, pair, pick, want, plus, minus):
        return {"case": case, "pair": pair, "condition": "base", "pick": pick, "want": want, "p+": plus, "p-": minus}
    rows = [row("a1", "a", "+", "+", 0.6, 0.1), row("a2", "a", "-", "-", 0.1, 0.7),   # both right, ordered right
            row("b1", "b", "0", "+", 0.2, 0.1), row("b2", "b", "0", "-", 0.3, 0.1)]   # same pick, ordered wrong
    lines = P.pair_summary(rows).splitlines()
    assert lines[1].split() == ["base", "1/2", "1/2", "1/2"]


def test_order_uses_the_wants_not_the_pick():
    def row(case, pair, want, plus, minus):
        return {"case": case, "pair": pair, "condition": "base", "pick": "0", "want": want, "p+": plus, "p-": minus}
    rows = [row("a1", "a", "-", 0.1, 0.3), row("a2", "a", "0", 0.2, 0.2)]            # "0" should score above "-"
    assert P.pair_summary(rows).splitlines()[1].split()[-1] == "1/1"


DRAFT = Path(__file__).resolve().parents[1] / "tools/probe_embed_swap_cases_draft.jsonl"


def test_draft_cases_are_pairs_whose_words_are_found():
    import json
    _, tok = setup()
    cases = [json.loads(line) for line in DRAFT.read_text().splitlines()]
    assert len(cases) == 20
    pairs: dict[str, list[dict]] = {}
    for c in cases:
        pairs.setdefault(c["pair"], []).append(c)
        assert c["answer"].count(c["word"]) == 1 and c["suggest"] in P.SIGNS and c["want"] == ""
    assert all(len(v) == 2 and v[0]["suggest"] != v[1]["suggest"] for v in pairs.values())
