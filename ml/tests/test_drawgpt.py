"""Drawing-GPT tokens: structure, round trip and error feedback (sketch/drawgpt.py)."""

import numpy as np

from sellanything_ml.sketch.drawgpt import END, FIRST_CLASS, LIFT, START, decode, encode_batch, sequence, tidy


def _drawing():
    # two strokes: (10,10)->(30,12)->(50,40) and (100,100)->(90,140); last column marks stroke ends
    return np.array([[10, 10, 0], [30, 12, 0], [50, 40, 1], [100, 100, 0], [90, 140, 1]], np.uint8)


def _centers():
    # coarse moves on a 10-px grid up to ±150 (961 moves, within K = 1000): forces rounding
    g = np.arange(-150, 151, 10, dtype=np.float32)
    return np.array([(x, y) for x in g for y in g], np.float32)


def test_sequence_layout():
    d = _drawing()
    seq = sequence(3, d, np.arange(len(d)))
    assert seq[0] == FIRST_CLASS + 3 and seq[1] == START
    assert list(seq[2:]) == [0, 1, 2, LIFT, 3, 4, END]   # lift after a stroke, end after the last


def test_round_trip_keeps_strokes_and_points():
    d, c = _drawing(), _centers()
    (toks,) = encode_batch([d], c)
    strokes = decode(sequence(0, d, toks)[2:], c)
    assert [len(xs) for xs, _ in strokes] == [3, 2]
    pts = np.array([p for xs, ys in strokes for p in zip(xs, ys)])
    assert np.abs(pts - d[:, :2]).max() <= 5              # half a grid step, never accumulating


def test_error_feedback_does_not_drift():
    rng = np.random.default_rng(0)
    xy = np.cumsum(rng.integers(1, 4, size=(60, 2)), axis=0)   # 60 small moves of 1-3 px
    d = np.c_[xy, np.zeros(60)].astype(np.uint8)
    c = _centers()
    (toks,) = encode_batch([d], c)
    cur, err = np.zeros(2), []
    for t, p in zip(toks, xy):
        cur = cur + c[t]
        err.append(np.abs(cur - p).max())
    assert max(err) <= 5                                   # without feedback this would grow


def test_decode_stops_at_end():
    c = _centers()
    assert decode(np.array([0, END, 0, 0]), c) == decode(np.array([0, END]), c)


def test_tidy_drops_the_stuck_tail_of_unfinished_drawings_only():
    face = ([0.0, 100.0, 100.0, 0.0], [0.0, 0.0, 100.0, 100.0])
    eye, dot, cut = ([30.0], [30.0]), ([50.0, 52.0], [50.0, 51.0]), ([10.0, 60.0], [10.0, 90.0])
    stuck = [face, eye, eye, cut] + [dot] * 13
    assert tidy(stuck, ended=True) == stuck                        # finished drawings are never touched
    assert tidy(stuck, ended=False) == [face, eye, eye, cut]       # 12 dots after the cut-off one: stuck
    assert tidy([face, eye, eye, dot], ended=False) == [face, eye, eye]   # a few dots may be eyes: kept
    assert tidy([face, cut], ended=False) == [face]                # the stroke cut off at the limit goes
    assert tidy([dot] * 15, ended=False) == [dot] * 15             # nothing real left: keep what there was


def _tiny_gpt(vocab):
    import torch

    from sellanything_ml.llm.model import GPT, GPTConfig
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32))


def test_guidance_one_is_plain_sampling_and_other_values_change_it():
    import torch
    m = _tiny_gpt(50)
    cond, free = torch.tensor([[7, 3]] * 2), torch.tensor([[9, 3]] * 2)
    plain = m.generate(cond, 12, temperature=0)
    assert torch.equal(m.generate(cond, 12, temperature=0, uncond=free, guidance=1.0), plain)
    assert not torch.equal(m.generate(cond, 12, temperature=0, uncond=free, guidance=8.0), plain)


def test_warm_start_keeps_old_tokens_and_starts_no_class_at_the_class_mean(tmp_path):
    import torch

    from sellanything_ml.sketch.drawgpt import warm_start
    old = _tiny_gpt(FIRST_CLASS + 5)
    torch.save({"model": old.state_dict()}, tmp_path / "best.pt")
    new = _tiny_gpt(FIRST_CLASS + 6)
    warm_start(new, tmp_path / "best.pt")
    w, w0 = new.embed.weight.detach(), old.embed.weight.detach()
    assert torch.equal(w[:FIRST_CLASS + 5], w0)
    assert torch.allclose(w[FIRST_CLASS + 5], w0[FIRST_CLASS:].mean(0))
    assert new.head.weight is new.embed.weight                  # still tied after loading
