"""The shelter painter (L8 + an eye model, stroke by stroke): it reads the same strokes as the decoders and its
finished canvas is the evaluation's layer-at-once render (tools/eye_eval_e1.py), blending layers included."""

import numpy as np
import pytest
import torch
from test_eyegpt_l8 import eye_material, l8_plan

from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import eyegpt_l8 as EG
from sellanything_ml.sketch import shelter as S

E, ELL, ER = S._tools()


def l8_sequence(seed: int = 0, per_layer: int = 6) -> np.ndarray:
    """l8_plan's plan, then all eight layers with random strokes; the detail layers' strokes split into regions."""
    rng = np.random.default_rng(seed)
    plan = list(l8_plan())
    t = plan[:2] + [L.SP["POSE"]] + [L.OFFSET["plan_angle"] + 32] * 3 + plan[2:]     # the grammar wants a pose first
    for k in range(L.MAX_LAYERS):
        t.append(L.SP[f"LAYER{k}"])
        for i in range(per_layer):
            if k in L.DETAIL and i and i % 2 == 0:
                t.append(L.SP["REGION_END"])
            fifth = "strength" if L.brush(k)["blend"] else "paint"
            t += [L.OFFSET[g] + int(rng.integers(L.FIELDS[g])) for g in ("x", "y", "angle", "size", fifth)]
        if k in L.DETAIL:
            t += [L.SP["REGION_END"]] * (L.REGIONS - (per_layer - 1) // 2)    # every region closed
        t.append(L.SP["LAYER_END"])
    return np.array(t + [L.SP["EOS"]])


def follow(track, tokens) -> list:
    return [s for s in (track.step(int(tok)) for tok in tokens) if s is not None]


def test_face_strokes_are_the_decoders():
    t = l8_sequence()
    L.allowed_groups(t, L.MAX_LAYERS)                                       # raises if the sequence breaks the grammar
    got = follow(S.FaceStrokes(), t[1:-1])
    dec = L.decode_layers(t)
    for k, rows in enumerate(dec):
        assert np.array_equal(np.concatenate([r for j, r in got if j == k]), rows)


@pytest.mark.parametrize("vocab", [EG.Vocab("split"), EG.Vocab("split", "polar"), EG.Vocab("split", "xy", "digital")])
def test_eye_strokes_are_the_decoders(vocab):
    t, _ = EG.encode(l8_plan(), eye_material(0, vocab.layer_names), vocab)
    t = [int(x) for x in t]
    plan = t[:t.index(EG.SP["PLAN_END"]) + 1]
    got = follow(S.EyeStrokes(vocab, plan), t[len(plan):-1])
    for k, bins in enumerate(EG.decode_layers(np.array(t), vocab)):
        mine = [b for j, b in got if j == k]
        assert np.array_equal(np.concatenate(mine) if mine else np.zeros((0, 9), int), bins)


def test_the_face_painted_stroke_by_stroke_is_the_render():
    t = l8_sequence(1)
    ys, xs = torch.meshgrid(torch.arange(256) + 0.5, torch.arange(256) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    paint = S.Layers(torch.ones(256, 256, 3), lambda c, j, rows: S.face_strokes(E, c, j, rows, grid),
                     lambda j: L.brush(j)["blend"])
    for stroke in follow(S.FaceStrokes(), t[1:-1]):
        canvas = paint.add(*stroke)
    assert np.abs(canvas.numpy() - L.render(L.decode_layers(t))).max() < 1e-5


def test_the_eyes_painted_stroke_by_stroke_are_the_evaluations():
    vocab = EG.Vocab("split", "xy", "digital")
    brushes = [{"length": 8.0, "width": 2.0, "edge_px": 0.5, "alpha_range": [0.3, 1.0], "bristle": 0.0,
                "blend": k in vocab.blend_layers} for k in range(vocab.n_layers)]
    t, _ = EG.encode(l8_plan(), eye_material(2, vocab.layer_names), vocab)
    t = [int(x) for x in t]
    plan = t[:t.index(EG.SP["PLAN_END"]) + 1]
    face = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    box = (60.0, 40.0, 190.0, 170.0)
    paint = S.Layers(face, lambda c, j, b: ER.layers_on_face(c, [np.concatenate(b)], [brushes[j]], box, EG.FACE_BOX),
                     lambda j: brushes[j]["blend"])
    for stroke in follow(S.EyeStrokes(vocab, plan), t[len(plan):-1]):
        live = paint.add(*stroke)
    ref = ER.layers_on_face(face, EG.decode_layers(np.array(t), vocab), brushes, box, EG.FACE_BOX)
    assert np.abs(live - ref).max() < 1e-5
