"""Eye GPT: the two paint encodings (joint = E1, split = E2) carry the same strokes, E1's layout is unchanged, and a
split model samples only what its grammar allows."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import eyegpt_l8 as EG

SPLIT = EG.Vocab("split")
TINY = {"d_model": 32, "n_layers": 1, "n_heads": 2, "n_kv_heads": 2, "dropout": 0.0}


def random_bins(n: int, seed: int = 0) -> np.ndarray:
    """(n, 9) bins inside their ranges: x, y, angle, length, width, r, g, b, opacity."""
    hi = [EG.NXY, EG.NXY, EG.NANGLE, EG.LEVELS_SIZE, EG.LEVELS_SIZE, EG.LEVELS_RGB, EG.LEVELS_RGB, EG.LEVELS_RGB,
          EG.LEVELS_ALPHA]
    return np.random.default_rng(seed).integers(0, hi, size=(n, 9))


def l8_plan() -> np.ndarray:
    """A minimal L8 sequence with a plan of two eye outlines and two pupils (face px)."""
    o = L.OFFSET
    eye = lambda cx: [L.SP["EYE"]] + [t for x, y in ((cx - 10, 100), (cx, 94), (cx + 10, 100), (cx, 106))  # noqa: E731
                                      for t in (o["x"] + x, o["y"] + y)] + [L.SP["OUT_END"]]
    pupil = lambda cx: [L.SP["PUPIL"], o["x"] + cx, o["y"] + 100, o["radius"] + 3, o["radius"] + 5,  # noqa: E731
                        o["plan_angle"] + 32]
    return np.array([L.SP["BOS"], L.SP["PLAN"]] + eye(90) + eye(160) + pupil(90) + pupil(160) + [L.SP["PLAN_END"]])


def eye_material(seed: int = 0, names: tuple = EG.LAYER_NAMES) -> dict:
    """Eye material for l8_plan's cat: the eye canvas is the face canvas, each layer's strokes first eye then second."""
    rng = np.random.default_rng(seed)
    out = {"fbox": EG.FACE_BOX, "ebox": EG.FACE_BOX, "plan_0": np.array([[90.0, 100.0]]), "plan_1": np.array([[160.0, 100.0]])}
    for k, name in enumerate(names):
        b = random_bins(6 + k, seed + k)
        b[:3, 0], b[3:, 0] = rng.integers(80, 100, 3), rng.integers(150, 170, len(b) - 3)
        out[f"bins_{name}"] = b
    return out


def test_the_joint_layout_is_e1s():
    assert EG.VOCAB == EG.Vocab("joint").size == 14545                     # runs/brushgpt/E1/config.json
    assert EG.TOKENS_PER_STROKE == 5 and EG.MAX_TOK == 1280
    assert EG.OFFSET == {"x": 13, "y": 269, "angle": 525, "size": 557, "paint": 593, "radius": 14417, "plan_angle": 14481}


def test_split_writes_the_colour_as_four_tokens_after_the_same_fields():
    assert SPLIT.stroke_fields == ("x", "y", "angle", "size", "r", "g", "b", "alpha")
    assert SPLIT.per_stroke == 8 and SPLIT.colour_fields == ("r", "g", "b", "alpha")
    assert [SPLIT.fields[k] for k in SPLIT.colour_fields] == [12, 12, 12, 8]
    assert all(SPLIT.offset[k] == EG.OFFSET[k] for k in ("x", "y", "angle", "size"))
    assert SPLIT.size == len(EG.SPECIALS) + 256 + 256 + 32 + 36 + 12 * 3 + 8 + 64 + 64
    with pytest.raises(ValueError):
        EG.Vocab("rgb")


@pytest.mark.parametrize("paint", EG.PAINTS)
def test_bins_round_trip_exactly(paint):
    v = EG.Vocab(paint)
    b = random_bins(500)
    rows = EG.bins_tokens(b, v)
    assert rows.shape == (500, v.per_stroke)
    assert np.array_equal(EG.tokens_bins(rows, v), b)
    groups = [[v.group_of(t) for t in r] for r in rows]
    assert all(g == list(v.stroke_fields) for g in groups)


def test_both_encodings_carry_the_same_strokes_and_pass_their_grammar():
    seq8, eye = l8_plan(), eye_material()
    seqs = {}
    for paint in EG.PAINTS:
        v = EG.Vocab(paint)
        t, w = EG.encode(seq8, eye, v)
        EG.allowed_groups(t, v)                                             # raises if a token breaks the grammar
        dec = EG.decode_layers(t, v)
        assert all(np.array_equal(d, eye[f"bins_{n}"]) for d, n in zip(dec, EG.LAYER_NAMES))
        plan_end = list(t).index(EG.SP["PLAN_END"])
        assert not w[:plan_end + 1].any() and w[plan_end + 1:].all()       # no loss on the given plan
        seqs[paint] = t
    strokes = sum(len(eye[f"bins_{n}"]) for n in EG.LAYER_NAMES)
    assert len(seqs["split"]) - len(seqs["joint"]) == 3 * strokes
    with pytest.raises(ValueError):
        EG.allowed_groups(seqs["split"])                                    # split tokens do not fit the joint grammar


def test_the_split_grammar_asks_for_r_g_b_alpha_in_order():
    g = EG.Grammar(SPLIT)
    for tk in EG.plan_tokens(l8_plan(), EG.FACE_BOX, tuple(EG.FACE_BOX), SPLIT):
        g.step(tk)
    g.step(EG.SP["LAYER0"])
    row = EG.bins_tokens(random_bins(1), SPLIT)[0]
    for field, tk in zip(SPLIT.stroke_fields, row):
        assert g.allowed() == ({field, "LAYER_END", "EYE_NEXT"} if field == "x" else {field})
        g.step(tk)


def test_a_model_without_paint_in_its_config_is_e1s():
    cfg = asdict(EG.Config())
    cfg.pop("paint")
    m = EG.make_model(EG.Config(**cfg))
    assert m.vocab.paint == "joint" and EG.model_config(m.eye_cfg).vocab_size == 14545


def test_a_tiny_split_model_samples_only_what_the_grammar_allows():
    import torch
    torch.manual_seed(0)
    model = EG.make_model(EG.Config(paint="split", **TINY)).eval()
    assert EG.model_config(model.eye_cfg).vocab_size == SPLIT.size
    plan = EG.plan_tokens(l8_plan(), EG.FACE_BOX, tuple(EG.FACE_BOX), SPLIT)
    pix = np.random.default_rng(0).integers(0, 256, (2, EG.NPATCH, EG.PATCH_DIM), dtype=np.uint8)
    out = EG.sample(model, pix, [plan, plan], temperature=1.0, seed=1, max_len=len(plan) + 80)
    for s in out:
        assert list(s[:len(plan)]) == plan
        EG.allowed_groups(s, SPLIT)
        assert all(len(b) == 0 or b.shape[1] == 9 for b in EG.decode_layers(s, SPLIT))


def write_data(path, paint: str, style: str = "layers") -> None:
    """A two-cat data file in the layout of eyegpt_l8.build (one train cat, one val cat)."""
    v = EG.Vocab(paint, "xy", style)
    sets: dict = {frozenset([EG.ALL]): 0}
    toks, wts, sids, offs = [], [], [], [0]
    for seed in (0, 1):
        t, w = EG.encode(l8_plan(), eye_material(seed, v.layer_names), v)
        sids.append(np.array([sets.setdefault(frozenset(a), len(sets)) for a in EG.allowed_groups(t, v)], np.uint8))
        toks.append(t.astype(np.int16))
        wts.append(w)
        offs.append(offs[-1] + len(t))
    np.savez(path, tokens=np.concatenate(toks), loss_w=np.concatenate(wts), set_ids=np.concatenate(sids),
             offsets=np.array(offs, np.int64), patches=np.zeros((2, EG.NPATCH, EG.PATCH_DIM), np.uint8),
             names=np.array(["a", "b"]), split=np.array(["train", "val"]),
             sets=np.array(json.dumps([sorted(s) for s in sorted(sets, key=sets.__getitem__)])), paint=np.array(paint),
             style=np.array(style))


def test_validation_reports_losses_that_compare_across_encodings(tmp_path):
    import torch
    torch.manual_seed(0)
    write_data(tmp_path / "d.npz", "split")
    data = EG.Data(tmp_path / "d.npz")
    assert data.vocab.paint == "split"
    model = EG.make_model(EG.Config(paint="split", **TINY))
    rows = np.array([0, 1])
    res = EG.evaluate_loss(model, data, rows, data.mask_table("cpu"), "cpu")
    weighted = float(data.loss_w.sum())
    assert res["nats_per_cat"] == pytest.approx(res["all"] * weighted / 2, rel=1e-5)
    assert res["paint"] == pytest.approx(sum(res[k] for k in SPLIT.colour_fields))
    assert res["stroke"] == pytest.approx(sum(res[k] for k in SPLIT.stroke_fields))


def test_a_data_file_and_a_run_of_different_encodings_do_not_train(tmp_path, monkeypatch):
    write_data(tmp_path / "d.npz", "joint")
    monkeypatch.setattr(EG, "RUN_DIR", tmp_path / "runs")
    with pytest.raises(ValueError, match="paint"):
        EG.train(EG.Config(name="x", paint="split", **TINY), tmp_path / "d.npz")
    assert not (tmp_path / "runs").exists()


def test_data_files_before_e2_read_as_joint(tmp_path):
    write_data(tmp_path / "d.npz", "joint")
    z = dict(np.load(tmp_path / "d.npz"))
    z.pop("paint")
    np.savez(tmp_path / "old.npz", **z)
    assert EG.Data(tmp_path / "old.npz").vocab.paint == "joint"


def test_a_tiny_split_run_trains_saves_and_logs_comparable_losses(tmp_path, monkeypatch):
    write_data(tmp_path / "d.npz", "split")
    monkeypatch.setattr(EG, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setenv("BRUSHGPT_DEVICE", "cpu")
    run = EG.train(EG.Config(name="t", paint="split", batch=1, max_steps=2, eval_every=1, warmup=1, **TINY), tmp_path / "d.npz")
    rows = [json.loads(x) for x in open(run / "metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2]
    assert {"nats_per_cat", "stroke", "paint", "r", "alpha"} <= set(rows[-1]["val"])
    assert (run / "best.pt").exists() and json.loads((run / "config.json").read_text())["max_tok"] == 2048
    assert EG.load_model(run / "best.pt", "cpu").vocab.paint == "split"


POLAR = EG.Vocab("split", "polar")


def test_split_xy_offsets_are_e2s_and_polar_keeps_x_y_for_the_plan():
    assert EG.Vocab("split").offset == {"x": 13, "y": 269, "angle": 525, "size": 557, "r": 593, "g": 605, "b": 617,
                                        "alpha": 629, "radius": 637, "plan_angle": 701}  # runs/brushgpt/E2
    assert POLAR.stroke_fields == ("rad", "theta", "angle", "size", "r", "g", "b", "alpha")
    assert POLAR.offset["x"] == 13 and POLAR.offset["y"] == 269 and POLAR.per_stroke == 8
    assert POLAR.fields["rad"] == EG.NRAD and POLAR.fields["theta"] == EG.NTHETA
    with pytest.raises(ValueError):
        EG.Vocab("split", "grid")


def test_eye_centres_are_the_pupils_nearest_each_outline():
    plan = EG.plan_tokens(l8_plan(), EG.FACE_BOX, tuple(EG.FACE_BOX), POLAR)
    assert np.allclose(EG.eye_centres(plan, POLAR), [[90.5, 100.5], [160.5, 100.5]])


def test_polar_strokes_come_back_within_one_grid_step_and_keep_every_other_field():
    seq8, eye = l8_plan(), eye_material()
    t, _ = EG.encode(seq8, eye, POLAR)
    EG.allowed_groups(t, POLAR)
    for d, n in zip(EG.decode_layers(t, POLAR), EG.LAYER_NAMES):
        b = eye[f"bins_{n}"]
        assert np.abs(d[:, :2] - b[:, :2]).max() <= 1
        assert np.array_equal(d[:, 2:], b[:, 2:])


def test_polar_measures_each_eyes_strokes_from_that_eye():
    eye = eye_material()
    eye["bins_iris"] = eye["bins_iris"].copy()
    eye["bins_iris"][:3, :2] = [[90, 110], [90, 110], [90, 110]]                    # 10 px below the first pupil
    eye["bins_iris"][3:, :2] = [160, 90]                                            # 10 px above the second
    t, _ = EG.encode(l8_plan(), eye, POLAR)
    o = POLAR.offset
    body = t[list(t).index(EG.SP["LAYER1"]) + 1:list(t).index(EG.SP["LAYER2"]) - 1]
    rows = body[body != EG.SP["EYE_NEXT"]].reshape(-1, POLAR.per_stroke)
    assert set((rows[:, 0] - o["rad"]).tolist()) == {20}                            # exactly 10 px, 0.5 px steps
    assert (rows[:3, 1] - o["theta"] < EG.NTHETA / 2).all() and (rows[3:, 1] - o["theta"] >= EG.NTHETA / 2).all()


def test_a_tiny_polar_model_samples_valid_strokes():
    import torch
    torch.manual_seed(0)
    model = EG.make_model(EG.Config(paint="split", coords="polar", **TINY)).eval()
    plan = EG.plan_tokens(l8_plan(), EG.FACE_BOX, tuple(EG.FACE_BOX), POLAR)
    pix = np.zeros((1, EG.NPATCH, EG.PATCH_DIM), np.uint8)
    s = EG.sample(model, pix, [plan], seed=2, max_len=len(plan) + 60)[0]
    EG.allowed_groups(s, POLAR)
    assert all(((b[:, :2] >= 0) & (b[:, :2] < EG.NXY)).all() for b in EG.decode_layers(s, POLAR))


DIGITAL = EG.Vocab("split", "xy", "digital")


def test_digital_has_eight_layers_and_blending_strokes_without_colour():
    assert DIGITAL.layer_names == ("lid", "base", "shade", "blend", "fibres", "blend2", "pupil", "highlight")
    assert DIGITAL.layer_fields[3] == DIGITAL.layer_fields[5] == ("x", "y", "angle", "size", "alpha")
    assert all(DIGITAL.layer_fields[k] == DIGITAL.stroke_fields for k in (0, 1, 2, 4, 6, 7))
    assert DIGITAL.max_tok == 8192 and DIGITAL.sp["LAYER7"] == 14 and DIGITAL.sp["LAYER_END"] == 16
    assert all(DIGITAL.sp[n] == EG.SP[n] for n in EG.PLAN_SPECIALS)         # the plan's tokens keep their ids
    assert EG.Vocab("split").sp["LAYER_END"] == EG.SP["LAYER_END"] == 12      # E2-E5's layout is unchanged
    with pytest.raises(ValueError):
        EG.Vocab("joint", "xy", "digital")


def test_blending_strokes_keep_position_size_and_opacity_and_drop_colour():
    b = random_bins(50, 3)
    rows = EG.bins_tokens(b, DIGITAL, layer=3)
    assert rows.shape == (50, 5)
    back = EG.tokens_bins(rows, DIGITAL, layer=3)
    assert np.array_equal(back[:, [0, 1, 2, 3, 4, 8]], b[:, [0, 1, 2, 3, 4, 8]]) and not back[:, 5:8].any()
    assert np.array_equal(EG.tokens_bins(EG.bins_tokens(b, DIGITAL, layer=4), DIGITAL, layer=4), b)


def test_a_digital_sequence_passes_its_grammar_and_decodes_every_layer():
    eye = eye_material(0, DIGITAL.layer_names)
    t, w = EG.encode(l8_plan(), eye, DIGITAL)
    EG.allowed_groups(t, DIGITAL)
    dec = EG.decode_layers(t, DIGITAL)
    for k, (d, n) in enumerate(zip(dec, DIGITAL.layer_names)):
        b = eye[f"bins_{n}"].copy()
        if k in DIGITAL.blend_layers:
            b[:, 5:8] = 0
        assert np.array_equal(d, b)
    strokes = [len(eye[f"bins_{n}"]) for n in DIGITAL.layer_names]
    plan_end = list(t).index(EG.SP["PLAN_END"])
    body = sum(s * len(f) for s, f in zip(strokes, DIGITAL.layer_fields)) + 3 * DIGITAL.n_layers + 1
    assert len(t) - plan_end - 1 == body and not w[:plan_end + 1].any()


def test_a_tiny_digital_model_samples_eight_valid_layers():
    import torch
    torch.manual_seed(0)
    model = EG.make_model(EG.Config(paint="split", style="digital", **TINY)).eval()
    plan = EG.plan_tokens(l8_plan(), EG.FACE_BOX, tuple(EG.FACE_BOX), DIGITAL)
    pix = np.zeros((1, EG.NPATCH, EG.PATCH_DIM), np.uint8)
    s = EG.sample(model, pix, [plan], seed=4, max_len=len(plan) + 200)[0]
    EG.allowed_groups(s, DIGITAL)
    assert len(EG.decode_layers(s, DIGITAL)) == 8


def test_a_tiny_digital_run_trains_and_reports_each_layer(tmp_path, monkeypatch):
    write_data(tmp_path / "d.npz", "split", "digital")
    assert EG.Data(tmp_path / "d.npz").vocab.style == "digital"
    monkeypatch.setattr(EG, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setenv("BRUSHGPT_DEVICE", "cpu")
    with pytest.raises(ValueError, match="style"):
        EG.train(EG.Config(name="w", paint="split", **TINY), tmp_path / "d.npz")
    run = EG.train(EG.Config(name="t", paint="split", style="digital", batch=1, max_steps=1, eval_every=1, warmup=1, **TINY),
                   tmp_path / "d.npz")
    val = json.loads(open(run / "metrics.jsonl").readline())["val"]
    assert {f"layer_{n}" for n in DIGITAL.layer_names} <= set(val)
