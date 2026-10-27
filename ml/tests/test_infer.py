"""The game's sketch classifier (sketch/infer.py): stroke packing, and the v2 ensemble on real
Quick, Draw! drawings when its checkpoints are present (they are not in git)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sellanything_ml.sketch.infer import V2_DIR, stroke_points

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "quickdraw_samples.json").read_text())["drawings"]


def test_stroke_points_marks_stroke_ends_and_cuts():
    strokes = [([0, 10, 20], [0, 5, 9]), ([100, 90], [100, 140])]
    pts, n = stroke_points(strokes, max_len=8)
    assert n == 5
    assert pts[:5].tolist() == [[0, 0, 0], [10, 5, 0], [20, 9, 1], [100, 100, 0], [90, 140, 1]]
    assert pts[5:].sum() == 0
    cut, n = stroke_points(strokes, max_len=4)
    assert n == 4 and cut[3].tolist() == [100, 100, 1]       # a cut drawing still ends its last stroke


@pytest.mark.skipif(not (V2_DIR / "meta.json").exists(), reason="v2 checkpoints not exported")
@pytest.mark.parametrize("d", FIXTURES, ids=lambda d: f"{d['word']}-{d['key_id']}")
def test_ensemble_recognises_raw_strokes(d):
    from sellanything_ml.sketch.infer import SketchEnsemble
    top = SketchEnsemble().predict([(s[0], s[1]) for s in d["raw"]], k=3)
    assert len(top) == 3 and abs(sum(t["prob"] for t in top) - 1) <= 1
    assert d["word"] in [t["label"] for t in top]
    assert np.isclose(top[0]["prob"], max(t["prob"] for t in top))
