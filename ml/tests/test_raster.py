"""Stroke → bitmap pipeline against real Quick, Draw! data (tests/fixtures, CC BY 4.0)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sellanything_ml.sketch.raster import rasterize, simplify, strokes_to_bitmap

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "quickdraw_samples.json").read_text())["drawings"]


def corr(a, b):
    a = a.astype(float).ravel() - a.mean()
    b = b.astype(float).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


@pytest.mark.parametrize("d", FIXTURES, ids=lambda d: f"{d['word']}-{d['key_id']}")
def test_rasterize_matches_official_bitmap(d):
    ours = rasterize(d["simplified"])
    assert ours.shape == (28, 28) and ours.dtype == np.uint8
    assert corr(ours, np.array(d["bitmap"])) > 0.98


@pytest.mark.parametrize("d", FIXTURES, ids=lambda d: f"{d['word']}-{d['key_id']}")
def test_raw_strokes_through_full_pipeline(d):
    assert corr(strokes_to_bitmap(d["raw"]), np.array(d["bitmap"])) > 0.98


def test_simplify_is_scale_and_position_invariant():
    square = [([0, 100, 100, 0, 0], [0, 0, 100, 100, 0])]
    moved = [([500 + 3 * x for x in xs], [200 + 3 * y for y in ys]) for xs, ys in square]
    assert simplify(square) == simplify(moved)
    xs, ys = simplify(square)[0]
    assert min(xs) == 0 and max(max(xs), max(ys)) == 255


def test_dot_and_empty_input():
    assert strokes_to_bitmap([([5], [5])]).max() > 0
    assert rasterize([]).max() == 0
