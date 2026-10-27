"""Cutting drawings at stroke boundaries for the partial-drawing classifier (design-v2 §4)."""
import torch

from sellanything_ml.sketch.partial import _rows_to_strokes, by_fraction, cut, strokes_of


def drawing():
    # 3 strokes of 2, 3 and 1 points, padded to 8 rows
    rows = [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 3, 0], [4, 4, 1], [5, 5, 1], [0, 0, 0], [0, 0, 0]]
    return torch.tensor([rows], dtype=torch.uint8), torch.tensor([6])


def test_cut_keeps_whole_strokes_and_never_more_than_the_drawing():
    pts, lens = drawing()
    assert strokes_of(pts, lens).tolist() == [3]
    assert [int(cut(pts, lens, torch.tensor([k]))) for k in (0, 1, 2, 3, 9)] == [2, 2, 5, 6, 6]


def test_by_fraction_rounds_up_and_includes_the_full_drawing():
    pts, lens = drawing()
    got = [int(by_fraction(pts, lens, torch.tensor([f]))) for f in (0.01, 0.34, 0.67, 1.0)]
    assert got == [2, 5, 6, 6]          # ceil(0.34 × 3) = 2 strokes, ceil(0.67 × 3) = 3


def test_padding_after_the_length_is_never_counted_as_a_stroke():
    pts, lens = drawing()
    pts[0, 7, 2] = 1                    # a stray end flag in the padding
    assert strokes_of(pts, lens).tolist() == [3]


def test_packed_rows_go_back_to_strokes():
    pts, lens = drawing()
    assert _rows_to_strokes(pts[0].numpy(), int(lens)) == [([0, 1], [0, 1]), ([2, 3, 4], [2, 3, 4]), ([5], [5])]
