"""Sketch v2 pipeline: splits identical to train.py, stroke storage, batching, and that an
interrupted training run resumes to exactly the weights an uninterrupted run reaches."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sellanything_ml.sketch import data, pipeline
from sellanything_ml.sketch.strokes import encode


def test_splits_match_train_py(monkeypatch):
    """Same drawings in train / val / test as data.load_splits (train.py), so v1 and v2 can be compared."""
    cats = ["a", "b", "c"]
    per_class, n_val, n_test = 60, 5, 5

    def fake(category, n):                       # row k of class c remembers (c, k)
        c = cats.index(category)
        x = np.zeros((n, 784), np.uint8)
        x[:, 0], x[:, 1] = c, np.arange(n)
        return x
    monkeypatch.setattr(data, "download_first_n", fake)
    train, val, test = data.load_splits(cats, per_class, n_val, n_test, seed=0)

    monkeypatch.setattr(pipeline, "load_categories", lambda: cats)
    cfg = pipeline.Config(per_class=70, n_train=per_class - n_val - n_test, n_val=n_val, n_stack=n_test)
    sp = pipeline.splits(cfg)

    def as_pairs(split):
        return sorted(zip(split.x[:, 0].tolist(), split.x[:, 1].tolist()))

    def to_pairs(idx):
        return sorted((int(i) // cfg.per_class, int(i) % cfg.per_class) for i in idx)

    assert as_pairs(train) == to_pairs(sp["train"])
    assert as_pairs(val) == to_pairs(sp["val"])
    assert as_pairs(test) == to_pairs(sp["stack"])
    assert to_pairs(sp["final"]) == [(c, k) for c in range(3) for k in range(60, 70)]


def test_strokeset_round_trip_and_padding():
    s = encode([([([1, 2, 3], [4, 5, 6]), ([7], [8])], True), ([([9, 10], [11, 12])], False)])
    assert len(s) == 2 and s.recognized.tolist() == [True, False]
    assert s.strokes(0) == [([1, 2, 3], [4, 5, 6]), ([7], [8])]
    pts, lens = s.padded(np.array([0, 1]), 3)
    assert lens.tolist() == [3, 2]
    assert pts[0, :, 2].tolist() == [0, 0, 1]          # cut after 3 points: the last kept point ends a stroke


def test_bucket_order_keeps_whole_batches_of_similar_length():
    rng = np.random.default_rng(0)
    order = rng.permutation(1000)
    lengths = rng.integers(1, 200, 1000)
    out = pipeline.bucket_order(order, lengths, 10, np.random.default_rng(1))
    assert len(out) == 1000 and len(set(out.tolist())) == 1000
    spread = [np.ptp(lengths[out[i:i + 10]]) for i in range(0, 1000, 10)]
    assert np.median(spread) < 40                       # batches are length-homogeneous


def _tiny_run(run_dir, monkeypatch, crash_after=None):
    monkeypatch.setattr(pipeline, "RUN_DIR", run_dir)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 4))
    X = torch.randn(400, 8, generator=torch.Generator().manual_seed(1))
    Y = (X[:, 0] > 0).long() + 2 * (X[:, 1] > 0).long()
    cfg = pipeline.Config(batch=20, checkpoint_minutes=0.0)       # checkpoint after every step
    calls = {"n": 0}

    def step(m, batch, gen):
        calls["n"] += 1
        if crash_after is not None and calls["n"] > crash_after:
            raise KeyboardInterrupt("simulated power cut")
        x, y = batch
        x = x + 0.1 * torch.randn(x.shape, generator=gen)            # augmentation from the step's generator
        return F.cross_entropy(m(x), y)

    def evaluate(m):
        with torch.no_grad():
            lp = F.log_softmax(m(X), -1)
        return pipeline.scores(lp, Y.numpy())

    pipeline.train("tiny", cfg, model, step, 3,
                   lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9),
                   lambda o, total: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=total),
                   lambda e: np.random.default_rng(e).permutation(400),
                   lambda idx: (X[torch.from_numpy(idx)], Y[torch.from_numpy(idx)]),
                   evaluate, torch.device("cpu"))
    return model


def test_interrupted_training_resumes_to_identical_weights(tmp_path, monkeypatch):
    straight = _tiny_run(tmp_path / "a", monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _tiny_run(tmp_path / "b", monkeypatch, crash_after=27)       # dies in the middle of epoch 2
    resumed = _tiny_run(tmp_path / "b", monkeypatch)
    for p, q in zip(straight.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(p, q)
