"""Load a trained sketch classifier and classify player strokes.

Two kinds of checkpoint:
- v2 (preferred): CNN v2 on 48×48 images + stroke Transformer, combined 0.4 / 0.6 in log-probability
  (the owner's choice after ml/runs/sketch_v2/final_report.json: 82.11% top-1 on 172,500 test
  drawings). Lives in checkpoints/sketch_v2/, made by `python -m sellanything_ml.sketch.infer export`.
- v1: one 28×28 CNN (sketch_cnn_345.pt).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import ML_DIR
from .model import make_model
from .raster import Stroke, rasterize, simplify, strokes_to_bitmap

DEFAULT_CHECKPOINTS = [ML_DIR / "checkpoints" / "sketch_cnn_345.pt",
                       ML_DIR / "checkpoints" / "sketch_cnn_10class.pt"]
V2_DIR = ML_DIR / "checkpoints" / "sketch_v2"
PARTIAL_DIR = ML_DIR / "checkpoints" / "sketch_partial"   # live guesses (sketch/partial.py)


class SketchClassifier:
    def __init__(self, path: Path, device: str = "cpu"):
        state = torch.load(path, map_location=device)
        weights = state["state_dict"]
        # Infer the architecture from the weights, so notebook and script checkpoints both load.
        width = weights["0.0.weight"].shape[0]
        linear = [k for k, v in weights.items() if v.dim() == 2]
        hidden = weights[linear[0]].shape[0]
        self.categories: list[str] = list(state["categories"])
        self.model = make_model(len(self.categories), width=width, hidden=hidden)
        self.model.load_state_dict(weights)
        self.model.eval().to(device)
        self.device = device
        self.path = path

    @torch.no_grad()
    def predict_bitmap(self, bitmap: np.ndarray, k: int = 3) -> list[dict]:
        x = torch.from_numpy(bitmap.astype(np.float32) / 255.0).view(1, 1, 28, 28).to(self.device)
        probs = self.model(x).softmax(-1)[0]
        top = probs.topk(min(k, len(self.categories)))
        return [{"label": self.categories[i], "prob": round(p, 4)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())]

    def predict(self, strokes: list[Stroke], k: int = 3) -> list[dict]:
        return self.predict_bitmap(strokes_to_bitmap(strokes), k)


def stroke_points(strokes: list[Stroke], max_len: int) -> tuple[np.ndarray, int]:
    """Simplified strokes (0..255) → (max_len, 3) uint8 points (x, y, 1 if the stroke ends here),
    cut at max_len the same way StrokeSet.padded cut the training data, and the length."""
    rows = []
    for xs, ys in strokes:
        for j, (x, y) in enumerate(zip(xs, ys)):
            rows.append((x, y, int(j == len(xs) - 1)))
    out = np.zeros((max_len, 3), np.uint8)
    d = np.asarray(rows[:max_len], np.int64).clip(0, 255).astype(np.uint8)
    out[:len(d)] = d
    if len(d):
        out[len(d) - 1, 2] = 1
    return out, len(d)


class SketchEnsemble:
    """CNN v2 + stroke Transformer, combined with one weight (docs/design-v2.md §4)."""

    def __init__(self, folder: Path = V2_DIR, device: str = "cpu"):
        from . import pipeline
        meta = json.loads((folder / "meta.json").read_text())
        self.categories: list[str] = meta["categories"]
        self.weights = meta["weights"]                      # [cnn, seq]
        self.side, self.max_len = meta["side"], meta["max_len"]
        n = len(self.categories)
        self.cnn = pipeline.CNN2(n, self.side)
        self.cnn.load_state_dict(torch.load(folder / "cnn.pt", map_location=device)["model"])
        self.seq = pipeline.StrokeTransformer(n, max_len=self.max_len)
        self.seq.load_state_dict(torch.load(folder / "seq.pt", map_location=device)["model"])
        self.cnn.eval().to(device)
        self.seq.eval().to(device)
        self._features = pipeline.seq_features
        self.device, self.path = device, folder

    @torch.no_grad()
    def logits_simplified(self, simple: list[Stroke]) -> torch.Tensor:
        """Combined scores for strokes already in Quick, Draw!'s simplified form (0..255)."""
        img = torch.from_numpy(rasterize(simple, side=self.side).astype(np.float32) / 255.0)
        lp_cnn = self.cnn(img.view(1, 1, self.side, self.side).to(self.device)).log_softmax(-1)
        pts, n = stroke_points(simple, self.max_len)
        pts_t = torch.from_numpy(pts[None, :max(n, 1)]).to(self.device)
        lens = torch.tensor([n], device=self.device)
        lp_seq = self.seq(self._features(pts_t, lens, None), lens).log_softmax(-1)
        return (self.weights[0] * lp_cnn + self.weights[1] * lp_seq)[0]

    def predict(self, strokes: list[Stroke], k: int = 3) -> list[dict]:
        simple = simplify(strokes)
        if not simple:
            return []
        probs = self.logits_simplified(simple).softmax(-1)
        top = probs.topk(min(k, len(self.categories)))
        return [{"label": self.categories[i], "prob": round(p, 4)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())]


class PartialSketch(SketchEnsemble):
    """Live guesses while drawing (design-v2 §4): the v2 models with the stroke Transformer replaced by
    the one fine-tuned on drawing prefixes. meta["mode"] says whether the CNN still takes part
    ("ensemble, tuned Transformer") or the tuned Transformer guesses alone ("tuned Transformer")."""

    def __init__(self, folder: Path = PARTIAL_DIR, device: str = "cpu"):
        super().__init__(V2_DIR, device)
        meta = json.loads((folder / "meta.json").read_text())
        self.mode = meta["mode"]
        self.seq.load_state_dict(torch.load(folder / "seq.pt", map_location=device)["model"])
        self.seq.eval()
        self.path = folder

    @torch.no_grad()
    def logits_simplified(self, simple: list[Stroke]) -> torch.Tensor:
        if self.mode != "tuned Transformer":
            return super().logits_simplified(simple)
        pts, n = stroke_points(simple, self.max_len)
        pts_t = torch.from_numpy(pts[None, :max(n, 1)]).to(self.device)
        lens = torch.tensor([n], device=self.device)
        return self.seq(self._features(pts_t, lens, None), lens).log_softmax(-1)[0]


def load_partial(device: str = "cpu"):
    """The live-guess model, or None if none has been exported (live guesses then use load_default)."""
    return PartialSketch(PARTIAL_DIR, device) if (PARTIAL_DIR / "meta.json").exists() else None


def export_v2(run: Path = ML_DIR / "runs" / "sketch_v2", folder: Path = V2_DIR) -> None:
    """Copy the chosen v2 checkpoints and the combination weight out of the training run."""
    from . import pipeline
    cfg = pipeline.Config()
    stack = torch.load(run / "stack.pt", weights_only=False)["cnn_v2+seq"]
    report = json.loads((run / "final_report.json").read_text())["models"]["cnn_v2+seq (one weight)"]
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("cnn", "seq"):
        st = torch.load(run / f"{name}_best.pt", map_location="cpu", weights_only=False)
        torch.save({"model": st["model"], "epoch": st["epoch"]}, folder / f"{name}.pt")
    meta = {"categories": pipeline.categories(cfg), "weights": [float(w) for w in stack["weights"]],
            "side": cfg.side, "max_len": cfg.max_len, "source": str(run.relative_to(ML_DIR)),
            "test_top1": report["top1"], "test_top3": report["top3"]}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"exported to {folder}: weights {meta['weights']}, test top-1 {meta['test_top1']:.4f}")


def load_default(device: str = "cpu"):
    if (V2_DIR / "meta.json").exists():
        return SketchEnsemble(V2_DIR, device)
    for path in DEFAULT_CHECKPOINTS:
        if path.exists():
            return SketchClassifier(path, device)
    return None


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        export_v2()
    else:
        raise SystemExit("usage: python -m sellanything_ml.sketch.infer export")
