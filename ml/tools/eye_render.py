"""Painting the eye model's strokes with each layer's own brush (owner, 2026-10-25: the evaluation must draw the digital
material's eight layers as they were fitted). tools/eye_data_l8.eyes_on_face knows only the four "layers" brushes; the
digital styles store each layer's brush in the material file ("layers": length, width, edge_px, alpha_range, bristle,
blend). These draw a list of (n, 9) bins (x, y, angle, length, width, r, g, b, opacity, as eye_data_l8.to_bins) with
those brushes, on the eye canvas or straight onto the face canvas (every stroke's centre, length, width and soft edge
scaled by the zoom); blending layers smudge (eye_crop_styles.blend_composite) instead of painting.

Kept apart from eye_data_l8 so the eye material's fit can keep running while this changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import eye_data_l8 as D
from sellanything_ml.sketch import catbrush as B


def brushes_of(material: Path | dict) -> list[dict]:
    """Each layer's brush, from a digital material file (or its loaded arrays). Four-layer files have none: then
    eye_data_l8.BRUSHES with catbrush's defaults."""
    z = np.load(material) if isinstance(material, (str, Path)) else material
    if "layers" in getattr(z, "files", z):
        return json.loads(str(z["layers"]))
    return [{"length": ln, "width": wd, "edge_px": B.EDGE_PX, "alpha_range": list(B.ALPHA_RANGE), "bristle": B.BRISTLE,
             "blend": False} for ln, wd in D.BRUSHES]


def _strokes(b: np.ndarray, brush: dict, k: float = 1.0, shift: np.ndarray | None = None) -> B.Strokes:
    """One layer's bins as strokes, scaled by k and moved by shift (eye canvas -> face canvas); call inside the layer's
    brush settings (the opacity bins are relative to its range)."""
    v = D.from_bins(b, (brush["length"], brush["width"]))
    v["xy"] = v["xy"] * k + (0 if shift is None else shift)
    v["length_width"] = v["length_width"] * k
    v = {key: np.asarray(val, np.float32) for key, val in v.items()}
    return B.Strokes.from_stage(v, (float(brush["length"] * k), float(brush["width"] * k)))


def _paint(canvas: torch.Tensor, bins: list[np.ndarray], brushes: list[dict], k: float, shift) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    with torch.no_grad():
        for b, brush in zip(bins, brushes):
            if not len(b):
                continue
            spec = {"edge_px": brush["edge_px"] * k, "alpha_range": tuple(brush["alpha_range"]), "bristle": brush["bristle"]}
            with E.brush_settings(spec):
                st = _strokes(b, brush, k, shift)
                canvas = E.blend_composite(canvas, st) if brush.get("blend") else B.composite(canvas, st, grid)
    return canvas


def layers_on_crop(base: np.ndarray, bins: list[np.ndarray], brushes: list[dict]) -> np.ndarray:
    """The layers painted on the eye canvas (base: (256, 256, 3) in 0-1)."""
    return _paint(torch.from_numpy(np.asarray(base, np.float32)), bins, brushes, 1.0, None).numpy().clip(0, 1)


def layers_on_face(face: np.ndarray, bins: list[np.ndarray], brushes: list[dict], ebox: tuple, fbox: np.ndarray) -> np.ndarray:
    """The layers painted straight onto the face canvas, as eye_data_l8.eyes_on_face but with each layer's brush."""
    k = float((ebox[2] - ebox[0]) / (fbox[2] - fbox[0]))              # face px per crop px
    shift = (np.array(ebox[:2], float) - fbox[:2]) / (fbox[2] - fbox[0]) * B.SIDE
    return _paint(torch.from_numpy(np.asarray(face, np.float32)), bins, brushes, k, shift).numpy().clip(0, 1)
