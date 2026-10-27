"""Sketch classifier architecture (stage 6 of notebooks/01_sketch_classifier.ipynb, widened)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


def make_model(num_classes: int, width: int = 64, hidden: int = 512, dropout: float = 0.3) -> nn.Sequential:
    """28x28 grayscale -> class logits. width=32, hidden=256 reproduces the notebook's model."""
    w = width
    return nn.Sequential(
        conv_block(1, w), conv_block(w, w), nn.MaxPool2d(2),                 # (w, 14, 14)
        conv_block(w, 2 * w), conv_block(2 * w, 2 * w), nn.MaxPool2d(2),     # (2w, 7, 7)
        conv_block(2 * w, 4 * w), nn.MaxPool2d(2),                           # (4w, 3, 3)
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(4 * w * 3 * 3, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )


def random_affine(x: torch.Tensor, max_degrees: float = 12, max_shift: float = 0.08,
                  scale_range: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """Random rotation / shift / scale per image, on whatever device x lives on."""
    n = x.size(0)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(max_degrees)
    scale = torch.empty(n, device=x.device).uniform_(*scale_range)
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, shift[:, 0]], dim=1),
        torch.stack([sin, cos, shift[:, 1]], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
