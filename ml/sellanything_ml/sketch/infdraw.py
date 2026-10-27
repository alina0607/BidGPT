"""Photo → line drawing with Informative Drawings (Chan, Durand, Isola, "Learning to generate line drawings that
convey geometry and semantics", CVPR 2022), using the authors' released generators.

The network below has the same layers as `model.py` in github.com/carolineec/informative-drawings (the released
weights load into it with strict key and shape checks); that file imports torchvision for its other networks,
which this project does not install.

    MIT License, Copyright (c) 2022 Caroline Chan. Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above copyright notice and this permission
    notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
    "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

Weights (`model.zip` from the README's Google Drive link, 47.7 MB) are unpacked to WEIGHTS_DIR, outside git.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import ML_DIR

WEIGHTS_DIR = ML_DIR / "data" / "raw" / "informative_drawings" / "model"
STYLES = ("contour_style", "opensketch_style", "anime_style")


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
                                        nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
                                        nn.InstanceNorm2d(ch))

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """RGB in [0, 1] → one channel in [0, 1], 1 = paper, 0 = ink."""

    def __init__(self, n_residual_blocks: int = 3):
        super().__init__()
        self.model0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.model1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True))
        self.model2 = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks)])
        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Sigmoid())

    def forward(self, x):
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


def load(style: str, dev="cpu") -> Generator:
    net = Generator()
    state = torch.load(WEIGHTS_DIR / style / "netG_A_latest.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.to(dev).eval()


@torch.no_grad()
def draw(net: Generator, rgb: np.ndarray, batch: int = 32) -> np.ndarray:
    """uint8 (N, H, W, 3) photos → float32 (N, H, W) line drawings, 1 = paper."""
    dev = next(net.parameters()).device
    out = []
    for i in range(0, len(rgb), batch):
        x = torch.from_numpy(rgb[i:i + batch]).to(dev).float().div_(255).permute(0, 3, 1, 2)
        out.append(net(x)[:, 0].float().cpu().numpy())
    return np.concatenate(out)
