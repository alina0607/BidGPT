"""VQ-VAE image tokenizer for pencil sketches (van den Oord et al. 2017; Razavi et al. 2019).

A 128×128 grey image goes in; the encoder shrinks it 8× to a 16×16 grid of vectors, and each vector is
replaced by the nearest of K learned codebook vectors. The image is then 256 integers in 0..K-1, which a
GPT can model exactly like text tokens (Esser et al. 2021, "Taming Transformers" §3.2). The decoder turns
the 16×16 grid of codebook vectors back into an image.

Training (VQ-VAE §3.2, eq. 3): reconstruction loss + β·commitment loss, with the straight-through
estimator copying the decoder's gradient past the nearest-neighbour lookup. The codebook is not trained
by gradient but by exponential moving averages of the encoder outputs assigned to each code (VQ-VAE-2 §2.1,
γ = 0.99). Codes no encoder output has used for a while are moved onto a random encoder output from the
current batch, so the whole codebook stays in use (checked by `usage` in the training log).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, chans: tuple[int, ...], dim: int, blocks: int, image_ch: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(image_ch, chans[0], 3, padding=1)]
        for i, ch in enumerate(chans):
            prev = chans[i - 1] if i else chans[0]
            if ch != prev:
                layers.append(nn.Conv2d(prev, ch, 1))
            layers += [ResBlock(ch) for _ in range(blocks)]
            layers.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))           # halve the side
        layers += [ResBlock(chans[-1]), nn.GroupNorm(8, chans[-1]), nn.SiLU(), nn.Conv2d(chans[-1], dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, chans: tuple[int, ...], dim: int, blocks: int, image_ch: int = 1):
        super().__init__()
        rev = chans[::-1]
        layers: list[nn.Module] = [nn.Conv2d(dim, rev[0], 3, padding=1), ResBlock(rev[0])]
        for i, ch in enumerate(rev):
            prev = rev[i - 1] if i else rev[0]
            if ch != prev:
                layers.append(nn.Conv2d(prev, ch, 1))
            layers += [ResBlock(ch) for _ in range(blocks)]
            layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))  # double the side
        layers += [nn.GroupNorm(8, rev[-1]), nn.SiLU(), nn.Conv2d(rev[-1], image_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Quantizer(nn.Module):
    """Nearest-codebook lookup with an EMA codebook and restarts of unused codes."""

    def __init__(self, k: int, dim: int, decay: float = 0.99, dead_after: int = 200):
        super().__init__()
        self.k, self.dim, self.decay, self.dead_after = k, dim, decay, dead_after
        embed = torch.randn(k, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.ones(k))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("idle", torch.zeros(k, dtype=torch.long))   # steps since last used

    def codes(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, D, H, W) → code indices (B, H, W)."""
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.dim)
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.T + self.embed.pow(2).sum(1)[None]
        return d.argmin(1).view(z.size(0), z.size(2), z.size(3))

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """Code indices (B, H, W) → vectors (B, D, H, W)."""
        return self.embed[idx].permute(0, 3, 1, 2).contiguous()

    def forward(self, z: torch.Tensor):
        z = z.float()                     # distances and the EMA in float32 even under bf16 autocast
        idx = self.codes(z)
        q = self.lookup(idx)
        commit = F.mse_loss(z, q.detach())
        if self.training:
            self._update(z, idx)
        q = z + (q - z).detach()          # straight-through: forward uses q, gradient flows to z
        return q, idx, commit

    @torch.no_grad()
    def _update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.dim)
        onehot = F.one_hot(idx.reshape(-1), self.k).type(flat.dtype)
        count = onehot.sum(0)
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
        n = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (n + self.k * 1e-5) * n     # Laplace smoothing
        self.embed.copy_(self.ema_sum / smoothed[:, None])
        self.idle.add_(1)
        self.idle[count > 0] = 0
        dead = (self.idle >= self.dead_after).nonzero().squeeze(1)
        if len(dead):
            pick = flat[torch.randint(0, len(flat), (len(dead),), device=flat.device)]
            self.embed[dead] = pick
            self.ema_sum[dead] = pick
            self.ema_count[dead] = 1.0
            self.idle[dead] = 0


class VQVAE(nn.Module):
    def __init__(self, k: int = 1024, dim: int = 64, chans: tuple[int, ...] = (32, 128, 256), blocks: int = 1,
                 image_ch: int = 1):
        super().__init__()
        self.encoder = Encoder(chans, dim, blocks, image_ch)
        self.quant = Quantizer(k, dim)
        self.decoder = Decoder(chans, dim, blocks, image_ch)
        self.factor = 2 ** len(chans)

    def forward(self, x: torch.Tensor):
        """x (B, image_ch, H, W) in [-1, 1] → reconstruction, code indices, commitment loss."""
        q, idx, commit = self.quant(self.encoder(x))
        return self.decoder(q).float(), idx, commit

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant.codes(self.encoder(x).float())

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quant.lookup(idx)).clamp(-1, 1)
