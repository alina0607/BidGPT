"""Pixel sketch GPT: draw a pencil sketch of a class as an image, one image patch at a time.

The drawing GPT (drawgpt.py) draws Quick, Draw! doodles as pen moves. This one draws in pixels, the way
DALL·E (Ramesh et al. 2021) and Taming Transformers (Esser et al. 2021) do: a VQ-VAE turns each 128×128
image into 16×16 = 256 codebook indices (vqvae.py), and a decoder-only GPT (llm/model.py) learns
p(code_i | class, codes before i) in raster order. Sampling a class token and 256 codes, then decoding them,
gives a new image; it grows patch by patch, top-left to bottom-right.

Data: DomainNet's sketch domain (Peng et al., ICCV 2019), the authors' cleaned lists, 345 classes,
48,212 train / 20,916 test images. Cat, dog and bird were reviewed by hand
(data/domainnet_review/README.md): 321 of 746 kept, 224 of them in the train split.

Stages (each resumable; `all` runs them in order):

  data     every listed image → grey, background levelled to white, padded square, 128×128 (CPU)
  vq       train the VQ-VAE on the train split (GPU)
  tokens   every image (and its mirror image) as 256 codes (GPU)
  judge    train an independent classifier on the train split images, to score generated images (GPU)
  gpt      train the class-conditional GPT on the codes of all 345 classes (GPU)
  tune     train it further on the hand-cleaned cat / dog / bird images only (GPU)

    python -m sellanything_ml.sketch.pixsketch data
    python -m sellanything_ml.sketch.pixsketch vq
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ML_DIR

RAW_DIR = ML_DIR / "data" / "raw" / "sketch_extra" / "domainnet"
REVIEW = ML_DIR / "data" / "domainnet_review" / "review.tsv"
DATA_DIR = ML_DIR / "data" / "processed" / "pixsketch"
RUN_DIR = ML_DIR / "runs" / "pixsketch"
SIDE = 128
DARK = 128             # a background darker than this is not levelled (photos, white-on-black: 2% of images)


@dataclass
class Config:
    seed: int = 0
    # VQ-VAE
    vq_k: int = 1024
    vq_dim: int = 64
    vq_chans: tuple = (32, 128, 256)     # few channels at full size: 0.58 s/step against 2.07 s for (64,128,256)×2
    vq_blocks: int = 1                   # blocks (measured on the M5, batch 64, bf16 autocast, 2026-10-07)
    vq_beta: float = 0.25                # commitment weight (VQ-VAE §3.2)
    vq_lr: float = 2e-4                  # VQ-VAE §4.1
    vq_batch: int = 64
    vq_epochs: int = 15
    checkpoint_minutes: float = 10.0


# --- data -----------------------------------------------------------------------------------------

def _listed(split: str) -> list[tuple[str, str]]:
    """(path, class name) of every image in the authors' cleaned list for a split, in list order."""
    out = []
    for line in (RAW_DIR / f"sketch_{split}.txt").read_text().splitlines():
        path = line.split()[0]
        out.append((path, path.split("/")[1]))
    return out


def categories() -> list[str]:
    return sorted({c for _, c in _listed("train")})


def prepare(path: Path) -> np.ndarray:
    """One image → 128×128 uint8, white background. The background level is the 90th percentile of the
    grey values (a sketch is mostly paper); dividing by it turns tinted or grey paper white. Very dark
    backgrounds are left alone. The image is padded to a square with white, keeping its aspect ratio."""
    from PIL import Image, ImageOps
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.float32)
    bg = np.percentile(a, 90)
    if bg >= DARK:
        a = np.clip(a * 255.0 / bg, 0, 255)
    img = Image.fromarray(a.astype(np.uint8))
    img = ImageOps.pad(img, (SIDE, SIDE), method=Image.Resampling.LANCZOS, color=255)
    return np.asarray(img, dtype=np.uint8)


def _prepare_chunk(paths: list[str]) -> np.ndarray:
    return np.stack([prepare(RAW_DIR / p) for p in paths])


def paths(split: str) -> dict[str, Path]:
    return {"images": DATA_DIR / f"{split}_images.u8", "labels": DATA_DIR / f"{split}_labels.npy",
            "files": DATA_DIR / f"{split}_files.txt"}


def stage_data() -> None:
    cats = categories()
    index = {c: i for i, c in enumerate(cats)}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "categories.json").write_text(json.dumps(cats) + "\n")
    for split in ("train", "test"):
        p = paths(split)
        if p["labels"].exists():
            print(f"data: {split} already done")
            continue
        rows = _listed(split)
        files = [r[0] for r in rows]
        chunks = [files[i:i + 500] for i in range(0, len(files), 500)]
        out = np.memmap(p["images"], dtype=np.uint8, mode="w+", shape=(len(files), SIDE, SIDE))
        t0, done = time.time(), 0
        with ProcessPoolExecutor(os.cpu_count()) as pool:
            for i, arr in enumerate(pool.map(_prepare_chunk, chunks)):
                out[i * 500:i * 500 + len(arr)] = arr
                done += len(arr)
                if (i + 1) % 20 == 0:
                    print(f"  data {split}: {done:,}/{len(files):,} ({time.time() - t0:.0f} s)", flush=True)
        out.flush()
        p["files"].write_text("\n".join(files) + "\n")
        np.save(p["labels"], np.array([index[c] for _, c in rows], dtype=np.int16))   # written last = done
        print(f"data: {split} {len(files):,} images in {time.time() - t0:.0f} s")


def load_images(split: str) -> np.ndarray:
    n = len(np.load(paths(split)["labels"]))
    return np.memmap(paths(split)["images"], dtype=np.uint8, mode="r", shape=(n, SIDE, SIDE))


def load_labels(split: str) -> np.ndarray:
    return np.load(paths(split)["labels"]).astype(np.int64)


def reviewed(split: str) -> dict[str, np.ndarray]:
    """Row numbers (into the split's arrays) of the hand-reviewed images: 'kept' and 'all'."""
    files = paths(split)["files"].read_text().split()
    row = {f: i for i, f in enumerate(files)}
    kept, seen = [], []
    for line in REVIEW.read_text().splitlines()[1:]:
        path, _, sp, decision, _ = line.split("\t")
        if sp != split:
            continue
        seen.append(row[path])
        if decision == "K":
            kept.append(row[path])
    return {"kept": np.array(sorted(kept)), "all": np.array(sorted(seen))}


# --- shared training helpers ----------------------------------------------------------------------

def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_tensor(batch: np.ndarray, dev) -> torch.Tensor:
    """uint8 (B, H, W) → float (B, 1, H, W) in [-1, 1]."""
    return torch.from_numpy(np.array(batch)).to(dev).float().div_(127.5).sub_(1).unsqueeze(1)


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _log(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# --- VQ-VAE ---------------------------------------------------------------------------------------

def build_vq(cfg: Config):
    from .vqvae import VQVAE
    return VQVAE(cfg.vq_k, cfg.vq_dim, tuple(cfg.vq_chans), cfg.vq_blocks)


@torch.no_grad()
def vq_eval(model, images: np.ndarray, dev, n: int = 2048, batch: int = 128) -> dict:
    """Reconstruction on the first n images of a split: mean absolute error in grey levels (0–255),
    the share of 'ink' pixels (darker than 128) that come back as ink, and codebook usage."""
    model.eval()
    l1, used = 0.0, torch.zeros(model.quant.k, dtype=torch.bool, device=dev)
    ink_hit: float = 0
    ink_all: float = 0
    for i in range(0, min(n, len(images)), batch):
        x = to_tensor(images[i:i + batch], dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            rec, idx, _ = model(x)
        rec = rec.clamp(-1, 1)
        l1 += (rec - x).abs().sum().item() * 127.5
        ink = x < 0
        ink_all += ink.sum().item()
        ink_hit += (ink & (rec < 0)).sum().item()
        used[idx.unique()] = True
    m = min(n, len(images))
    return {"mae_grey": round(l1 / (m * SIDE * SIDE), 3), "ink_recall": round(ink_hit / max(ink_all, 1), 4),
            "codes_used": int(used.sum().item())}


def stage_vq(cfg: Config) -> Path:
    dev = device()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    last, best_path, log = RUN_DIR / "vq_last.pt", RUN_DIR / "vq_best.pt", RUN_DIR / "vq_metrics.jsonl"
    train, test = load_images("train"), load_images("test")
    model = build_vq(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vq_lr)
    steps = len(train) // cfg.vq_batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.vq_lr, total_steps=cfg.vq_epochs * steps, pct_start=0.05)
    epoch, step, best = 0, 0, float("inf")
    if last.exists():
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        epoch, step, best = st["epoch"], st["step"], st["best"]
        print(f"vq: resumed at epoch {epoch + 1}, step {step}/{steps}", flush=True)
    print(f"vq: {sum(p.numel() for p in model.parameters()):,} parameters, {steps} steps/epoch", flush=True)
    while epoch < cfg.vq_epochs:
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train))
        t0, t_ck, sums, seen = time.time(), time.time(), np.zeros(2), 0
        for s in range(step, steps):
            ix = np.sort(order[s * cfg.vq_batch:(s + 1) * cfg.vq_batch])   # sorted reads from the memmap
            x = to_tensor(train[ix], dev)
            flip = torch.rand(len(x), device=dev) < 0.5                     # mirror half the batch
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                rec, _, commit = model(x)
            rec_loss = F.l1_loss(rec, x)
            loss = rec_loss + cfg.vq_beta * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if s % 50 == 0:
                sums += (rec_loss.item(), commit.item())
                seen += 1
            if time.time() - t_ck > cfg.checkpoint_minutes * 60:
                _save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                       "epoch": epoch, "step": s + 1, "best": best, "config": asdict(cfg)}, last)
                t_ck = time.time()
                print(f"  vq: epoch {epoch + 1} step {s + 1}/{steps} rec {sums[0] / seen:.4f}", flush=True)
        rec = {"epoch": epoch + 1, "train_rec_l1": round(sums[0] / max(seen, 1), 5),
               "train_commit": round(sums[1] / max(seen, 1), 5), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0)} | {f"test_{k}": v for k, v in vq_eval(model, test, dev).items()}
        _log(log, rec)
        if rec["test_mae_grey"] < best:
            best = rec["test_mae_grey"]
            _save({"model": model.state_dict(), "config": asdict(cfg), **rec}, best_path)
        epoch, step = epoch + 1, 0
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "epoch": epoch, "step": 0, "best": best, "config": asdict(cfg)}, last)
    return best_path


# --- CLI ------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["data", "vq"])
    ap.add_argument("--vq-epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = Config()
    if a.vq_epochs:
        cfg.vq_epochs = a.vq_epochs
    if a.stage == "data":
        stage_data()
    elif a.stage == "vq":
        stage_vq(cfg)


if __name__ == "__main__":
    main()
