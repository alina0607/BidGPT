"""Cat faces as line drawings, drawn by a GPT one image patch at a time.

Data: the CAT dataset (Zhang, Sun, Tang, "Cat Head Detection", ECCV 2008; archive.org/details/CAT_DATASET),
about 10,000 cat photos, each with 9 head points: the two eyes, the mouth and 3 points on each ear. For research
use only: the authors state they do not hold the photos' copyright, so nothing derived from them leaves this
machine. Each face is aligned by its points (eyes level, a square around all 9 points with a 20% margin), turned
into a line drawing by Informative Drawings (infdraw.py) in two styles, "contour" (bold ink) and "anime" (fine
pencil), and shrunk to 128×128. Faces whose square reaches more than 10% outside the photo are left out.

The image model follows pixsketch.py (DALL·E, Taming Transformers): a VQ-VAE (vqvae.py) turns a 128×128 drawing
into 16×16 = 256 codes, and a decoder-only GPT (llm/model.py) learns the sequence

    [BOS] x0 y0 … x8 y8  c0 … c255

the 9 points first (2 px bins on the 128 canvas), then the codes in raster order. With the points first the
model lays out the face (where the ears and eyes go) before drawing it, as B2 laid in a sketch with one stroke
per part; nothing makes the image follow the points except what the model learns from the data.

    python -m sellanything_ml.sketch.catface data
    python -m sellanything_ml.sketch.catface vq --style anime
    python -m sellanything_ml.sketch.catface tokens --style anime
    python -m sellanything_ml.sketch.catface gpt --style anime
    python -m sellanything_ml.sketch.catface sample --style anime

The first VQ-VAE, trained on L1 alone, gives blurred lines: unsure whether a line is here or a pixel over, L1
pays least for a grey band in between. `vqgan` (variant "gan") continues its training with the adversarial
loss of Taming Transformers (§3.1): a PatchGAN discriminator learns to tell real drawing patches from
reconstructed ones and the decoder learns to fool it, so it draws a line somewhere instead of a grey band.

    python -m sellanything_ml.sketch.catface vqgan --style anime        # → runs/catface/anime_gan/
    python -m sellanything_ml.sketch.catface tokens --style anime --variant gan   (then gpt, sample)

Style "color" (stage `color`, after `data` and after tools/fgmask.swift has written a subject mask for every
photo): the cat is cut out with macOS Vision's subject mask, kept only inside a face outline drawn from the 9
points (ears, then a smooth curve round the cheeks and chin, so a person or the wall between the ears is cut
away), set on white, drawn in contour lines, and coloured with flat colours: 6 for the fur and, separately, 3
inside a circle round each eye, so the eyes keep their yellow, green or blue instead of merging into the fur. The
darkest colour is lifted to 40 of 255 so black lines still show on black fur. 128×128 RGB, same splits.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from . import infdraw
from .data import ML_DIR
from .pixsketch import _log, _save, device, to_tensor, vq_eval

RAW_DIR = ML_DIR / "data" / "raw" / "cat_dataset"
DATA_DIR = ML_DIR / "data" / "processed" / "catface"
RUN_DIR = ML_DIR / "runs" / "catface"
STYLES = {"contour": "contour_style", "anime": "anime_style"}
COLOR = "color"
EYE_R = 0.22                            # eye circle radius, share of the eye-to-eye distance (checked on a sheet)
DARKEST = 40                            # the darkest fill colour, of 255
BIG, SIDE = 256, 128                    # Informative Drawings runs at 256; the models see 128
MIN_INSIDE = 0.9
PBINS = 64                              # point bins of 2 px on the 128 canvas


@dataclass
class Config:
    style: str = "anime"
    variant: str = ""                   # "" = L1-only VQ-VAE; "gan" = continued with the adversarial loss
    seed: int = 0
    vq_k: int = 1024
    vq_dim: int = 64
    vq_chans: tuple = (32, 128, 256)
    vq_blocks: int = 1
    vq_beta: float = 0.25
    vq_lr: float = 2e-4
    vq_batch: int = 64
    vq_epochs: int = 30
    gan_epochs: int = 15
    disc_weight: float = 0.8            # Taming Transformers' default
    disc_start: int = 1                 # epochs the discriminator learns alone first: a random one's gradient
                                        # undid the trained decoder (L1 0.082 → 0.193 in one step, CPU test)
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 64
    epochs: int = 20                    # about 6 min each at 1.5 s/step (linegpt's rate on this Mac)
    warmup_steps: int = 200


# --- data -----------------------------------------------------------------------------------------

def align(img, p: np.ndarray):
    """Photo and its 9 points → (RGB face BIG×BIG, which pixels came from the photo, the points on the face)."""
    from PIL import Image
    (lx, ly), (rx, ry) = p[0], p[1]
    ang = math.atan2(ry - ly, rx - lx)
    c, s = math.cos(-ang), math.sin(-ang)
    cx, cy = p[:2].mean(0)
    q = (p - [cx, cy]) @ np.array([[c, -s], [s, c]]).T
    lo, hi = q.min(0), q.max(0)
    side = (hi - lo).max() * 1.2
    ox, oy = (lo + hi) / 2 - side / 2
    scale = side / BIG
    a, b = scale * c, scale * s
    coeffs = (a, b, cx + c * ox + s * oy, -b, a, cy - s * ox + c * oy)
    face = img.convert("RGB").transform((BIG, BIG), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    inside = np.asarray(Image.new("L", img.size, 255).transform((BIG, BIG), Image.Transform.AFFINE, coeffs, fillcolor=0)) > 0
    return np.asarray(face), inside, (q - [ox, oy]) / scale


def _members() -> list[tuple[Path, str]]:
    out = []
    for z in sorted(RAW_DIR.glob("CAT_DATASET_*.zip")):
        names = set(zipfile.ZipFile(z).namelist())
        out += [(z, m) for m in sorted(names) if m.lower().endswith(".jpg") and m + ".cat" in names]
    return out


def mirror_order(points: np.ndarray) -> list[int]:
    """Which point each point becomes in the mirror image: the eyes swap, the mouth stays, and each left-ear
    point becomes the right-ear point whose mean position is nearest its own mirrored mean (the annotation
    does not say how the two ears' points correspond, so the data decides)."""
    from itertools import permutations
    m = points.mean(0)
    mirrored_left = np.c_[SIDE - m[3:6, 0], m[3:6, 1]]
    best = min(permutations(range(3)), key=lambda pm: np.abs(mirrored_left - m[[6 + i for i in pm]]).sum())
    right = [6 + i for i in best]
    order = [1, 0, 2] + right + [0] * 3
    for k, r in enumerate(right):
        order[r] = 3 + k
    return order


def stage_data(batch: int = 256) -> None:
    from PIL import Image
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    members = _members()
    dev = device()
    nets = {k: infdraw.load(v, dev) for k, v in STYLES.items()}
    keep: list[str] = []
    lines: dict[str, list[np.ndarray]] = {k: [] for k in STYLES}
    pts: list[np.ndarray] = []
    zips: dict[Path, zipfile.ZipFile] = {}
    t0 = time.time()
    for i in range(0, len(members), batch):
        faces, masks = [], []
        for zpath, m in members[i:i + batch]:
            z = zips.setdefault(zpath, zipfile.ZipFile(zpath))
            v = [float(x) for x in z.read(m + ".cat").decode().split()]
            p = np.array(v[1:1 + 2 * int(v[0])]).reshape(-1, 2)
            face, inside, q = align(Image.open(io.BytesIO(z.read(m))), p)
            if inside.mean() < MIN_INSIDE:
                continue
            faces.append(face)
            masks.append(inside)
            pts.append(q / (BIG / SIDE))
            keep.append(f"{zpath.stem}/{m}")
        if not faces:
            continue
        rgb, mask = np.stack(faces), np.stack(masks)
        for k, net in nets.items():
            y = np.where(mask, infdraw.draw(net, rgb), 1.0)
            small = [np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.Resampling.LANCZOS)) for v in y]
            lines[k].append(np.stack(small))
        print(f"  data: {min(i + batch, len(members))}/{len(members)} photos, {len(keep)} kept, "
              f"{time.time() - t0:.0f} s", flush=True)
    pts_all = np.stack(pts).astype(np.float32)
    order = np.random.default_rng(0).permutation(len(keep))
    n_val = n_test = len(keep) // 10
    splits = {"validation": order[:n_val], "test": order[n_val:n_val + n_test], "train": order[n_val + n_test:]}
    arrays = {k: np.concatenate(v) for k, v in lines.items()}
    for split, ix in splits.items():
        np.savez_compressed(DATA_DIR / f"{split}.npz", points=pts_all[ix], ids=np.array(keep)[ix],
                            **cast(dict[str, Any], {k: a[ix] for k, a in arrays.items()}))
    report = {"photos": len(members), "kept": len(keep), "left_out_outside": len(members) - len(keep),
              "min_inside": MIN_INSIDE, "splits": {k: len(v) for k, v in splits.items()},
              "mirror_order": mirror_order(pts_all[splits["train"]]), "seconds": round(time.time() - t0)}
    (DATA_DIR / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


def load(split: str) -> dict[str, np.ndarray]:
    z = np.load(DATA_DIR / f"{split}.npz")
    out = {k: z[k] for k in z.files}
    if (DATA_DIR / f"color_{split}.npz").exists():
        out[COLOR] = np.load(DATA_DIR / f"color_{split}.npz")[COLOR]
    return out


def image_ch(style: str) -> int:
    return 3 if style == COLOR else 1


def to_x(batch: np.ndarray, dev) -> torch.Tensor:
    """uint8 (B, H, W) or (B, H, W, 3) → float (B, C, H, W) in [-1, 1]."""
    if batch.ndim == 3:
        return to_tensor(batch, dev)
    return torch.from_numpy(np.array(batch)).to(dev).float().div_(127.5).sub_(1).permute(0, 3, 1, 2)


def grey(images: np.ndarray) -> np.ndarray:
    return images if images.ndim == 3 else (images @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


@torch.no_grad()
def eval_rec(model, images: np.ndarray, dev, n: int = 2048) -> dict:
    """Grey style: pixsketch's measures. Colour: mean absolute error per channel (0–255) and codebook use."""
    if images.ndim == 3:
        return vq_eval(model, images, dev, n)
    model.eval()
    err, used = 0.0, torch.zeros(model.quant.k, dtype=torch.bool, device=dev)
    for i in range(0, min(n, len(images)), 128):
        x = to_x(images[i:i + 128], dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            rec, idx, _ = model(x)
        err += (rec.clamp(-1, 1) - x).abs().mean(dim=(1, 2, 3)).sum().item() * 127.5
        used[idx.unique()] = True
    return {"mae_grey": round(err / min(n, len(images)), 3), "codes_used": int(used.sum().item())}


# --- colour data ----------------------------------------------------------------------------------

def face_region(q: np.ndarray, side: int) -> np.ndarray:
    """On the aligned face (eyes level): the ear points, then a smooth curve from the right ear's base round the
    right cheek, the chin and the left cheek to the left ear's base (cheeks and chin placed from the eyes and the
    mouth), filled and grown by a tenth of the eye distance."""
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import binary_dilation
    from skimage.draw import polygon
    (lx, _), (rx, _), (mx, my) = q[0], q[1], q[2]
    d = rx - lx
    jaw = np.array([q[8], (rx + 0.55 * d, my - 0.1 * d), (rx + 0.3 * d, my + 0.3 * d), (mx, my + 0.45 * d),
                    (lx - 0.3 * d, my + 0.3 * d), (lx - 0.55 * d, my - 0.1 * d), q[3]])
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(jaw, axis=0), axis=1))]
    jaw = CubicSpline(t, jaw, bc_type="natural")(np.linspace(0, t[-1], 60))
    outline = np.vstack([q[3:9], jaw[1:-1]])
    m = np.zeros((side, side), bool)
    rr, cc = polygon(outline[:, 1], outline[:, 0], (side, side))
    m[rr, cc] = True
    return binary_dilation(m, iterations=max(1, int(0.1 * d)))


def _kmeans(px: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    c = px[np.random.default_rng(0).choice(len(px), min(k, len(px)), replace=False)]
    for _ in range(iters):
        lab = ((px[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([px[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(len(c))])
    return c[lab]


def _subject_mask(photo_name: str, p: np.ndarray, shape) -> np.ndarray:
    """The Vision subject instance that covers both eye points (every pixel if none does)."""
    from PIL import Image
    for m in sorted((RAW_DIR / "masks").glob(Path(photo_name).stem + "_mask*.png")):
        a = np.asarray(Image.open(m).convert("L")) > 127
        if all(a[min(int(round(y)), a.shape[0] - 1), min(int(round(x)), a.shape[1] - 1)] for x, y in p[:2]):
            return a
    return np.ones(shape, bool)


def stage_color(batch: int = 128, dev_name: str = "") -> None:
    from PIL import Image
    from scipy.ndimage import median_filter
    from skimage.draw import disk
    dev = torch.device(dev_name) if dev_name else device()
    net = infdraw.load(STYLES["contour"], dev)
    t0, stats = time.time(), {"no_subject": 0}
    for split in ("train", "validation", "test"):
        ids = load(split)["ids"]
        out = np.zeros((len(ids), SIDE, SIDE, 3), np.uint8)
        for i in range(0, len(ids), batch):
            faces, keeps, flats = [], [], []
            for full in ids[i:i + batch]:
                zstem, m = full.split("/", 1)
                name = f"{zstem}__{m.replace('/', '__')}"
                photo = Image.open(RAW_DIR / "photos" / name)
                v = [float(x) for x in zipfile.ZipFile(RAW_DIR / f"{zstem}.zip").read(m + ".cat").decode().split()]
                p = np.array(v[1:1 + 2 * int(v[0])]).reshape(-1, 2)
                fg = _subject_mask(name, p, (photo.height, photo.width))
                stats["no_subject"] += int(fg.all())
                face, inside, q = align(photo, p)
                fgal = align(Image.fromarray((fg * 255).astype(np.uint8)), p)[0][..., 0] > 127
                keep = fgal & inside & face_region(q, BIG)
                face = np.where(keep[..., None], face, 255).astype(np.uint8)
                eye = np.zeros(keep.shape, bool)
                for x, y in q[:2]:
                    rr, cc = disk((y, x), EYE_R * (q[1, 0] - q[0, 0]), shape=keep.shape)
                    eye[rr, cc] = True
                eye &= keep
                flat = np.full(face.shape, 255.0)
                sm = median_filter(face, size=(7, 7, 1)).astype(np.float64)
                if (keep & ~eye).any():
                    flat[keep & ~eye] = _kmeans(sm[keep & ~eye], 6)
                if eye.any():
                    flat[eye] = _kmeans(face[eye].astype(np.float64), 3)
                flat = np.where(keep[..., None], DARKEST + flat * (255 - DARKEST) / 255, 255)
                faces.append(face)
                keeps.append(keep)
                flats.append(flat)
            lines = infdraw.draw(net, np.stack(faces))
            for j, (keep, flat, ln) in enumerate(zip(keeps, flats, lines)):
                img = np.where(keep[..., None], flat * np.where(keep, ln, 1.0)[..., None], 255)
                out[i + j] = np.asarray(Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((SIDE, SIDE), Image.Resampling.LANCZOS))
            print(f"  color {split}: {min(i + batch, len(ids))}/{len(ids)}, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(DATA_DIR / f"color_{split}.npz", color=out, ids=ids)
    stats["seconds"] = round(time.time() - t0)
    (DATA_DIR / "color_report.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats), flush=True)


def mirror(images: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = json.loads((DATA_DIR / "report.json").read_text())["mirror_order"]
    p = points.copy()
    p[..., 0] = SIDE - p[..., 0]
    out = np.empty_like(p)
    out[:, order] = p
    return images[:, :, ::-1].copy(), out


# --- VQ-VAE ---------------------------------------------------------------------------------------

def run_dir(cfg: Config) -> Path:
    return RUN_DIR / (f"{cfg.style}_{cfg.variant}" if cfg.variant else cfg.style)


def vq_file(cfg: Config) -> Path:
    # with the adversarial loss the last epoch is kept: blur lowers the pixel error, so the lowest-error
    # epoch would be the blurriest one
    return run_dir(cfg) / ("vq_gan_last.pt" if cfg.variant == "gan" else "vq_best.pt")


def build_vq(cfg: Config):
    from .vqvae import VQVAE
    return VQVAE(cfg.vq_k, cfg.vq_dim, tuple(cfg.vq_chans), cfg.vq_blocks, image_ch(cfg.style))


def stage_vq(cfg: Config) -> Path:
    """Train on the train split (half of each batch mirrored), keep the weights with the lowest validation error."""
    dev, out = device(), run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    train, val = load("train")[cfg.style], load("validation")[cfg.style]
    model = build_vq(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vq_lr)
    steps = len(train) // cfg.vq_batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.vq_lr, total_steps=cfg.vq_epochs * steps, pct_start=0.05)
    log, best = out / "vq_metrics.jsonl", float("inf")
    log.write_text("")
    print(f"vq {cfg.style}: {sum(p.numel() for p in model.parameters()):,} parameters, {len(train)} images, "
          f"{steps} steps/epoch", flush=True)
    for epoch in range(cfg.vq_epochs):
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train))
        t0, sums = time.time(), np.zeros(2)
        for s in range(steps):
            x = to_x(train[np.sort(order[s * cfg.vq_batch:(s + 1) * cfg.vq_batch])], dev)
            flip = torch.rand(len(x), device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                rec, _, commit = model(x)
            rec_loss = F.l1_loss(rec, x)
            opt.zero_grad(set_to_none=True)
            (rec_loss + cfg.vq_beta * commit).backward()
            opt.step()
            sched.step()
            sums += (rec_loss.item(), commit.item())
        rec = {"epoch": epoch + 1, "train_rec_l1": round(sums[0] / steps, 5), "train_commit": round(sums[1] / steps, 5),
               "seconds": round(time.time() - t0)} | {f"val_{k}": v for k, v in eval_rec(model, val, dev).items()}
        _log(log, rec)
        if rec["val_mae_grey"] < best:
            best = rec["val_mae_grey"]
            _save({"model": model.state_dict(), "config": asdict(cfg), **rec}, out / "vq_best.pt")
    return out / "vq_best.pt"


class PatchDisc(torch.nn.Module):
    """PatchGAN discriminator (Isola et al. 2017), the one Taming Transformers uses: one logit per output cell,
    each judging a patch of about 70×70 px of a 128×128 drawing."""

    def __init__(self, ch: int = 64, layers: int = 3, image_ch: int = 1):
        super().__init__()
        nn = torch.nn
        seq: list = [nn.Conv2d(image_ch, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = ch
        for _ in range(1, layers):
            seq += [nn.Conv2d(c, c * 2, 4, 2, 1), nn.GroupNorm(8, c * 2), nn.LeakyReLU(0.2, True)]
            c *= 2
        seq += [nn.Conv2d(c, c * 2, 4, 1, 1), nn.GroupNorm(8, c * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(c * 2, 1, 4, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def stage_vqgan(cfg: Config) -> Path:
    """Continue the style's L1 VQ-VAE with Taming's hinge adversarial loss. The adversarial term's weight is set
    every step from the gradients at the decoder's last layer, λ = ‖∇L_rec‖ / (‖∇L_gan‖ + 1e-4) × disc_weight
    (Taming eq. 7), so neither loss drowns the other."""
    dev, out = device(), run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    train, val = load("train")[cfg.style], load("validation")[cfg.style]
    model = load_vq(Config(style=cfg.style), dev).train()
    disc = PatchDisc(image_ch=image_ch(cfg.style)).to(dev)
    opt_g = torch.optim.Adam(model.parameters(), lr=cfg.vq_lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.vq_lr, betas=(0.5, 0.9))
    last_layer = model.decoder.net[-1].weight
    steps = len(train) // cfg.vq_batch
    log = out / "vq_gan_metrics.jsonl"
    log.write_text("")
    _log(log, {"epoch": 0} | {f"val_{k}": v for k, v in eval_rec(model, val, dev).items()})
    for epoch in range(cfg.gan_epochs):
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train))
        t0, sums = time.time(), np.zeros(4)
        for s in range(steps):
            x = to_x(train[np.sort(order[s * cfg.vq_batch:(s + 1) * cfg.vq_batch])], dev)
            flip = torch.rand(len(x), device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                rec, _, commit = model(x)
                g_loss = -disc(rec).float().mean()
            rec_loss = F.l1_loss(rec, x)
            g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_gan = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            lam = (g_rec.norm() / (g_gan.norm() + 1e-4)).clamp(0, 1e4).detach() * cfg.disc_weight
            lam = lam * (epoch >= cfg.disc_start)
            opt_g.zero_grad(set_to_none=True)
            (rec_loss + cfg.vq_beta * commit + lam * g_loss).backward()
            opt_g.step()
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                d_loss = F.relu(1 - disc(x).float()).mean() + F.relu(1 + disc(rec.detach()).float()).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            sums += (rec_loss.item(), g_loss.item(), d_loss.item(), lam.item())
        rec = {"epoch": epoch + 1, "train_rec_l1": round(sums[0] / steps, 5), "g_loss": round(sums[1] / steps, 4),
               "d_loss": round(sums[2] / steps, 4), "lambda": round(sums[3] / steps, 4),
               "seconds": round(time.time() - t0)} | {f"val_{k}": v for k, v in eval_rec(model, val, dev).items()}
        _log(log, rec)
        _save({"model": model.state_dict(), "config": asdict(cfg), **rec}, vq_file(cfg))
    return vq_file(cfg)


def load_vq(cfg: Config, dev):
    st = torch.load(vq_file(cfg), map_location="cpu", weights_only=False)
    model = build_vq(Config(**{k: v for k, v in st["config"].items() if k in Config.__dataclass_fields__}))
    model.load_state_dict(st["model"])
    return model.to(dev).eval()


# --- sequences ------------------------------------------------------------------------------------

def layout(cfg: Config) -> dict[str, int]:
    k = cfg.vq_k
    return {"x0": k, "y0": k + PBINS, "bos": k + 2 * PBINS, "vocab": k + 2 * PBINS + 1}


def point_tokens(points: np.ndarray, cfg: Config) -> np.ndarray:
    lay = layout(cfg)
    b = np.clip((points / (SIDE / PBINS)).astype(np.int64), 0, PBINS - 1)
    return np.stack([b[..., 0] + lay["x0"], b[..., 1] + lay["y0"]], -1).reshape(len(points), -1)


def stage_tokens(cfg: Config) -> None:
    """Every image of every split (and the mirror images of the train split) as [BOS] points codes."""
    dev = device()
    vq = load_vq(cfg, dev)
    lay = layout(cfg)
    for split in ("train", "validation", "test"):
        d = load(split)
        sets = [(d[cfg.style], d["points"])] + ([mirror(d[cfg.style], d["points"])] if split == "train" else [])
        rows = []
        for imgs, pts in sets:
            codes = np.concatenate([vq.encode(to_x(imgs[i:i + 256], dev)).reshape(-1, 256).cpu().numpy()
                                    for i in range(0, len(imgs), 256)])
            rows.append(np.concatenate([np.full((len(imgs), 1), lay["bos"]), point_tokens(pts, cfg), codes], 1))
        np.save(run_dir(cfg) / f"tokens_{split}.npy", np.concatenate(rows).astype(np.int16))
        print(split, sum(len(r) for r in rows), "rows", flush=True)


# --- GPT ------------------------------------------------------------------------------------------

def gpt_config(cfg: Config):
    from ..llm.model import GPTConfig
    return GPTConfig(vocab_size=layout(cfg)["vocab"], d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                     n_kv_heads=cfg.n_kv_heads, max_seq_len=1 + 18 + 256, dropout=cfg.dropout, abs_pos=True)


@torch.no_grad()
def val_loss(model, rows: np.ndarray, dev) -> dict:
    """Mean cross-entropy of the points and of the codes, separately."""
    model.eval()
    pts, codes = [], []
    for i in range(0, len(rows), 128):
        t = torch.from_numpy(rows[i:i + 128].astype(np.int64)).to(dev)
        logits, _ = model(t[:, :-1])
        nll = F.cross_entropy(logits.float().transpose(1, 2), t[:, 1:], reduction="none")
        pts.append(nll[:, :18].cpu())
        codes.append(nll[:, 18:].cpu())
    return {"val_points": round(float(torch.cat(pts).mean()), 4), "val_codes": round(float(torch.cat(codes).mean()), 4)}


def stage_gpt(cfg: Config) -> Path:
    from ..llm.model import GPT
    dev, out = device(), run_dir(cfg)
    train = np.load(out / "tokens_train.npy").astype(np.int64)
    val = np.load(out / "tokens_validation.npy").astype(np.int64)
    torch.manual_seed(cfg.seed)
    model = GPT(gpt_config(cfg)).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    steps = len(train) // cfg.batch
    total = steps * cfg.epochs
    lr_at = lambda st: cfg.lr * min(1.0, (st + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / total)))  # noqa: E731
    log, best, step = out / "gpt_metrics.jsonl", float("inf"), 0
    log.write_text("")
    print(f"gpt {cfg.style}: {model.num_params():,} parameters, {len(train)} train rows, {steps} steps/epoch", flush=True)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        order, t0, run = rng.permutation(len(train)), time.time(), 0.0
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            t = torch.from_numpy(train[order[s * cfg.batch:(s + 1) * cfg.batch]]).to(dev)
            _, loss = model(t[:, :-1], t[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
        rec = {"epoch": epoch + 1, "train_loss": round(run / steps, 4), "lr": round(lr_at(step), 7),
               "seconds": round(time.time() - t0)} | val_loss(model, val, dev)
        _log(log, rec)
        score = rec["val_points"] * 18 + rec["val_codes"] * 256
        if score < best:
            best = score
            _save({"model": model.state_dict(), "config": asdict(cfg), **rec}, out / "gpt_best.pt")
    return out / "gpt_best.pt"


# --- sampling and evaluation ----------------------------------------------------------------------

def stage_sample(cfg: Config, n: int = 48, temperature: float = 1.0, seed: int = 1) -> dict:
    """n new faces, a sheet of them, and how they compare with held-out real faces: distance to the nearest
    training drawing (copying), variety, and tokens of the wrong kind (a code where a point belongs or back)."""
    from PIL import Image
    from ..llm.model import GPT
    from .linegpt import diversity, nearest
    dev, out = device(), run_dir(cfg)
    st = torch.load(out / "gpt_best.pt", map_location="cpu", weights_only=False)
    model = GPT(gpt_config(cfg))
    model.load_state_dict(st["model"])
    model = model.to(dev).eval()
    vq, lay = load_vq(cfg, dev), layout(cfg)
    torch.manual_seed(seed)
    seq = model.generate(torch.full((n, 1), lay["bos"], device=dev), 18 + 256, temperature=temperature)[:, 1:].cpu()
    pts, codes = seq[:, :18], seq[:, 18:]
    wrong = int(((pts < cfg.vq_k) | (pts >= lay["bos"])).sum() + (codes >= cfg.vq_k).sum())
    imgs = vq.decode(codes.clamp(max=cfg.vq_k - 1).view(n, 16, 16).to(dev)).float().cpu().permute(0, 2, 3, 1).numpy()
    imgs = ((imgs + 1) * 127.5).clip(0, 255).astype(np.uint8)
    imgs = imgs[..., 0] if imgs.shape[-1] == 1 else imgs
    train, val = load("train")[cfg.style], load("validation")[cfg.style]
    sheet_imgs = imgs
    imgs, train, val = grey(imgs), grey(train), grey(val)      # the distances and variety are on grey levels
    report = {"style": cfg.style, "temperature": temperature, "seed": seed, "n": n, "wrong_kind_tokens": wrong,
              "nearest_train_median": round(float(np.median(nearest(imgs, train)[1])), 4),
              "real_val_nearest_train_median": round(float(np.median(nearest(val[:n], train)[1])), 4),
              "diversity": round(diversity(imgs), 4), "real_val_diversity": round(diversity(val[:200]), 4),
              "gpt_epoch": st["epoch"], "val_points": st["val_points"], "val_codes": st["val_codes"]}
    cols, side = 8, 132
    sheet = Image.new("RGB", (cols * side, math.ceil(n / cols) * side), "white")
    for i, im in enumerate(sheet_imgs):
        sheet.paste(Image.fromarray(im), ((i % cols) * side + 2, (i // cols) * side + 2))
    sheet.save(out / f"sample_t{temperature}_s{seed}.png")
    (out / f"sample_t{temperature}_s{seed}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["data", "color", "vq", "vqgan", "tokens", "gpt", "sample"])
    ap.add_argument("--style", default="anime", choices=list(STYLES) + [COLOR])
    ap.add_argument("--variant", default="", choices=["", "gan"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--vq-epochs", type=int, default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="", help="color stage: e.g. cpu, while the GPU trains something else")
    a = ap.parse_args()
    cfg = Config(style=a.style, variant="gan" if a.stage == "vqgan" else a.variant)
    if a.epochs:
        cfg.epochs = a.epochs
    if a.vq_epochs:
        cfg.vq_epochs = a.vq_epochs
    {"data": lambda: stage_data(), "color": lambda: stage_color(dev_name=a.device), "vq": lambda: stage_vq(cfg), "vqgan": lambda: stage_vqgan(cfg),
     "tokens": lambda: stage_tokens(cfg),
     "gpt": lambda: stage_gpt(cfg), "sample": lambda: stage_sample(cfg, a.n, a.temperature, a.seed)}[a.stage]()


if __name__ == "__main__":
    main()
