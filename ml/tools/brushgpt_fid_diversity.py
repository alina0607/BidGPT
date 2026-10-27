"""Brush GPT quality and diversity against held-out paintings: FID and KID to the AFHQ test paintings (never used for
training or early stopping by any run), and IconShop's Uniqueness and Novelty (R11 in docs/paper-notes).

brushgpt eval measures FID against the *training* paintings with 256 samples, which says how close a run gets to what
it trained on but not how well it generalises, and its two FIDs (generated vs validation) have different sample sizes.
Here every set is scored against the same test paintings with the same number of images, and next to each number is
the same number for *real* paintings the model never saw (the validation paintings, and an equal-size draw of training
paintings), so a reader can see how far a run is from real data on the same scale:

- FID (Heusel et al. 2017) with pytorch-fid's Inception weights, as R1 and brushgpt eval do. FID is biased upward at
  a few hundred images, so it compares only at equal n, which this tool keeps.
- KID (Binkowski et al. 2018): the unbiased squared MMD with a cubic polynomial kernel on the same Inception features,
  mean and standard deviation over random subsets; unlike FID it does not depend on n on average.
- Uniqueness: the share of generated paintings with no *other* generated painting above a cosine similarity threshold
  (IconShop: "occurring only once among all generated samples"). Novelty: the share with no training painting (or its
  mirror image, which the model also trains on) above it. IconShop uses CLIP image features and 0.98, a value it chose
  for black-on-white icons; the same scores for the real validation paintings are given beside them so the threshold
  can be judged on our paintings, and also at 0.90, 0.95 and 0.99. IconShop warns that high Uniqueness and Novelty
  can be "fake diversity" from broken outputs: read them with FID.

Renders and features of the real paintings are cached per token file (runs/brushgpt/_features/), so a second run
costs only its own samples.

    nice -n 19 env BRUSHGPT_DEVICE=cpu PYTHONPATH=. python tools/brushgpt_fid_diversity.py --run I_aligned_points_eyes \
        --data data/processed/brushgpt/tokens_runs_HI.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brushgpt_by_source import source_of  # noqa: E402

from sellanything_ml.sketch import brushgpt as G  # noqa: E402

CACHE = G.RUN_DIR / "_features"
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
CLIP_NAME = "openai/clip-vit-base-patch32"


def inception_features(images: np.ndarray, dev: str) -> np.ndarray:
    """(n, 2048) pool features of pytorch-fid's Inception (the FID network), images (n, H, W, 3) in 0–1."""
    import torch
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 50):
            t = torch.from_numpy(images[i:i + 50]).permute(0, 3, 1, 2).float().to(dev)
            out.append(net(t)[0].flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def clip_features(images: np.ndarray, dev: str) -> np.ndarray:
    """(n, 512) unit-length CLIP image embeddings (ViT-B/32), images (n, H, W, 3) in 0–1."""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(CLIP_NAME).to(dev).eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[None, :, None, None]
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 50):
            t = torch.from_numpy(images[i:i + 50]).permute(0, 3, 1, 2).float().to(dev)
            t = F.interpolate(t, size=224, mode="bicubic", antialias=True).clamp(0, 1)   # 256 → 224, square already
            e = model.get_image_features(pixel_values=(t - mean) / std)
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
            out.append(F.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


def fid(fa: np.ndarray, fb: np.ndarray) -> float:
    """‖μa − μb‖² + Tr(Σa + Σb − 2(ΣaΣb)^½) of two feature sets."""
    from scipy import linalg
    mu, sa, sb = fa.mean(0) - fb.mean(0), np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    return float(mu @ mu + np.trace(sa + sb - 2 * np.real(linalg.sqrtm(sa @ sb))))


def kid(fa: np.ndarray, fb: np.ndarray, subsets: int = 100, size: int = 100, seed: int = 0) -> tuple[float, float]:
    """KID × 1000 (mean, std over random subsets): unbiased MMD² with k(x, y) = (x·y / d + 1)³."""
    rng = np.random.default_rng(seed)
    d, size = fa.shape[1], min(size, len(fa), len(fb))
    vals = []
    for _ in range(subsets):
        x, y = fa[rng.choice(len(fa), size, replace=False)], fb[rng.choice(len(fb), size, replace=False)]
        kxx, kyy, kxy = ((x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3)
        m = size
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return round(float(np.mean(vals)) * 1000, 3), round(float(np.std(vals)) * 1000, 3)


def unique_novel(e: np.ndarray, train: np.ndarray) -> dict[str, dict[str, float]]:
    """Uniqueness and Novelty (%) of unit embeddings e at each threshold: no other row of e, and no row of train,
    with cosine similarity above it. Also the highest similarities themselves (median)."""
    within = e @ e.T
    np.fill_diagonal(within, -1)
    best_within, best_train = within.max(1), (e @ train.T).max(1)
    out = {f"{t:.2f}": {"uniqueness": round(100 * float((best_within <= t).mean()), 2),
                        "novelty": round(100 * float((best_train <= t).mean()), 2)} for t in THRESHOLDS}
    out["median_max_cos"] = {"within": round(float(np.median(best_within)), 4),
                             "to_train": round(float(np.median(best_train)), 4)}
    return out


def real_features(data: Path, dev: str) -> dict[str, np.ndarray]:
    """Renders' Inception and CLIP features of the train (and mirrored train), val and test paintings of a token file,
    cached by the file's hash. Real paintings are rendered from the plain stroke order, which paints the same picture
    as the overlap order (brushgpt checks this when it makes the file)."""
    digest = hashlib.sha256(data.read_bytes()).hexdigest()[:16]
    path = CACHE / f"{data.stem}_{digest}.npz"
    if path.exists():
        return dict(np.load(path))
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        rows, ids = G.load_rows(data, split)
        t0 = time.time()
        img = G.render_all(rows, dev)
        print(json.dumps({"rendered": split, "n": len(img), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        out[f"{split}_ids"] = ids
        out[f"{split}_inception"] = inception_features(img, dev)
        out[f"{split}_clip"] = clip_features(img, dev)
        if split == "train":
            out["train_mirror_clip"] = clip_features(img[:, :, ::-1].copy(), dev)
    np.savez(path, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=Path, required=True, help="the run's own token file")
    ap.add_argument("--n", type=int, default=0, help="generated paintings (default: as many as AFHQ validation cats)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = G.device()
    t0 = time.time()
    real = real_features(a.data, dev)
    afhq = {s: source_of(real[f"{s}_ids"]) == "afhq" for s in ("val", "test")}
    test_inc = real["test_inception"][afhq["test"]]
    val_inc, val_clip = real["val_inception"][afhq["val"]], real["val_clip"][afhq["val"]]
    n = a.n or len(val_inc)
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", dev)
    gen = G.sample(model, n, a.temperature, seed=a.seed, batch=a.batch)
    ok = G.valid(gen).all(1)
    img = G.render_all(gen[ok], dev)
    gen_inc, gen_clip = inception_features(img, dev), clip_features(img, dev)
    train_clip = np.concatenate([real["train_clip"], real["train_mirror_clip"]])
    draw = np.random.default_rng(0).choice(len(real["train_inception"]), len(val_inc), replace=False)
    train_inc = real["train_inception"][draw]

    sets = {"generated": gen_inc, "real_val": val_inc[:len(gen_inc)], "real_train_draw": train_inc[:len(gen_inc)]}
    report = {"run": a.run, "data": str(a.data), "temperature": a.temperature, "seed": a.seed, "device": dev,
              "n": {"generated": int(ok.sum()), "decoded_rate": round(float(ok.mean()), 4), "test_afhq": len(test_inc),
                    "each_set": len(gen_inc)},
              "fid_to_test": {k: round(fid(v, test_inc), 2) for k, v in sets.items()},
              "kid_x1000_to_test": {k: kid(v, test_inc) for k, v in sets.items()},
              "clip": CLIP_NAME,
              "generated": unique_novel(gen_clip, train_clip),
              "real_val": unique_novel(val_clip[:len(gen_clip)], train_clip)}
    report["minutes"] = round((time.time() - t0) / 60, 1)
    out = a.out or G.RUN_DIR / a.run / f"fid_diversity_t{a.temperature}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), tokens=gen, inception=gen_inc, clip=gen_clip)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
