"""Quick, Draw! bitmap data: download, cache and split.

Only the first N drawings of each class are fetched, using an HTTP Range request, so a class costs
N x 784 bytes instead of the full 100+ MB file. See notebooks/01_sketch_classifier.ipynb for the
step-by-step explanation.
"""

from __future__ import annotations

import csv
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[2]  # .../ml
ITEMS_TSV = ML_DIR / "data" / "items.tsv"
RAW_DIR = ML_DIR / "data" / "raw" / "quickdraw"
BITMAP_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
IMAGE_BYTES = 28 * 28


def load_categories() -> list[str]:
    """All 345 Quick, Draw! class names, in items.tsv order (alphabetical)."""
    with open(ITEMS_TSV, encoding="utf-8") as f:
        return [row["en"] for row in csv.DictReader(f, delimiter="\t")]


def cache_path(category: str, n: int) -> Path:
    return RAW_DIR / f"{category.replace(' ', '_')}_{n}.npy"


def download_first_n(category: str, n: int, retries: int = 4) -> np.ndarray:
    """First n drawings of one class as a (n, 784) uint8 array, cached on disk.

    A cache file for a larger n is reused by slicing, so 12,000-image caches serve 6,000 requests.
    """
    for existing in sorted(RAW_DIR.glob(f"{category.replace(' ', '_')}_*.npy")):
        have = int(existing.stem.rsplit("_", 1)[1])
        if have >= n:
            return np.load(existing, mmap_mode="r")[:n].copy()

    url = BITMAP_URL + urllib.parse.quote(category) + ".npy"
    want = 128 + n * IMAGE_BYTES  # header is < 128 bytes
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes=0-{want - 1}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            header_len = int.from_bytes(raw[8:10], "little")
            data = np.frombuffer(raw, np.uint8, offset=10 + header_len, count=n * IMAGE_BYTES)
            images = data.reshape(n, IMAGE_BYTES)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path(category, n), images)
            return images
        except Exception:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


@dataclass
class Split:
    x: np.ndarray  # (N, 784) uint8
    y: np.ndarray  # (N,) int64


def load_splits(
    categories: list[str], per_class: int, n_val: int, n_test: int, seed: int = 0, workers: int = 8
) -> tuple[Split, Split, Split]:
    """Download (in parallel) and split every class into train / val / test.

    Within each class the order is shuffled with a fixed seed before splitting, so the split is
    reproducible and doesn't depend on the order drawings appear in the source file.
    """
    with ThreadPoolExecutor(workers) as pool:
        per_class_images = list(pool.map(lambda c: download_first_n(c, per_class), categories))

    rng = np.random.default_rng(seed)
    parts: dict[str, tuple[list, list]] = {"train": ([], []), "val": ([], []), "test": ([], [])}
    n_train = per_class - n_val - n_test
    for label, images in enumerate(per_class_images):
        images = images[rng.permutation(len(images))]
        bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val),
                  "test": (n_train + n_val, per_class)}
        for name, (a, b) in bounds.items():
            parts[name][0].append(images[a:b])
            parts[name][1].append(np.full(b - a, label, dtype=np.int64))

    def build(name: str) -> Split:
        return Split(np.concatenate(parts[name][0]), np.concatenate(parts[name][1]))

    return build("train"), build("val"), build("test")
