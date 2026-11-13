"""Quick, Draw! drawings as strokes, in drawing order (the 'simplified' ndjson files).

The 28x28 bitmaps (data.py) lose the order the strokes were drawn in. These files keep it: every
drawing is a list of strokes, each a list of x and a list of y in 0..255, already simplified with
Ramer-Douglas-Peucker (epsilon 2). Line k of a class's file is row k of its bitmap file (checked:
pixel correlation 0.995 between the two, 0.19 for a row off by one), so a stroke model and a bitmap
model can be scored on exactly the same drawings.

    download_first_n_strokes("cat", 15500) -> StrokeSet  (cached in data/raw/quickdraw_strokes/)

A StrokeSet stores every point of every drawing in one array, (x, y, end of stroke), with offsets
saying where each drawing starts. `padded()` turns a selection into fixed-size model input.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

from .data import ML_DIR
from .raster import Stroke

STROKES_DIR = ML_DIR / "data" / "raw" / "quickdraw_strokes"
SIMPLIFIED_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
CHUNK = 8 << 20          # bytes per range request


@dataclass
class StrokeSet:
    points: np.ndarray       # (total points, 3) uint8: x, y, 1 if the stroke ends at this point
    offsets: np.ndarray      # (n + 1,) int64: drawing i is points[offsets[i]:offsets[i + 1]]
    recognized: np.ndarray   # (n,) bool: whether the Quick, Draw! game recognised it at the time

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def drawing(self, i: int) -> np.ndarray:
        return self.points[self.offsets[i]:self.offsets[i + 1]]

    def strokes(self, i: int) -> list[Stroke]:
        """Drawing i as ([xs], [ys]) strokes, the format raster.rasterize takes."""
        d = self.drawing(i)
        ends = np.flatnonzero(d[:, 2]) + 1
        return [(seg[:, 0].tolist(), seg[:, 1].tolist()) for seg in np.split(d, ends) if len(seg)]

    def padded(self, idx: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """(len(idx), max_len, 3) uint8 points (longer drawings cut at max_len) and their lengths."""
        out = np.zeros((len(idx), max_len, 3), np.uint8)
        lengths = np.zeros(len(idx), np.int64)
        for row, i in enumerate(idx):
            d = self.drawing(int(i))[:max_len]
            out[row, :len(d)] = d
            if len(d):
                out[row, len(d) - 1, 2] = 1          # a cut drawing still ends its last stroke
            lengths[row] = len(d)
        return out, lengths


def encode(drawings: list[tuple[list, bool]]) -> StrokeSet:
    """Parsed 'drawing' fields (+ recognized flags) -> StrokeSet."""
    pts, offsets, rec = [], [0], []
    for strokes, recognized in drawings:
        for xs, ys in strokes:
            seg = np.zeros((len(xs), 3), np.uint8)
            seg[:, 0], seg[:, 1] = np.clip(xs, 0, 255), np.clip(ys, 0, 255)
            seg[-1, 2] = 1
            pts.append(seg)
        offsets.append(offsets[-1] + sum(len(xs) for xs, _ in strokes))
        rec.append(recognized)
    points = np.concatenate(pts) if pts else np.zeros((0, 3), np.uint8)
    return StrokeSet(points, np.array(offsets, np.int64), np.array(rec, bool))


def cache_path(category: str, n: int):
    return STROKES_DIR / f"{category.replace(' ', '_')}_{n}.npz"


def load_cached(category: str, n: int) -> StrokeSet | None:
    for f in sorted(STROKES_DIR.glob(f"{category.replace(' ', '_')}_*.npz")):
        have = int(f.stem.rsplit("_", 1)[1])
        if have >= n:
            z = np.load(f)
            s = StrokeSet(z["points"], z["offsets"], z["recognized"])
            return _first(s, n)
    return None


def _first(s: StrokeSet, n: int) -> StrokeSet:
    end = s.offsets[n]
    return StrokeSet(s.points[:end], s.offsets[:n + 1], s.recognized[:n])


def download_first_n_strokes(category: str, n: int, retries: int = 5) -> StrokeSet:
    """The first n drawings of a class, fetched a few MB at a time from the start of its file.
    Cached; the file is written atomically, so an interrupted download leaves nothing half-done."""
    cached = load_cached(category, n)
    if cached is not None:
        return cached
    url = SIMPLIFIED_URL + urllib.parse.quote(category) + ".ndjson"
    drawings: list[tuple[list, bool]] = []
    start, tail = 0, b""
    while len(drawings) < n:
        chunk = _get_range(url, start, start + CHUNK - 1, retries)
        if not chunk:
            break                                            # end of file
        lines = (tail + chunk).split(b"\n")
        tail = lines.pop()                                   # the last line may be cut off
        for line in lines:
            if line.strip():
                d = json.loads(line)
                drawings.append((d["drawing"], bool(d.get("recognized", True))))
        start += len(chunk)
        if len(chunk) < CHUNK and tail.strip():              # file ended without a newline
            d = json.loads(tail)
            drawings.append((d["drawing"], bool(d.get("recognized", True))))
            break
    if len(drawings) < n:
        raise ValueError(f"{category} has only {len(drawings):,} drawings")
    s = encode(drawings[:n])
    STROKES_DIR.mkdir(parents=True, exist_ok=True)
    final = cache_path(category, n)
    tmp = final.with_name(final.stem + ".tmp.npz")
    np.savez(tmp, points=s.points, offsets=s.offsets, recognized=s.recognized)
    os.replace(tmp, final)
    return s


def _get_range(url: str, first: int, last: int, retries: int) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 416:                                # asked past the end of the file
                return b""
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")
