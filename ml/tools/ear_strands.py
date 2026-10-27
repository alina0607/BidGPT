"""Can an ear crop teach the ear model its fine detail? (owner, 2026-10-18: ears where no fine detail shows should go;
whatever the coat, the inner ear's tufts are fine light strands, so an ear where no fine light strand shows is too
dark, too blurred or too small.) The score is the share of the crop's pixels (white padding left out) that are a fine
light line: brighter than the brighter of their two sides 1 or 2 px away, in the best of four directions, by more
than LINE. Checked on seven ears the owner judged (2026-10-18): the two without detail scored 0.004 and 0.014, the five
good ones 0.070-0.228 (one of them only 165 px wide: resolution alone would have dropped it). On all 29,903 ears
(2026-10-18) a sheet of six random ears per band (ear_strands_bands.png) showed: below 0.03 almost all blurred, dark
or seen from behind; 0.03-0.04 mixed; above 0.04 mostly good. The threshold 0.03 is the conservative cut; the mixed
band stays for the owner. Not caught by the score: ears seen from behind (no inner ear), left to the owner.

Score every ear in the review folder (ml/data/processed/review/2_耳朵), write ear_strands.csv, and with --move put the
ears below --threshold into a 建議不要/ folder next to them for the owner to sweep (drag back any good one). The
review's collect step counts 建議不要 as rejected, like 不要.

    PYTHONPATH=. python tools/ear_strands.py                       # scores and a sheet of borderline ears
    PYTHONPATH=. python tools/ear_strands.py --move --threshold 0.03
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

EARS = Path(__file__).resolve().parents[1] / "data" / "processed" / "review" / "2_耳朵"
LINE = 0.06
SUGGEST = "建議不要"


def bright_lines(lum: np.ndarray, s: int) -> np.ndarray:
    """Per pixel, how much brighter it is than the brighter of its two sides s px away, best of four directions."""
    p = np.pad(lum, s, mode="edge")
    h, w = lum.shape
    out = np.zeros_like(lum)
    for dy, dx in ((0, s), (s, 0), (s, s), (s, -s)):
        a = p[s + dy:s + dy + h, s + dx:s + dx + w]
        b = p[s - dy:s - dy + h, s - dx:s - dx + w]
        out = np.maximum(out, lum - np.maximum(a, b))
    return out.clip(0)


def strand_score(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), np.float32) / 255
    real = a.min(-1) < 0.97                              # not the white padding beyond the photo
    if real.sum() < 100:
        return 0.0
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    lines = np.maximum(bright_lines(lum, 1), bright_lines(lum, 2))
    return float(((lines > LINE) & real).sum() / real.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--move", action="store_true")
    a = ap.parse_args()
    rows = []
    for source in ("afhq", "cat"):
        for f in sorted((EARS / source).glob("*.jpg")):
            rows.append({"source": source, "file": f.name, "score": round(strand_score(Image.open(f)), 4)})
    with open(EARS / "ear_strands.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["source", "file", "score"])
        wr.writeheader()
        wr.writerows(rows)
    s = np.array([r["score"] for r in rows])
    print(f"{len(rows)} ears; score percentiles 5/10/25/50/75: {np.percentile(s, [5, 10, 25, 50, 75]).round(4)}")
    for t in (0.01, 0.02, 0.03, 0.04, 0.05):
        print(f"  below {t}: {(s < t).sum()} ({(s < t).mean():.1%})")
    if a.move:
        moved = 0
        for r in rows:
            if r["score"] < a.threshold:
                d = EARS / r["source"] / SUGGEST
                d.mkdir(exist_ok=True)
                shutil.move(str(EARS / r["source"] / r["file"]), str(d / r["file"]))
                moved += 1
        print(f"moved {moved} ears below {a.threshold} into {SUGGEST}/")


if __name__ == "__main__":
    main()
