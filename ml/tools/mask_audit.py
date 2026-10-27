"""Are the subject masks clean where it matters, on the face? (owner, 2026-10-18: some generated faces have odd
shapes; the suspicion is that some paintings were made from photos whose cat was not cut out cleanly.) The painter's
target is the photo with everything outside the subject mask (Apple Vision's subject lifting, section 4.1.1) turned
to white paper, so a mask that misses part of the face puts white paper into the painting's face.

For every painted cat, the share of the face that the mask calls cat: the face is the convex hull of the eyes, the
mouth and the four ear-base points (the ear tips are left out: the hull between two ear tips is background). A
healthy mask covers all of it. Written to <out>/<source>.json, sorted from the worst, with a figure of the worst.

Tried first and dropped (2026-10-18, 10 AFHQ cats): cutting each cat out a second time with SAM (facebook/sam-vit-base)
and comparing. Prompted with the face points SAM cuts out pieces of the face; prompted with a box it takes in the
background. On those cats Apple's mask was the right one each time, so SAM is no reference for a whole cat.

    PYTHONPATH=. python tools/mask_audit.py --out runs/mask_audit
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy.spatial import ConvexHull

ML = Path(__file__).resolve().parents[1]
AFHQ, AFHQ_MASKS = ML / "data" / "raw" / "afhq", ML / "data" / "processed" / "afhq_masks"
LABELS = ML / "data" / "processed" / "afhq_cats_mae_pad1.5.npz"
CAT = ML / "data" / "raw" / "cat_dataset"
CAT_PAINTED = ML / "data" / "processed" / "cat_brush_face"
FACE = [0, 1, 2, 3, 5, 6, 8]       # eyes, mouth, ear bases (CAT order: 3-5 and 6-8 are the ears, 4 and 7 their tips)


def face_cover(mask: np.ndarray, points: np.ndarray) -> float:
    """Share of the face hull (FACE points) inside the mask (H, W bool, the photo's pixels)."""
    q = points[FACE]
    hull = Polygon(q[ConvexHull(q).vertices])
    x0, y0 = np.floor(q.min(0)).astype(int).clip(0)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = hull.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(yy.shape)
    return float(mask[y0:y1, x0:x1][inside].mean()) if inside.any() else 1.0


def afhq() -> list[dict]:
    z = np.load(LABELS)
    rows = []
    for rel, p in zip([str(i) for i in z["ids"]], z["points"], strict=True):
        m = np.asarray(Image.open(AFHQ_MASKS / f"{Path(rel).stem}_mask1.png").convert("L").resize((512, 512))) > 127
        rows.append({"id": rel, "face_cover": round(face_cover(m, p), 4)})
    return rows


def cat() -> list[dict]:
    names = {}
    for k in ("CAT_DATASET_01", "CAT_DATASET_02"):
        zf = zipfile.ZipFile(CAT / f"{k}.zip")
        names.update({n: zf for n in zf.namelist()})
    rows = []
    for stem in sorted(p.stem for p in CAT_PAINTED.glob("*.npz")):
        _, sub, f = stem.split("__")
        member = next(c for c in (f"{sub}/{f}.jpg.cat", f"CAT_DATASET_01/{sub}/{f}.jpg.cat",
                                  f"CAT_DATASET_02/{sub}/{f}.jpg.cat") if c in names)
        p = np.array([float(x) for x in names[member].read(member).decode().split()][1:19]).reshape(9, 2)
        size = Image.open(CAT / "photos" / f"{stem}.jpg").size
        m = np.asarray(Image.open(CAT / "masks" / f"{stem}_mask1.png").convert("L").resize(size)) > 127
        rows.append({"id": stem, "face_cover": round(face_cover(m, p), 4)})
    return rows


def figure(rows: list[dict], source: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Heiti TC"]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.6))
    for a, r in zip(ax.ravel(), rows[:8]):
        if source == "afhq":
            photo = Image.open(AFHQ / r["id"]).convert("RGB")
            mpath = AFHQ_MASKS / f"{Path(r['id']).stem}_mask1.png"
        else:
            photo = Image.open(CAT / "photos" / f"{r['id']}.jpg").convert("RGB")
            mpath = CAT / "masks" / f"{r['id']}_mask1.png"
        img = np.asarray(photo).astype(float) / 255
        m = np.asarray(Image.open(mpath).convert("L").resize(photo.size)) > 127
        img[~m] = 0.35 * img[~m] + 0.65 * np.array([1, 0, 0])
        a.imshow(img)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{source} …{r['id'][-14:]}  臉被算成貓 {r['face_cover']:.3f}\n紅＝當成背景（換成白紙）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for source, fn in (("afhq", afhq), ("cat", cat)):
        rows = sorted(fn(), key=lambda r: r["face_cover"])
        cov = np.array([r["face_cover"] for r in rows])
        summary = {"cats": len(rows), "median": round(float(np.median(cov)), 4),
                   "below": {str(t): int((cov < t).sum()) for t in (0.99, 0.97, 0.95, 0.9, 0.8)}}
        (a.out / f"{source}.json").write_text(json.dumps({"summary": summary, "cats": rows}, indent=0))
        figure(rows, source, a.out / f"{source}_worst8.png")
        print(source, json.dumps(summary))


if __name__ == "__main__":
    main()
