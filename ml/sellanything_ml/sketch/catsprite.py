"""The cat sprite: import a drawn sheet, find the 9 face points on every frame, and bake real cats into it.

The sheet (drawn from the brief in design-v2 §14.6) has two rows of frames on a transparent background: the
colourless cat above, the same frames with one flat colour per part below (head red, body green, tail blue,
paws yellow). Every frame is cut out around the same anchor, the bottom centre of the paws, so the cat stands
still from frame to frame, and scaled by SCALE.

The 9 points (eyes, mouth, 3 per ear) are not marked by hand: the ear tips are the head's highest pixel on each
side; the other points sit where the dataset's mean face has them, relative to the face's own outline box, laid
on the head's box. catbake.py then pins the warp at those points and at 36 outline pairs, so the face's edge
lands on the head's edge.

    python -m sellanything_ml.sketch.catsprite <sheet.png> ../game/assets/cat runs/catface/sprite_demo.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.ndimage import label as cc_label

from . import catbake
from . import catface as C

SCALE = 0.5
BOX = (300, 400)                                  # source pixels around a frame's anchor, before SCALE
ANCHOR_IN_BOX = (150, 392)
NAMES = ["left_eye", "right_eye", "mouth", "left_ear_outer", "left_ear_tip", "left_ear_inner",
         "right_ear_inner", "right_ear_tip", "right_ear_outer"]
PARTS = {"head": 1, "body": 2, "tail": 3, "paws": 4}


def _runs(idx, gap=10):
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i > p + gap:
            out.append((int(s), int(p)))
            s = i
        p = i
    return out + [(int(s), int(p))]


def _labels(rgba: np.ndarray) -> np.ndarray:
    rgb, op = rgba[..., :3].astype(int), rgba[..., 3] > 200
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lab = np.zeros(op.shape, np.uint8)
    lab[op & (r > 200) & (g < 130) & (b < 130)] = PARTS["head"]
    lab[op & (g > 120) & (r < 120) & (b < 120)] = PARTS["body"]
    lab[op & (b > 160) & (r < 130)] = PARTS["tail"]
    lab[op & (r > 200) & (g > 160) & (b < 120)] = PARTS["paws"]
    return lab


def mean_face_layout(n: int = 2000) -> np.ndarray:
    """The 9 points of the training faces, each relative to its own face's outline box (0–1), averaged."""
    tr = C.load("train")
    rel = []
    for face, pts in zip(tr[C.COLOR][:n], tr["points"][:n]):
        sil = catbake.silhouette(face)
        ys, xs = np.nonzero(sil)
        if len(xs) < 100:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        rel.append((pts - [x0, y0]) / [max(x1 - x0, 1), max(y1 - y0, 1)])
    return np.mean(rel, axis=0)


def head_points(head: np.ndarray, layout: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(head)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pts = np.c_[x0 + layout[:, 0] * (x1 - x0), y0 + layout[:, 1] * (y1 - y0)]
    cx = (x0 + x1) / 2
    for idx, side in ((4, xs < cx), (7, xs >= cx)):                 # ear tips: the head's top on each side
        k = np.argmin(np.where(side, ys, 10 ** 6))
        pts[idx] = (xs[k], ys[k])
    return pts


def import_sheet(sheet_path: Path, out_dir: Path, state: str = "idle", frame_time: float = 0.25):
    a = np.asarray(Image.open(sheet_path).convert("RGBA"))
    op = a[..., 3] > 40
    rows = [r for r in _runs(np.nonzero(op.any(1))[0]) if r[1] - r[0] > 50]
    if len(rows) != 2:
        raise SystemExit(f"expected 2 rows of frames, found {len(rows)}")
    (t0, t1), (b0, b1) = rows
    cols = [c for c in _runs(np.nonzero(op[t0:t1 + 1].any(0))[0]) if c[1] - c[0] > 50]
    dy = b0 - t0                                                     # the label row repeats the colour row
    w, h = round(BOX[0] * SCALE), round(BOX[1] * SCALE)
    frames, labels, points = [], [], []
    layout = mean_face_layout()
    for c0, c1 in cols:
        lab_full = _labels(a[b0:b1 + 1, c0:c1 + 1])
        ys, xs = np.nonzero(lab_full == PARTS["paws"])
        ax, ay = c0 + xs.mean(), t0 + ys.max()                       # anchor: bottom centre of the paws
        left, top = int(round(ax - ANCHOR_IN_BOX[0])), int(round(ay - ANCHOR_IN_BOX[1]))
        crop = np.zeros((BOX[1], BOX[0], 4), np.uint8)
        lcrop = np.zeros((BOX[1], BOX[0]), np.uint8)
        sy0, sx0 = max(top, 0), max(left, 0)
        sy1, sx1 = min(top + BOX[1], a.shape[0]), min(left + BOX[0], a.shape[1])
        crop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = a[sy0:sy1, sx0:sx1]
        lab_src = _labels(a[sy0 + dy:sy1 + dy, sx0:sx1]) if sy1 + dy <= a.shape[0] else None
        lcrop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = lab_src
        # a tail reaching over from the next frame lands in this box too: keep only this cat, the opaque piece
        # that holds the anchor's paws
        pieces, _ = cc_label(crop[..., 3] > 40)
        own = pieces[lcrop == PARTS["paws"]]
        if own.size:
            mine = binary_dilation(pieces == np.bincount(own).argmax(), iterations=3)
            crop[~mine] = 0
            lcrop[~mine] = 0
        frame = Image.fromarray(crop).resize((w, h), Image.LANCZOS)
        lab = np.asarray(Image.fromarray(lcrop).resize((w, h), Image.NEAREST)).copy()
        # the anti-aliased band beside the outline is in no part's flat colour; it goes to the nearest part, or
        # it stays the sheet's grey and shows as a light rim round every part once the cat is coloured
        fa = np.asarray(frame)
        orphan = (fa[..., 3] > 40) & (fa[..., :3].max(-1) > 110) & (lab == 0)
        if orphan.any() and (lab > 0).any():
            _, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
            lab[orphan] = lab[iy[orphan], ix[orphan]]
        # lines drawn inside the head (the inner ears) belong to the sheet's face, not the cat's: the cat's own
        # face covers them
        inner = binary_fill_holes(binary_closing(lab == PARTS["head"], iterations=2)) & (lab == 0) & (fa[..., 3] > 40)
        lab[inner] = PARTS["head"]
        frames.append(frame)
        labels.append(lab)
        points.append(head_points(np.asarray(lab) == PARTS["head"], layout))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGBA", (w * len(frames), h), (0, 0, 0, 0))
    lsheet = Image.new("L", (w * len(frames), h), 0)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        sheet.paste(f, (i * w, 0))
        lsheet.paste(Image.fromarray(lab), (i * w, 0))
    sheet.save(out_dir / f"cat_{state}.png")
    lsheet.save(out_dir / f"cat_{state}_parts.png")
    meta_path = out_dir / "cat.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"look": "cat", "cell": [w, h], "anchor": [round(ANCHOR_IN_BOX[0] * SCALE), round(ANCHOR_IN_BOX[1] * SCALE)],
                 "face_points": NAMES, "parts": PARTS})
    meta.setdefault("states", {})[state] = {
        "file": f"cat_{state}.png", "parts_file": f"cat_{state}_parts.png", "cols": len(frames), "rows": 1,
        "frames": len(frames), "frame_time": frame_time, "loop": True,
        "face": [[[round(float(x), 2), round(float(y), 2)] for x, y in p] for p in points]}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return frames, labels, points


def demo(frames, labels, points, out_png: Path, picks=(0, 4, 5, 6, 10, 13)) -> None:
    """Real colour faces (validation) baked into every frame."""
    val = C.load("validation")
    w, h = frames[0].size
    rows = [[f for f in frames]]
    for i in picks:
        face, pts = val[C.COLOR][i], val["points"][i]
        row = []
        for f, lab, hp in zip(frames, labels, points):
            head = lab == PARTS["head"]
            body = np.isin(lab, [PARTS["body"], PARTS["tail"], PARTS["paws"]])
            ys, xs = np.nonzero(head)
            scale = (xs.max() - xs.min()) / max(np.ptp(np.nonzero(catbake.silhouette(face))[1]), 1)
            row.append(Image.fromarray(catbake.bake(np.asarray(f), head, body, hp, face, pts, scale,
                                                    np.random.default_rng(i))))
        rows.append(row)
    sheet = Image.new("RGBA", (len(frames) * w * 2, len(rows) * h * 2), (255, 255, 255, 255))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.alpha_composite(im.resize((w * 2, h * 2), Image.LANCZOS), (c * w * 2, r * h * 2))
    sheet.convert("RGB").save(out_png)


if __name__ == "__main__":
    fr, la, po = import_sheet(Path(sys.argv[1]), Path(sys.argv[2]))
    demo(fr, la, po, Path(sys.argv[3]))
    print(len(fr), "frames, cell", fr[0].size, "points frame 0:", np.round(po[0], 1).tolist())
