"""Folders for the owner's manual review (owner, 2026-10-18: maybe the model paints badly because the data is lacking;
first look at every cat by hand). The painting is split between models (CLAUDE.md, part models): a whole-face model
for the block-in (shape, outline, ears' shape, where the features are, big shadows) and part models for the detail
(three: ears, eyes, nose and mouth; whether the fur is a fourth, over the whole face, or part of the three is still
open). Each learns only from photos where its part is whole, so a photo whose ear tips are
cut off still teaches the eyes, the nose and the fur. Five folders, from every background-removed cat (AFHQ and the
CAT dataset); a photo may be in several (the owner reviews each folder on its own):

  1_整體打底  the head alone (the subject mask cut to head_shape: no back behind the ears, no chest below the
             chin), on white, at the photo's own resolution: the whole face inside the mask (mask_audit.face_cover >= --min-face), each ear tip at most
             TIP_SLACK eye distances outside the photo (AFHQ is framed on the face: a third of its ear tips lie beyond
             the edge), each ear inside the mask TIP_INWARD of the way from tip to base, not too dark. Closed eyes fine.
  2_耳朵      each ear on its own (1.5 x its three points' box), from the photo: its base inside the photo, at least
             MIN_EAR_PX wide, its middle inside the mask. A cut-off tip is fine: the inner ear and its fur show.
  3_眼睛      the eye crop of tools/eye_crop_styles.py (two eye distances around the eyes' midpoint), from the photo.
  4_鼻子嘴巴  nose and mouth (NOSE_SIDE eye distances, centred NOSE_AT of the way from the eyes to the mouth).
  5_毛流      the same head crop as 1_整體打底, for every cat whose face is inside the mask and not too dark,
             whatever its ears: reviewed now, used by whichever model ends up painting the fur.

3-5 need the eyes at least --min-eye-px apart (smaller crops show no detail).
Darkness: the standard deviation of the luminance within 0.35 eye distances of the eyes ("contrast"); a photo too
dark to see the eyes has almost none. 0.035 was chosen by looking at the darkest kept faces (2026-10-18): the two
darkest of 12,760 hide their eyes, a black cat at 0.026 still shows them; only the extreme ones are left out, the
rest is for the owner's eye. The face points are not drawn: they are in the data files already.
Every cat measured goes into manifest.csv with its numbers and, per folder, why it was left out (not_<folder>). The
owner moves rejected images into the 不要/ subfolder next to them; --collect then writes assignments.csv: for every
photo and every training, used, rejected (by the owner) or empty (never offered).

    PYTHONPATH=. python tools/review_export.py --measure-only     # numbers only
    PYTHONPATH=. python tools/review_export.py                    # numbers and the five folders
    PYTHONPATH=. python tools/review_export.py --collect          # after the review
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image

import mask_audit as M

ML = Path(__file__).resolve().parents[1]
OUT = ML / "data" / "processed" / "review"
TIPS = (4, 7)                       # ear tips (CAT order); 3, 5 and 6, 8 are the ears' bases
TIP_SLACK = 0.10                    # eye distances an ear tip may lie outside the photo (a few px cut off the tip)
EARS = ((3, 4, 5), (6, 7, 8))     # each ear: base, tip, base (CAT order)
MIN_EAR_PX = 20.0                   # an ear's base at least this wide to be worth a crop
NOSE_AT, NOSE_SIDE = 0.8, 1.2       # nose and mouth crop: centre this far from the eyes' midpoint to the mouth; side
FOLDERS = {"whole": "1_整體打底", "ear": "2_耳朵", "eyes": "3_眼睛", "nose_mouth": "4_鼻子嘴巴", "fur": "5_毛流"}
HEAD_CHIN, HEAD_SIDE, HEAD_POWER = 0.45, 0.10, 2.2   # head_shape: chin below the mouth, cheeks, roundness
EAR_SCALE, EAR_PAD = 1.3, 0.08      # head_shape: how generous the ear triangles are (eye distances)
TIP_INWARD = 0.15                   # the ear is checked this far from its tip towards its base: inside a whole ear


def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def cats():
    """(source, id, photo path, mask path, 9 points in photo pixels) for every cat with a mask and points."""
    z = np.load(M.LABELS)
    for rel, p in zip([str(i) for i in z["ids"]], z["points"], strict=True):
        yield "afhq", Path(rel).stem, M.AFHQ / rel, M.AFHQ_MASKS / f"{Path(rel).stem}_mask1.png", np.asarray(p, float)
    names = {}
    for k in ("CAT_DATASET_01", "CAT_DATASET_02"):
        zf = zipfile.ZipFile(M.CAT / f"{k}.zip")
        names.update({n: (k, zf) for n in zf.namelist() if n.endswith(".jpg.cat")})
    for member, (k, zf) in sorted(names.items()):
        sub, f = member.split("/")[-2], member.split("/")[-1][:-len(".jpg.cat")]
        stem = f"{k}__{sub}__{f}"
        mask = M.CAT / "masks" / f"{stem}_mask1.png"
        if not mask.exists():
            continue
        p = np.array([float(x) for x in zf.read(member).decode().split()][1:19]).reshape(9, 2)
        yield "cat", stem, M.CAT / "photos" / f"{stem}.jpg", mask, p


def measure(photo: Image.Image, mask: np.ndarray, p: np.ndarray) -> dict:
    w, h = photo.size
    d = float(np.linalg.norm(p[0] - p[1]))
    img = np.asarray(photo, np.float32) / 255
    yy, xx = np.mgrid[:h, :w] + 0.5
    near = np.min([np.hypot(xx - x, yy - y) for x, y in p[:2]], 0) < 0.35 * d
    slack = TIP_SLACK * d                                     # an ear tip this far outside the photo still counts
    tips_in_photo = all(-slack <= x <= w + slack and -slack <= y <= h + slack for x, y in p[list(TIPS)])
    inner = [p[t] + TIP_INWARD * ((p[b0] + p[b1]) / 2 - p[t]) for t, b0, b1 in ((4, 3, 5), (7, 6, 8))]
    tip_cover = 0.0
    if tips_in_photo and all(0 <= x < w and 0 <= y < h for x, y in inner):
        tip_cover = min(float(mask[int(y), int(x)]) for x, y in inner)
    return {"eye_dist_px": round(d, 1), "face_cover": round(M.face_cover(mask, p), 4), "tip_cover": round(tip_cover, 3),
            "tips_in_photo": tips_in_photo, "eye_contrast": round(float(luminance(img)[near].std()), 4),
            "eye_lum": round(float(luminance(img)[near].mean()), 4), "photo_w": w, "photo_h": h}


def head_shape(p: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The head alone, from the nine face points (owner, 2026-10-18: Apple's subject mask keeps the whole cat, so the
    back behind the ears and the chest below the chin made every face a different, odd shape). In the face's frame
    (origin the eyes' midpoint, x along the eyes, y towards the mouth, unit the eye distance): a rounded face, a
    superellipse (power HEAD_POWER) from the highest ear base down to HEAD_CHIN below the mouth and as wide as the
    widest ear base or one eye distance plus HEAD_SIDE; plus each ear's triangle (base, tip, base) scaled by
    EAR_SCALE about its centre and widened by EAR_PAD, generous on purpose: the ear's own edge comes from the subject
    mask it is intersected with, the triangle only keeps the body behind the ear out."""
    h, w = shape
    d = float(np.linalg.norm(p[0] - p[1]))
    mid = p[:2].mean(0)
    ex = (p[1] - p[0]) / d
    ey = np.array([-ex[1], ex[0]])
    if np.dot(p[2] - mid, ey) < 0:
        ey = -ey
    P = np.stack([(p - mid) @ ex, (p - mid) @ ey], -1) / d
    bases = P[[3, 5, 6, 8]]
    a = max(float(np.abs(bases[:, 0]).max()), 1.0) + HEAD_SIDE
    top, bot = float(bases[:, 1].min()), float(P[2, 1]) + HEAD_CHIN
    cy, b = (top + bot) / 2, (bot - top) / 2
    yy, xx = np.mgrid[:h, :w] + 0.5
    rel = np.stack([xx - mid[0], yy - mid[1]], -1)
    qx, qy = rel @ ex / d, rel @ ey / d
    head = np.abs(qx / a) ** HEAD_POWER + np.abs((qy - cy) / b) ** HEAD_POWER <= 1
    q = np.stack([qx, qy], -1).reshape(-1, 2)
    for ear in EARS:
        T = P[list(ear)]
        c = T.mean(0)
        v = T - c
        T = c + v * EAR_SCALE + v / np.linalg.norm(v, axis=1, keepdims=True) * EAR_PAD
        head |= Polygon(T).contains_points(q).reshape(h, w)
    return head


def face_crop(photo: Image.Image, mask: np.ndarray, p: np.ndarray) -> Image.Image:
    """The head alone (subject mask x head_shape) on white, in a square around the head with a small margin."""
    head = head_shape(p, mask.shape)
    ys, xs = np.nonzero(head)
    lo, hi = np.array([xs.min(), ys.min()], float), np.array([xs.max(), ys.max()], float) + 1
    box = square((lo + hi) / 2, 1.08 * float((hi - lo).max()))
    m = mask * head
    img = np.asarray(photo, np.float32) / 255
    out = img * m[..., None] + (1 - m[..., None])                       # white paper outside the head
    return crop(Image.fromarray((out * 255).round().astype(np.uint8)), box)


def square(c: np.ndarray, side: float) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in (c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2))


def crop(photo: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """photo cut to box; where the box leaves the photo, white."""
    sheet = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    sheet.paste(photo, (-box[0], -box[1]))
    return sheet


def eye_box(p: np.ndarray) -> tuple[int, int, int, int]:
    return square(p[:2].mean(0), 2 * float(np.linalg.norm(p[0] - p[1])))


def ear_box(p: np.ndarray, ear: tuple[int, int, int]) -> tuple[int, int, int, int]:
    q = p[list(ear)]
    lo, hi = q.min(0), q.max(0)
    return square((lo + hi) / 2, 1.5 * float((hi - lo).max()))


def nose_mouth_box(p: np.ndarray) -> tuple[int, int, int, int]:
    mid = p[:2].mean(0)
    return square(mid + NOSE_AT * (p[2] - mid), NOSE_SIDE * float(np.linalg.norm(p[0] - p[1])))


def ear_usable(p: np.ndarray, ear: tuple[int, int, int], mask: np.ndarray) -> str:
    """'' when the ear can teach the ear model, else why not: its base inside the photo, wide enough, and its middle
    inside the subject mask (not hidden or cut away). Its tip may be outside the photo."""
    h, w = mask.shape
    b0, tip, b1 = (p[k] for k in ear)
    if not all(0 <= x < w and 0 <= y < h for x, y in (b0, b1)):
        return "ear base outside photo"
    if np.linalg.norm(b0 - b1) < MIN_EAR_PX:
        return "ear too small"
    x, y = (b0 + b1) / 2 + 0.35 * (tip - (b0 + b1) / 2)
    if not (0 <= x < w and 0 <= y < h) or mask[int(y), int(x)] < 0.5:
        return "ear hidden or cut"
    return ""


def offers(row: dict, p: np.ndarray, mask: np.ndarray, a: argparse.Namespace) -> dict[str, str]:
    """For each folder, '' when the photo goes into it, else why not (the automatic checks; the owner reviews the rest)."""
    face = [k for k, bad in (("face cut", row["face_cover"] < a.min_face), ("too dark", row["eye_contrast"] < a.min_contrast))
            if bad]
    small = row["eye_dist_px"] < a.min_eye_px
    whole = face + [k for k, bad in (("ear outside photo", not row["tips_in_photo"]),
                                    ("ear cut", row["tips_in_photo"] and row["tip_cover"] < a.min_tip)) if bad]
    h, w = mask.shape
    return {"whole": "; ".join(whole),
            "ear_1": ear_usable(p, EARS[0], mask), "ear_2": ear_usable(p, EARS[1], mask),
            "eyes": "eyes too small" if small else "",
            "nose_mouth": "eyes too small" if small else ("mouth outside photo" if not (0 <= p[2][0] < w and 0 <= p[2][1] < h)
                                                          else ""),
            "fur": "; ".join(face + (["eyes too small"] if small else []))}


def export(folder: str, source: str, name: str, img: Image.Image, out: Path, only: str = "") -> None:
    if only and folder not in only.split(","):
        return
    d = out / FOLDERS[folder] / source
    (d / "不要").mkdir(parents=True, exist_ok=True)
    img.save(d / f"{name}.jpg", quality=92)


def collect(out: Path) -> None:
    """After the review: which photo goes into which training. A photo offered to a folder and left outside its 不要/
    is used ("used"), one moved into 不要/ by the owner is not ("rejected"), nor one an automatic check moved into
    建議不要/ and the owner left there ("suggested_out", e.g. tools/ear_strands.py); one never offered stays empty
    (manifest.csv says why)."""
    rows = {(r["source"], r["id"]): r for r in csv.DictReader(open(out / "manifest.csv"))}
    table = {k: {"source": k[0], "id": k[1], **{f: "" for f in FOLDERS}} for k in rows}
    for f, dirname in FOLDERS.items():
        for source in ("afhq", "cat"):
            d = out / dirname / source
            for state, files in (("used", d.glob("*.jpg")), ("rejected", (d / "不要").glob("*.jpg")),
                                 ("suggested_out", (d / "建議不要").glob("*.jpg"))):
                for g in files:
                    stem, part = g.stem, f
                    if f == "ear":
                        stem, part = g.stem.rsplit("_", 1)[0], "ear_" + g.stem.rsplit("_", 1)[1][-1]
                    if (source, stem) in table:
                        table[(source, stem)][part] = state
    cols = ["source", "id", "whole", "ear_1", "ear_2", "eyes", "nose_mouth", "fur"]
    with open(out / "assignments.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(table.values())
    counts = {c: {s: sum(r.get(c) == s for r in table.values()) for s in ("used", "rejected", "suggested_out")}
              for c in cols[2:]}
    (out / "assignments_summary.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False))
    print(json.dumps(counts, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--collect", action="store_true", help="after the review: write assignments.csv")
    ap.add_argument("--min-face", type=float, default=0.97)
    ap.add_argument("--min-tip", type=float, default=0.5)
    ap.add_argument("--min-contrast", type=float, default=0.035)
    ap.add_argument("--min-eye-px", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--folders", default=",".join(FOLDERS), help="which folders to (re)write")
    a = ap.parse_args()
    if a.collect:
        collect(a.out)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, cid, photo_path, mask_path, p in cats():
        photo = Image.open(photo_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L").resize(photo.size, Image.BILINEAR), np.float32) / 255
        row = {"source": source, "id": cid, **measure(photo, mask > 0.5, p)}
        why = offers(row, p, mask, a)
        row.update({f"not_{k}": v for k, v in why.items()})
        if not a.measure_only:
            head = face_crop(photo, mask, p) if not (why["whole"] and why["fur"]) else None
            if not why["whole"]:
                export("whole", source, cid, head, a.out, a.folders)
            for k, ear in (("ear_1", EARS[0]), ("ear_2", EARS[1])):
                if not why[k]:
                    export("ear", source, f"{cid}_耳{k[-1]}", crop(photo, ear_box(p, ear)), a.out, a.folders)
            if not why["eyes"]:
                export("eyes", source, cid, crop(photo, eye_box(p)), a.out, a.folders)
            if not why["nose_mouth"]:
                export("nose_mouth", source, cid, crop(photo, nose_mouth_box(p)), a.out, a.folders)
            if not why["fur"]:
                export("fur", source, cid, head, a.out, a.folders)
        rows.append(row)
        if len(rows) % 2000 == 0:
            print(len(rows), "cats", flush=True)
    with open(a.out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summary = {}
    for source in ("afhq", "cat"):
        rs = [r for r in rows if r["source"] == source]
        summary[source] = {"cats": len(rs)}
        for k in ("whole", "ear_1", "ear_2", "eyes", "nose_mouth", "fur"):
            reasons = {}
            for r in rs:
                for why_ in filter(None, r[f"not_{k}"].split("; ")):
                    reasons[why_] = reasons.get(why_, 0) + 1
            summary[source][k] = {"in": sum(not r[f"not_{k}"] for r in rs), "left_out": reasons}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
