"""The whole face in digital painting (owner, 2026-10-18: the digital painter's way, smooth brushes without bristle
streaks, for the whole face, so the fine fur of the photo can be drawn too). Order: whatever the colour, the finer and
the higher in contrast a mark is, the later it is painted (owner). Eight layers, every brush smooth (bristle 0) and
soft-edged, fitted to the photo with catbrush's L1 as before:

  1 washes        24 x 96 x 40 px, edge 6 px      the big masses of light and shade
  2 blocks        64 x 48 x 20, edge 3
  3 blend         80 blending strokes (30 x 10) along the fur
  4 small blocks  160 x 24 x 8, edge 1.5
  5 blend         120 blending strokes (16 x 5) along the fur
  6 fur           400 x 12 x 2, edge 0.7, starting along the fur's direction
  7 dark hairs    500 x 10 x 1, edge 0.5, starting on the photo's fine dark lines
  8 light hairs   500 x 12 x 1, edge 0.5, starting on the photo's fine light lines (light fur, tufts, whiskers)

Within each paint layer the strokes are then laid in order of their contrast with what is under them, lowest first,
so the sharpest marks land last, and fitted a while longer in that order (eye_crop_styles, order_by_contrast).
Re-laying them without refitting (tried first, 2 cats) raised the L1 from 0.034 to 0.045. Changing a pixel outside
the head costs PROTECT times the change (the soft washes otherwise left a grey halo), and no hair starts within
EDGE_KEEP px of the head's edge (the light-hair layer otherwise drew a dashed white line along it).
The target is the head alone (tools/review_export.face_crop: no body, white around), at 256 px like the old painter,
which is painted on the same target for comparison. CPU only; a painter pilot, not training data.

    PYTHONPATH=.:tools python tools/digital_face.py --n 6 --out runs/digital_face/pilot6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

import ear_strands as S
import eye_crop_styles as E
import review_export as RX
from sellanything_ml.sketch import catbrush as B

PROTECT = 10.0              # changing a pixel outside the head costs this many times the change (no halo)
EDGE_KEEP = 3               # px inside the head's edge where no hair starts (the edge itself is not a hair)
LAYERS = [  # name, length, width, count, edge px, kind
    ("washes", 96, 40, 24, 6.0, "paint"), ("blocks", 48, 20, 64, 3.0, "paint"), ("blend", 30, 10, 80, 0.5, "blend"),
    ("small blocks", 24, 8, 160, 1.5, "paint"), ("blend", 16, 5, 120, 0.5, "blend"), ("fur", 12, 2, 400, 0.7, "paint"),
    ("dark hairs", 10, 1, 500, 0.5, "dark"), ("light hairs", 12, 1, 500, 0.5, "light")]


def head_target(rel: str, points: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """The head alone on white, and its mask, at the painter's 256 px."""
    photo = Image.open(E.AFHQ / rel).convert("RGB")
    apple = np.asarray(Image.open(E.MASKS / f"{Path(rel).stem}_mask1.png").convert("L").resize(photo.size), np.float32) / 255
    img = RX.face_crop(photo, apple, points).resize((B.SIDE, B.SIDE), Image.LANCZOS)
    head = RX.head_shape(points, apple.shape)
    ys, xs = np.nonzero(head)
    lo, hi = np.array([xs.min(), ys.min()], float), np.array([xs.max(), ys.max()], float) + 1
    box = RX.square((lo + hi) / 2, 1.08 * float((hi - lo).max()))          # the box face_crop uses
    sheet = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)           # beyond the photo: not cat
    sheet.paste(Image.fromarray(((head * apple) * 255).astype(np.uint8)), (-box[0], -box[1]))
    mask = np.asarray(sheet.resize((B.SIDE, B.SIDE), Image.BILINEAR), np.float32) / 255
    return torch.from_numpy(np.asarray(img, np.float32) / 255), torch.from_numpy(mask.clip(0, 1))


def specs(target: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    lum = E.luminance(target.numpy())
    m = mask.numpy()
    inner = binary_erosion(m > 0.5, iterations=EDGE_KEEP)       # no hair starts on the head's own edge
    dark = E.dark_lines(lum) * inner
    light = np.maximum(S.bright_lines(lum, 1), S.bright_lines(lum, 2)) * inner
    out = []
    for _name, length, width, n, edge, kind in LAYERS:
        spec = {"length": length, "width": width, "n": n, "edge_px": edge, "bristle": 0.0, "alpha_range": (0.2, 1.0),
                "protect": (torch.from_numpy((1 - m).astype(np.float32)), PROTECT), "order_by_contrast": kind != "blend"}
        if kind == "blend":
            spec["blend"] = True
        if kind == "dark":
            spec["place"] = dark ** 2 + 1e-6 * m
        if kind == "light":
            spec["place"] = light ** 2 + 1e-6 * m
        out.append(spec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--skip-old", action="store_true", help="do not paint the old painter (timing runs)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    labels = np.load(E.LABELS)
    index = {str(i): k for k, i in enumerate(labels["ids"])}
    z = np.load(E.SPLIT)
    ids = [str(i) for i in z["ids"][z["split"] == "val"][:a.n]]
    rows = []
    for rel in ids:
        name = Path(rel).stem
        target, mask = head_target(rel, labels["points"][index[rel]])
        t0 = time.time()
        old = target.numpy() if a.skip_old else E.paint(target, mask, [{"length": ln, "width": w, "n": n}
                                                                     for ln, w, n in B.STAGES[:4]], dev=a.device)[-1]
        t_old = time.time() - t0
        t0 = time.time()
        record = []
        digital = torch.from_numpy(E.paint(target, mask, specs(target, mask), record=record, dev=a.device)[-1])
        t_dig = time.time() - t0
        tg = target.numpy()
        row = {"id": rel, "device": a.device, "old": {"l1": round(float(np.abs(old - tg).mean()), 4), "strokes": 608, "seconds": round(t_old, 1)},
               "digital": {"l1": round(float(np.abs(digital.numpy() - tg).mean()), 4),
                           "strokes": sum(n for *_, n, _e, _k in LAYERS), "seconds": round(t_dig, 1)}}
        torch.save({"record": [(spec, {k: getattr(s, k).detach().cpu() for k in ("xy", "size", "angle", "colour", "alpha")},
                                s.base)
                               for spec, s in record]}, a.out / f"{name}_strokes.pt")
        np.savez_compressed(a.out / f"{name}.npz", photo=tg, mask=mask.numpy(), old=old, digital=digital.numpy())
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "summary.json").write_text(json.dumps({"cats": rows, "layers": LAYERS}, indent=1))


if __name__ == "__main__":
    main()
