"""Checks and shows the digital eye material once the Kaggle run and the Mac are merged (owner, 2026-10-24): every cat of
tools/eye_pack.py's list has a file in data/processed/brush_l/eye_l8_digital/ that opens, is style digital and has its
eight layers with the stored stroke counts; a figure of the first n cats in order (not picked): the photo crop, the
layers material (E-runs) and the digital material, each eye zoomed. CPU only.

    PYTHONPATH=.:tools python tools/eye_digital_merge.py --n 12
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eye_data_l8 as D
import eye_pack as P

OUT = D.OUT.parent / "eye_l8_digital"
PACK = D.ML / "data" / "kaggle" / "eye_digital" / "pack"


def main() -> None:
    """Check every cat's file, write check.json and a figure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    want = P.names()
    names = D.layer_names("digital")
    missing, bad, l1 = [], [], []
    for n in want:
        f = OUT / f"{n}.npz"
        if not f.exists():
            missing.append(n)
            continue
        try:
            z = np.load(f)
            ok = str(z["style"]) == "digital" and all(len(z[f"bins_{k}"]) == s for k, s in
                                                     zip(names, [x["n"] for x in json.loads(str(z["layers"]))]))
            info = json.loads(str(z["info"]))
            l1.append((info["base"]["eye_l1"], info["painted"]["eye_l1"]))
        except Exception as e:                                  # a broken file (e.g. a cut-off download)
            ok, info = False, str(e)
        if not ok:
            bad.append(n)
    res = {"cats": len(want), "have": len(want) - len(missing), "missing": len(missing), "bad": bad[:20], "n_bad": len(bad),
           "eye_l1_base_painted": [round(float(np.mean([x[0] for x in l1])), 4), round(float(np.mean([x[1] for x in l1])), 4)]}
    (OUT / "check.json").write_text(json.dumps({**res, "missing_names": missing}, indent=1))
    print(json.dumps(res))
    packs = {}
    for f in sorted(PACK.glob("pack_*.npz")):
        if not f.name.endswith(".tmp.npz"):
            z = np.load(f)
            packs.update({str(x): (f, j) for j, x in enumerate(z["names"])})
    S, font = 220, ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 18)
    shown = [n for n in want if (OUT / f"{n}.npz").exists()][:a.n]
    img = Image.new("RGB", (3 * 2 * (S + 4) + 12, len(shown) * (S + 4) + 40), "white")
    d = ImageDraw.Draw(img)
    for j, cap in enumerate(("照片", "layers（E1–E6 的教材）", "digital（新教材）")):
        d.text((6 + j * 2 * (S + 4), 8), cap, fill=(0, 0, 0), font=font)
    for i, n in enumerate(shown):
        dz = np.load(OUT / f"{n}.npz")
        f, j = packs[n]
        photo = np.load(f)["target"][j]
        old = np.load(D.OUT / f"{n}.npz")["painted"]
        for c, im in enumerate((photo, old, dz["painted"])):
            for e, (x, y) in enumerate(dz["eyes"]):
                r = 0.42 * float(dz["eye_dist"])
                crop = Image.fromarray(im).crop((int(x - r), int(y - r), int(x + r), int(y + r))).resize((S, S), Image.LANCZOS)
                img.paste(crop, (6 + (2 * c + e) * (S + 4), 34 + i * (S + 4)))
    img.save(OUT / "first_cats.png")
    print(OUT / "first_cats.png")


if __name__ == "__main__":
    main()
