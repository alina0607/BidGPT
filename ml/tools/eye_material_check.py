"""Checks an eye material folder before it is trained on (owner, 2026-10-25: after the Mac repaint, check every cat, then
build the data and train automatically): every cat of tools/eye_pack.py's list has a file that opens, has the given
style and base, and has each layer's stored number of strokes. Writes check.json in the folder; exits 1 if anything is
missing or wrong, so a chain stops there.

    PYTHONPATH=.:tools python tools/eye_material_check.py --dir data/processed/brush_l/eye_l8_rim_hybrid \\
        --style digital_rim_long --base hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import eye_data_l8 as D
import eye_pack as P


def check(path: Path, style: str, base: str) -> str | None:
    """None when the file is right, else what is wrong with it."""
    try:
        z = np.load(path)
        if str(z["style"]) != style:
            return f"style {z['style']}"
        if str(z["base_kind"]) != base:
            return f"base {z['base_kind']}"
        for layer in json.loads(str(z["layers"])):
            if len(z[f"bins_{layer['name']}"]) != layer["n"]:
                return f"layer {layer['name']}"
        if z["painted"].shape != (256, 256, 3):
            return "painting"
    except Exception as e:                                       # a cut-off or broken file
        return f"broken: {e}"
    return None


def main() -> None:
    """Check the folder; exit 1 on any missing or wrong file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--base", default="hybrid")
    a = ap.parse_args()
    names = P.names()
    missing = [n for n in names if not (a.dir / f"{n}.npz").exists()]
    bad = {n: why for n in names if n not in missing for why in [check(a.dir / f"{n}.npz", a.style, a.base)] if why}
    l1 = [json.loads(str(np.load(a.dir / f"{n}.npz")["info"]))["painted"]["eye_l1"] for n in names[::50] if n not in missing and n not in bad]
    res = {"cats": len(names), "missing": len(missing), "bad": len(bad), "eye_l1_painted_sample": round(float(np.mean(l1)), 4) if l1 else None,
           "layers": D.layer_names(a.style)}
    (a.dir / "check.json").write_text(json.dumps({**res, "missing_names": missing, "bad_files": bad}, indent=1))
    print(json.dumps(res))
    sys.exit(1 if missing or bad else 0)


if __name__ == "__main__":
    main()
