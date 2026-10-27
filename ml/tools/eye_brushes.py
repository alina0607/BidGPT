"""Save an eye run's brushes next to its checkpoint (owner, 2026-10-27: the game paints with L8 + the eye model). The
eye strokes carry no brush; each layer's brush is the one of the material the run learned from (tools/eye_render.py
brushes_of: a digital material stores them, a four-layer one uses eye_data_l8's). The game (sketch/shelter.py) reads
runs/brushgpt/<run>/brushes.json, so it does not need the material folder.

    PYTHONPATH=.:tools python tools/eye_brushes.py --run E5 --eyes data/processed/brush_l/eye_l8
    PYTHONPATH=.:tools python tools/eye_brushes.py --run E8L --eyes data/processed/brush_l/eye_l8_rim_hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import eye_render as ER
from sellanything_ml.sketch import eyegpt_l8 as EG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--eyes", type=Path, required=True, help="the eye material the run learned from")
    a = ap.parse_args()
    first = next(f for f in sorted(a.eyes.glob("afhq__*.npz")) if not f.name.endswith((".tmp.npz", ".compare.npz")))
    brushes = ER.brushes_of(first)
    model = EG.load_model(EG.RUN_DIR / a.run / "best.pt", "cpu")
    if len(brushes) != model.vocab.n_layers:
        raise SystemExit(f"{a.eyes} has {len(brushes)} layers, the run {model.vocab.n_layers}")
    out = EG.RUN_DIR / a.run / "brushes.json"
    out.write_text(json.dumps(brushes, indent=1) + "\n")
    print(json.dumps({"run": a.run, "layers": len(brushes), "from": str(first), "out": str(out)}))


if __name__ == "__main__":
    main()
