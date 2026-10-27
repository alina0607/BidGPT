"""tools/eye_judge.py's eye judge for Brush GPT L (owner, 2026-10-20: are L5's eyes better than J's?). Same judge (the
cat-face landmark model, trained on photos only), same n, seed and temperatures as J and K's runs. References, all on
the first n validation cats: the photos, the L5 training paintings (five layers, binned) and the L8 ones (eight).

    PYTHONPATH=.:tools python tools/eye_judge_l.py --run L5 --data data/kaggle/brush_l5/data/data_L5.npz --n 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eye_judge as EJ
from sellanything_ml.sketch import brushgpt_l as G
from sellanything_ml.sketch import brushtokens_l as L

ML = Path(__file__).resolve().parents[1]
CATS = ML / "data" / "processed" / "brush_l"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="L5")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.8, 0.6])
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    z = np.load(a.data)
    val = [str(n) for n, s in zip(z["names"], z["split"]) if s == "val"][:a.n]
    out = {"n": len(val)}
    photos = []
    for n in val:
        f = next((ML / "data/processed/repaint").glob(f"**/shard_*/{n}.npz"))
        photos.append(np.load(f)["target"] / 255)
    out["photos"] = EJ.eye_score(np.stack(photos).astype(np.float32))
    for name, d in (("L5_paintings", "cats"), ("L8_paintings", "cats8")):
        out[name] = EJ.eye_score(np.stack([np.load(CATS / d / f"{n}.npz")["painting"] / 255 for n in val]).astype(np.float32))
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", G.device())
    layers = int(z["n_layers"])
    for t in a.temperatures:
        rows = G.sample(model, len(val), t, a.seed, max_len=3072 if layers == 5 else G.MAX_SEQ)
        gen = np.stack([L.render(L.decode_layers(r)[:layers]) for r in rows]).astype(np.float32)
        out[f"{a.run}_t{t}"] = {**EJ.eye_score(gen), "ended": int(sum(r[-1] == L.SP["EOS"] for r in rows))}
        if layers > L.LAYERS:                  # the same samples with only their first five layers (L8 vs its own block-in)
            out[f"{a.run}_t{t}_first5"] = EJ.eye_score(np.stack([L.render(L.decode_layers(r)[:L.LAYERS]) for r in rows]).astype(np.float32))
        np.savez_compressed(G.RUN_DIR / a.run / f"eye_judge_samples_t{t}_s{a.seed}.npz", paintings=(gen * 255).round().astype(np.uint8))
        print(json.dumps({t: out[f"{a.run}_t{t}"]}), flush=True)
    (G.RUN_DIR / a.run / "eye_judge.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
