"""Brush GPT L's validation loss split by painting layer and stroke field (owner, 2026-10-21: why is L8's fur messy?).
brushgpt_l.evaluate_loss reports each field over all layers together; here every position is also given the layer it
belongs to (the last LAYERk token before it), so e.g. the angle loss of the fine hairs can be read on its own.

For each layer and field, next to the model's loss: the loss of guessing uniformly over the field's tokens (ln of
their number), and the entropy of the field's tokens in the training data of that layer (what a model that only knew
how often each value occurs would get). A model that has learned where strokes go and which way they point is well
below that entropy. Validation cats, the same weights as training (strokes centred in the uncertain region weigh 0).
CPU by default (the Mac GPU is busy).

    PYTHONPATH=.:tools python tools/loss_by_layer_l.py --run L8 --data data/processed/brush_l/data_L8.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
FIELDS = ("x", "y", "angle", "size", "paint", "strength")


def layer_of(tok: np.ndarray, layer_ids: dict[int, int]) -> np.ndarray:
    """For each position, the painting layer it belongs to (-1 in the plan, before the first layer)."""
    mark = np.full(len(tok), -1)
    for t, k in layer_ids.items():
        mark[tok == t] = k
    idx = np.where(mark >= 0, np.arange(len(tok)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, mark[np.maximum(last, 0)], -1)


def field_mask(tok: np.ndarray, f: str, L) -> np.ndarray:
    return (tok >= L.OFFSET[f]) & (tok < L.OFFSET[f] + L.FIELDS[f])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    os.environ["BRUSHGPT_DEVICE"] = a.device
    import torch
    torch.set_num_threads(a.threads)
    from sellanything_ml.sketch import brushgpt_l as G
    from sellanything_ml.sketch import brushtokens_l as L
    data = G.Data(a.data)
    layer_ids = {L.SP[f"LAYER{k}"]: k for k in range(data.n_layers)}
    # the entropy of each field's tokens per layer in the training cats
    counts: dict[tuple[int, str], np.ndarray] = {}
    for i in data.rows("train"):
        tok = data.tokens[data.offsets[i]:data.offsets[i + 1]]
        lay = layer_of(tok, layer_ids)
        for k in range(data.n_layers):
            for f in FIELDS:
                v = tok[(lay == k) & field_mask(tok, f, L)].astype(int) - L.OFFSET[f]
                if len(v):
                    c = counts.setdefault((k, f), np.zeros(L.FIELDS[f]))
                    np.add.at(c, v, 1)
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", a.device)
    table = data.mask_table(a.device)
    tot, cnt = {}, {}
    rows = data.rows("val")
    with torch.no_grad():
        for b in range(0, len(rows), a.batch):
            tok, w, sid = data.batch(rows[b:b + a.batch])
            _, ce, wt = G.losses(model, tok, w, sid, table, a.device)
            ce, wt = ce.cpu().numpy(), wt.cpu().numpy()
            for r in range(len(tok)):
                lay = layer_of(tok[r], layer_ids)[1:]
                for k in range(data.n_layers):
                    for f in FIELDS:
                        m = (lay == k) & field_mask(tok[r, 1:], f, L) & (wt[r] > 0)
                        if m.any():
                            tot[(k, f)] = tot.get((k, f), 0.0) + float(ce[r][m].sum())
                            cnt[(k, f)] = cnt.get((k, f), 0) + int(m.sum())
            print(json.dumps({"done": min(b + a.batch, len(rows)), "of": len(rows)}), flush=True)
    out = []
    for (k, f), n in sorted(cnt.items()):
        c = counts[(k, f)]
        p = c[c > 0] / c.sum()
        loss = tot[(k, f)] / n
        out.append({"layer": k, "field": f, "tokens": n, "loss": round(loss, 4), "perplexity": round(math.exp(loss), 2),
                    "uniform": round(math.log(L.FIELDS[f]), 4), "train_entropy": round(float(-(p * np.log(p)).sum()), 4)})
    res = {"run": a.run, "val_cats": int(len(rows)), "rows": out}
    path = G.RUN_DIR / a.run / "loss_by_layer.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
