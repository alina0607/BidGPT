"""Does Brush GPT use the tokens before the one it predicts? Three measurements on a trained run, saved for the notebook:

1. Attention: one full forward pass over real validation paintings; for chosen tokens (the first stroke of stage 1, the
   first eye-region stroke of stage 3 and of stage 4), how much attention each layer's heads give to each earlier part
   of the sequence (BOS, face points, the strokes of each stage), and the weights themselves over every earlier
   position for one cat. Also, for every query segment, the mean share it gives to every key segment, per layer.
2. Same randomness, different first stage: the model continues after BOS, the face points and the real stage 1 of
   several cats, each with the same seed, so the continuations differ only because the first stage differs.
3. Moved face points: the same cats with their nine face points moved left or right by `--shift` px and nothing else,
   the same seed for every shift; the eye judge (tools/eye_judge.py) then finds the eyes in each painting. If the
   strokes follow the points, the judge's eyes move with them.

    nice -n 19 env BRUSHGPT_DEVICE=cpu PYTHONPATH=. python tools/brushgpt_autoregress_demo.py \
        --run I_aligned_points_eyes --data data/processed/brushgpt/tokens_runs_HI.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eye_judge import judge  # noqa: E402
from stage_continuation import stage_marks  # noqa: E402

from sellanything_ml.sketch import brushgpt as G  # noqa: E402
from sellanything_ml.sketch import brushtokens as T  # noqa: E402

SEGMENTS = ("BOS", "臉部點", "第1階段", "第2階段", "第3階段", "第4階段")


def segment_of(points: bool) -> np.ndarray:
    """For every position of a row, the index into SEGMENTS (stage markers count with their stage; EOS with stage 4)."""
    marks = stage_marks(points)
    seg = np.zeros(marks[-1] + 1, int)
    seg[1:marks[0]] = 1
    for k in range(4):
        seg[marks[k]:marks[k + 1]] = 2 + k
    seg[marks[-1]] = 5
    return seg


def attention(model, row: np.ndarray, keep: list[int]) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """One forward pass over row[:-1] (as in training, causal). Per layer, the attention weights averaged over heads
    (T, T); and for each query position in `keep`, every head's weights over the keys up to it (layers, heads, q+1).
    Computed from the queries and keys each layer's attention module makes, one cat at a time to bound memory."""
    import torch
    from sellanything_ml.llm.model import apply_rope
    mean, rows_kept = [], {q: [] for q in keep}

    def hook(mod, args):
        x, cos, sin = args[0], args[1], args[2]
        B_, T_, _ = x.shape
        c = mod.cfg
        q = mod.q(x).view(B_, T_, c.n_heads, c.head_dim).transpose(1, 2)
        k, _ = mod.kv(x).view(B_, T_, 2, c.n_kv_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        s = (q @ k.transpose(-1, -2)) / c.head_dim ** 0.5
        w = s.masked_fill(torch.ones(T_, T_, dtype=torch.bool).triu(1), float("-inf")).softmax(-1)[0]   # (H, T, T)
        mean.append(w.mean(0).numpy())
        for qi in keep:
            rows_kept[qi].append(w[:, qi, :qi + 1].numpy())

    hs = [b.attn.register_forward_pre_hook(hook) for b in model.blocks]
    with torch.no_grad():
        model(torch.from_numpy(row[None, :-1]))
    for h in hs:
        h.remove()
    return mean, {q: np.stack(v) for q, v in rows_kept.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="I_aligned_points_eyes")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-attn", type=int, default=8, help="validation cats for the segment-to-segment attention")
    ap.add_argument("--n-swap", type=int, default=4)
    ap.add_argument("--n-shift", type=int, default=16)
    ap.add_argument("--shift", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", "cpu")
    cfg = model.brush_cfg
    rows, ids = G.load_rows(a.data, "val", 0, cfg.order, points=cfg.points)
    pts = G.load_points(a.data, "val")
    marks = stage_marks(cfg.points)
    seg = segment_of(cfg.points)
    _, _, slot = G.LAYOUTS[cfg.points]
    out, report = {}, {"run": a.run, "data": str(a.data), "seed": a.seed, "marks": marks, "segments": SEGMENTS}

    # 1. attention ----------------------------------------------------------------------------------------------
    eyes = G.eye_strokes(rows[:1], pts[:1])[0]
    xslot = 1 + G.FIELDS.index("x")
    first_s1 = marks[0] + 1
    eye_s3 = int(next(p for p in range(marks[2] + 1, marks[3]) if eyes[p] and slot[p] == xslot))
    eye_s4 = int(next(p for p in range(marks[3] + 1, marks[4]) if eyes[p] and slot[p] == xslot))
    targets = {"第1階段第1筆的 x": first_s1, "第3階段第一筆眼睛區域筆的 x": eye_s3, "第4階段第一筆眼睛區域筆的 x": eye_s4}
    report["attention_targets"] = targets
    q_seg, k_seg = seg[1:], seg[:-1]                            # the query at position i predicts token i + 1
    share = np.zeros((cfg.n_layers, len(SEGMENTS), len(SEGMENTS)))   # layer, segment of the predicted token, key segment
    for c_ in range(a.n_attn):
        mean, kept = attention(model, rows[c_], [p - 1 for p in targets.values()] if c_ == 0 else [])
        for li, m in enumerate(mean):
            for qs in range(1, len(SEGMENTS)):
                share[li, qs] += np.array([m[q_seg == qs][:, k_seg == ks].sum(1).mean() for ks in range(len(SEGMENTS))]) / a.n_attn
        if c_ == 0:
            for p in targets.values():
                out[f"attn_{p}"] = kept[p - 1].astype(np.float32)          # (layers, heads, p): keys 0..p-1
        print(json.dumps({"attention cat": c_, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out["segment_share"] = share
    import torch                                                # the model's distribution for each token of one stroke
    mask = G.allowed("cpu", cfg.points)
    with torch.no_grad():
        logits = G.field_logits(model(torch.from_numpy(rows[:1, :-1]))[0].float(), 1, mask)[0]
    for f_ in range(T.TOKENS_PER_STROKE):
        p = eye_s3 + f_
        prob = logits[p - 1].softmax(-1).numpy()
        lo_, hi_ = int(G.LAYOUTS[cfg.points][0][p]), int(G.LAYOUTS[cfg.points][1][p])
        out[f"probs_{p}"] = prob[lo_:hi_ + 1].astype(np.float32)   # over the field's own tokens, in token order
    out["attn_row"] = rows[0]
    out["attn_points"] = pts[0]

    # 2. same seed, different first stage -------------------------------------------------------------------------
    sw = []
    for i in range(a.n_swap):
        sw.append(G.sample(model, 1, 1.0, seed=a.seed, batch=1, prefix=rows[i:i + 1, :marks[1]])[0])
    out["swap_tokens"] = np.stack(sw)
    out["swap_real"] = rows[:a.n_swap]
    report["swap_minutes"] = round((time.time() - t0) / 60, 1)

    # 3. moved face points ---------------------------------------------------------------------------------------
    shifts = [-a.shift, 0, a.shift]
    res = {}
    for dx in shifts:
        p_ = pts[:a.n_shift].copy()
        p_[..., 0] += dx
        prefix = np.concatenate([rows[:a.n_shift, :1], G.point_tokens(p_)], 1)
        gen = G.sample(model, a.n_shift, 1.0, seed=a.seed, batch=a.n_shift, prefix=prefix)
        found, conf = judge(np.stack([G.render(r) for r in gen]), "cpu")
        given_eyes = p_[:, :2]                                    # where the points told the eyes to be
        moved = found[:, :2, 0].mean(1) - pts[:a.n_shift, :2, 0].mean(1)   # judge's eye x minus the original eye x
        res[str(dx)] = {"judge_eye_x_shift_mean": round(float(moved.mean()), 2),
                        "judge_eye_x_shift_se": round(float(moved.std(ddof=1) / np.sqrt(len(moved))), 2),
                        "judge_to_given_eye_px_median": round(float(np.median(np.linalg.norm(found[:, :2] - given_eyes, axis=-1))), 2),
                        "eye_conf_mean": round(float(conf[:, :2].mean()), 4)}
        out[f"shift_{dx}_tokens"], out[f"shift_{dx}_found"] = gen, found
        print(json.dumps({"shift": dx, **res[str(dx)], "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out["shift_points"] = pts[:a.n_shift]
    report["shift"] = res
    report["shifts"] = shifts
    report["n_shift"] = a.n_shift
    report["minutes"] = round((time.time() - t0) / 60, 1)
    path = G.RUN_DIR / a.run / "autoregress_demo.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    np.savez_compressed(path.with_suffix(".npz"), ids=ids[:max(a.n_attn, a.n_swap, a.n_shift)], **out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
