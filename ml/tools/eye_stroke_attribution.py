"""Which strokes make the eyes clearer and which make them blurrier? For real paintings and for a run's own paintings,
paint the strokes one at a time and, for every stroke that changes the eyes' pixels, measure the eyes just before and
just after it: the eye judge's confidence (tools/eye_judge.py) and the eyes' sharpness (the standard deviation of the
luminance inside each eye's core disc). Each such stroke is either an eye stroke (its centre lies in the eye region,
brushgpt.EYE_ZONE eye distances, the region eye weighting uses) or a spill stroke (its centre lies outside the region
but it paints over an eye). Summed per stage and per kind, the changes say which strokes build the eyes and which
blur them, for the real paintings (the data) and for the model's (the model) side by side.

The paintings come from tools/stage_continuation.py's saved file: "real" (validation paintings) and "free" (the model
painting the whole cat, face points too). Each painting's eyes are its own two face points (for "free", the points the
model wrote). Diagnostic only: nothing here is used in training or sampling.

    PYTHONPATH=. python tools/eye_stroke_attribution.py --run J_cat_aligned_points_eyes --source afhq
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

from sellanything_ml.sketch import brushgpt as G  # noqa: E402
from sellanything_ml.sketch import brushtokens as T  # noqa: E402
from sellanything_ml.sketch import catbrush as B  # noqa: E402

CORE = 0.2          # the eye itself: a disc of this many eye distances around each eye point
TOUCH = 2e-3        # a stroke touches an eye when it changes the core discs' pixels by more than this on average
KINDS = ("eye", "spill")


def eyes_of(row: np.ndarray) -> np.ndarray:
    """(2, 2) canvas pixels of the two eye points written at the start of a row (face points on)."""
    lo = G.LAYOUTS[True][0]
    pts = (row[1:1 + G.PREFIX] - lo[1:1 + G.PREFIX]).reshape(-1, 2) + 0.5
    return pts[:2].astype(np.float64)


def attribute(row: np.ndarray, dev: str) -> list[dict]:
    """Paint one row stroke by stroke; for every stroke that touches an eye, the eyes before and after it."""
    import torch
    eyes = eyes_of(row)
    d = float(np.linalg.norm(eyes[0] - eyes[1]))
    yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
    core = [np.hypot(xx - x, yy - y) < CORE * d for x, y in eyes]
    core_any = core[0] | core[1]
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1).to(dev)
    canvas = torch.ones(B.SIDE, B.SIDE, 3, device=dev)
    def sharp(im: np.ndarray) -> float:
        lum = im @ np.array([0.299, 0.587, 0.114])
        return float(np.mean([lum[c].std() for c in core]))
    out, idx = [], 0
    with torch.no_grad():
        for k, st in enumerate(T.decode(G.strokes_only(row).tolist())):
            for i in range(len(st["xy"])):
                one = {key: v[i:i + 1] for key, v in st.items()}
                before = canvas
                canvas = B.composite(canvas, B.Strokes.from_stage(one, B.STAGES[k][:2]).to(dev), grid)
                b_, a_ = before.cpu().numpy(), canvas.cpu().numpy()
                if np.abs(a_[core_any] - b_[core_any]).mean() > TOUCH:
                    centre = one["xy"][0]
                    inside = min(np.linalg.norm(centre - e) for e in eyes) < G.EYE_ZONE * d
                    out.append({"stroke": idx, "stage": k + 1, "kind": "eye" if inside else "spill",
                                "before": b_.clip(0, 1), "after": a_.clip(0, 1),
                                "sharp_before": sharp(b_.clip(0, 1)), "sharp_after": sharp(a_.clip(0, 1))})
                idx += 1
    return out


def summarise(records: list[list[dict]], n_paintings: int) -> dict:
    """Per stage and kind: strokes that touch an eye (per painting), mean change of the judge and of sharpness per
    stroke, the share of strokes that lower the judge, and the summed change per painting."""
    flat = [r for rec in records for r in rec]
    res = {}
    for stage in range(1, len(G.STROKES) + 1):
        for kind in KINDS:
            sel = [r for r in flat if r["stage"] == stage and r["kind"] == kind]
            if not sel:
                continue
            dc = np.array([r["d_conf"] for r in sel])
            ds = np.array([r["sharp_after"] - r["sharp_before"] for r in sel])
            res[f"{stage}_{kind}"] = {"strokes_per_painting": round(len(sel) / n_paintings, 2),
                                      "d_conf_mean": round(float(dc.mean()), 5), "d_conf_se": round(float(dc.std(ddof=1) / np.sqrt(len(dc))), 5) if len(dc) > 1 else None,
                                      "share_lowering_conf": round(float((dc < 0).mean()), 3),
                                      "d_conf_sum_per_painting": round(float(dc.sum() / n_paintings), 4),
                                      "d_sharp_mean": round(float(ds.mean()), 5),
                                      "share_lowering_sharp": round(float((ds < 0).mean()), 3),
                                      "d_sharp_sum_per_painting": round(float(ds.sum() / n_paintings), 4)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="J_cat_aligned_points_eyes")
    ap.add_argument("--source", default="afhq", help="which stage_continuation file: afhq or cat")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    dev = G.device()
    run_dir = G.RUN_DIR / a.run
    z = np.load(run_dir / f"stage_continuation_{a.source}_t1.0.npz")
    t0 = time.time()
    report = {"run": a.run, "source": a.source, "device": dev, "core": CORE, "zone": G.EYE_ZONE, "touch": TOUCH}
    saved = {}
    for name in ("real", "free"):
        rows = z[name][:a.n]
        recs = []
        for j, row in enumerate(rows):
            rec = attribute(row, dev)
            if rec:
                imgs = np.stack([r["before"] for r in rec] + [r["after"] for r in rec]).astype(np.float32)
                _, conf = judge(imgs, dev)
                c = conf[:, :2].mean(1)
                for i, r in enumerate(rec):
                    r["conf_before"], r["conf_after"] = float(c[i]), float(c[len(rec) + i])
                    r["d_conf"] = r["conf_after"] - r["conf_before"]
                    del r["before"], r["after"]
            recs.append(rec)
            if j % 8 == 0:
                print(json.dumps({"set": name, "painting": j, "touching": len(rec), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        report[name] = summarise(recs, len(rows))
        keys = ("stroke", "stage", "d_conf", "sharp_before", "sharp_after")
        saved[name] = np.array([(j, r["stroke"], r["stage"], KINDS.index(r["kind"]), r["d_conf"], r["sharp_before"], r["sharp_after"])
                                for j, rec in enumerate(recs) for r in rec], dtype=np.float32)
        report[name + "_columns"] = ["painting", *keys[:2], "kind (0 eye, 1 spill)", *keys[2:]]
    report["minutes"] = round((time.time() - t0) / 60, 1)
    out = run_dir / f"eye_strokes_{a.source}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(out.with_suffix(".npz"), **saved)
    print(json.dumps({k: report[k] for k in ("real", "free")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
