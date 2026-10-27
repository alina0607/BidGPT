"""Side by side for eye runs (owner, 2026-10-22): one table and two figures from what each run's evaluation saved
(tools/eye_eval_e1.py, tools/eye_measure_e1.py, metrics.jsonl). CPU only, nothing is sampled again; the figures
show the first validation cats in order (not picked) for every run at eye temperature 1.0, next to the photo, L8
alone and the eye material each run learned from.

    PYTHONPATH=.:tools python tools/eye_compare_runs.py --runs E1 E2 E3 --out runs/brushgpt/compare_E1_E2_E3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eye_data_l8 as D
from sellanything_ml.sketch import eyegpt_l8 as EG

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
S = 256
CROP = (40, 95, 216, 175)                       # the eye region of the 256 px face canvas, shown enlarged
ZH = round(S * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0]))


def best_val(run: str) -> dict:
    """The validation row of the run's best step (lowest overall loss), with nats_per_cat for runs logged before it
    existed (E1: overall loss x weighted tokens per validation cat)."""
    rows = [json.loads(x) for x in open(EG.RUN_DIR / run / "metrics.jsonl") if '"val"' in x]
    b = min(rows, key=lambda r: r["val"]["all"])
    v = dict(b["val"], step=b["step"], minutes=b["minutes"], last_step=rows[-1]["step"])
    if "nats_per_cat" not in v:
        data = EG.Data(Path(json.loads((EG.RUN_DIR / run / "config.json").read_text())["data"]))
        rows_v = data.rows("val")
        w = sum(float(data.loss_w[data.offsets[i]:data.offsets[i + 1]].sum()) for i in rows_v)
        v["nats_per_cat"] = v["all"] * w / len(rows_v)
        v["stroke"] = sum(v[k] for k in data.vocab.stroke_fields)
    return v


def table(runs: list[str]) -> str:
    out = ["| | " + " | ".join(runs) + " |", "|---" * (len(runs) + 1) + "|"]
    vals = {r: best_val(r) for r in runs}
    real = {r: json.loads((EG.RUN_DIR / r / "eval" / "real.json").read_text()) for r in runs}
    free = {r: json.loads((EG.RUN_DIR / r / "eval" / "free.json").read_text()) for r in runs}
    meas = {r: json.loads((EG.RUN_DIR / r / "eval" / "measures.json").read_text()) for r in runs}
    row = lambda name, f: out.append(f"| {name} | " + " | ".join(f(r) for r in runs) + " |")   # noqa: E731
    row("best step (stopped at)", lambda r: f"{vals[r]['step']} ({vals[r]['last_step']}), {vals[r]['minutes']} min")
    row("nats per val cat", lambda r: f"{vals[r]['nats_per_cat']:.1f}")
    row("nats per stroke (colour)", lambda r: f"{vals[r]['stroke']:.2f} ({vals[r]['paint']:.2f})")
    for t in ("1.0", "0.8"):
        row(f"judge gain, real, eye t{t}", lambda r: "{:+.4f} ({}/{})".format(*real[r][f"eye_t{t}"]["gain"][:1],
                                                                             *real[r][f"eye_t{t}"]["gain"][2:]))
    row("judge gain, real, material", lambda r: "{:+.4f}".format(real[r]["material_upper_bound"]["gain"][0]))
    for t in ("1.0", "0.8", "0.6"):
        row(f"judge gain, free, L8 t{t}", lambda r: "{:+.4f}".format(free[r][f"l8_t{t}"]["gain"][0]))
    for key, label in (("with_t1.0", "eye t1.0"), ("with_t0.8", "eye t0.8"), ("material", "material")):
        m = {r: meas[r]["real"][key] for r in runs}
        row(f"real {label}: pupil contrast", lambda r: f"{m[r]['contrast'][0]:.3f}")
        row(f"real {label}: lighter outwards", lambda r: f"{m[r]['outward']:.0%}")
        row(f"real {label}: black-hole eyes", lambda r: f"{m[r]['holes'][0] / m[r]['holes'][1]:.1%}")
        row(f"real {label}: speckle", lambda r: f"{m[r]['speckle'][0]:.3f}")
        row(f"real {label}: left-right colour", lambda r: f"{m[r]['lr_diff'][0]:.3f}")
    for t in ("1.0", "0.8", "0.6"):
        m = {r: meas[r]["free"][f"with_l8t{t}"] for r in runs}
        row(f"free L8 t{t}: pupil contrast / lighter outwards / holes",
            lambda r: f"{m[r]['contrast'][0]:.3f} / {m[r]['outward']:.0%} / {m[r]['holes'][0] / m[r]['holes'][1]:.1%}")
    return "\n".join(out)


def figure(cols: list, title: str, path: Path, n: int, zoom: bool) -> None:
    big, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    h = ZH if zoom else S
    out = Image.new("RGB", (len(cols) * (S + 8) + 8, n * (h + 8) + 80), "white")
    d = ImageDraw.Draw(out)
    d.text((8, 8), title, fill=(0, 0, 0), font=big)
    for j, (cap, _) in enumerate(cols):
        d.text((8 + j * (S + 8), 46), cap, fill=(0, 0, 0), font=small)
    for i in range(n):
        for j, (_, get) in enumerate(cols):
            im = Image.fromarray(get(i))
            if zoom:
                im = im.crop(CROP).resize((S, ZH), Image.LANCZOS)
            out.paste(im, (8 + j * (S + 8), 74 + i * (h + 8)))
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table.md").write_text(table(a.runs) + "\n")
    print(table(a.runs))
    real = {r: np.load(EG.RUN_DIR / r / "eval" / "real_images.npz") for r in a.runs}
    names = real[a.runs[0]]["names"]
    assert all((real[r]["names"] == names).all() for r in a.runs), "runs evaluated on different cats"
    rel = D.photo_paths()
    photo = lambda i: np.asarray(Image.open(D.E.AFHQ / rel[str(names[i]).split("__", 1)[1]]).convert("RGB").resize((S, S)))  # noqa: E731
    cols = [("照片（參考）", photo), ("L8 單獨", lambda i: real[a.runs[0]]["alone"][i])]
    cols += [(f"+ {r}（溫度 1.0）", lambda i, r=r: real[r]["with_t1.0"][i]) for r in a.runs]
    cols += [(f"{r} 的教材", lambda i, r=r: real[r]["material"][i]) for r in a.runs
             if r == a.runs[0] or not np.array_equal(real[r]["material"][:a.n], real[a.runs[0]]["material"][:a.n])]
    t = f"真實 plan：驗證貓第 1–{a.n} 隻（照順序，沒有挑）"
    figure(cols, t, a.out / "real.png", a.n, False)
    figure(cols, t + "，眼睛放大", a.out / "real_eyes.png", a.n, True)
    free = {r: np.load(EG.RUN_DIR / r / "eval" / "free_images.npz") for r in a.runs}
    cols = [("L8 自由生成（溫度 1.0）", lambda i: free[a.runs[0]]["alone_t1.0"][i])]
    cols += [(f"+ {r}", lambda i, r=r: free[r]["with_t1.0"][i]) for r in a.runs]
    t = f"自由生成：同一批 L8 臉，第 1–{a.n} 張（沒有挑）"
    figure(cols, t, a.out / "free.png", a.n, False)
    figure(cols, t + "，眼睛放大", a.out / "free_eyes.png", a.n, True)


if __name__ == "__main__":
    main()
