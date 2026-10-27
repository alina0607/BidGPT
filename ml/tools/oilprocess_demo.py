"""Paint a few AFHQ cats with the oil-process painter and set them beside catbrush's paintings of the same cats.

The cats are drawn (seeded) from those that pass the training-data rules and already have a catbrush painting in
data/processed/afhq_brush_face; each is cut with that painting's own box, so both painters see the same canvas.
For every cat it writes the stage-by-stage frames of both painters and a fur-direction figure, and for all cats a
JSON of L1 per stage and of how far the fur strokes turn from (a) the anatomy-only fur template and (b) the photo's
own fine hair texture where that texture is coherent, and one figure of all cats side by side (compare_6cats.png).

    cd ml && PYTHONPATH=. python tools/oilprocess_demo.py [--cats 6] [--seed 0] [--iters 150]   # → runs/oilprocess/
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from sellanything_ml.sketch import brushtokens as T
from sellanything_ml.sketch import catbrush as B
from sellanything_ml.sketch import oilprocess as O

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ML = Path(__file__).resolve().parents[1]
AFHQ, MASKS = ML / "data/raw/afhq", ML / "data/processed/afhq_masks"
OLD = ML / "data/processed/afhq_brush_face"
OUT = ML / "runs/oilprocess"
STEPS, DEST = O.STEPS, OUT          # the painter being run and where its files go (main sets them)
OLD_FUR = [2, 3]                   # catbrush's two finest stages (32×12 and 18×7): its fur
COHERENT = 0.4
plt.rcParams.update({"font.sans-serif": ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"], "font.size": 8,
                     "axes.unicode_minus": False})


def old_frames(stages: list[B.Stage]) -> list[np.ndarray]:
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1)
    canvas, out = torch.ones(B.SIDE, B.SIDE, 3), []
    with torch.no_grad():
        for k, st in enumerate(stages):
            canvas = B.composite(canvas, B.Strokes.from_stage(st, B.STAGES[k][:2]), grid)
            out.append(canvas.numpy())
    return out


def at(img: torch.Tensor | np.ndarray, xy: np.ndarray) -> np.ndarray:
    ix = np.clip(xy.astype(int), 0, B.SIDE - 1)
    return np.asarray(img)[ix[:, 1], ix[:, 0]]


def fur_stats(strokes: list[B.Stage], guides: O.Guides, fine: torch.Tensor, coh: torch.Tensor) -> dict:
    """Turn of the fur strokes away from the anatomy template and from the photo's coherent fine texture, in °."""
    xy = np.concatenate([s["xy"] for s in strokes])
    ang = np.concatenate([s["angle"] for s in strokes])
    keep = at(guides.regions["fur"], xy) > 0.5
    xy, ang = xy[keep], ang[keep]
    tmpl = at(O._angle(guides.template), xy)
    d_tmpl = np.degrees(O.angle_diff(ang, tmpl))
    sure = at(coh, xy) > COHERENT
    d_tex = np.degrees(O.angle_diff(ang[sure], at(fine, xy[sure])))
    return {"strokes": int(keep.sum()), "template_mean_deg": float(d_tmpl.mean()),
            "template_within_20_pct": float((d_tmpl <= 20).mean() * 100),
            "texture_strokes": int(sure.sum()), "texture_mean_deg": float(d_tex.mean()) if sure.any() else None,
            "texture_within_20_pct": float((d_tex <= 20).mean() * 100) if sure.any() else None}


def field_figure(path: Path, photo: B.Photo, guides: O.Guides, old: list[B.Stage], new: list[B.Stage]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    img = photo.target.numpy()
    theta = O._angle(guides.field).numpy()
    ys, xs = np.mgrid[6:B.SIDE:12, 6:B.SIDE:12]
    m = guides.photo.mask.numpy()[ys, xs] > 0.5
    t = theta[ys, xs]
    ax[0].imshow(img)
    ax[0].quiver(xs[m], ys[m], np.cos(t[m]), np.sin(t[m]), angles="xy", pivot="mid", headwidth=0, headlength=0,
                 headaxislength=0, color="#00e5ff", scale=28, width=0.004)
    ax[0].scatter(*guides.face.p.numpy().T, s=10, c="red")
    ax[0].set_title("毛流場（由 9 個臉部點 + 照片細紋理）")
    tmpl = O._angle(guides.template).numpy()
    for a, strokes, title in ((ax[1], [old[k] for k in OLD_FUR], "舊畫家：第 3–4 階段的筆"),
                              (ax[2], [new[k] for k in O.fur_steps(STEPS)], "新畫家：6a–6c 毛髮筆")):
        a.imshow(img * 0.35 + 0.65)
        xy = np.concatenate([s["xy"] for s in strokes])
        ang = np.concatenate([s["angle"] for s in strokes])
        length = np.concatenate([s["length_width"][:, 0] for s in strokes])
        keep = at(guides.regions["fur"], xy) > 0.5
        dev = np.degrees(O.angle_diff(ang, at(tmpl, xy)))
        for (x, y), th, ln, d, k in zip(xy, ang, length, dev, keep):
            if not k:
                continue
            c = plt.cm.RdYlGn_r(min(d, 60) / 60)
            dx, dy = 0.5 * ln * math.cos(th), 0.5 * ln * math.sin(th)
            a.plot([x - dx, x + dx], [y - dy, y + dy], color=c, lw=0.8)
        a.set_title(f"{title}（綠＝順毛、紅＝偏 ≥60°）")
    for a in ax:
        a.set_xlim(0, B.SIDE)
        a.set_ylim(B.SIDE, 0)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def stages_figure(path: Path, name: str, photo: B.Photo, old: list[np.ndarray], new: list[np.ndarray],
                  l1_old: list[float], l1_new: list[float]) -> None:
    cols = 1 + len(new)
    fig, ax = plt.subplots(2, cols, figsize=(cols * 1.55, 3.6))
    ax[0, 0].imshow(photo.target.numpy())
    ax[0, 0].set_title(name, fontsize=6)
    for k, f in enumerate(old):
        ax[0, k + 1].imshow(f.clip(0, 1))
        ax[0, k + 1].set_title(f"舊 第{k + 1}階段\nL1 {l1_old[k]:.3f}", fontsize=7)
    for k, (f, step) in enumerate(zip(new, STEPS)):
        ax[1, k + 1].imshow(f.clip(0, 1))
        ax[1, k + 1].set_title(f"{step.name}\nL1 {l1_new[k]:.3f}", fontsize=7)
    ax[1, 0].imshow(photo.target.numpy())
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def compare_figure(report: dict, labels) -> None:
    """Photo, both finished paintings and a close-up of the cheek below the right eye, one row a cat."""
    ids = [str(r) for r in labels["ids"]]
    cats = report["cats"]
    fig, ax = plt.subplots(len(cats), 5, figsize=(15, 3.08 * len(cats)))
    for r, c in enumerate(cats):
        rel, name = c["id"], Path(c["id"]).stem
        box = tuple(int(v) for v in np.load(OLD / f"{name}.npz")["box"])
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box).target.numpy()
        old = old_frames(T.load_npz(str(OLD / f"{name}.npz")))[-1]
        z = np.load(OUT / f"{name}.npz")
        new = O.render([{k: z[f"s{i}_{k}"] for k in ("xy", "length_width", "angle", "colour", "alpha")}
                        for i in range(len(O.STEPS))])[-1]
        face = O.Face.from_photo(labels["points"][ids.index(rel)], box)
        cx, cy = (face.eye_r - 0.5 * face.d * face.up + 0.25 * face.d * face.right).numpy().astype(int)
        h = 40
        x0, y0 = np.clip(cx - h, 0, B.SIDE - 2 * h), np.clip(cy - h, 0, B.SIDE - 2 * h)
        crop = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        ims = [photo, old, new, old[crop], new[crop]]
        titles = [name, f"舊畫家（L1 {c['l1_old'][-1]:.3f}）", f"新畫家（L1 {c['l1_new'][-1]:.3f}）", "舊：臉頰放大",
                  "新：臉頰放大"]
        for j, (im, t) in enumerate(zip(ims, titles)):
            ax[r, j].imshow(im.clip(0, 1), interpolation="nearest" if j > 2 else "antialiased")
            ax[r, j].set_title(t, fontsize=9 if j else 8)
            ax[r, j].axis("off")
        for j in (1, 2):
            ax[r, j].add_patch(plt.Rectangle((x0, y0), 2 * h, 2 * h, fill=False, ec="red", lw=1))
    fig.tight_layout()
    fig.savefig(OUT / "compare_6cats.png", dpi=110)
    plt.close(fig)


def eye_box(face: O.Face) -> tuple[slice, slice]:
    """A crop around both eyes: half an eye distance beyond each, 0.4 above and below."""
    (xl, yl), (xr, yr), d = face.eye_l.numpy(), face.eye_r.numpy(), face.d
    x0, x1 = max(int(min(xl, xr) - 0.5 * d), 0), min(int(max(xl, xr) + 0.5 * d), B.SIDE)
    yc = (yl + yr) / 2
    y0, y1 = max(int(yc - 0.4 * d), 0), min(int(yc + 0.4 * d), B.SIDE)
    return slice(y0, y1), slice(x0, x1)


def compare3_figure(labels) -> None:
    """Photo, catbrush, the oil process and the oil process with the eye steps, then the three eye close-ups."""
    ids = [str(r) for r in labels["ids"]]
    cats = json.loads((OUT / "report.json").read_text())["cats"]
    eyes = {c["id"]: c for c in json.loads((OUT / "oil_eyes" / "report.json").read_text())["cats"]}
    fig, ax = plt.subplots(len(cats), 7, figsize=(19, 2.75 * len(cats)), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3, 1.3, 1.3]})
    for r, c in enumerate(cats):
        rel, name = c["id"], Path(c["id"]).stem
        box = tuple(int(v) for v in np.load(OLD / f"{name}.npz")["box"])
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box).target.numpy()
        old = old_frames(T.load_npz(str(OLD / f"{name}.npz")))[-1]
        paints = []
        for folder, steps in ((OUT, O.STEPS), (OUT / "oil_eyes", O.STEPS_EYES)):
            z = np.load(folder / f"{name}.npz")
            paints.append(O.render([{k: z[f"s{i}_{k}"] for k in ("xy", "length_width", "angle", "colour", "alpha")}
                                    for i in range(len(steps))], steps)[-1])
        crop = eye_box(O.Face.from_photo(labels["points"][ids.index(rel)], box))
        l1 = [c["l1_old"][-1], c["l1_new"][-1], eyes[rel]["l1_new"][-1]]
        ims = [photo, old, *paints, old[crop], paints[0][crop], paints[1][crop]]
        titles = [name, f"舊畫家（L1 {l1[0]:.3f}）", f"油畫流程（L1 {l1[1]:.3f}）", f"油畫流程 + 眼睛（L1 {l1[2]:.3f}）",
                  "舊：眼睛", "油畫流程：眼睛", "+ 眼睛：眼睛"]
        for j, (im, t) in enumerate(zip(ims, titles)):
            ax[r, j].imshow(im.clip(0, 1), interpolation="nearest" if j > 3 else "antialiased")
            ax[r, j].set_title(t, fontsize=9 if j else 8)
            ax[r, j].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "compare3_6cats.png", dpi=105)
    plt.close(fig)


def main() -> None:
    global STEPS, DEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=O.ITERS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--compare-only", action="store_true", help="only redraw compare_6cats.png from saved paintings")
    ap.add_argument("--compare3-only", action="store_true", help="only redraw compare3_6cats.png (needs both painters)")
    ap.add_argument("--painter", choices=sorted(O.PAINTERS), default="oil", help="oil_eyes writes to runs/oilprocess/oil_eyes/")
    args = ap.parse_args()
    STEPS = O.PAINTERS[args.painter]
    DEST = OUT if args.painter == "oil" else OUT / args.painter
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.load(ML / "data/processed/afhq_cats_mae_pad1.5.npz")
    if args.compare3_only:
        compare3_figure(labels)
        return
    if args.compare_only:
        compare_figure(json.loads((OUT / "report.json").read_text()), labels)
        return
    checks = np.load(ML / "data/processed/afhq_cats_mae_pad1.5_checks.npz")
    mirror = np.delete(checks["mirror"], [4, 7], axis=1).mean(1)          # the same rules as tools/paint_afhq.py
    kept = (checks["tip_outside"] <= 0.1) & (mirror <= np.quantile(mirror, 0.95))
    have = np.array([(OLD / f"{Path(str(r)).stem}.npz").exists() for r in labels["ids"]])
    pool = np.flatnonzero(kept & have)
    chosen = np.random.default_rng(args.seed).choice(pool, args.cats, replace=False)
    report = {"painter": args.painter, "steps": [s.name for s in STEPS], "iters": args.iters, "cats": []}
    for i in chosen:
        rel = str(labels["ids"][i])
        name = Path(rel).stem
        z = np.load(OLD / f"{name}.npz")
        box = tuple(int(v) for v in z["box"])
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box)
        face = O.Face.from_photo(labels["points"][i], box)
        print(name, flush=True)
        t0 = time.time()
        new, frames, guides = O.paint(photo, face, iters=args.iters, steps=STEPS)
        seconds = time.time() - t0
        redrawn = O.render(new, STEPS)
        old = T.load_npz(str(OLD / f"{name}.npz"))
        oframes = old_frames(old)
        target = photo.target.numpy()
        l1_new = [float(np.abs(f - target).mean()) for f in frames]
        l1_old = [float(np.abs(f - target).mean()) for f in oframes]
        fine, coh = O.texture(photo.target)
        stats = {"id": rel, "seconds": round(seconds, 1),
                 "redraw_max_diff": float(np.abs(redrawn[-1] - frames[-1]).max()),
                 "l1_old": l1_old, "l1_new": l1_new,
                 "strokes_old": int(sum(len(s["alpha"]) for s in old)),
                 "strokes_new": int(sum(len(s["alpha"]) for s in new)),
                 "fur_old": fur_stats([old[k] for k in OLD_FUR], guides, fine, coh),
                 "fur_new": fur_stats([new[k] for k in O.fur_steps(STEPS)], guides, fine, coh)}
        report["cats"].append(stats)
        print(f"  {seconds:.0f} s; L1 old {l1_old[-1]:.4f} new {l1_new[-1]:.4f}; "
              f"fur vs template old {stats['fur_old']['template_mean_deg']:.1f}° "
              f"new {stats['fur_new']['template_mean_deg']:.1f}°", flush=True)
        np.savez(DEST / f"{name}.npz", box=np.array(box),
                 **{f"s{k}_{key}": v for k, st in enumerate(new) for key, v in st.items()})
        stages_figure(DEST / f"{name}_stages.png", name, photo, oframes, frames, l1_old, l1_new)
        field_figure(DEST / f"{name}_fur.png", photo, guides, old, new)
        (DEST / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if args.painter == "oil":
        compare_figure(report, labels)
    else:
        compare3_figure(labels)


if __name__ == "__main__":
    main()
