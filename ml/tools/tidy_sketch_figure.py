"""Figures for the owner (2026-10-19): the tidy-up step at several "enough" thresholds (tools/repaint_tidy.py) and the
sketch plan (tools/sketch_plan.py), on the same 6 cats.

    PYTHONPATH=.:tools python tools/tidy_sketch_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import repaint as R
from repaint_tidy import REGION_ORDER, region_of

plt.rcParams["font.family"] = ["Heiti TC"]
ML = Path(__file__).resolve().parents[1]
TIDY = ML / "runs" / "repaint_tidy"
SKETCH = ML / "runs" / "sketch_plan" / "six"
MAC = ML / "data" / "processed" / "repaint" / "mac" / "shard_010"
RUNS = [("q100", "只重排＋再擬合\n（不刪筆）"), ("q93", "比例門檻 93%"), ("q85", "比例門檻 85%"),
        ("tau05", "絕對門檻 0.5"), ("tau1", "絕對門檻 1"), ("tau2", "絕對門檻 2")]
REGION_ZH = {"ears": "耳朵", "forehead": "額頭", "left_cheek": "左臉頰", "right_cheek": "右臉頰", "chin_mouth": "下巴與嘴周圍"}
REGION_COL = {"ears": "tab:orange", "forehead": "tab:blue", "left_cheek": "tab:green", "right_cheek": "tab:red",
              "chin_mouth": "tab:purple"}


def tidy_figure() -> None:
    runs = [(k, t, {r["file"]: r for r in json.loads((TIDY / k / "summary.json").read_text())}) for k, t in RUNS
            if (TIDY / k / "summary.json").exists()]
    files = list(runs[0][2])
    fig, axes = plt.subplots(len(files), 2 + len(runs), figsize=(3.2 * (2 + len(runs)), 3.5 * len(files)))
    for i, f in enumerate(files):
        cat = R.load(MAC / f)
        base = runs[0][2][f]
        panels = [(cat["target"], "照片（只留頭）"), (cat["painting"], f"原本 1,848 筆\n誤差 {base['full_l1_cat']:.4f}")]
        for k, t, rows in runs:
            r = rows[f]
            panels.append((np.load(TIDY / k / f)["painting"], f"{t}\n{r['strokes']:,} 筆　誤差 {r['l1_cat']:.4f}"))
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=11)
        axes[i, 0].set_ylabel(f.replace("afhq__flickr_cat_", "貓 ").replace(".npz", ""), fontsize=13, rotation=0,
                              ha="right", va="center")
    fig.text(0.5, 0.004, "誤差：只算貓的像素（不含背景與不確定區）的平均絕對誤差。每一欄都在新的順序下再擬合 100 步；"
             "「只重排＋再擬合」是同樣處理但不刪筆的對照組。", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(TIDY / "compare_zh.png", dpi=80)
    plt.close(fig)


HAIR_RUNS = [("tau1", "絕對門檻 1"), ("tau1_gain1", "絕對門檻 1\n＋毛層細線覆蓋（增益 1）"),
             ("tau1_gain05", "絕對門檻 1\n＋毛層細線覆蓋（增益 0.5）")]


def hair_figure() -> None:
    """The hair layers' own rule: whole face and a zoom on the forehead (the top middle of the head)."""
    runs = [(k, t, {r["file"]: r for r in json.loads((TIDY / k / "summary.json").read_text())}) for k, t in HAIR_RUNS]
    files = list(runs[0][2])
    cols = 2 + len(runs)
    fig, axes = plt.subplots(2 * len(files), cols, figsize=(3.4 * cols, 7.0 * len(files)))
    for i, f in enumerate(files):
        cat = R.load(MAC / f)
        pts = cat["points"]
        d = float(np.linalg.norm(pts[0] - pts[1]))
        cx, cy = pts[:2].mean(0) - [0, 0.55 * d]
        half = 0.55 * d
        box = (cx - half, cx + half, cy + half, cy - half)
        panels = [(cat["target"], "照片（只留頭）"), (cat["painting"], f"原本 1,848 筆\n誤差 {runs[0][2][f]['full_l1_cat']:.4f}")]
        for k, t, rows in runs:
            r = rows[f]
            hairs = sum(c["kept"] for c in r["counts"][5:])
            panels.append((np.load(TIDY / k / f)["painting"],
                           f"{t}\n{r['strokes']:,} 筆（細毛三層 {hairs}）　誤差 {r['l1_cat']:.4f}"))
        for j, (img, title) in enumerate(panels):
            for row, zoom in ((2 * i, False), (2 * i + 1, True)):
                ax = axes[row, j]
                ax.imshow(img)
                if zoom:
                    ax.set_xlim(box[0], box[1])
                    ax.set_ylim(box[2], box[3])
                else:
                    ax.set_title(title, fontsize=11)
                    ax.add_patch(plt.Rectangle((box[0], box[3]), 2 * half, 2 * half, fill=False, color="yellow", lw=1.5))
                ax.set_xticks([])
                ax.set_yticks([])
        axes[2 * i, 0].set_ylabel(f.replace("afhq__flickr_cat_", "貓 ").replace(".npz", ""), fontsize=13, rotation=0,
                                  ha="right", va="center")
        axes[2 * i + 1, 0].set_ylabel("黃框放大", fontsize=12, rotation=0, ha="right", va="center")
    fig.text(0.5, 0.003, "細毛三層＝細毛、最細的深色毛、最細的亮色毛（原本共 1,400 筆）。增益：一筆至少要蓋到多少「還沒被蓋到」的照片細線才保留。",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.008, 1, 1))
    fig.savefig(TIDY / "hair_rule_zh.png", dpi=72)
    plt.close(fig)


def line_check(target: np.ndarray, painting: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, dict]:
    """The photo's thin dark and light lines, green where the painting has them at the same place (min of the two
    maps), red where it does not; and the recall per map."""
    from repaint_tidy import line_maps
    ph, pa = line_maps(target, tri), line_maps(painting, tri)
    img = np.ones((256, 256, 3))
    rec = {}
    for k, n in ((6, "dark"), (7, "light")):
        a, b = ph[k], pa[k]
        got = np.minimum(a, b)
        rec[n] = float(got.sum() / a.sum())
        top = max(np.percentile(a[a > 0], 95), 1e-6)
        img -= (got / top).clip(0, 1)[..., None] * np.array([0.9, 0.2, 0.9])
        img -= ((a - got) / top).clip(0, 1)[..., None] * np.array([0.2, 0.9, 0.9])
    return img.clip(0, 1), rec


def big_compare(title: str, columns: list[tuple[np.ndarray, str]], pts: np.ndarray, out: Path) -> None:
    """Large panels side by side, whole (top) and a zoom on the forehead (bottom)."""
    d = float(np.linalg.norm(pts[0] - pts[1]))
    cx, cy = pts[:2].mean(0) - [0, 0.45 * d]
    half = 0.5 * d
    n = len(columns)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 17.2))
    for j, (img, t) in enumerate(columns):
        for row in (0, 1):
            ax = axes[row, j]
            ax.imshow(img, interpolation="lanczos" if img.dtype == np.uint8 else "nearest")
            if row == 1:
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy + half, cy - half)
                ax.set_title("黃框放大（額頭）", fontsize=16)
            else:
                ax.set_title(t, fontsize=16)
                ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, color="yellow", lw=2))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=21, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=70)
    plt.close(fig)


def full_hairs_figure(names: tuple[str, ...] = ("pixabay_cat_004798", "pixabay_cat_003341", "pixabay_cat_000417")) -> None:
    """Tabbies with clear fur, large: the photo the painter aims at (256 px), the painting with every stroke (1,848:
    fur 400, dark hairs 500, light hairs 500), and which of the photo's fine lines the painting has (green) or lacks
    (red)."""
    for name in names:
        cat = R.load(next(MAC.parent.glob(f"shard_*/afhq__{name}.npz")))
        tri = cat["tri"]
        l1 = float(np.abs(cat["painting"].astype(np.float32) / 255 - cat["target"].astype(np.float32) / 255)[tri == 1].mean())
        img, rec = line_check(cat["target"], cat["painting"], tri)
        big_compare(f"虎斑貓 {name.replace('pixabay_cat_', '')}：細毛全部用上（1,848 筆）vs 照片",
                    [(cat["target"], "照片（畫家的目標，256 px）"),
                     (cat["painting"], f"畫作：全部 1,848 筆\n（細毛 400＋深色毛 500＋亮色毛 500）　誤差 {l1:.4f}"),
                     (img, f"照片的細線：畫作有畫出來（綠）／沒畫出來（紅）\n深色細線 {rec['dark']:.0%}　亮色細線 {rec['light']:.0%}")],
                    cat["points"], TIDY / f"full_hairs_{name}_zh.png")


SIMPLE = (("tau1_simple", "afhq__pixabay_cat_001917.npz", "黑貓"), ("tau1_simple", "afhq__pixabay_cat_004525.npz", "奶油色貓"),
          ("tau1_simple", "afhq__pixabay_cat_004542.npz", "白貓"), ("tau1_simple", "afhq__pixabay_cat_004563.npz", "淺橘貓"),
          ("tau1", "afhq__flickr_cat_000016.npz", "白貓"))


def simple_figure() -> None:
    """Simple (few fine lines) cats, large: the photo, the painting with every stroke, the tidy step at tau 1, and
    which of the photo's fine lines the tau 1 painting has."""
    for run, f, kind in SIMPLE:
        row = {r["file"]: r for r in json.loads((TIDY / run / "summary.json").read_text())}[f]
        cat = R.load(next(MAC.parent.glob(f"shard_*/{f}")))
        tidy = np.load(TIDY / run / f)["painting"]
        img, rec = line_check(cat["target"], tidy, cat["tri"])
        _, rec_full = line_check(cat["target"], cat["painting"], cat["tri"])
        hairs = [c["kept"] for c in row["counts"][5:]]
        name = f.replace("afhq__", "").replace(".npz", "")
        big_compare(f"簡單的貓（{kind} {name[-6:]}）：絕對門檻 1 vs 照片",
                    [(cat["target"], "照片（畫家的目標，256 px）"),
                     (cat["painting"], f"原本：全部 1,848 筆　誤差 {row['full_l1_cat']:.4f}\n細線畫出：深色 {rec_full['dark']:.0%}　亮色 {rec_full['light']:.0%}"),
                     (tidy, f"絕對門檻 1：{row['strokes']:,} 筆　誤差 {row['l1_cat']:.4f}\n（細毛 {hairs[0]}、深色毛 {hairs[1]}、亮色毛 {hairs[2]}）"),
                     (img, f"絕對門檻 1 的細線：畫出（綠）／沒畫出（紅）\n深色細線 {rec['dark']:.0%}　亮色細線 {rec['light']:.0%}")],
                    cat["points"], TIDY / f"simple_tau1_{name}_zh.png")
        print(name, row["strokes"], row["l1_cat"], row["full_l1_cat"], {k: round(v, 2) for k, v in rec.items()},
              {k: round(v, 2) for k, v in rec_full.items()})


def hair_count_figure() -> None:
    """The same tabbies painted with the hair layers x1 (the repaint), x2 and x4 (tools/hair_count_test.py)."""
    hc = ML / "runs" / "hair_count_test"
    rows = {(r["name"], r["times"]): r for t in (2, 4) if (hc / f"summary_x{t}.json").exists()
            for r in json.loads((hc / f"summary_x{t}.json").read_text())}
    for name in sorted({n for n, _ in rows}):
        cat = R.load(next(MAC.parent.glob(f"shard_*/afhq__{name}.npz")))
        x1 = rows[(name, 2)]["x1"]
        cols = [(cat["target"], "照片（畫家的目標，256 px）"),
                (cat["painting"], f"×1：{x1['strokes']:,} 筆　誤差 {x1['l1']:.4f}\n細線畫出：深色 {x1['recall']['dark']:.0%}　亮色 {x1['recall']['light']:.0%}")]
        for t in (2, 4):
            if (name, t) in rows:
                r = rows[(name, t)]
                cols.append((np.load(hc / f"{name}_x{t}.npz")["painting"],
                             f"細毛三層 ×{t}：{r['strokes']:,} 筆　誤差 {r['l1']:.4f}\n細線畫出：深色 {r['recall']['dark']:.0%}　亮色 {r['recall']['light']:.0%}"))
        big_compare(f"虎斑貓 {name[-6:]}：細毛加倍有沒有用", cols, cat["points"], hc / f"hair_count_{name}_zh.png")


def regions_figure(name: str = "afhq__flickr_cat_000011.npz") -> None:
    """The fur layer's strokes coloured by region, in the order they are laid (the tidy step at tau 1)."""
    cat = R.load(MAC / name)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(cat["target"])
    axes[0].set_title("照片", fontsize=13)
    for ax, k, t in ((axes[1], 5, "細毛層（原本 400 筆）"), (axes[2], 6, "最細的深色毛層（原本 500 筆）")):
        ax.imshow(np.full_like(cat["target"], 255))
        xy = cat["layers"][k]["xy_px"]
        lab = region_of(cat["points"], xy)
        for r in REGION_ORDER:
            m = lab == r
            ax.scatter(xy[m, 0], xy[m, 1], s=6, c=REGION_COL[r], label=f"{REGION_ORDER.index(r) + 1} {REGION_ZH[r]}（{m.sum()}）")
        ax.legend(fontsize=10, loc="lower right", markerscale=3)
        ax.set_title(t + "：每一筆依中心點分區", fontsize=13)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.01, "細節層照 1 → 5 的順序一區一區畫完；每一區有自己的結束 token。", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(TIDY / "regions_zh.png", dpi=80)
    plt.close(fig)


def sketch_figure() -> None:
    summ = {c["id"]: c for c in json.loads((SKETCH / "summary.json").read_text())["cats"]}
    ids = list(summ)
    fig, axes = plt.subplots(len(ids), 3, figsize=(15, 5.2 * len(ids)))
    for i, cid in enumerate(ids):
        z = np.load(SKETCH / f"{cid}.npz", allow_pickle=True)
        s = summ[cid]
        pts = z["points"]
        d = float(np.linalg.norm(pts[0] - pts[1]))
        ax = axes[i, 0]
        ax.imshow(Image.open(SKETCH / f"{cid}_overlay.png"), extent=(0, 256, 256, 0))
        nose = pts[:2].mean(0) + 0.8 * (pts[2] - pts[:2].mean(0))
        face_dir = -z["R"][:2, 2] * 0.9 * d                          # where the face points (towards the viewer)
        ax.annotate("", nose + face_dir, nose, arrowprops={"arrowstyle": "-|>", "color": "yellow", "lw": 2.5})
        ax.set_title(f"計畫（檢查用，各部位不同顏色）\n左右轉 {s['yaw']:+.0f}°　上下 {s['pitch']:+.0f}°　歪頭 {s['roll']:+.0f}°",
                     fontsize=13)
        axes[i, 1].imshow(Image.open(SKETCH / f"{cid}_sketch.png"))
        axes[i, 1].set_title(f"B　淡淡的草稿層（給人看）\nA　計畫 token：{s['tokens']} 個", fontsize=13)
        painting = R.load(MAC / f"afhq__{cid}.npz")["painting"]
        axes[i, 2].imshow(painting)
        axes[i, 2].set_title("畫作（原本 1,848 筆）", fontsize=13)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(cid.replace("flickr_cat_", "貓 "), fontsize=14, rotation=0, ha="right", va="center")
    fig.text(0.5, 0.003, "藍：頭的輪廓　橘：耳朵　綠：眼眶　紅：瞳孔　紫：鼻子　青：嘴　黃色箭頭：臉朝的方向。"
             "下巴接脖子的地方是不確定區，不畫線。", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    fig.savefig(SKETCH / "figure_zh.png", dpi=72)
    plt.close(fig)


if __name__ == "__main__":
    hair_figure()
    tidy_figure()
    regions_figure()
    sketch_figure()
