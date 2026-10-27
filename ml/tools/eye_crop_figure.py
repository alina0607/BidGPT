"""The labelled figure of an eye-crop painter comparison (tools/eye_crop_styles.py): one eye per panel, zoomed, the
photo with the traced eye outline, and for each painting whether it has a bright dot where the photo has a catchlight.

A catchlight is a peak of the photo's light map (saved by eye_crop_styles) that lies inside the traced outline and is
brighter than its surroundings (mean within 1.6 px minus the mean of the ring 4-6 px away, above 0.08); a painting has
it when its own contrast there is above 0.05 and above 0.4 of the photo's. The eye shown is the one whose outline the
tracer found best. Labels are in Chinese (the owner's notebook).

    PYTHONPATH=. python tools/eye_crop_figure.py runs/eye_crop_styles/pilot4 runs/eye_crop_styles/pilot4/compare.png
    PYTHONPATH=. python tools/eye_crop_figure.py runs/eye_crop_styles/24cats out.png --show 6 --bold rim_trace_hl \\
        --cols "photo=照片,old=①,rim_fit=②a,..."
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Heiti TC"]

COLS = "photo=照片,old=① 舊畫家（現在的訓練資料）,old_edge=② 舊畫家＋暗線目標,old_contour=③ 描眼框＋點高光（新）"
NOTE = ("黃圈＋箭頭＝照片高光的位置，這張畫在那裡真的有亮點（用亮度量的）　紅色虛線圈＝照片有高光，這張畫沒畫出來\n"
        "青點＝程式在照片上描到的眼框　暗框強度：越接近照片的數字越好　眼睛誤差：顏色跟照片差多少，越小越好")
NOTE_TRACED = "\n注意：③ 的高光點本來就是照著照片最亮的點放的，所以它一定 100% 對得上；要看的是它自不自然，以及暗框有沒有出來"


def lum(im: np.ndarray) -> np.ndarray:
    return im @ np.array([0.299, 0.587, 0.114])


def contrast(L: np.ndarray, x: float, y: float) -> float:
    """Mean brightness within 1.6 px of (x, y) minus the mean of the ring 4-6 px away."""
    yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
    r = np.hypot(xx - x, yy - y)
    return float(L[r <= 1.6].mean() - L[(r >= 4) & (r <= 6)].mean())


def inside(p: np.ndarray, e: np.ndarray, C: np.ndarray) -> bool:
    """Whether p lies within 0.9 of the traced outline's radius in its direction from the eye point e."""
    ok = C[:, 3] > 0
    if ok.sum() < 20:
        return False
    ang = np.arctan2(C[ok, 1] - e[1], C[ok, 0] - e[0])
    rad = np.hypot(C[ok, 0] - e[0], C[ok, 1] - e[1])
    k = np.argmin(np.abs(np.angle(np.exp(1j * (ang - np.arctan2(p[1] - e[1], p[0] - e[0]))))))
    return bool(np.hypot(p[0] - e[0], p[1] - e[1]) < 0.9 * rad[k])


def catchlights(z) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    """The eye the tracer found best, its outline, the photo's catchlights inside it (at most 3), and the eye distance."""
    eyes = z["eyes"]
    j = int(np.argmax([(z[f"contour_{k}"][:, 3] > 0).sum() for k in (0, 1)]))
    e, C = eyes[j], z[f"contour_{j}"]
    Lp = lum(z["photo"])
    hl = [p for p in z["highlights"] if inside(p, e, C) and contrast(Lp, *p) > 0.08][:3]
    return e, C, hl, float(np.linalg.norm(eyes[1] - eyes[0]))


def painted(z, k, hl) -> list[bool]:
    """Whether painting k has a bright dot at each catchlight."""
    Lp, L = lum(z["photo"]), lum(z[k])
    return [contrast(L, *p) > max(0.05, 0.4 * contrast(Lp, *p)) for p in hl]


def panel(ax, z, k, title, rec, e, C, hl, box, bold=False) -> None:
    x0, x1, y0, y1 = box
    ax.imshow(z[k].clip(0, 1), interpolation="lanczos")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    if k == "photo":
        ax.set_title(f"{title}　（暗框強度 {rec['photo_rim']:.3f}）", fontsize=14, fontweight="bold")
        Cv = C[C[:, 3] > 0]
        ax.plot(Cv[:, 0], Cv[:, 1], ".", ms=4, color="#00e5ff")
        for p in hl:
            ax.add_patch(plt.Circle(p, 4.5, fill=False, ec="yellow", lw=2.5))
        return
    r = rec[k]
    ax.set_title(f"{title}\n暗框 {r['rim']:.3f}　眼睛誤差 {r['eye_l1']:.4f}", fontsize=13,
                 fontweight="bold" if bold else "normal")
    got = painted(z, k, hl)
    lab = (x0 + 0.62 * (x1 - x0), y0 + 0.10 * (y1 - y0))
    if any(got):
        ax.text(*lab, "高光點", fontsize=14, fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="#ffd400", ec="black", lw=0.8))
    for p, g in zip(hl, got):
        if g:
            ax.add_patch(plt.Circle(p, 4.5, fill=False, ec="#ffd400", lw=2.5))
            ax.annotate("", xy=p, xytext=lab, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=2.4, shrinkA=14, shrinkB=6))
        else:
            ax.add_patch(plt.Circle(p, 4.5, fill=False, ec="#ff3b30", lw=2.5, ls="--"))
    n = sum(got)
    tag = f"高光 {n}/{len(hl)} 有畫出" if hl else "照片這隻眼睛沒有高光"
    col = "#0a7d27" if hl and n == len(hl) else ("#c0392b" if hl and n == 0 else "#b36b00")
    ax.text(x0 + 2, y1 - 2, tag, fontsize=13, color="white", fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out")
    ap.add_argument("--cols", default=COLS, help="style=label,... in order; photo first")
    ap.add_argument("--show", type=int, default=0, help="draw only the first N cats (0: all); counts use all")
    ap.add_argument("--bold", default="old_contour", help="the column whose title is bold")
    a = ap.parse_args()
    cols = [tuple(c.split("=", 1)) for c in a.cols.split(",")]
    recs = {Path(c["id"]).stem: c for c in json.loads((a.src / "summary.json").read_text())["cats"]}
    files = sorted(a.src.glob("*.npz"))
    counts: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        *_, hl, _ = catchlights(z)
        for k, _ in cols[1:]:
            counts.setdefault(k, []).append([int(sum(painted(z, k, hl))), len(hl)])
    shown = files[:a.show] if a.show else files
    fig, axes = plt.subplots(len(shown), len(cols), figsize=(4.25 * len(cols), 4.6 * len(shown) + 1.6), squeeze=False)
    for i, f in enumerate(shown):
        z = np.load(f)
        e, C, hl, d = catchlights(z)
        h = 0.45 * d
        for c, (k, title) in enumerate(cols):
            panel(axes[i, c], z, k, title, recs[f.stem], e, C, hl, (e[0] - h, e[0] + h, e[1] - h, e[1] + h), bold=k == a.bold)
        axes[i, 0].set_ylabel(f.stem.replace("flickr_cat_", "貓 "), fontsize=15, rotation=0, ha="right", va="center")
    note = NOTE + (NOTE_TRACED if any(k == "old_contour" for k, _ in cols) else "")
    fig.text(0.5, 0.012, note, ha="center", fontsize=13, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.03 + 0.015 * note.count("\n"), 1, 1))
    fig.savefig(a.out, dpi=100)
    totals = {k: [sum(x for x, _ in v), sum(y for _, y in v)] for k, v in counts.items()}
    report = {"cats": len(files), "per_cat": counts, "total": totals}
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"cats": len(files), "total": totals}))


if __name__ == "__main__":
    main()
