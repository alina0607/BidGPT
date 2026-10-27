"""Stroke-direction rules from the painterly-rendering literature, run on the oil-process pilot cats.

Each rule gives an orientation at every canvas pixel (θ and θ + π are the same stroke). They are compared on the
cat's fur (the subject mask minus the eye cores) with two references, neither of which is ground truth:
(a) the anatomical template from the nine face points (oilprocess.template_field) — this favours our field;
(b) the photo's fine hair texture, where it is coherent (oilprocess.texture, coherence > 0.4) — this favours the
image-only rules. Nobody has hand-marked the fur direction of these cats, so there is no independent reference yet.

Rules (parameters from the papers where they give them; ours where they do not, marked "chosen"):
  litwinowicz97  normal to the Sobel gradient of the blurred luminance; Gaussian whose support ends at 15 px
                 (edge kernel 11 px + 4 px, p.3–4; σ = 15 / 3, chosen); gradients with |Gx| < 3 and |Gy| < 3 (0–255
                 luminance, unnormalised Sobel, chosen) replaced by a thin-plate spline fit of Gx and Gy from the
                 rest (2,000 random good samples, 64 nearest neighbours — the paper fits all of them, globally)
  hertzmann98    normal to the Sobel gradient of the reference blurred with σ = f_σ·R, f_σ = 0.5, for R = 2 and 8
                 (the Impressionist style's finest and largest brushes); no interpolation (a stroke stops where the
                 gradient vanishes)
  hays04         blur σ = 2 (∝ brush width; constant chosen); basis points: the strongest 10% of gradients on the
                 cat, at least 12 px apart, at most 200 (paper: 10–200 a layer, threshold and spacing unstated);
                 Gx and Gy each interpolated with a linear RBF φ(r) = r over all basis points; angle = gradient + 90°
  catbrush       the old painter's starting angle: structure tensor with a 9 × 9 mean (catbrush.orientation)
  template       ours, anatomy only (oilprocess.template_field)
  field          ours, the template bent toward coherent fine texture (oilprocess.fur_field)

    cd ml && PYTHONPATH=. python tools/fur_field_compare.py     # → runs/oilprocess/fields.json, fields_<cat>.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from sellanything_ml.sketch import catbrush as B
from sellanything_ml.sketch import oilprocess as O

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ML = Path(__file__).resolve().parents[1]
OUT = ML / "runs/oilprocess"
AFHQ, MASKS, OLD = ML / "data/raw/afhq", ML / "data/processed/afhq_masks", ML / "data/processed/afhq_brush_face"
COHERENT = 0.4
SOBEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)       # unnormalised
plt.rcParams.update({"font.sans-serif": ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"], "font.size": 9})


def lum255(img: torch.Tensor) -> np.ndarray:
    """Luminance on 0–255 with Litwinowicz's weights (30, 59, 11)/100, which Hertzmann also uses."""
    return (img.numpy() @ np.array([0.30, 0.59, 0.11])) * 255


def blur(g: np.ndarray, sigma: float) -> np.ndarray:
    return O._gauss(torch.as_tensor(g, dtype=torch.float32), sigma).numpy() if sigma > 0 else g


def sobel(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import correlate
    return correlate(g, SOBEL, mode="nearest"), correlate(g, SOBEL.T, mode="nearest")


def normal_angle(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.arctan2(gy, gx) + math.pi / 2


def litwinowicz97(img: torch.Tensor, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    gx, gy = sobel(blur(lum255(img), 15 / 3))
    good = (np.abs(gx) >= 3.0) | (np.abs(gy) >= 3.0)
    ys, xs = np.nonzero(good)
    pick = rng.choice(len(xs), min(2000, len(xs)), replace=False)
    pts = np.c_[xs[pick], ys[pick]].astype(float)
    tps = RBFInterpolator(pts, np.c_[gx[good][pick], gy[good][pick]], kernel="thin_plate_spline", neighbors=64)
    bad = np.argwhere(~good)
    gx, gy = gx.copy(), gy.copy()
    if len(bad):
        fill = tps(bad[:, ::-1].astype(float))
        gx[~good], gy[~good] = fill[:, 0], fill[:, 1]
    return normal_angle(gx, gy), float(1 - good.mean())


def hertzmann98(img: torch.Tensor, radius: int) -> np.ndarray:
    gx, gy = sobel(blur(lum255(img), 0.5 * radius))
    return normal_angle(gx, gy)


def hays04(img: torch.Tensor, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = sobel(blur(lum255(img), 2.0))
    mag = np.hypot(gx, gy) * (mask > 0.5)
    ys, xs = np.nonzero(mag >= np.quantile(mag[mask > 0.5], 0.9))
    order = np.argsort(-mag[ys, xs])
    basis: list[tuple[int, int]] = []
    for i in order:
        if all((xs[i] - bx) ** 2 + (ys[i] - by) ** 2 >= 12 ** 2 for bx, by in basis):
            basis.append((xs[i], ys[i]))
            if len(basis) == 200:
                break
    pts = np.array(basis, float)
    rbf = RBFInterpolator(pts, np.c_[gx[pts[:, 1].astype(int), pts[:, 0].astype(int)],
                                     gy[pts[:, 1].astype(int), pts[:, 0].astype(int)]], kernel="linear")
    yy, xx = np.mgrid[0:B.SIDE, 0:B.SIDE]
    g = rbf(np.c_[xx.ravel(), yy.ravel()].astype(float)).reshape(B.SIDE, B.SIDE, 2)
    return normal_angle(g[..., 0], g[..., 1]), pts


def stats(theta: np.ndarray, ref: np.ndarray, where: np.ndarray) -> dict:
    d = np.degrees(O.angle_diff(theta[where], ref[where]))
    return {"mean_deg": float(d.mean()), "within_20_pct": float((d <= 20).mean() * 100),
            "over_60_pct": float((d > 60).mean() * 100)}


def quiver(ax, img, theta, mask, title, basis=None):
    ax.imshow(img * 0.55 + 0.45)
    ys, xs = np.mgrid[6:B.SIDE:12, 6:B.SIDE:12]
    m = mask[ys, xs] > 0.5
    t = theta[ys, xs]
    ax.quiver(xs[m], ys[m], np.cos(t[m]), np.sin(t[m]), angles="xy", pivot="mid", headwidth=0, headlength=0,
              headaxislength=0, color="#0b6e99", scale=26, width=0.005)
    if basis is not None:
        ax.scatter(basis[:, 0], basis[:, 1], s=6, c="#d62728")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, B.SIDE)
    ax.set_ylim(B.SIDE, 0)
    ax.axis("off")


def main() -> None:
    torch.set_num_threads(4)
    report = json.loads((OUT / "report.json").read_text())
    labels = np.load(ML / "data/processed/afhq_cats_mae_pad1.5.npz")
    ids = [str(r) for r in labels["ids"]]
    names = ["litwinowicz97", "hertzmann98_R2", "hertzmann98_R8", "hays04", "catbrush", "template", "field"]
    out = {"coherent": COHERENT, "cats": []}
    for c in report["cats"]:
        rel, name = c["id"], Path(c["id"]).stem
        box = tuple(int(v) for v in np.load(OLD / f"{name}.npz")["box"])
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box)
        face = O.Face.from_photo(labels["points"][ids.index(rel)], box)
        guides = O.Guides(photo, face)
        rng = np.random.default_rng(0)
        mask = photo.mask.numpy()
        fur = guides.regions["fur"].numpy() > 0.5
        fine, coh = (t.numpy() for t in O.texture(photo.target))
        lit, flat_share = litwinowicz97(photo.target, rng)
        hay, basis = hays04(photo.target, mask)
        th = {"litwinowicz97": lit, "hertzmann98_R2": hertzmann98(photo.target, 2),
              "hertzmann98_R8": hertzmann98(photo.target, 8), "hays04": hay,
              "catbrush": B.orientation(photo.target).numpy(), "template": O._angle(guides.template).numpy(),
              "field": O._angle(guides.field).numpy()}
        sure = fur & (coh > COHERENT)
        tmpl2, img2 = guides.template.numpy(), O._doubled(torch.as_tensor(fine)).numpy()
        weight = 0.6 * coh * np.clip((tmpl2 * img2).sum(-1), 0, None) ** 2    # oilprocess.fur_field's w
        row = {"id": rel, "fur_pixels": int(fur.sum()), "coherent_pixels": int(sure.sum()),
               "litwinowicz_flat_share": flat_share, "hays_basis": len(basis),
               "blend_weight_mean": float(weight[fur].mean()), "blend_weight_over_0.2_pct": float((weight[fur] > 0.2).mean() * 100),
               "coherence_mean": float(coh[fur].mean()),
               "vs_template": {k: stats(v, th["template"], fur) for k, v in th.items()},
               "vs_texture": {k: stats(v, fine, sure) for k, v in th.items()}}
        out["cats"].append(row)
        print(name, {k: round(row["vs_template"][k]["mean_deg"], 1) for k in names},
              {k: round(row["vs_texture"][k]["mean_deg"], 1) for k in names}, flush=True)
        fig, ax = plt.subplots(2, 4, figsize=(15, 7.8))
        img = photo.target.numpy()
        ax[0, 0].imshow(img)
        ax[0, 0].scatter(*face.p.numpy().T, s=10, c="red")
        ax[0, 0].set_title(f"{name}：照片與 9 個臉部點", fontsize=9)
        ax[0, 0].axis("off")
        titles = {"litwinowicz97": "Litwinowicz 1997：垂直梯度 + 薄板樣條補平坦處",
                  "hertzmann98_R2": "Hertzmann 1998：垂直梯度（R = 2）",
                  "hertzmann98_R8": "Hertzmann 1998：垂直梯度（R = 8）",
                  "hays04": "Hays & Essa 2004：最強梯度（紅點）的 RBF",
                  "catbrush": "舊畫家：結構張量 9 × 9", "template": "我們：臉部點的解剖模板",
                  "field": "我們：模板 + 一致的細紋理"}
        for a, k in zip(ax.flat[1:], names):
            quiver(a, img, th[k], mask, f"{titles[k]}\n偏離模板 {row['vs_template'][k]['mean_deg']:.0f}°，"
                   f"偏離細紋理 {row['vs_texture'][k]['mean_deg']:.0f}°", basis if k == "hays04" else None)
        fig.tight_layout()
        fig.savefig(OUT / f"fields_{name}.png", dpi=110)
        plt.close(fig)
    out["blend_weight_mean"] = float(np.mean([r["blend_weight_mean"] for r in out["cats"]]))
    out["blend_weight_over_0.2_pct"] = float(np.mean([r["blend_weight_over_0.2_pct"] for r in out["cats"]]))
    out["mean"] = {ref: {k: {s: float(np.mean([r[ref][k][s] for r in out["cats"]]))
                             for s in ("mean_deg", "within_20_pct", "over_60_pct")} for k in names}
                   for ref in ("vs_template", "vs_texture")}
    (OUT / "fields.json").write_text(json.dumps(out, indent=1))
    for ref in ("vs_template", "vs_texture"):
        print(ref, {k: round(v["mean_deg"], 1) for k, v in out["mean"][ref].items()})
    print("blend weight on fur: mean", round(out["blend_weight_mean"], 3), "share > 0.2", round(out["blend_weight_over_0.2_pct"], 1), "%")


if __name__ == "__main__":
    main()
