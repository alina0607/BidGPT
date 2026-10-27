"""Why the oil-process painter's eyes are worse than catbrush's: the eye region, stage by stage, for the pilot cats.

For each cat in runs/oilprocess/report.json, with the eye zone of brushgpt (strokes whose centre is within 0.35 eye
distances of an eye point):
  * how many strokes of each stage land in the eye zone (old painter and new painter);
  * L1 to the photo inside the eye zone after each stage, so the stages that make the eyes worse show up as a rise;
  * the independent eye judge (tools/eye_judge.py, the landmark model trained on photos) on the photo and on every
    finished painting, and how far the eyes it finds are from the true eye points.

    cd ml && PYTHONPATH=. python tools/oilprocess_eyes.py     # → runs/oilprocess/eyes.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import brushtokens as T
from sellanything_ml.sketch import catbrush as B
from sellanything_ml.sketch import oilprocess as O

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eye_judge import judge  # noqa: E402
from oilprocess_demo import AFHQ, MASKS, OLD, OUT, old_frames  # noqa: E402

EYE_ZONE = 0.35                    # brushgpt.EYE_ZONE
# the painters measured besides catbrush ("old"): the oil process ("new") and, once painted, the oil process with
# the eye steps ("eyes", runs/oilprocess/oil_eyes/)
PAINTERS = [("new", OUT, O.STEPS)] + ([("eyes", OUT / "oil_eyes", O.STEPS_EYES)]
                                      if (OUT / "oil_eyes" / "report.json").exists() else [])
KINDS = ["photo", "old"] + [k for k, *_ in PAINTERS]


def in_zone(xy: np.ndarray, eyes: np.ndarray, r: float) -> np.ndarray:
    return np.min(np.linalg.norm(xy[:, None, :] - eyes[None], axis=-1), axis=1) <= r


def main() -> None:
    ML = Path(__file__).resolve().parents[1]
    report = json.loads((OUT / "report.json").read_text())
    labels = np.load(ML / "data/processed/afhq_cats_mae_pad1.5.npz")
    ids = [str(r) for r in labels["ids"]]
    yy, xx = np.mgrid[0:B.SIDE, 0:B.SIDE] + 0.5
    grid = np.stack([xx, yy], -1).reshape(-1, 2)
    rows, images = [], []
    for c in report["cats"]:
        rel, name = c["id"], Path(c["id"]).stem
        box = tuple(int(v) for v in np.load(OLD / f"{name}.npz")["box"])
        photo = B.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box).target.numpy()
        face = O.Face.from_photo(labels["points"][ids.index(rel)], box)
        eyes = face.p.numpy()[:2]
        r = EYE_ZONE * face.d
        zone = in_zone(grid, eyes, r).reshape(B.SIDE, B.SIDE)
        old = T.load_npz(str(OLD / f"{name}.npz"))
        of = old_frames(old)
        eye_l1 = lambda f: float(np.abs(f - photo)[zone].mean())             # noqa: E731
        row = {"id": rel, "eye_zone_px": int(zone.sum()),
               "old_strokes_in_zone": [int(in_zone(s["xy"], eyes, r).sum()) for s in old],
               "old_eye_l1": [eye_l1(f) for f in of]}
        images += [photo, of[-1]]
        for kind, folder, steps in PAINTERS:
            z = np.load(folder / f"{name}.npz")
            new = [{k: z[f"s{i}_{k}"] for k in ("xy", "length_width", "angle", "colour", "alpha")} for i in range(len(steps))]
            nf = O.render(new, steps)
            row[f"{kind}_strokes_in_zone"] = [int(in_zone(s["xy"], eyes, r).sum()) for s in new]
            row[f"{kind}_eye_l1"] = [eye_l1(f) for f in nf]
            images.append(nf[-1])
        rows.append(row)
        print(name, {k: round(v[-1], 4) for k, v in row.items() if k.endswith("eye_l1")}, flush=True)
    pts, conf = judge(np.stack(images).astype(np.float32).clip(0, 1))
    for k, row in enumerate(rows):
        face = O.Face.from_photo(labels["points"][ids.index(row["id"])], tuple(int(v) for v in np.load(OLD / f"{Path(row['id']).stem}.npz")["box"]))
        eyes = face.p.numpy()[:2]
        for j, kind in enumerate(KINDS):
            i = len(KINDS) * k + j
            row[f"judge_{kind}_eye_conf"] = float(conf[i, :2].mean())
            row[f"judge_{kind}_eye_err_d"] = float(np.linalg.norm(pts[i, :2] - eyes, axis=1).mean() / face.d)
    summary = {kind: {"eye_conf": float(np.mean([r[f"judge_{kind}_eye_conf"] for r in rows])),
                      "eye_err_d": float(np.mean([r[f"judge_{kind}_eye_err_d"] for r in rows]))} for kind in KINDS}
    for kind in KINDS[1:]:
        summary[f"{kind}_eye_l1"] = float(np.mean([r[f"{kind}_eye_l1"][-1] for r in rows]))
        summary[f"{kind}_strokes_in_zone"] = np.mean([r[f"{kind}_strokes_in_zone"] for r in rows], 0).tolist()
        summary[f"{kind}_eye_l1_by_step"] = np.mean([r[f"{kind}_eye_l1"] for r in rows], 0).tolist()
    summary["old_eye_l1_by_stage"] = summary.pop("old_eye_l1_by_step")
    (OUT / "eyes.json").write_text(json.dumps({"eye_zone": EYE_ZONE, "summary": summary, "cats": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
