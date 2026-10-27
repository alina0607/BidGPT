"""Checks of the eye model's sequence before training (zh notebook 17.28), saved to runs/eye_data_l8/eyegpt_checks.json:

  round trip   material bins -> tokens -> bins is exact, every sequence passes the grammar, and the strokes decoded
               from the tokens paint the same face as the material's bins;
  EYE_NEXT     strokes on the wrong side of the cut between the two eyes (nearest plan eye centre);
  box          where the photo's eye points put the eye canvas vs where L8's plan eye outlines put it (centre shift in
               eye-canvas px, size ratio), on the first 1,022 cats made with the photo's points (eye_l8_landmarkbox/),
               and how many cats' plan box would leave the photo (every 5th cat of data_L8);
  model        parameters of the E1 configuration; tokens per sequence.

    PYTHONPATH=.:tools python tools/eyegpt_l8_checks.py --n 200
    PYTHONPATH=.:tools python tools/eyegpt_l8_checks.py --n 200 --paint split     # E2's encoding: eyegpt_checks_split.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import eye_crop_styles as E
import eye_data_l8 as D
import eye_layers_l as ELL
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B
from sellanything_ml.sketch import eyegpt_l8 as EG

OLD = EG.EYES.parent / "eye_l8_landmarkbox"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--paint", choices=EG.PAINTS, default="joint")
    a = ap.parse_args()
    vocab = EG.Vocab(a.paint)
    files = sorted(OLD.glob("afhq__*.npz"))
    res = {}
    bad = wrong = strokes = 0
    lens, loss_tok, face_diff = [], [], 0.0
    for f in files[:a.n]:
        eye = dict(np.load(f))
        seq8 = np.load(EG.BASES / f.name)["tokens"]
        t, w = EG.encode(seq8, eye, vocab)
        EG.allowed_groups(t, vocab)
        dec = EG.decode_layers(t, vocab)
        bad += sum(not np.array_equal(dec[k], eye[f"bins_{n}"]) for k, n in enumerate(EG.LAYER_NAMES))
        c = np.array([eye["plan_0"].mean(0), eye["plan_1"].mean(0)])
        for n in EG.LAYER_NAMES:
            b = eye[f"bins_{n}"]
            cut = EG.eye_split(b, c)
            sec = np.linalg.norm(b[:, :2] + 0.5 - c[1], axis=1) < np.linalg.norm(b[:, :2] + 0.5 - c[0], axis=1)
            wrong += int(sec[:cut].sum() + (~sec[cut:]).sum())
            strokes += len(b)
        lens.append(len(t))
        loss_tok.append(int(w.sum()))
    for f in files[:8]:                                         # painting from the tokens = painting from the bins
        eye = dict(np.load(f))
        seq8 = np.load(EG.BASES / f.name)["tokens"]
        face = L.render(L.decode_layers(seq8))
        t, _ = EG.encode(seq8, eye, vocab)
        a_ = D.eyes_on_face(face, EG.decode_layers(t, vocab), tuple(eye["ebox"]), eye["fbox"])
        b_ = D.eyes_on_face(face, [eye[f"bins_{n}"] for n in EG.LAYER_NAMES], tuple(eye["ebox"]), eye["fbox"])
        face_diff = max(face_diff, float(np.abs(a_ - b_).max()))
    res["round_trip"] = {"cats": min(a.n, len(files)), "layers_mismatched": bad, "face_max_diff_8cats": face_diff,
                         "tokens": [int(min(lens)), int(np.median(lens)), int(max(lens))], "loss_tokens_median": int(np.median(loss_tok))}
    res["eye_next"] = {"wrong_side": wrong, "strokes": strokes}
    shift, ratio = [], []
    for f in files:
        z = np.load(f)
        seq8 = np.load(EG.BASES / f.name)["tokens"]
        fbox = z["fbox"]
        k = (fbox[2] - fbox[0]) / B.SIDE
        cen = np.array([e.mean(0) + 0.5 for e in L.decode_plan(seq8)["eyes"]]) * k + fbox[:2]
        eb, ea = np.array(E.eye_box(np.r_[cen, np.zeros((3, 2))]), float), z["ebox"].astype(float)
        s = ea[2] - ea[0]
        shift.append(float(np.linalg.norm((eb[:2] + eb[2:]) / 2 - (ea[:2] + ea[2:]) / 2) / s * B.SIDE))
        ratio.append(float((eb[2] - eb[0]) / s))
    z8 = np.load(EG.DATA_L8)
    n_clip = n_all = 0
    for name in [str(x) for x in z8["names"]][::5]:
        rp = np.load(ELL.repaint_file(name))
        fbox = rp["box"].astype(float)
        k = (fbox[2] - fbox[0]) / B.SIDE
        cen = np.array([e.mean(0) + 0.5 for e in L.decode_plan(np.load(ELL.CATS8 / f"{name}.npz")["tokens"])["eyes"]]) * k + fbox[:2]
        side = 2 * np.linalg.norm(cen[0] - cen[1])
        lt = cen.mean(0) - side / 2
        n_all += 1
        n_clip += bool((lt < 0).any() or (lt + side > E.SOURCE).any())
    res["box"] = {"cats": len(files), "centre_shift_px_p50_p90_p99": [round(float(np.percentile(shift, q)), 1) for q in (50, 90, 99)],
                  "size_ratio_p5_p50_p95": [round(float(np.percentile(ratio, q)), 3) for q in (5, 50, 95)],
                  "clipped": n_clip, "checked_for_clip": n_all}
    model = EG.make_model(EG.Config(paint=a.paint))
    res["model"] = {"paint": a.paint, "parameters": model.num_params(), "vocab": vocab.size, "patches": EG.NPATCH,
                    "max_tok": vocab.max_tok}
    out = EG.ML_DIR / "runs" / "eye_data_l8" / ("eyegpt_checks.json" if a.paint == "joint" else f"eyegpt_checks_{a.paint}.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
