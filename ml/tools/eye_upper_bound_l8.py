"""Before generating eye-model data for every cat (about a day of GPU), how much could a perfect eye model add at most?
(owner, 2026-10-21: eye data on L8's own paintings drawn from each cat's real plan; test first.)

The eye model will learn to paint the four eye layers (tools/eye_layers.py "layers") over a painting L8 drew. The
best it could ever do is what the eye painter fits to the photo there, so that fit is its upper bound. On the first
n validation cats:
  gen   L8 is handed each cat's real plan (its tokens up to PLAN_END) and paints the rest (Mac GPU; seconds per cat
        are recorded: test C, the time for every cat);
  fit   on each painting, rendered into the zoomed eye crop, the four layers are fitted to the photo inside the plan's
        eye outlines (CPU, as tools/eye_layers_l.py); the change they make is scaled back to the face canvas and added
        to L8's painting. The eye judge (tools/eye_judge.py) then scores L8's painting with and without it (test A),
        next to the cat's L8 training painting and photo; and how far the eyes the judge finds are from the plan's
        eye outlines, for L8's painting and the training painting (test B: are the plan's eyes where L8 painted them).
Decision rule fixed before the run (owner, 2026-10-21): a gain of 0.02 or more in the judge's eye score, visible in
the pictures, is worth the day; 0.01-0.02 is decided by looking; under 0.01 is not.

    PYTHONPATH=.:tools python tools/eye_upper_bound_l8.py --stage gen --n 64 --device mps
    PYTHONPATH=.:tools python tools/eye_upper_bound_l8.py --stage fit --n 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import eye_crop_styles as E
import eye_judge as EJ
import eye_layers as EL
import eye_layers_l as ELL
from sellanything_ml.sketch import brushgpt_l as G
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import catbrush as B

ML = Path(__file__).resolve().parents[1]
OUT = ML / "runs" / "eye_upper_bound_l8"
DATA = ML / "data" / "processed" / "brush_l" / "data_L8.npz"


def val_names(n: int) -> list[str]:
    z = np.load(DATA)
    return [str(x) for x, s in zip(z["names"], z["split"]) if s == "val"][:n]


def stage_gen(n: int, temperature: float, seed: int) -> None:
    names = val_names(n)
    real = [np.load(ELL.CATS8 / f"{x}.npz")["tokens"] for x in names]
    prefix = [s[:int(np.flatnonzero(s == L.SP["PLAN_END"])[0]) + 1] for s in real]
    model = G.load_model(G.RUN_DIR / "L8" / "best.pt", G.device())
    t0 = time.time()
    seqs = G.sample(model, n, temperature, seed, max_len=G.MAX_SEQ, prefix=prefix)
    sec = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"gen_t{temperature}_s{seed}.npz", names=np.array(names),
                        tokens=np.array([np.pad(s, (0, G.MAX_SEQ - len(s)), constant_values=-1) for s in seqs]),
                        lengths=np.array([len(s) for s in seqs]))
    info = {"cats": n, "seconds": round(sec, 1), "seconds_per_cat": round(sec / n, 2), "device": G.device(),
            "ended": int(sum(s[-1] == L.SP["EOS"] for s in seqs)),
            "hours_for_5547": round(sec / n * 5547 / 3600, 1)}
    (OUT / f"gen_t{temperature}_s{seed}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def crop_to_face(delta: np.ndarray, ebox: tuple, fbox: np.ndarray) -> tuple[np.ndarray, tuple]:
    """A change made on the eye crop, resampled to the face canvas: (h, w, 3) and its top-left corner (x, y)."""
    k_face = (fbox[2] - fbox[0]) / B.SIDE
    x0, y0 = (np.array(ebox[:2]) - fbox[:2]) / k_face
    side = (ebox[2] - ebox[0]) / k_face
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    size = int(np.ceil(x0 + side)) - ix0
    # sample the crop's change at the face pixels' centres
    ys, xs = np.mgrid[iy0:iy0 + size, ix0:ix0 + size] + 0.5
    u, v = (xs - x0) / side * 2 - 1, (ys - y0) / side * 2 - 1
    grid = torch.from_numpy(np.stack([u, v], -1)[None].astype(np.float32))
    # average the crop down first (it is finer than the face canvas), then sample
    t = torch.from_numpy(delta.transpose(2, 0, 1)[None].astype(np.float32))
    t = F.interpolate(t, size=max(8, int(round(side))), mode="area")
    out = F.grid_sample(t, grid, mode="bilinear", align_corners=False, padding_mode="zeros")[0].numpy().transpose(1, 2, 0)
    return out, (ix0, iy0)


def add_at(img: np.ndarray, patch: np.ndarray, corner: tuple) -> np.ndarray:
    out = img.copy()
    x0, y0 = corner
    h, w = patch.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(B.SIDE, x0 + w), min(B.SIDE, y0 + h)
    out[ya:yb, xa:xb] += patch[ya - y0:yb - y0, xa - x0:xb - x0]
    return out.clip(0, 1)


def plan_eye_centres(tokens: np.ndarray) -> np.ndarray:
    return np.array([e.mean(0) + 0.5 for e in L.decode_plan(tokens)["eyes"]])


def stage_fit(n: int, temperature: float, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    g = np.load(OUT / f"gen_t{temperature}_s{seed}.npz")
    names = [str(x) for x in g["names"]][:n]
    split = np.load(E.SPLIT)
    rel_of = {Path(str(i)).stem: str(i) for i in split["ids"]}
    faces, fixed, train_p, photos, crops, rows = [], [], [], [], [], []
    for i, name in enumerate(names):
        seq = g["tokens"][i][:g["lengths"][i]]
        real = np.load(ELL.CATS8 / f"{name}.npz")
        rp = np.load(ELL.repaint_file(name))
        fbox = rp["box"].astype(float)
        pts = fbox[:2] + rp["points"] * (fbox[2] - fbox[0]) / B.SIDE
        ebox = E.eye_box(pts)
        stem = name.split("__", 1)[1]
        photo = B.load(str(E.AFHQ / rel_of[stem]), str(E.MASKS / f"{stem}_mask1.png"), ebox)
        target = photo.target.numpy()
        eyes = (pts[:2] - np.array(ebox[:2])) * (B.SIDE / (ebox[2] - ebox[0]))
        d = float(np.linalg.norm(eyes[0] - eyes[1]))
        base = ELL.l8_on_crop(seq, fbox, ebox)
        outline = ELL.plan_outlines(seq, fbox, ebox)                        # the real plan (written in)
        t0 = time.time()
        final = E.paint(photo.target, photo.mask, EL.variants(target, eyes, d, outline)[0]["layers"], start=base)[-1]
        face = L.render(L.decode_layers(seq))
        patch, corner = crop_to_face(final - base, ebox, fbox)
        faces.append(face)
        fixed.append(add_at(face, patch, corner))
        train_p.append(real["painting"] / 255)
        photos.append(rp["target"] / 255)
        crops.append((target, base, final))
        yy, xx = np.mgrid[:B.SIDE, :B.SIDE] + 0.5
        disc = np.min([np.hypot(xx - x, yy - y) for x, y in eyes], 0) < 0.35 * d
        rows.append({"name": name, "seconds_fit": round(time.time() - t0, 1),
                     "crop_eye_l1_base": round(float(np.abs(base - target)[disc].mean()), 4),
                     "crop_eye_l1_fixed": round(float(np.abs(final - target)[disc].mean()), 4),
                     "crop_rim_base": round(E.rim_strength(base, eyes, d), 4), "crop_rim_fixed": round(E.rim_strength(final, eyes, d), 4),
                     "crop_rim_photo": round(E.rim_strength(target, eyes, d), 4)})
        print(json.dumps(rows[-1]), flush=True)
    res = {"cats": len(names), "temperature": temperature}
    judged = {}
    for key, imgs in (("l8_gen", faces), ("l8_gen_plus_bound", fixed), ("l8_training", train_p), ("photo", photos)):
        arr = np.stack(imgs).astype(np.float32)
        pts_j, conf = EJ.judge(arr)
        judged[key] = (pts_j, conf)
        res[key] = EJ.eye_score(arr)
    for key in ("l8_gen", "l8_training"):                                   # test B
        offs = []
        for i, name in enumerate(names):
            seq = g["tokens"][i][:g["lengths"][i]] if key == "l8_gen" else np.load(ELL.CATS8 / f"{name}.npz")["tokens"]
            c = plan_eye_centres(seq)
            d = float(np.linalg.norm(c[0] - c[1]))
            offs.append(float(np.linalg.norm(judged[key][0][i, :2] - c, axis=1).mean() / d))
        res[f"{key}_judge_off_plan"] = [round(float(np.mean(offs)), 4), round(float(np.std(offs) / np.sqrt(len(offs))), 4)]
    gain = judged["l8_gen_plus_bound"][1][:, :2].mean(1) - judged["l8_gen"][1][:, :2].mean(1)
    res["gain_eye"] = [round(float(gain.mean()), 4), round(float(gain.std() / np.sqrt(len(gain))), 4), int((gain > 0).sum())]
    for k in ("crop_eye_l1_base", "crop_eye_l1_fixed", "crop_rim_base", "crop_rim_fixed", "crop_rim_photo"):
        res[k] = round(float(np.mean([r[k] for r in rows])), 4)
    (OUT / "summary.json").write_text(json.dumps({"summary": res, "cats": rows}, indent=1))
    np.savez_compressed(OUT / "images.npz", faces=(np.stack(faces) * 255).round().astype(np.uint8),
                        fixed=(np.stack(fixed) * 255).round().astype(np.uint8),
                        crops=(np.stack([np.stack(c) for c in crops[:8]]) * 255).round().astype(np.uint8))
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("gen", "fit"), required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    if a.device:
        import os
        os.environ["BRUSHGPT_DEVICE"] = a.device
    if a.stage == "gen":
        stage_gen(a.n, a.temperature, a.seed)
    else:
        stage_fit(a.n, a.temperature, a.seed, a.threads)


if __name__ == "__main__":
    main()
