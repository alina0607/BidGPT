"""Evaluate the eye model (sellanything_ml/sketch/eyegpt_l8.py) as the owner set it before training (zh notebook 17.27):
every result is reported for L8 alone and for L8 + the eye model, on the same paintings, with the same eye judge
(tools/eye_judge.py, the cat-face landmark model trained on photos only). Nothing is picked: every cat is sampled once
per temperature (seed fixed) and every one is counted.

The eye model gets only what L8 produced: the eye canvas is placed by the eye outlines in L8's plan
(eyegpt_l8.plan_box), the pixels are L8's painting on that canvas (eye_layers_l.l8_on_crop), the plan part is L8's
eye outlines and pupils. Its strokes are painted straight onto L8's face canvas (eye_data_l8.eyes_on_face).

  real   validation cats, L8 painting from the cat's real plan (the bases, tools/gen_l8_bases.py): L8 alone,
         + eye model (temperature 1.0 and 0.8), and + the eye material (fitted to the photo: the upper bound);
         also on the zoomed eye canvas, the eye-disc L1 to the photo and the rim strength (measures only).
  free   L8 sampled freely (its own plan): the first n validation slots, seed 3, temperatures 1.0 / 0.8 / 0.6, as
         tools/eye_judge_l.py (zh notebook 17.19); + eye model at temperature 1.0. A painting whose plan does not have
         two eye outlines gets no eye strokes and is still counted (reported).

    PYTHONPATH=.:tools python tools/eye_eval_e1.py --run E1 --stage real
    PYTHONPATH=.:tools python tools/eye_eval_e1.py --run E1 --stage free --n 128

Any eye run works (--run E2: the split paint encoding); the plan part and the decoding use the run's own vocabulary.
--eyes <folder> (both stages): the eye material the run learned from; its brushes draw the run's strokes, and in the
real stage its paintings are the upper bound (default eye_l8/; E3: eye_l8_inside/). --l8-from <free_images.npz> (free only) takes the L8 paintings an earlier run saved instead of sampling L8 again: the
eye models are then compared on the very same faces, and L8's 3 x 128 paintings (about 90 GPU minutes) are not redone.

    PYTHONPATH=.:tools python tools/eye_eval_e1.py --run E2 --stage free --n 128 --l8-from runs/brushgpt/E1/eval/free_images.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import eye_crop_styles as E
import eye_render as ER
import eye_data_l8 as D
import eye_judge as EJ
import eye_layers_l as ELL
from sellanything_ml.sketch import brushgpt_l as G
from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import eyegpt_l8 as EG


def eye_inputs(l8_tokens: np.ndarray, vocab: EG.Vocab = EG.JOINT) -> tuple[np.ndarray, list[int], np.ndarray] | None:
    """What the eye model is given for one L8 painting: patches, plan part (in the eye model's vocabulary), eye canvas
    box (face px); None when L8's plan does not have two eye outlines."""
    if len(L.decode_plan(l8_tokens)["eyes"]) != 2:
        return None
    box = EG.plan_box(l8_tokens)
    base = ELL.l8_on_crop(l8_tokens, EG.FACE_BOX, tuple(box))
    return EG.patches((base * 255).round().astype(np.uint8)), EG.plan_tokens(l8_tokens, EG.FACE_BOX, tuple(box), vocab), box


def run_eye_model(model, inputs: list, temperature: float, seed: int, batch: int = 16) -> list[np.ndarray | None]:
    """Eye sequences for the inputs (None stays None); cats with plans of the same length are sampled together."""
    out: list = [None] * len(inputs)
    by_len: dict[int, list[int]] = {}
    for i, x in enumerate(inputs):
        if x is not None:
            by_len.setdefault(len(x[1]), []).append(i)
    for n, idx in sorted(by_len.items()):
        seqs = EG.sample(model, np.stack([inputs[i][0] for i in idx]), [inputs[i][1] for i in idx],
                         temperature, seed + n, batch=batch)
        for i, s in zip(idx, seqs):
            out[i] = s
    return out


BRUSHES: list[dict] = ER.brushes_of({})                         # the run's layers' brushes (main: from --eyes)


def with_eyes(face: np.ndarray, seq: np.ndarray | None, box: np.ndarray | None, vocab: EG.Vocab = EG.JOINT) -> np.ndarray:
    if seq is None:
        return face
    return ER.layers_on_face(face, EG.decode_layers(seq, vocab), BRUSHES, tuple(box), EG.FACE_BOX)


def judge_pair(base: list, other: list) -> dict:
    a, b = np.stack(base).astype(np.float32), np.stack(other).astype(np.float32)
    ca, cb = EJ.judge(a)[1][:, :2].mean(1), EJ.judge(b)[1][:, :2].mean(1)
    g = cb - ca
    return {"alone": EJ.eye_score(a), "with": EJ.eye_score(b),
            "gain": [round(float(g.mean()), 4), round(float(g.std(ddof=1) / np.sqrt(len(g))), 4), int((g > 0).sum()), len(g)]}


def stage_real(model, out: Path, temperatures: tuple, seed: int, n: int, eyes: Path = EG.EYES) -> dict:
    z = np.load(EG.DATA_L8)
    names = [str(x) for x, s in zip(z["names"], z["split"]) if s == "val" and (eyes / f"{x}.npz").exists()][:n or None]
    seqs8 = [np.load(EG.BASES / f"{x}.npz")["tokens"] for x in names]
    faces = [L.render(L.decode_layers(s)) for s in seqs8]
    inputs = [eye_inputs(s, model.vocab) for s in seqs8]
    res = {"cats": len(names), "no_two_eyes": sum(x is None for x in inputs)}
    material = []
    for x, face, seq8 in zip(names, faces, seqs8):
        e = dict(np.load(eyes / f"{x}.npz"))                    # through the run's own tokens (polar is not exact)
        bins = EG.decode_layers(EG.encode(seq8, e, model.vocab)[0], model.vocab)
        material.append(ER.layers_on_face(face, bins, ER.brushes_of(e), tuple(e["ebox"]), e["fbox"]))
    res["material_upper_bound"] = judge_pair(faces, material)
    rel_of = D.photo_paths()
    cats = [D.Cat(x, rel_of) for x in names]                     # the photo crop, for the zoomed measures only
    res["crop_base"] = {m: round(float(np.mean([c.measure(c.base)[m] for c in cats])), 4) for m in ("eye_l1", "rim")}
    res["crop_material"] = {m: round(float(np.mean([c.measure(np.load(eyes / f"{c.name}.npz")["painted"] / 255)[m]
                                                    for c in cats])), 4) for m in ("eye_l1", "rim")}
    res["crop_photo_rim"] = round(float(np.mean([E.rim_strength(c.target, c.eyes, c.d) for c in cats])), 4)
    keep = {}
    for t in temperatures:
        t0 = time.time()
        eye = run_eye_model(model, inputs, t, seed)
        withe = [with_eyes(f, s, None if x is None else x[2], model.vocab) for f, s, x in zip(faces, eye, inputs)]
        r = judge_pair(faces, withe)
        crops = []
        for c, s in zip(cats, eye):                              # the model's strokes on the material's eye canvas
            img = np.asarray(c.base, np.float32) if s is None else \
                ER.layers_on_crop(c.base, EG.decode_layers(s, model.vocab), BRUSHES)
            crops.append(c.measure(img))
        r["crop"] = {m: round(float(np.mean([x[m] for x in crops])), 4) for m in ("eye_l1", "rim")}
        r["ended"] = int(sum(s is not None and s[-1] == EG.SP["EOS"] for s in eye))
        r["strokes"] = [round(float(np.mean([len(b) for s in eye if s is not None
                                             for b in [EG.decode_layers(s, model.vocab)[k]]])), 1)
                        for k in range(model.vocab.n_layers)]
        r["seconds"] = round(time.time() - t0, 1)
        res[f"eye_t{t}"] = r
        keep[f"with_t{t}"] = (np.stack(withe) * 255).round().astype(np.uint8)
        print(json.dumps({t: r}), flush=True)
    np.savez_compressed(out / "real_images.npz", names=np.array(names), alone=(np.stack(faces) * 255).round().astype(np.uint8),
                        material=(np.stack(material) * 255).round().astype(np.uint8), **keep)
    return res


def stage_free(model, out: Path, n: int, l8_temps: tuple, eye_t: float, seed: int, l8_device: str,
               l8_from: Path | None = None) -> dict:
    res = {"n": n, "seed": seed, "eye_temperature": eye_t, "l8_from": str(l8_from) if l8_from else None}
    saved = np.load(l8_from) if l8_from else None
    l8 = None if saved is not None else G.load_model(G.RUN_DIR / "L8" / "best.pt", l8_device)
    keep = {}
    for t in l8_temps:
        t0 = time.time()
        if saved is not None:                                    # the earlier run's L8 paintings, padding (-1) removed
            rows = [x[x >= 0] for x in saved[f"tokens_t{t}"][:n]]
        else:
            rows = G.sample(l8, n, t, seed, max_len=G.MAX_SEQ)
        sec_l8 = time.time() - t0
        faces = [L.render(L.decode_layers(r)) for r in rows]
        inputs = [eye_inputs(r, model.vocab) for r in rows]
        t1 = time.time()
        eye = run_eye_model(model, inputs, eye_t, seed)
        withe = [with_eyes(f, s, None if x is None else x[2], model.vocab) for f, s, x in zip(faces, eye, inputs)]
        r = judge_pair(faces, withe)
        r.update({"l8_ended": int(sum(x[-1] == L.SP["EOS"] for x in rows)), "no_two_eyes": sum(x is None for x in inputs),
                  "eye_ended": int(sum(s is not None and s[-1] == EG.SP["EOS"] for s in eye)),
                  "seconds_l8": round(sec_l8, 1), "seconds_eye": round(time.time() - t1, 1)})
        res[f"l8_t{t}"] = r
        keep[f"tokens_t{t}"] = np.array([np.pad(x, (0, G.MAX_SEQ - len(x)), constant_values=-1) for x in rows])
        keep[f"alone_t{t}"] = (np.stack(faces) * 255).round().astype(np.uint8)
        keep[f"with_t{t}"] = (np.stack(withe) * 255).round().astype(np.uint8)
        print(json.dumps({t: r}), flush=True)
    np.savez_compressed(out / "free_images.npz", **keep)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="E1")
    ap.add_argument("--stage", choices=("real", "free"), required=True)
    ap.add_argument("--n", type=int, default=0, help="real: the first n validation cats (0: all); free: paintings")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eyes", type=Path, default=EG.EYES, help="the eye material the run learned from (its brushes; real: its upper bound)")
    ap.add_argument("--l8-from", type=Path, default=None, help="free: an earlier run's free_images.npz (L8 not resampled)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = a.device or EG.device()
    run = EG.RUN_DIR / a.run
    out = run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    model = EG.load_model(run / "best.pt", dev)
    global BRUSHES
    first = next((f for f in sorted(a.eyes.glob("afhq__*.npz")) if not f.name.endswith((".tmp.npz", ".compare.npz"))), None)
    BRUSHES = ER.brushes_of(first) if first else ER.brushes_of({})
    if len(BRUSHES) != model.vocab.n_layers:
        raise SystemExit(f"--eyes {a.eyes} has {len(BRUSHES)} layers, the run {model.vocab.n_layers}")
    if a.stage == "real":
        res = stage_real(model, out, (1.0, 0.8), a.seed, a.n, a.eyes)
    else:
        res = stage_free(model, out, a.n or 128, (1.0, 0.8, 0.6), 1.0, a.seed, dev, a.l8_from)
    (out / f"{a.stage}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
