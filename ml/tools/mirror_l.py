"""Left-right mirror images of Brush GPT L's training sequences (owner, 2026-10-20: mirror both L5 and L8, then train
L8 from L5's weights). Runs D-K doubled their data the same way; for L the sketch plan has to be mirrored too.

The mirror is made on the tokens, so the mirrored painting is exactly the original flipped (the bins are symmetric:
x bin i -> 255 - i, angle bin i -> 31 - i over [0, pi)); nothing is repainted. What changes:
  strokes      x -> 255 - x, angle -> 31 - angle (a direction (cos t, sin t) becomes (-cos t, sin t));
  plan         points x -> 255 - x (plan points are whole pixels, so a part may move by one pixel), every outline's
               points in reverse order (the outlines keep their direction around the part) and the head's runs in
               reverse order; the two eyes and the two pupils swap (the first eye is always the one on the left of
               the picture: 300 of 300 cats checked); yaw and roll change sign (bin i -> 63 - i), pitch does not; a
               pupil's angle changes sign;
  detail       (L8) the left and right cheek regions swap, so the regions stay in the order ears, forehead, left
               cheek, right cheek, chin and mouth (left = left of the picture, as repaint_tidy.region_of).
Loss weights move with their tokens. Only training cats are mirrored (a mirrored validation or test cat in training
would leak it). A mirrored cat is named <name>__mirror.

    PYTHONPATH=.:tools python tools/mirror_l.py --check 20           # flip check on 20 cats (L5 and L8)
    PYTHONPATH=.:tools python tools/mirror_l.py --write              # every training cat, into cats_mirror/ and cats8_mirror/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import brushtokens_l as L

ML = Path(__file__).resolve().parents[1]
BL = ML / "data" / "processed" / "brush_l"
SPLIT_FROM = ML / "data" / "kaggle" / "brush_l5" / "data" / "data_L5.npz"
SP, OFF = L.SP, L.OFFSET
KINDS = {SP["HEAD"], SP["EYE"], SP["NOSE"], SP["MOUTH"]}


def _field(t: int, name: str) -> bool:
    return OFF[name] <= t < OFF[name] + L.FIELDS[name]


def _flip(t: int, name: str) -> int:
    return OFF[name] + L.FIELDS[name] - 1 - (t - OFF[name])


def _plan(pos: list[int], tok: np.ndarray) -> list[int]:
    """The plan's positions (between PLAN and PLAN_END) in mirrored order; the values are flipped by the caller."""
    out = []
    assert tok[pos[0]] == SP["POSE"]
    out.append(pos[:4])
    i = 4
    heads, eyes, pupils, rest = [], [], [], []
    while i < len(pos):
        k = tok[pos[i]]
        if k == SP["PUPIL"]:
            pupils.append(pos[i:i + 6])
            i += 6
            continue
        assert k in KINDS, k
        j = i + 1
        while tok[pos[j]] != SP["OUT_END"]:
            j += 1
        pts = [pos[a:a + 2] for a in range(i + 1, j, 2)][::-1]                 # (x, y) pairs, reversed
        seg = [pos[i]] + [p for xy in pts for p in xy] + [pos[j]]
        (heads if k == SP["HEAD"] else eyes if k == SP["EYE"] else rest).append(seg)
        i = j + 1
    for group in (heads[::-1], eyes[::-1], pupils[::-1], rest):
        out += group
    return [p for seg in out for p in seg]


def mirror(tok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mirrored sequence and, for each of its positions, the position it came from (to move loss weights)."""
    tok = np.asarray(tok)
    p0, p1 = int(np.flatnonzero(tok == SP["PLAN"])[0]), int(np.flatnonzero(tok == SP["PLAN_END"])[0])
    order = list(range(p0 + 1)) + _plan(list(range(p0 + 1, p1)), tok) + [p1]
    k = p1 + 1
    while k < len(tok):
        t = tok[k]
        if t in (SP["LAYER5"], SP["LAYER6"], SP["LAYER7"]):
            e = k + 1 + int(np.flatnonzero(tok[k + 1:] == SP["LAYER_END"])[0])
            ends = [k] + [int(x) for x in k + 1 + np.flatnonzero(tok[k + 1:e] == SP["REGION_END"])]
            blocks = [list(range(ends[r] + 1, ends[r + 1] + 1)) for r in range(len(ends) - 1)]
            assert len(blocks) == L.REGIONS, len(blocks)
            blocks[2], blocks[3] = blocks[3], blocks[2]                       # left and right cheek
            order += [k] + [p for b in blocks for p in b] + [e]
            k = e + 1
        else:
            order.append(k)
            k += 1
    order = np.array(order)
    assert sorted(order.tolist()) == list(range(len(tok)))
    out = tok[order].copy()
    pose = out[p0 + 1:p0 + 5]                                                # POSE yaw pitch roll
    pose[1], pose[3] = _flip(pose[1], "plan_angle"), _flip(pose[3], "plan_angle")
    q = p0 + 5
    while q < p1:
        if out[q] == SP["PUPIL"]:
            out[q + 1] = _flip(out[q + 1], "x")
            out[q + 5] = _flip(out[q + 5], "plan_angle")
            q += 6
        elif out[q] in KINDS:
            q += 1
            while out[q] != SP["OUT_END"]:
                out[q] = _flip(out[q], "x")
                q += 2
            q += 1
        else:
            raise ValueError(f"plan token {out[q]} at {q}")
    for q in range(p1 + 1, len(out)):                                       # strokes: x and angle
        if _field(out[q], "x") and q + 2 < len(out) and _field(out[q + 2], "angle") and _field(out[q + 1], "y"):
            out[q] = _flip(out[q], "x")
            out[q + 2] = _flip(out[q + 2], "angle")
    return out, order


def check(names: list[str]) -> dict:
    res = {"cats": len(names), "max_flip_gap": {}, "grammar": True, "l8_starts_with_l5": True, "twice_is_identity": True}
    for d, nl in (("cats", 5), ("cats8", 8)):
        gaps = []
        for n in names:
            t = np.load(BL / d / f"{n}.npz")["tokens"]
            m, _ = mirror(t)
            L.allowed_groups(m, nl)
            res["twice_is_identity"] &= bool((mirror(m)[0] == t).all())
            a = L.render(L.decode_layers(t)[:nl])
            b = L.render(L.decode_layers(m)[:nl])
            gaps.append(float(np.abs(a[:, ::-1] - b).max()))
            if d == "cats8":
                m5, _ = mirror(np.load(BL / "cats" / f"{n}.npz")["tokens"])
                res["l8_starts_with_l5"] &= bool((m[:len(m5) - 1] == m5[:-1]).all())
        res["max_flip_gap"][d] = round(max(gaps), 6)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    z = np.load(SPLIT_FROM)
    train = [str(n) for n, s in zip(z["names"], z["split"]) if s == "train"]
    if a.check:
        r = check(train[:a.check])
        print(json.dumps(r))
        (ML / "runs" / "brush_l_mirror").mkdir(parents=True, exist_ok=True)
        (ML / "runs" / "brush_l_mirror" / "check.json").write_text(json.dumps(r, indent=1))
    if a.write:
        for d in ("cats", "cats8"):
            out = BL / f"{d}_mirror"
            out.mkdir(exist_ok=True)
            for n in train:
                f = np.load(BL / d / f"{n}.npz")
                m, order = mirror(f["tokens"])
                np.savez_compressed(out / f"{n}__mirror.npz", tokens=m, loss_w=f["loss_w"][order], kept=f["kept"],
                                    painting=f["painting"][:, ::-1])
            print(d, len(train), "mirrored into", out)


if __name__ == "__main__":
    main()
