"""Left-right mirror images of the eye model's training cats (owner, 2026-10-23, run E5: E3 with its training data
doubled by mirroring, everything else the same). Made on the tokens, as tools/mirror_l.py does for Brush GPT L, so
nothing is repainted; only training cats are mirrored (a mirrored validation or test cat would leak it), named
<name>__mirror. What changes, for an E3 sequence (split paint, xy coords, inside-out order):

  patches   L8's painting on the eye canvas flipped left to right;
  plan      outline points x -> 255 - x, each outline's points in reverse order, the two eyes swap (the first eye is
            always the one on the left: 5,546 of 5,546 cats), the two pupils swap, a pupil's angle changes sign
            (bin a -> 63 - a), its radii stay;
  strokes   x -> 255 - x, angle -> 31 - angle (bins over [0, pi), symmetric), and each layer's two eye blocks swap;
  order     the order each layer's strokes are written in is put back to the material's rule for the mirrored eye:
            the iris, pupil and highlight layers were sorted from the eye's centre outwards on their final
            positions (tools/eye_data_l8.py --order inside_out), so they are sorted the same way again; the outline
            layer was ordered around the eye only on its starting positions and the strokes moved while fitted
            (order-to-angle rank correlation 0.32), so it is not sorted: a clockwise order starting on the left,
            mirrored, runs from the left clockwise again once each half (above / below the starting line) is
            reversed, which keeps the outline layer as loosely ordered as the material is.

Changing the order changes which stroke is painted over which; --check measures that on the eye canvas against the
flipped material painting, and the flip alone. 24 training cats (2026-10-23): all 24 mirrored sequences pass the
grammar; the flip alone differs from the flipped painting by 0.04/255 on average per layer (under 0.006% of pixels
over 5/255, at most 9-20: the brush is not exactly symmetric), the flip and the re-ordering by 0.175/255.

    PYTHONPATH=.:tools python tools/eye_data_mirror.py --data data/processed/brush_l/data_E3.npz \\
        --eyes data/processed/brush_l/eye_l8_inside --out data/processed/brush_l/data_E5.npz
    PYTHONPATH=.:tools python tools/eye_data_mirror.py --data ... --eyes ... --check 24
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import eyegpt_l8 as EG

SP = EG.SP


def _plan(tokens: list[int]) -> tuple[list[list[int]], list[list[int]], int]:
    """The plan part's eye outlines (point tokens) and pupils (5 value tokens), and the position after PLAN_END."""
    i = tokens.index(SP["EYES"]) + 1
    eyes, pupils = [], []
    while tokens[i] != SP["PLAN_END"]:
        if tokens[i] == SP["EYE"]:
            j = tokens.index(SP["OUT_END"], i)
            eyes.append(tokens[i + 1:j])
            i = j + 1
        else:
            pupils.append(tokens[i + 1:i + 6])
            i += 6
    return eyes, pupils, i + 1


def mirror_plan(tokens: list[int], vocab: EG.Vocab) -> list[int]:
    """BOS EYES … PLAN_END of the mirrored cat (two outlines first, then the pupils, as in every cat's plan)."""
    eyes, pupils, _ = _plan(tokens)
    o, f = vocab.offset, vocab.fields
    flip_x = lambda t: o["x"] + f["x"] - 1 - (t - o["x"])                         # noqa: E731
    out = [SP["BOS"], SP["EYES"]]
    for pts in eyes[::-1]:
        pairs = [pts[a:a + 2] for a in range(0, len(pts), 2)][::-1]
        out += [SP["EYE"]] + [t for x, y in pairs for t in (flip_x(x), y)] + [SP["OUT_END"]]
    for x, y, r0, r1, a in pupils[::-1]:
        out += [SP["PUPIL"], flip_x(x), y, r0, r1, o["plan_angle"] + f["plan_angle"] - 1 - (a - o["plan_angle"])]
    return out + [SP["PLAN_END"]]


def blocks(tokens: list[int], vocab: EG.Vocab) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each layer, the (n, 9) bins of the first eye's strokes and of the second eye's (split at EYE_NEXT)."""
    t = np.asarray(tokens)
    sp = vocab.sp
    out = []
    for k in range(vocab.n_layers):
        per = len(vocab.layer_fields[k])
        s = int(np.flatnonzero(t == sp[f"LAYER{k}"])[0]) + 1
        e = s + int(np.flatnonzero(t[s:] == sp["LAYER_END"])[0])
        body = t[s:e]
        cut = int(np.flatnonzero(body == sp["EYE_NEXT"])[0])
        two = [body[:cut], body[cut + 1:]]
        out.append(tuple(EG.tokens_bins(b.reshape(-1, per), vocab, layer=k) if len(b) else np.zeros((0, 9), int)
                         for b in two))
    return out


def _angle_key(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """eye_data_l8.around_eyes's key: the angle around the centre from the left (pi), clockwise on screen."""
    return np.mod(np.arctan2(b[:, 1] + 0.5 - c[1], b[:, 0] + 0.5 - c[0]) - math.pi, 2 * math.pi)


def flip_bins(b: np.ndarray) -> np.ndarray:
    m = np.array(b, int, copy=True)
    m[:, 0] = EG.NXY - 1 - m[:, 0]
    m[:, 2] = EG.NANGLE - 1 - m[:, 2]
    return m


def mirror_layer(k: int, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """One eye block of layer k, already flipped, in the material's order for its (mirrored) centre c."""
    if not len(b):
        return b
    if k == 0:                                     # outline: each half of the mirrored clockwise order reversed
        return b[_reverse_halves(b, c)]
    d = np.hypot(b[:, 0] + 0.5 - c[0], b[:, 1] + 0.5 - c[1])
    return b[np.lexsort((_angle_key(b, c), d))]    # inside out, ties by angle (eye_data_l8.inside_out)


def _reverse_halves(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The material wrote the outline clockwise from the left, by the angle key k in [0, 2 pi) of each stroke's start.
    Mirroring maps k to pi - k (k < pi) and 3 pi - k (k >= pi): each half stays in its half with its order reversed.
    So the block (in the material's order) is split by the mirrored strokes' own key (< pi: first half) and each
    half is reversed, which gives ascending mirrored keys without sorting strokes that moved while fitted."""
    key = _angle_key(b, c)
    pos = np.arange(len(b))
    first, second = pos[key < math.pi], pos[key >= math.pi]
    return np.r_[first[::-1], second[::-1]]


def mirror(tokens: list[int], eye: dict[str, np.ndarray], vocab: EG.Vocab) -> tuple[np.ndarray, np.ndarray]:
    """The mirrored sequence (after the patches) and its loss weights."""
    seq = mirror_plan(tokens, vocab)
    w = [0] * len(seq)
    c = np.array([eye["plan_0"].mean(0), eye["plan_1"].mean(0)])
    cm = np.array([[EG.SIDE - c[1, 0], c[1, 1]], [EG.SIDE - c[0, 0], c[0, 1]]])   # mirrored, eyes swapped
    for k, (b0, b1) in enumerate(blocks(tokens, vocab)):
        new = [mirror_layer(k, flip_bins(b1), cm[0]), mirror_layer(k, flip_bins(b0), cm[1])]
        body = ([vocab.sp[f"LAYER{k}"]] + EG.bins_tokens(new[0], vocab, layer=k).reshape(-1).tolist()
                + [vocab.sp["EYE_NEXT"]] + EG.bins_tokens(new[1], vocab, layer=k).reshape(-1).tolist()
                + [vocab.sp["LAYER_END"]])
        seq += body
        w += [1] * len(body)
    seq.append(SP["EOS"])
    w.append(1)
    return np.array(seq, np.int32), np.array(w, np.uint8)


def check(data: EG.Data, eyes: Path, n: int) -> dict[str, object]:
    """On n training cats' eye canvases: the flipped material painted from the flipped strokes in the material's own
    order (the flip alone) and in the mirrored sequence's order (the flip and the re-ordering), each against the
    flipped stored painting."""
    import eye_render as ER
    rows = data.rows("train")[:n]
    res = {"cats": len(rows), "flip_only_max": 0, "reordered_mean": [], "reordered_max": 0, "grammar_ok": 0}

    for i in rows:
        name = str(data.names[i])
        t = data.tokens[data.offsets[i]:data.offsets[i + 1]].tolist()
        e = dict(np.load(eyes / f"{name}.npz"))
        brushes = ER.brushes_of(e)

        def paint(base: np.ndarray, layers: list[np.ndarray], brushes: list[dict] = brushes) -> np.ndarray:
            return (ER.layers_on_crop(base.astype(np.float32) / 255, layers, brushes) * 255).round()

        base_m = e["base"][:, ::-1].copy()
        stored = paint(e["base"], EG.decode_layers(np.array(t), data.vocab))
        flipped = stored[:, ::-1]
        own = paint(base_m, [flip_bins(b) for b in EG.decode_layers(np.array(t), data.vocab)])
        m, _ = mirror(t, e, data.vocab)
        EG.allowed_groups(m, data.vocab)
        res["grammar_ok"] += 1
        new = paint(base_m, EG.decode_layers(m, data.vocab))
        res["flip_only_max"] = max(res["flip_only_max"], float(np.abs(own - flipped).max()))
        res["reordered_mean"].append(float(np.abs(new - flipped).mean()))
        res["reordered_max"] = max(res["reordered_max"], float(np.abs(new - flipped).max()))
    res["reordered_mean"] = round(float(np.mean(res["reordered_mean"])), 3)
    return res


def write(data_path: Path, eyes: Path, out: Path) -> dict[str, object]:
    z = dict(np.load(data_path))
    data = EG.Data(data_path)
    vocab = data.vocab
    if (vocab.paint, vocab.coords) != ("split", "xy"):
        raise ValueError(f"{data_path}: mirroring is written for split paint and xy coords, not {vocab.paint}/{vocab.coords}")
    sets = {frozenset(s): j for j, s in enumerate(data.sets)}
    toks, wts, sids, offs = [z["tokens"]], [z["loss_w"]], [z["set_ids"]], list(z["offsets"])
    pats, names, split = [z["patches"]], list(z["names"]), list(z["split"])
    added = []
    for i in data.rows("train"):
        name = str(data.names[i])
        e = dict(np.load(eyes / f"{name}.npz"))
        t, w = mirror(data.tokens[data.offsets[i]:data.offsets[i + 1]].tolist(), e, vocab)
        ids = [sets.setdefault(frozenset(a), len(sets)) for a in EG.allowed_groups(t, vocab)]
        toks.append(t.astype(np.int16))
        wts.append(w)
        sids.append(np.array(ids, np.uint8))
        offs.append(offs[-1] + len(t))
        added.append(EG.patches(e["base"][:, ::-1].copy()))
        names.append(f"{name}__mirror")
        split.append("train")
    lens = np.diff(offs)
    assert lens.max() <= vocab.max_tok
    np.savez_compressed(out, tokens=np.concatenate(toks), loss_w=np.concatenate(wts), set_ids=np.concatenate(sids),
                        offsets=np.array(offs, np.int64), patches=np.concatenate([pats[0], np.stack(added)]),
                        names=np.array(names), split=np.array(split),
                        sets=np.array(json.dumps([sorted(s) for s in sorted(sets, key=sets.get)])),
                        paint=np.array(vocab.paint), coords=np.array(vocab.coords), style=np.array(vocab.style))
    return {"cats": len(names), "mirrored": len(added), "split": {s: split.count(s) for s in ("train", "val", "test")},
            "tokens": [int(lens.min()), int(np.median(lens)), int(lens.max())], "sets": len(sets)}


def main() -> None:
    """--check n: the mirror check on n training cats; --out: write the data file with the mirrored cats added."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--eyes", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        print(json.dumps(check(EG.Data(a.data), a.eyes, a.check)))
    if a.out:
        print(json.dumps(write(a.data, a.eyes, a.out)))


if __name__ == "__main__":
    main()
