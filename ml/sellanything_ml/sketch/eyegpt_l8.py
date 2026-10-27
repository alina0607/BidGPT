"""Eye GPT (owner, 2026-10-21): the eye part model on top of L8. It sees only what L8 produced (L8's painting on the
zoomed eye canvas, as pixels, and the eye outlines and pupils L8 wrote in its plan) and writes the four eye layers
(outline, iris, pupil, highlight) as strokes, token by token. It never sees the photo; nothing is retrieved or pasted.
Its training material is tools/eye_data_l8.py (the eye painter's four layers fitted to the photo on L8's painting of
the cat's real plan, zh notebook 17.26); condition A of zh notebook 17.27 (owner's choice, 2026-10-21), no mirror.

A sequence (coordinates on the eye canvas: two eye distances wide, 256 px, tools/eye_crop_styles.eye_box):

    [256 pixel patches]  BOS  EYES  EYE x y … OUT_END  EYE … OUT_END  PUPIL x y r r a  PUPIL …  PLAN_END
                         LAYER0 stroke… EYE_NEXT stroke… LAYER_END  LAYER1 …  LAYER2 …  LAYER3 …  EOS

The pixel patches: L8's painting rendered on the eye canvas (eye_layers_l.l8_on_crop), averaged to 64 x 64, cut into
16 x 16 patches of 4 x 4 px; each patch's 48 values go through one linear layer, plus a learned position per patch
(as ViT). Then the plan part, copied from L8's plan (no loss). A stroke is five tokens as in Brush GPT L: x, y, angle,
size (length x width, 6 x 6 steps of 0.5-1.5 times the layer's brush) and paint (colour 12^3 x opacity 8 over
catbrush's 0.3-1.0). Paint "split" (owner, 2026-10-22, run E2) writes the colour as four tokens instead, r, g, b
(12 levels each) and opacity (8), so a stroke is eight tokens: in E1's single paint token near colours were unrelated
classes, the light iris colours spread over thousands of them while black sat in a few, and sampling fell onto black
at lower temperatures (tools/eye_measure_e1.py). The bins, the order and everything else are the same; only the
tokens of the colour differ (joint, the default, is E1's layout). Coords "polar" (owner, 2026-10-22, run E4) writes a
stroke's position as its distance and direction from its own eye's centre (the plan pupil nearest to that eye's
outline, which the model is given) instead of x, y on the canvas: distance in 0.5 px steps up to 162 px, direction in
512 steps; the plan keeps x, y. Decoding goes back to the canvas grid, so this one is not exact (median 0.1 px off
before the grid). Strokes go eye by eye in the plan's order (EYE_NEXT between them) and, within an eye, around it
(tools/eye_data_l8.around_eyes); the model ends each layer itself. The grammar fixes only the order of the fields.

Same network code as Brush GPT L (llm.model.GPT, RoPE over patches and tokens, causal), smaller: width 384, 6 layers.

    python -m sellanything_ml.sketch.eyegpt_l8 data --out data/processed/brush_l/data_E1.npz
    python -m sellanything_ml.sketch.eyegpt_l8 train --name E1 --data data/processed/brush_l/data_E1.npz
    python -m sellanything_ml.sketch.eyegpt_l8 data --paint split --out data/processed/brush_l/data_E2.npz
    python -m sellanything_ml.sketch.eyegpt_l8 train --name E2 --paint split --data data/processed/brush_l/data_E2.npz
    python -m sellanything_ml.sketch.eyegpt_l8 data --paint split --coords polar --eyes <material> --out <data_E4.npz>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import brushtokens_l as L

if TYPE_CHECKING:
    from ..llm.model import GPTConfig

ML_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = ML_DIR / "runs" / "brushgpt"
BASES = ML_DIR / "data" / "processed" / "brush_l" / "l8_gen"
EYES = ML_DIR / "data" / "processed" / "brush_l" / "eye_l8"
DATA_L8 = ML_DIR / "data" / "processed" / "brush_l" / "data_L8.npz"
SIDE = 256
NXY, NANGLE, LEVELS_SIZE, LEVELS_RGB, LEVELS_ALPHA, NRADIUS, NPLAN_ANGLE = 256, 32, 6, 12, 8, 64, 64
LAYERS = 4
LAYER_NAMES = ("outline", "iris", "pupil", "highlight")
# The digital painter's eight layers (owner, 2026-10-24: tools/eye_layers.py "digital", the look of
# runs/eye_layers/digital6/steps_000031.png; material from tools/eye_data_l8.py --style digital): blend and blend2 carry
# no paint, they smudge, so their strokes have no colour tokens (x y angle size alpha: the opacity is how strongly
# they smudge).
DIGITAL_LAYERS = ("lid", "base", "shade", "blend", "fibres", "blend2", "pupil", "highlight")
DIGITAL_BLEND = frozenset({3, 5})
STYLES = ("layers", "digital")
GRID, PATCH = 64, 4                      # pixels averaged to 64 x 64, patches of 4 x 4
NPATCH, PATCH_DIM = (GRID // PATCH) ** 2, PATCH * PATCH * 3

PLAN_SPECIALS = ["BOS", "EOS", "EYES", "EYE", "OUT_END", "PUPIL", "PLAN_END"]   # the same ids in every vocabulary


def specials(n_layers: int) -> list[str]:
    return PLAN_SPECIALS + [f"LAYER{k}" for k in range(n_layers)] + ["EYE_NEXT", "LAYER_END"]


SPECIALS = specials(LAYERS)                                 # E1-E5's layout; LAYERk / EYE_NEXT / LAYER_END ids differ
SP = {name: i for i, name in enumerate(SPECIALS)}           # with the number of layers: use Vocab.sp for those
NAME = {i: n for n, i in SP.items()}
PAINTS = ("joint", "split")
COORDS = ("xy", "polar")
NRAD, RAD_STEP, NTHETA = 324, 0.5, 512   # polar: distance 0-162 px in 0.5 px steps, direction in 512 steps


class Vocab:
    """The token layout of one paint encoding: specials, then the stroke fields, then the plan's radius and angle.
    joint: x y angle size paint (E1). split: x y angle size r g b alpha. polar coords: rad theta instead of a stroke's
    x y (the plan keeps x, y). style layers: E1-E5's four layers; digital: the digital painter's eight (split paint
    only), whose blending layers write x y angle size alpha. MAX_TOK (tokens after the patches) keeps the same headroom
    over the material (192 strokes: joint 1,056 tokens of 1,280, split 1,632 of 2,048); digital's 8,192 is what one
    Kaggle T4 trains at batch 8 (runs/eye_t4_bench; the material is about 6,500)."""

    def __init__(self, paint: str = "joint", coords: str = "xy", style: str = "layers"):
        if paint not in PAINTS:
            raise ValueError(f"paint {paint!r} not in {PAINTS}")
        if coords not in COORDS:
            raise ValueError(f"coords {coords!r} not in {COORDS}")
        if style not in STYLES:
            raise ValueError(f"style {style!r} not in {STYLES}")
        if style == "digital" and paint != "split":
            raise ValueError("the digital style needs split paint (its blending strokes keep only the opacity)")
        self.paint, self.coords, self.style = paint, coords, style
        self.layer_names = LAYER_NAMES if style == "layers" else DIGITAL_LAYERS
        self.n_layers = len(self.layer_names)
        self.blend_layers = frozenset() if style == "layers" else DIGITAL_BLEND
        self.specials = specials(self.n_layers)
        self.sp = {name: i for i, name in enumerate(self.specials)}
        self.name = {i: n for n, i in self.sp.items()}
        colour = {"paint": LEVELS_RGB ** 3 * LEVELS_ALPHA} if paint == "joint" else \
            {"r": LEVELS_RGB, "g": LEVELS_RGB, "b": LEVELS_RGB, "alpha": LEVELS_ALPHA}
        where = {"x": NXY, "y": NXY} if coords == "xy" else {"rad": NRAD, "theta": NTHETA}
        stroke = {**where, "angle": NANGLE, "size": LEVELS_SIZE ** 2, **colour}
        self.fields = {"x": NXY, "y": NXY, **stroke, "radius": NRADIUS, "plan_angle": NPLAN_ANGLE}
        self.stroke_fields = tuple(stroke)
        self.per_stroke = len(stroke)
        self.colour_fields = tuple(colour)
        blend = tuple(where) + ("angle", "size", "alpha")
        self.layer_fields = [blend if k in self.blend_layers else self.stroke_fields for k in range(self.n_layers)]
        self.max_tok = 1280 if paint == "joint" else 2048 if style == "layers" else 8192
        self.offset: dict[str, int] = {}
        o = len(self.specials)
        for k, n in self.fields.items():
            self.offset[k] = o
            o += n
        self.size = o

    def group_of(self, tok: int) -> str:
        tok = int(tok)
        if tok in self.name:
            return self.name[tok]
        for g, n in self.fields.items():
            if self.offset[g] <= tok < self.offset[g] + n:
                return g
        raise ValueError(f"token {tok} out of the vocabulary")


JOINT = Vocab("joint")
FIELDS, OFFSET, VOCAB = JOINT.fields, JOINT.offset, JOINT.size      # E1's layout, as before
STROKE_FIELDS, TOKENS_PER_STROKE, MAX_TOK = JOINT.stroke_fields, JOINT.per_stroke, JOINT.max_tok


def group_of(tok: int, vocab: Vocab = JOINT) -> str:
    return vocab.group_of(tok)


ALL = "ALL"                              # the allowed set of padding positions (weight 0)


def token_mask(groups, vocab: Vocab = JOINT) -> np.ndarray:
    if ALL in groups:
        return np.ones(vocab.size, bool)
    m = np.zeros(vocab.size, bool)
    for g in groups:
        if g in vocab.fields:
            m[vocab.offset[g]:vocab.offset[g] + vocab.fields[g]] = True
        else:
            m[vocab.sp[g]] = True
    return m


class Grammar:
    """The order of the fields as a state machine (allowed() / step(tok)), as brushtokens_l.Grammar."""

    def __init__(self, vocab: Vocab = JOINT):
        self.vocab = vocab
        self.state, self.n, self.layer, self.next_used = "start", 0, -1, False

    def allowed(self) -> set[str]:
        st, n = self.state, self.n
        if st == "start":
            return {"BOS"}
        if st == "after_bos":
            return {"EYES"}
        if st == "plan":
            return {"EYE", "PUPIL", "PLAN_END"}
        if st == "outline_x":
            return {"x", "OUT_END"}
        if st == "outline_y":
            return {"y"}
        if st == "pupil":
            return [{"x"}, {"y"}, {"radius"}, {"radius"}, {"plan_angle"}][n]
        if st == "layer_start":
            return {f"LAYER{self.layer + 1}"} if self.layer + 1 < self.vocab.n_layers else {"EOS"}
        if st == "stroke":
            fields = self.vocab.layer_fields[self.layer]
            if n:
                return {fields[n]}
            return {fields[0], "LAYER_END"} | (set() if self.next_used else {"EYE_NEXT"})
        return set()

    def step(self, tok: int) -> None:
        g = self.vocab.group_of(tok)
        if g not in self.allowed():
            raise ValueError(f"token {tok} ({g}) not in {sorted(self.allowed())} (state {self.state})")
        st = self.state
        if g == "BOS":
            self.state = "after_bos"
        elif g == "EYES":
            self.state = "plan"
        elif g == "EYE":
            self.state = "outline_x"
        elif g == "OUT_END":
            self.state = "plan"
        elif st == "outline_x":
            self.state = "outline_y"
        elif st == "outline_y":
            self.state = "outline_x"
        elif g == "PUPIL":
            self.state, self.n = "pupil", 0
        elif st == "pupil":
            self.n += 1
            if self.n == 5:
                self.state = "plan"
        elif g in ("PLAN_END", "LAYER_END"):
            self.state = "layer_start"
        elif g.startswith("LAYER"):
            self.layer, self.state, self.n, self.next_used = self.layer + 1, "stroke", 0, False
        elif g == "EYE_NEXT":
            self.next_used = True
        elif st == "stroke":
            self.n = (self.n + 1) % len(self.vocab.layer_fields[self.layer])
        elif g == "EOS":
            self.state = "done"


def allowed_groups(tokens: np.ndarray, vocab: Vocab = JOINT) -> list[set[str]]:
    g, out = Grammar(vocab), []
    for i, tok in enumerate(np.asarray(tokens)):
        out.append(g.allowed())
        try:
            g.step(tok)
        except ValueError as e:
            raise ValueError(f"position {i}: {e}") from None
    return out


# ------------------------------------------------------------------ from L8's plan and the eye material
FACE_BOX = np.array([0.0, 0.0, SIDE, SIDE])     # the face canvas in its own pixels (generation has no photo)


def plan_box(l8_tokens: np.ndarray) -> np.ndarray:
    """The eye canvas on L8's face canvas (face px), from the eye outlines in L8's plan only: a square two eye
    distances wide around the midpoint of the two outlines' centres (eye_crop_styles.eye_box, WIDTH 2). The eye
    material (tools/eye_data_l8.py) places it the same way in the photo, rounded to photo pixels and kept inside the
    photo (4 of 1,110 cats would leave it); here nothing is rounded or clipped."""
    c = np.array([e.mean(0) + 0.5 for e in L.decode_plan(l8_tokens)["eyes"]])
    side = 2.0 * float(np.linalg.norm(c[0] - c[1]))
    lt = c.mean(0) - side / 2
    return np.r_[lt, lt + side]


def crop_map(fbox: np.ndarray, ebox) -> tuple[np.ndarray, float]:
    """Face-canvas px -> eye-canvas px: (offset, scale) with crop = face * scale + offset (eye_layers_l.to_crop)."""
    k_face, k_eye = (fbox[2] - fbox[0]) / SIDE, SIDE / (ebox[2] - ebox[0])
    return (np.asarray(fbox[:2], float) - np.asarray(ebox[:2], float)) * k_eye, k_face * k_eye


def _xy_tokens(xy_crop: np.ndarray, vocab: Vocab = JOINT) -> list[int]:
    b = np.clip(np.floor(xy_crop).astype(int), 0, NXY - 1)
    return [t for x, y in b for t in (vocab.offset["x"] + x, vocab.offset["y"] + y)]


def plan_tokens(l8_tokens: np.ndarray, fbox: np.ndarray, ebox, vocab: Vocab = JOINT) -> list[int]:
    """L8's eye outlines and pupils (in the plan's order), moved to the eye canvas: BOS EYES … PLAN_END."""
    off, k = crop_map(fbox, ebox)
    t = [int(x) for x in np.asarray(l8_tokens)]
    i = t.index(L.SP["PLAN"]) + 1
    out = [SP["BOS"], SP["EYES"]]
    while t[i] != L.SP["PLAN_END"]:
        name = L.NAME.get(t[i])
        if name == "POSE":
            i += 4
        elif name == "PUPIL":
            v = t[i + 1:i + 6]
            c = np.array([[v[0] - L.OFFSET["x"], v[1] - L.OFFSET["y"]]], float) + 0.5
            r = [int(np.clip(round((x - L.OFFSET["radius"]) * k), 0, NRADIUS - 1)) for x in v[2:4]]
            out += [SP["PUPIL"]] + _xy_tokens(c * k + off, vocab) + [vocab.offset["radius"] + r[0], vocab.offset["radius"] + r[1],
                                                                      vocab.offset["plan_angle"] + v[4] - L.OFFSET["plan_angle"]]
            i += 6
        else:
            j = t.index(L.SP["OUT_END"], i)
            if name == "EYE":
                xy = np.array(t[i + 1:j], float).reshape(-1, 2) - [L.OFFSET["x"], L.OFFSET["y"]] + 0.5
                out += [SP["EYE"]] + _xy_tokens(xy * k + off, vocab) + [SP["OUT_END"]]
            i = j + 1
    return out + [SP["PLAN_END"]]


def eye_centres(tokens: np.ndarray | list[int], vocab: Vocab = JOINT) -> np.ndarray:
    """(2, 2) eye-canvas px: for each eye outline of a sequence's plan part (the plan's order), the centre of the plan
    pupil nearest to the outline's centre (the outline's centre when the plan has no pupils). What polar strokes are
    measured from; the model is given it (the plan part comes first)."""
    t = [int(x) for x in np.asarray(tokens)]
    i, o = t.index(SP["EYES"]) + 1, vocab.offset
    outlines, pupils = [], []
    while t[i] != SP["PLAN_END"]:
        if t[i] == SP["EYE"]:
            j = t.index(SP["OUT_END"], i)
            xy = np.array(t[i + 1:j], float).reshape(-1, 2) - [o["x"], o["y"]] + 0.5
            outlines.append(xy.mean(0))
            i = j + 1
        else:                                                     # PUPIL x y r r a
            pupils.append(np.array([t[i + 1] - o["x"], t[i + 2] - o["y"]], float) + 0.5)
            i += 6
    if not pupils:
        return np.array(outlines)
    return np.array([min(pupils, key=lambda p: float(np.linalg.norm(p - c))) for c in outlines])


def _to_polar(xy_bins: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d = (np.asarray(xy_bins, float) + 0.5) * SIDE / NXY - centres
    rad = np.clip(np.floor(np.hypot(d[:, 0], d[:, 1]) / RAD_STEP), 0, NRAD - 1)
    theta = np.floor(np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * math.pi) / (2 * math.pi / NTHETA)) % NTHETA
    return np.stack([rad, theta], 1).astype(int)


def _from_polar(polar: np.ndarray, centres: np.ndarray) -> np.ndarray:
    r = (polar[:, 0] + 0.5) * RAD_STEP
    a = (polar[:, 1] + 0.5) * (2 * math.pi / NTHETA)
    xy = centres + np.stack([r * np.cos(a), r * np.sin(a)], 1)
    return np.clip(np.floor(xy * NXY / SIDE), 0, NXY - 1).astype(int)


def bins_tokens(b: np.ndarray, vocab: Vocab = JOINT, centres: np.ndarray | None = None, layer: int = 0) -> np.ndarray:
    """(n, 9) eye-material bins (x, y, angle, length, width, r, g, b, opacity) -> (n, 5) tokens (joint) or (n, 8)
    tokens (split); a blending layer's strokes (n, 5): no r g b. Polar coords need each stroke's eye centre ((n, 2)
    eye-canvas px, eye_centres)."""
    b = np.asarray(b, int)
    o = vocab.offset
    if vocab.coords == "xy":
        where = [o["x"] + b[:, 0], o["y"] + b[:, 1]]
    else:
        p = _to_polar(b[:, :2], centres)
        where = [o["rad"] + p[:, 0], o["theta"] + p[:, 1]]
    head = where + [o["angle"] + b[:, 2], o["size"] + b[:, 3] * LEVELS_SIZE + b[:, 4]]
    if layer in vocab.blend_layers:
        colour = [o["alpha"] + b[:, 8]]
    elif vocab.paint == "joint":
        colour = [o["paint"] + ((b[:, 5] * LEVELS_RGB + b[:, 6]) * LEVELS_RGB + b[:, 7]) * LEVELS_ALPHA + b[:, 8]]
    else:
        colour = [o["r"] + b[:, 5], o["g"] + b[:, 6], o["b"] + b[:, 7], o["alpha"] + b[:, 8]]
    return np.stack(head + colour, 1)


def tokens_bins(rows: np.ndarray, vocab: Vocab = JOINT, centres: np.ndarray | None = None, layer: int = 0) -> np.ndarray:
    """The inverse of bins_tokens; a blending stroke's r g b bins come back as 0 (it carries no paint)."""
    r = np.asarray(rows, int) - np.array([vocab.offset[f] for f in vocab.layer_fields[layer]])
    if layer in vocab.blend_layers:
        z = np.zeros(len(r), int)
        colour = [z, z, z, r[:, 4]]
    elif vocab.paint == "joint":
        rgb, a = r[:, 4] // LEVELS_ALPHA, r[:, 4] % LEVELS_ALPHA
        colour = [rgb // LEVELS_RGB ** 2, rgb // LEVELS_RGB % LEVELS_RGB, rgb % LEVELS_RGB, a]
    else:
        colour = [r[:, 4], r[:, 5], r[:, 6], r[:, 7]]
    xy = r[:, :2] if vocab.coords == "xy" else _from_polar(r[:, :2], centres)
    return np.stack([xy[:, 0], xy[:, 1], r[:, 2], r[:, 3] // LEVELS_SIZE, r[:, 3] % LEVELS_SIZE, *colour], 1)


def eye_split(b: np.ndarray, centres: np.ndarray) -> int:
    """Where a layer's strokes (ordered first eye, then second) move to the second eye: the cut with the fewest strokes
    on the wrong side of it, by each stroke's nearest plan eye centre."""
    xy = (np.asarray(b[:, :2], float) + 0.5) * SIDE / NXY
    second = np.linalg.norm(xy - centres[1], axis=1) < np.linalg.norm(xy - centres[0], axis=1)
    wrong = [int(second[:c].sum() + (~second[c:]).sum()) for c in range(len(b) + 1)]
    return int(np.argmin(wrong))


def encode(l8_tokens: np.ndarray, eye: dict, vocab: Vocab = JOINT) -> tuple[np.ndarray, np.ndarray]:
    """A cat's sequence (after the patches) and its loss weights (0 on the plan part, which is given)."""
    seq = plan_tokens(l8_tokens, eye["fbox"], tuple(eye["ebox"]), vocab)
    w = [0] * len(seq)
    centres = np.array([eye["plan_0"].mean(0), eye["plan_1"].mean(0)])
    polar = eye_centres(seq, vocab) if vocab.coords == "polar" else None
    sp = vocab.sp
    for k, name in enumerate(vocab.layer_names):
        b = eye[f"bins_{name}"]
        cut = eye_split(b, centres)
        at = None if polar is None else polar[np.r_[np.zeros(cut, int), np.ones(len(b) - cut, int)]]
        rows = bins_tokens(b, vocab, at, k)
        body = [sp[f"LAYER{k}"]] + rows[:cut].reshape(-1).tolist() + [sp["EYE_NEXT"]] + rows[cut:].reshape(-1).tolist() + [sp["LAYER_END"]]
        seq += body
        w += [1] * len(body)
    seq.append(SP["EOS"])
    w.append(1)
    return np.array(seq, np.int32), np.array(w, np.uint8)


def decode_layers(tokens: np.ndarray, vocab: Vocab = JOINT) -> list[np.ndarray]:
    """The (n, 9) bins of each eye layer in a sequence that follows the grammar (EYE_NEXT dropped)."""
    t = np.asarray(tokens)
    polar = eye_centres(t, vocab) if vocab.coords == "polar" else None
    out = []
    sp = vocab.sp
    for k in range(vocab.n_layers):
        per = len(vocab.layer_fields[k])
        s = np.flatnonzero(t == sp[f"LAYER{k}"])
        if not len(s):
            out.append(np.zeros((0, 9), int))
            continue
        e = s[0] + 1 + np.flatnonzero(t[s[0] + 1:] == sp["LAYER_END"])
        body = t[s[0] + 1:e[0] if len(e) else len(t)]
        nxt = np.flatnonzero(body == sp["EYE_NEXT"])
        first = (nxt[0] if len(nxt) else len(body)) // per                   # strokes of the plan's first eye
        body = body[body != sp["EYE_NEXT"]]
        body = body[:len(body) - len(body) % per]
        n = len(body) // per
        at = None if polar is None else polar[np.r_[np.zeros(min(first, n), int), np.ones(n - min(first, n), int)]]
        out.append(tokens_bins(body.reshape(-1, per), vocab, at, k) if len(body) else np.zeros((0, 9), int))
    return out


def patches(base_u8: np.ndarray) -> np.ndarray:
    """L8's painting on the eye canvas (256, 256, 3) uint8 -> (NPATCH, PATCH_DIM) uint8: averaged to GRID x GRID,
    patches in row order, each patch's pixels in row order, RGB."""
    f = SIDE // GRID
    small = base_u8.astype(np.float32).reshape(GRID, f, GRID, f, 3).mean((1, 3))
    n = GRID // PATCH
    p = small.reshape(n, PATCH, n, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(NPATCH, PATCH_DIM)
    return np.round(p).astype(np.uint8)


# ------------------------------------------------------------------ data
def read_exclude(path: Path | None) -> set[str]:
    """Names in an exclusion list (first column of each non-comment line, as data/processed/brush_l/eye_exclude_*.tsv)."""
    if path is None:
        return set()
    lines = Path(path).read_text().splitlines()
    return {ln.split("\t")[0].strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def build(out: Path, eyes_dir: Path = EYES, paint: str = "joint", coords: str = "xy", exclude: set[str] | None = None,
          style: str = "layers") -> dict:
    """Every cat with eye material -> one file: tokens, loss weights, allowed-set ids, offsets, patches, names, splits
    (the split of data_L8.npz) and the paint and coords encodings. Training cats named in `exclude` are left out
    (owner, 2026-10-24: closed or unseen eyes and the like, tools/eye_quality.py); validation and test cats never are,
    so the evaluation keeps every cat."""
    vocab = Vocab(paint, coords, style)
    z = np.load(DATA_L8)
    split_of = dict(zip(map(str, z["names"]), map(str, z["split"])))
    exclude = exclude or set()
    files = sorted(f for f in eyes_dir.glob("afhq__*.npz") if not f.name.endswith((".tmp.npz", ".compare.npz"))
                   and not (f.stem in exclude and split_of.get(f.stem) == "train"))
    sets: dict[frozenset, int] = {frozenset([ALL]): 0}
    toks, wts, sids, offs, pats, names, split = [], [], [], [0], [], [], []
    for f in files:
        eye = dict(np.load(f))
        t, w = encode(np.load(BASES / f.name)["tokens"], eye, vocab)
        ids = [sets.setdefault(frozenset(a), len(sets)) for a in allowed_groups(t, vocab)]
        toks.append(t.astype(np.int16))
        wts.append(w)
        sids.append(np.array(ids, np.uint8))
        offs.append(offs[-1] + len(t))
        pats.append(patches(eye["base"]))
        names.append(f.stem)
        split.append(split_of[f.stem])
    lens = np.diff(offs)
    info = {"cats": len(files), "split": {s: split.count(s) for s in ("train", "val", "test")},
            "tokens": [int(lens.min()), int(np.median(lens)), int(lens.max())], "sets": len(sets), "vocab": vocab.size, "paint": paint, "coords": coords, "style": style,
            "excluded_train": sum(split_of.get(x) == "train" for x in exclude)}
    assert lens.max() <= vocab.max_tok, info
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, tokens=np.concatenate(toks), loss_w=np.concatenate(wts), set_ids=np.concatenate(sids),
                        offsets=np.array(offs, np.int64), patches=np.stack(pats), names=np.array(names), split=np.array(split),
                        sets=np.array(json.dumps([sorted(s) for s in sorted(sets, key=sets.get)])), paint=np.array(paint),
                        coords=np.array(coords), style=np.array(style))
    return info


class Data:
    def __init__(self, path: Path):
        z = np.load(path)
        self.tokens, self.loss_w, self.set_ids, self.offsets = z["tokens"], z["loss_w"], z["set_ids"], z["offsets"]
        self.patches, self.names, self.split = z["patches"], z["names"], z["split"]
        self.sets = [set(s) for s in json.loads(str(z["sets"]))]
        self.vocab = Vocab(str(z["paint"]) if "paint" in z.files else "joint",       # files before E2: joint, xy,
                           str(z["coords"]) if "coords" in z.files else "xy",      # four layers
                           str(z["style"]) if "style" in z.files else "layers")

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def batch(self, idx: np.ndarray):
        """(B, NPATCH, PATCH_DIM) patches and (B, T) tokens, weights, set ids, padded with EOS / 0 / ALL."""
        lens = self.offsets[idx + 1] - self.offsets[idx]
        T = int(lens.max())
        tok = np.full((len(idx), T), SP["EOS"], np.int64)
        w = np.zeros((len(idx), T), np.float32)
        sid = np.zeros((len(idx), T), np.int64)
        for r, (i, n) in enumerate(zip(idx, lens)):
            a = self.offsets[i]
            tok[r, :n] = self.tokens[a:a + n]
            w[r, :n] = self.loss_w[a:a + n]
            sid[r, :n] = self.set_ids[a:a + n]
        return self.patches[idx], tok, w, sid

    def mask_table(self, dev: str):
        import torch
        return torch.from_numpy(np.stack([token_mask(s, self.vocab) for s in self.sets])).to(dev)


# ------------------------------------------------------------------ model
@dataclass
class Config:
    name: str = "E1"
    paint: str = "joint"
    coords: str = "xy"
    style: str = "layers"
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 16
    accum: int = 1
    max_steps: int = 6000
    warmup: int = 100
    eval_every: int = 100
    patience: int = 8
    stop_after_min: float = 0.0
    seed: int = 0
    resume: bool = False


def model_config(cfg: Config) -> GPTConfig:
    from ..llm.model import GPTConfig
    vocab = Vocab(cfg.paint, cfg.coords, cfg.style)
    return GPTConfig(vocab_size=vocab.size, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                     n_kv_heads=cfg.n_kv_heads, max_seq_len=NPATCH + vocab.max_tok, dropout=cfg.dropout, abs_pos=False)


def make_model(cfg: Config):
    import torch
    from torch import nn

    from ..llm.model import GPT

    class EyeGPT(nn.Module):
        """llm.model.GPT with the pixel patches in front: patch -> linear + learned patch position."""

        def __init__(self):
            super().__init__()
            self.gpt = GPT(model_config(cfg))
            self.patch = nn.Linear(PATCH_DIM, cfg.d_model)
            self.patch_pos = nn.Parameter(torch.randn(NPATCH, cfg.d_model) * 0.02)
            nn.init.normal_(self.patch.weight, std=0.02)
            nn.init.zeros_(self.patch.bias)
            self.blocks = self.gpt.blocks
            self.eye_cfg = cfg
            self.vocab = Vocab(cfg.paint, cfg.coords, cfg.style)

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(self, idx, pix=None, cache=None, start_pos: int = 0):
            """idx (B, T) after the patches; pix (B, NPATCH, PATCH_DIM) uint8 when the patches come first in this call.
            Returns the logits of idx's positions only."""
            g = self.gpt
            x = g.embed(idx)
            p = 0
            if pix is not None:
                pe = self.patch(pix.float() / 255 - 0.5) + self.patch_pos
                x = torch.cat([pe.to(x.dtype), x], 1)
                p = NPATCH
            T = x.size(1)
            cos, sin = g.rope_cos[start_pos:start_pos + T], g.rope_sin[start_pos:start_pos + T]
            for i, block in enumerate(g.blocks):
                x = block(x, cos, sin, None if cache is None else cache[i])
            return g.head(g.norm(x[:, p:]))

    return EyeGPT()


def device() -> str:
    import os

    import torch
    if os.environ.get("BRUSHGPT_DEVICE"):
        return os.environ["BRUSHGPT_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def autocast(dev: str):
    """fp16 on CUDA (as brushgpt_l: L8 trained on Kaggle's T4 this way); nothing on the Mac or the CPU."""
    import contextlib

    import torch
    return torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else contextlib.nullcontext()


def losses(model, pix, tok, w, sid, table, dev: str):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(tok).to(dev)
    with autocast(dev):
        logits = model(t[:, :-1], torch.from_numpy(pix).to(dev))
    logits = logits.float()
    logits = logits.masked_fill(~table[torch.from_numpy(sid[:, 1:]).to(dev)], float("-inf"))
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1), reduction="none").view(len(t), -1)
    wt = torch.from_numpy(w[:, 1:]).to(dev)
    return (ce * wt).sum() / wt.sum().clamp(min=1), ce, wt


def evaluate_loss(model, data: Data, rows: np.ndarray, table, dev: str, batch: int = 16) -> dict:
    """Validation loss, overall, per field and per layer (positions with weight 1). The overall value is per token, so
    it is not comparable between paint encodings (split has 8 tokens a stroke, joint 5); nats_per_cat (the summed loss
    of a cat) and stroke (the summed loss of one stroke's fields, the colour as "paint" in both) are."""
    import torch
    model.eval()
    vocab = data.vocab
    report = vocab.stroke_fields + ("end",)
    keys = report + tuple(f"layer_{n}" for n in vocab.layer_names)
    tot, cnt = dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0.0)
    allt = allc = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            pix, tok, w, sid = data.batch(rows[i:i + batch])
            _, ce, wt = losses(model, pix, tok, w, sid, table, dev)
            ce, wt = ce.cpu().numpy(), wt.cpu().numpy()
            allt += float((ce * wt).sum())
            allc += float(wt.sum())
            nxt = tok[:, 1:]
            grp = np.vectorize(lambda x: vocab.group_of(x) if vocab.group_of(x) in vocab.stroke_fields else "end")(nxt)
            lay = np.cumsum(np.isin(nxt, [vocab.sp[f"LAYER{k}"] for k in range(vocab.n_layers)]), 1) - 1
            for g in report:
                m = (grp == g) & (wt > 0)
                tot[g] += float(ce[m].sum())
                cnt[g] += float(m.sum())
            for k, n in enumerate(vocab.layer_names):
                m = (lay == k) & (wt > 0)
                tot[f"layer_{n}"] += float(ce[m].sum())
                cnt[f"layer_{n}"] += float(m.sum())
    model.train()
    res = {"all": allt / max(allc, 1), "nats_per_cat": allt / max(len(rows), 1), **{k: tot[k] / cnt[k] for k in keys if cnt[k]}}
    if vocab.paint == "split" and all(k in res for k in vocab.colour_fields):
        res["paint"] = sum(res[k] for k in vocab.colour_fields)
    if all(k in res for k in vocab.stroke_fields):
        res["stroke"] = sum(res[k] for k in vocab.stroke_fields)
    return res


def train(cfg: Config, data_path: Path) -> Path:
    import torch
    dev = device()
    torch.manual_seed(cfg.seed)
    data = Data(data_path)
    have = (data.vocab.paint, data.vocab.coords, data.vocab.style)
    if have != (cfg.paint, cfg.coords, cfg.style):
        raise ValueError(f"{data_path} is paint/coords/style {have}, the run is {(cfg.paint, cfg.coords, cfg.style)}")
    run = RUN_DIR / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    model = make_model(cfg).to(dev)
    step0, best = 0, float("inf")
    ck = None
    if cfg.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        step0, best = ck["step"], ck["best"]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    if ck is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    table = data.mask_table(dev)
    train_rows, val_rows = data.rows("train"), data.rows("val")
    (run / "config.json").write_text(json.dumps({"config": asdict(cfg), "model": model_config(cfg).to_dict(),
                                                 "parameters": model.num_params(), "train_cats": len(train_rows),
                                                 "val_cats": len(val_rows), "device": dev, "data": str(data_path),
                                                 "patches": [NPATCH, PATCH_DIM], "max_tok": data.vocab.max_tok}, indent=1))
    rng = np.random.default_rng(cfg.seed + step0)
    t_start, bad = time.time(), 0
    with open(run / "metrics.jsonl", "a") as log:
        model.train()
        for step in range(step0, cfg.max_steps):
            lr = cfg.lr * (step + 1) / cfg.warmup if step < cfg.warmup else \
                cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup))))
            for gr in opt.param_groups:
                gr["lr"] = lr
            tl = 0.0
            for _ in range(cfg.accum):
                pix, tok, w, sid = data.batch(rng.choice(train_rows, cfg.batch, replace=False))
                loss, _, _ = losses(model, pix, tok, w, sid, table, dev)
                scaler.scale(loss / cfg.accum).backward()
                tl += float(loss.detach()) / cfg.accum
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps:
                val = evaluate_loss(model, data, val_rows, table, dev)
                row = {"step": step + 1, "train_loss": round(tl, 4), "val": {k: round(v, 4) for k, v in val.items()},
                       "lr": lr, "minutes": round((time.time() - t_start) / 60, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if val["all"] < best:
                    best, bad = val["all"], 0
                    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step + 1, "val": val}, run / "best.pt")
                else:
                    bad += 1
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "best": best}, run / "last.pt")
                if bad >= cfg.patience:
                    print(json.dumps({"stopped": "patience", "step": step + 1}), flush=True)
                    break
            if cfg.stop_after_min and time.time() - t_start > 60 * cfg.stop_after_min:
                print(json.dumps({"stopped": "time", "step": step + 1}), flush=True)
                break
    return run


# ------------------------------------------------------------------ sampling
def load_model(path: Path, dev: str):
    import torch
    ck = torch.load(path, map_location=dev)
    model = make_model(Config(**ck["config"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def sample(model, pix: np.ndarray, plans: list[list[int]], temperature: float = 1.0, seed: int = 0, batch: int = 16,
           max_len: int | None = None) -> list[np.ndarray]:
    """For each cat: its patches (NPATCH, PATCH_DIM) and its plan part (BOS … PLAN_END, from L8's plan, in the model's
    vocabulary) are given; the model writes every token after it, each among those its grammar state allows, until EOS
    (or max_len, default the model's MAX_TOK)."""
    import torch
    vocab = model.vocab
    max_len = max_len or vocab.max_tok
    dev = next(model.parameters()).device.type
    torch.manual_seed(seed)
    masks: dict[frozenset, torch.Tensor] = {}
    mask = lambda s: masks.setdefault(frozenset(s), torch.from_numpy(token_mask(s, vocab)).to(dev))   # noqa: E731
    out = []
    for i in range(0, len(plans), batch):
        b = min(batch, len(plans) - i)
        pl = plans[i:i + b]
        n0 = len(pl[0])
        assert all(len(p) == n0 for p in pl), "plans in a batch must have the same length"
        rows = torch.full((b, max_len), SP["EOS"], dtype=torch.long, device=dev)
        rows[:, :n0] = torch.tensor(pl, device=dev)
        gram = [Grammar(vocab) for _ in range(b)]
        for g, p in zip(gram, pl):
            for tk in p:
                g.step(tk)
        done = np.zeros(b, bool)
        cache = [{"size": NPATCH + max_len} for _ in model.blocks]
        with torch.no_grad():
            logits = model(rows[:, :n0], torch.from_numpy(pix[i:i + b]).to(dev), cache=cache)
            for p in range(n0, max_len):
                m = torch.stack([mask({"EOS"}) if done[r] else mask(g.allowed()) for r, g in enumerate(gram)])
                last = logits[:, -1].float().masked_fill(~m, float("-inf")) / max(temperature, 1e-6)
                nxt = torch.multinomial(last.softmax(-1), 1)[:, 0]
                rows[:, p] = nxt
                for r, tk in enumerate(nxt.tolist()):
                    if not done[r]:
                        gram[r].step(tk)
                        done[r] = tk == SP["EOS"]
                if done.all():
                    break
                logits = model(rows[:, p:p + 1], cache=cache, start_pos=NPATCH + p)
        for r in range(b):
            row = rows[r].cpu().numpy()
            e = np.flatnonzero(row[n0:] == SP["EOS"])
            out.append(row[:n0 + e[0] + 1] if len(e) else row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--eyes", type=Path, default=EYES)
    d.add_argument("--paint", choices=PAINTS, default="joint")
    d.add_argument("--coords", choices=COORDS, default="xy")
    d.add_argument("--style", choices=STYLES, default="layers", help="digital: the eight-layer material (--paint split)")
    d.add_argument("--exclude", type=Path, default=None, help="training cats to leave out (data/processed/brush_l/eye_exclude_*.tsv)")
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    for k, v in asdict(Config()).items():
        t.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()
    if a.cmd == "data":
        print(json.dumps(build(a.out, a.eyes, a.paint, a.coords, read_exclude(a.exclude), a.style)))
    else:
        cfg = Config(**{k: (bool(getattr(a, k)) if isinstance(v, bool) else getattr(a, k)) for k, v in asdict(Config()).items()})
        print(train(cfg, a.data))


if __name__ == "__main__":
    main()
