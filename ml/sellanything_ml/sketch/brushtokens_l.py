"""Tokens for Brush GPT L (owner, 2026-10-19): the whole-face model on the digital repaint, variable length, with a
sketch plan first. The tokens of runs A-K (brushtokens.py) are left as they are.

A painting is

    BOS  PLAN  plan…  PLAN_END  LAYER0 stroke… LAYER_END  …  LAYER4 stroke… LAYER_END  [LAYER5 … LAYER7]  EOS

L5 (run L5) has the first five layers; L8 adds the three detail layers (fur, dark hairs, light hairs), whose strokes
come region by region in a fixed order (ears, forehead, left cheek, right cheek, chin and mouth; owner 2026-10-19),
each region closed by REGION_END. One vocabulary and one grammar serve both (`n_layers`), and positions are RoPE only
(nothing learned per position), so L8 starts from L5's weights unchanged: the tokens L8 adds (LAYER5-7, REGION_END)
are already in L5's vocabulary, after all the others, so sequences prepared before they were added keep their ids.

plan (tools/sketch_plan.py): POSE and three angle tokens (yaw, pitch, roll); then outlines, each a kind (HEAD, EYE,
NOSE, MOUTH) followed by points (an x and a y token each) and OUT_END; a pupil is PUPIL x y radius radius angle.

The five layers are the digital repaint's first five (tools/repaint.py, digital_face.LAYERS): washes, blocks, blend,
small blocks, blend. A layer token carries the layer's brush (size, edge, opacity range, paint or blend), so a stroke
never spells it out. The model ends each layer itself (LAYER_END): the number of strokes is not fixed.

A paint stroke is five tokens, as before: X Y ANGLE SIZE PAINT (colour LEVELS_RGB steps a channel, opacity
LEVELS_ALPHA steps over the layer's range). A blending stroke carries no paint (it smudges what is under it), so its
fifth token is STRENGTH (its opacity, LEVELS_STRENGTH steps).

The grammar fixes, from the tokens so far, which tokens may come next (`allowed_groups`): the model still chooses
every value, including when a layer or an outline ends; what is fixed is the order of the fields (the data's layout,
as in run C, owner 2026-10-11). Training and sampling use the same rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from . import catbrush as B

NXY, NANGLE, LEVELS_SIZE, LEVELS_RGB, LEVELS_ALPHA, LEVELS_STRENGTH = 256, 32, 6, 12, 8, 8
# Opacity has 8 levels (runs A-K: 4). Binning the digital repaint's first five layers (10 cats, 2026-10-19) raised the
# L1 to the photo from 0.0511 to 0.0569 with 12 x 4 levels, 0.0555 with 12 x 8; binning each layer before the next is
# refitted on top (tools/prepare_l.py) brings 12 x 8 to +0.0022 over the unbinned painting (6 cats; saved in
# runs/brush_l_tokenize/refit.json by tools/tokenize_l_checks.py). Owner's choice.
NPLAN_ANGLE, NRADIUS = 64, 64
LAYERS = 5                     # the first five (L5); MAX_LAYERS for L8
MAX_LAYERS = 8
DETAIL = (5, 6, 7)             # layers whose strokes come region by region
REGIONS = 5                    # repaint_tidy.REGION_ORDER: ears, forehead, left cheek, right cheek, chin and mouth
TOKENS_PER_STROKE = 5
BRUSHES = [  # (length, width, edge px, blend): digital_face.LAYERS; every layer: bristle 0, opacity 0.2-1.0
    (96, 40, 6.0, False), (48, 20, 3.0, False), (30, 10, 0.5, True), (24, 8, 1.5, False), (16, 5, 0.5, True),
    (12, 2, 0.7, False), (10, 1, 0.5, False), (12, 1, 0.5, False)]
ALPHA_RANGE = (0.2, 1.0)

SPECIALS = ["BOS", "EOS", "PLAN", "PLAN_END", "POSE", "HEAD", "EYE", "PUPIL", "NOSE", "MOUTH", "OUT_END"] + \
    [f"LAYER{k}" for k in range(LAYERS)] + ["LAYER_END"]
SP = {name: i for i, name in enumerate(SPECIALS)}
FIELDS = {"x": NXY, "y": NXY, "angle": NANGLE, "size": LEVELS_SIZE ** 2, "paint": LEVELS_RGB ** 3 * LEVELS_ALPHA,
          "strength": LEVELS_STRENGTH, "plan_angle": NPLAN_ANGLE, "radius": NRADIUS}


EXTRA = [f"LAYER{k}" for k in range(LAYERS, MAX_LAYERS)] + ["REGION_END"]     # L8's tokens, after the fields


def _offsets() -> dict[str, int]:
    out, start = {}, len(SPECIALS)
    for name, n in FIELDS.items():
        out[name] = start
        start += n
    out["extra"] = start
    out["vocab"] = start + len(EXTRA)
    return out


OFFSET = _offsets()
VOCAB = OFFSET["vocab"]
SP.update({name: OFFSET["extra"] + i for i, name in enumerate(EXTRA)})
NAME = {i: name for name, i in SP.items()}
PLAN_TOKEN = {"[plan]": "PLAN", "[/plan]": "PLAN_END", "[pose]": "POSE", "[head]": "HEAD", "[eye]": "EYE",
              "[pupil]": "PUPIL", "[nose]": "NOSE", "[mouth]": "MOUTH", "[end]": "OUT_END"}
PLAN_FIELD = {"x": "x", "y": "y", "a": "plan_angle", "r": "radius"}


def _bin(x: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    return np.clip(((np.asarray(x, float) - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _centre(i: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    return lo + (np.asarray(i) + 0.5) * (hi - lo) / n


# ------------------------------------------------------------------ strokes
def stroke_tokens(layer: dict, k: int) -> np.ndarray:
    """(n, 5) token ids of layer k's strokes from their pixel values (xy_px, lw_px, angle_rad, colour_rgb, alpha_val)."""
    length, width, _edge, blend = BRUSHES[k]
    scale = layer["lw_px"] / np.array([length, width], float)
    five = (OFFSET["strength"] + _bin(layer["alpha_val"], *ALPHA_RANGE, LEVELS_STRENGTH)) if blend else \
        (OFFSET["paint"] + ((lambda rgb: (rgb[:, 0] * LEVELS_RGB + rgb[:, 1]) * LEVELS_RGB + rgb[:, 2])(
            _bin(layer["colour_rgb"], 0, 1, LEVELS_RGB)) * LEVELS_ALPHA + _bin(layer["alpha_val"], *ALPHA_RANGE, LEVELS_ALPHA)))
    return np.stack([OFFSET["x"] + _bin(layer["xy_px"][:, 0], 0, B.SIDE, NXY),
                     OFFSET["y"] + _bin(layer["xy_px"][:, 1], 0, B.SIDE, NXY),
                     OFFSET["angle"] + _bin(np.mod(layer["angle_rad"], math.pi), 0, math.pi, NANGLE),
                     OFFSET["size"] + _bin(scale[:, 0], *B.SIZE_RANGE, LEVELS_SIZE) * LEVELS_SIZE
                     + _bin(scale[:, 1], *B.SIZE_RANGE, LEVELS_SIZE), five], 1)


def stroke_values(rows: np.ndarray, k: int) -> dict:
    """Layer k's stroke values at the centres of the bins in rows (n, 5)."""
    length, width, _edge, blend = BRUSHES[k]
    size = rows[:, 3] - OFFSET["size"]
    out = {"xy": np.stack([_centre(rows[:, 0] - OFFSET["x"], 0, B.SIDE, NXY), _centre(rows[:, 1] - OFFSET["y"], 0, B.SIDE, NXY)], 1),
           "angle": _centre(rows[:, 2] - OFFSET["angle"], 0, math.pi, NANGLE),
           "length_width": np.stack([_centre(size // LEVELS_SIZE, *B.SIZE_RANGE, LEVELS_SIZE),
                                     _centre(size % LEVELS_SIZE, *B.SIZE_RANGE, LEVELS_SIZE)], 1) * np.array([length, width], float)}
    if blend:
        out["colour"] = np.full((len(rows), 3), 0.5)
        out["alpha"] = _centre(rows[:, 4] - OFFSET["strength"], *ALPHA_RANGE, LEVELS_STRENGTH)
    else:
        paint = rows[:, 4] - OFFSET["paint"]
        rgb_i = paint // LEVELS_ALPHA
        out["colour"] = _centre(np.stack([rgb_i // LEVELS_RGB ** 2, rgb_i // LEVELS_RGB % LEVELS_RGB, rgb_i % LEVELS_RGB], 1), 0, 1, LEVELS_RGB)
        out["alpha"] = _centre(paint % LEVELS_ALPHA, *ALPHA_RANGE, LEVELS_ALPHA)
    return out


def brush(k: int) -> dict:
    length, width, edge, blend = BRUSHES[k]
    return {"length": length, "width": width, "edge_px": edge, "bristle": 0.0, "alpha_range": ALPHA_RANGE, "blend": blend}


def render(layers: list[np.ndarray], dev: str = "cpu") -> np.ndarray:
    """Paint layers of stroke rows (n_k, 5) on white paper, as tools/repaint.replay does: (SIDE, SIDE, 3) in 0-1."""
    import eye_crop_styles as E                       # tools/: the digital painter's brushes and blending strokes
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1).to(dev)
    canvas = torch.ones(B.SIDE, B.SIDE, 3, device=dev)
    with torch.no_grad():
        for k, rows in enumerate(layers):
            if len(rows) == 0:
                continue
            br = brush(k)
            with E.brush_settings(br):
                s = B.Strokes.from_stage(stroke_values(rows, k), (br["length"], br["width"])).to(dev)
                canvas = E.blend_composite(canvas, s) if br["blend"] else B.composite(canvas, s, grid)
    return canvas.cpu().numpy()


# ------------------------------------------------------------------ plan
def plan_ids(words: list[str]) -> list[int]:
    """sketch_plan's plan words ("[plan]", "x12", "a40", …) as token ids."""
    out = []
    for w in words:
        if w in PLAN_TOKEN:
            out.append(SP[PLAN_TOKEN[w]])
        else:
            out.append(OFFSET[PLAN_FIELD[w[0]]] + min(int(w[1:]), FIELDS[PLAN_FIELD[w[0]]] - 1))
    return out


def decode_plan(tokens: np.ndarray) -> dict:
    """The plan of a sequence as sketch_plan's parts (pixels): pose (yaw, pitch, roll in degrees, bin centres), head
    runs, eye outlines, pupils (centre, radii, angle in radians) and the nose outline and mouth line (None if absent)."""
    t = [int(x) for x in np.asarray(tokens)]
    i = t.index(SP["PLAN"]) + 1
    ang = lambda v: (v - OFFSET["plan_angle"] + 0.5) * 180 / NPLAN_ANGLE - 90          # noqa: E731
    out: dict[str, Any] = {"pose": None, "head": [], "eyes": [], "pupils": [], "nose": None, "mouth": None}
    while t[i] != SP["PLAN_END"]:
        k = NAME.get(t[i])
        if k == "POSE":
            out["pose"] = [ang(v) for v in t[i + 1:i + 4]]
            i += 4
        elif k == "PUPIL":
            v = t[i + 1:i + 6]
            out["pupils"].append({"centre": np.array([v[0] - OFFSET["x"], v[1] - OFFSET["y"]], float),
                                  "radii": np.array([v[2] - OFFSET["radius"], v[3] - OFFSET["radius"]], float),
                                  "angle": math.radians(ang(v[4]) + 90)})
            i += 6
        else:
            j = t.index(SP["OUT_END"], i)
            xy = np.array(t[i + 1:j], float).reshape(-1, 2) - [OFFSET["x"], OFFSET["y"]]
            if k == "HEAD":
                out["head"].append(xy)
            elif k == "EYE":
                out["eyes"].append(xy)
            else:
                out[cast(str, k).lower()] = xy
            i = j + 1
    return out


# ------------------------------------------------------------------ whole sequences
def encode(plan_words: list[str], layers: list[dict], n_layers: int = LAYERS,
           keep: list[np.ndarray] | None = None, regions: dict[int, list[np.ndarray]] | None = None) -> np.ndarray:
    """A painting: BOS, the plan, each of the first n_layers layers between its LAYER token and LAYER_END, EOS.
    keep[k]: indices of layer k's strokes to include, in order (default all). regions[k] (detail layers, required
    when n_layers > LAYERS): REGIONS index arrays, in region order; each region is closed by REGION_END."""
    out = [SP["BOS"]] + plan_ids(plan_words)
    for k in range(n_layers):
        rows = stroke_tokens(layers[k], k)
        out.append(SP[f"LAYER{k}"])
        if k in DETAIL:
            for idx in cast(dict[int, list[np.ndarray]], regions)[k]:
                out += rows[idx].reshape(-1).tolist() + [SP["REGION_END"]]
        else:
            out += (rows if keep is None else rows[keep[k]]).reshape(-1).tolist()
        out.append(SP["LAYER_END"])
    return np.array(out + [SP["EOS"]], np.int32)


def decode_layers(tokens: np.ndarray) -> list[np.ndarray]:
    """The stroke rows (n_k, 5) of each layer present in a sequence that follows the grammar (region ends dropped)."""
    t = np.asarray(tokens)
    layers = []
    for k in range(MAX_LAYERS):
        s = np.flatnonzero(t == SP[f"LAYER{k}"])
        if not len(s):
            if k < LAYERS:
                layers.append(np.zeros((0, 5), int))
                continue
            break
        e = s[0] + 1 + np.flatnonzero(t[s[0] + 1:] == SP["LAYER_END"])
        body = t[s[0] + 1:e[0] if len(e) else len(t)]
        body = body[body != SP["REGION_END"]]
        body = body[:len(body) - len(body) % TOKENS_PER_STROKE]
        layers.append(body.reshape(-1, TOKENS_PER_STROKE))
    return layers


# ------------------------------------------------------------------ grammar
KINDS = ("HEAD", "EYE", "NOSE", "MOUTH")


class Grammar:
    """The grammar as a state machine: `allowed()` is the set of groups the next token may come from, `step(tok)`
    takes a token (ValueError if it is not allowed). Groups are token names, "KIND" (HEAD, EYE, NOSE, MOUTH) or field
    names. Used for training masks (`allowed_groups`) and, token by token, for sampling."""

    def __init__(self, n_layers: int = LAYERS):
        self.n_layers = n_layers
        self.state, self.n, self.layer, self.region = "start", 0, -1, 0

    def allowed(self) -> set[str]:
        st, n = self.state, self.n
        if st == "start":
            return {"BOS"}
        if st == "after_bos":
            return {"PLAN"}
        if st == "after_plan":
            return {"POSE"}
        if st == "pose":
            return {"plan_angle"}
        if st == "outlines":
            return {"KIND", "PUPIL", "PLAN_END"}
        if st == "outline_x":
            return {"x", "OUT_END"}
        if st == "outline_y":
            return {"y"}
        if st == "pupil":
            return [{"x"}, {"y"}, {"radius"}, {"radius"}, {"plan_angle"}][n]
        if st == "layer_start":
            return {f"LAYER{self.layer + 1}"} if self.layer + 1 < self.n_layers else {"EOS"}
        if st == "stroke":
            if n:
                return cast(set[str], [None, {"y"}, {"angle"}, {"size"}, {"strength" if BRUSHES[self.layer][3] else "paint"}][n])
            if self.layer in DETAIL:
                return {"x", "REGION_END"} if self.region < REGIONS else {"LAYER_END"}
            return {"x", "LAYER_END"}
        return set()

    def step(self, tok: int) -> None:
        g = group_of(tok)
        if g not in self.allowed():
            raise ValueError(f"token {tok} ({g}) not in {sorted(self.allowed())} (state {self.state})")
        st = self.state
        if g == "BOS":
            self.state = "after_bos"
        elif g == "PLAN":
            self.state = "after_plan"
        elif g == "POSE":
            self.state, self.n = "pose", 0
        elif st == "pose":
            self.n += 1
            self.state = "outlines" if self.n == 3 else "pose"
        elif g == "KIND":
            self.state = "outline_x"
        elif g == "PUPIL":
            self.state, self.n = "pupil", 0
        elif st == "pupil":
            self.n += 1
            if self.n == 5:
                self.state = "outlines"
        elif g == "OUT_END":
            self.state = "outlines"
        elif st == "outline_x":
            self.state = "outline_y"
        elif st == "outline_y":
            self.state = "outline_x"
        elif g in ("PLAN_END", "LAYER_END"):
            self.state = "layer_start"
        elif g.startswith("LAYER"):
            self.layer, self.state, self.n, self.region = self.layer + 1, "stroke", 0, 0
        elif g == "REGION_END":
            self.region += 1
        elif st == "stroke":
            self.n = (self.n + 1) % TOKENS_PER_STROKE
        elif g == "EOS":
            self.state = "done"


def allowed_groups(tokens: np.ndarray, n_layers: int = LAYERS) -> list[set[str]]:
    """For each position i of a sequence, the groups the token at i may come from given tokens[:i] (index 0: BOS).
    Raises ValueError where a token is not allowed (so encode's output is checked by the same rule)."""
    g, out = Grammar(n_layers), []
    for i, tok in enumerate(np.asarray(tokens)):
        out.append(g.allowed())
        try:
            g.step(tok)
        except ValueError as e:
            raise ValueError(f"position {i}: {e}") from None
    return out


def group_of(tok: int) -> str:
    tok = int(tok)
    if tok in NAME:
        name = NAME[tok]
        return "KIND" if name in KINDS else name
    for g, n in FIELDS.items():
        if OFFSET[g] <= tok < OFFSET[g] + n:
            return g
    raise ValueError(f"token {tok} out of the vocabulary")


def token_mask(groups: set[str]) -> np.ndarray:
    """Boolean (VOCAB,) of the token ids in the given groups."""
    m = np.zeros(VOCAB, bool)
    for g in groups:
        if g == "KIND":
            for kind in KINDS:
                m[SP[kind]] = True
        elif g in FIELDS:
            m[OFFSET[g]:OFFSET[g] + FIELDS[g]] = True
        else:
            m[SP[g]] = True
    return m


def load_cat(repaint_npz: Path, plan_npz: Path) -> tuple[list[str], list[dict], dict]:
    """A repainted cat (all eight layers) (tools/repaint.py) and its plan (tools/sketch_plan.py --batch): plan words, layers, extras."""
    z = np.load(repaint_npz, allow_pickle=True)
    layers = []
    for k in range(MAX_LAYERS):
        layers.append({q: z[f"l{k}_{q}"] for q in ("xy_px", "lw_px", "angle_rad", "colour_rgb", "alpha_val")})
        layers[-1]["brush"] = json.loads(str(z[f"l{k}_brush"]))
    p = np.load(plan_npz, allow_pickle=True)
    return [str(w) for w in p["tokens"]], layers, {"target": z["target"], "tri": z["tri"], "points": z["points"]}
