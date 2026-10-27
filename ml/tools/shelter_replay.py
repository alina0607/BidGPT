"""Paint a stretch of a shelter cat again from its record (owner, 2026-10-27: pause the game, say from which second to
which second it looks wrong, and look at exactly those strokes). sketch/shelter.py records every stroke of every run
in ml/logs/shelter/<date-time>_run<N>/cat<k>.jsonl with the cat's clock (the seconds the game shows, pauses left
out); this replays them with the same painters, so the canvas at any second is what the game showed then.

Writes, next to the record:
  replay_cat<k>_<from>-<to>.png  the canvas at <from>, at <to>, and at <to> with the window's strokes ringed (and the
                                 eye canvas the same way when the window has eye strokes)
  replay_cat<k>_<from>-<to>.gif  the window, a frame every --step seconds
and prints the window's strokes by layer, with how far each layer's colours are from the paint under them.

    PYTHONPATH=.:tools python tools/shelter_replay.py --cat 1 --from 40 --to 50            # the latest run
    PYTHONPATH=.:tools python tools/shelter_replay.py --record ml/logs/shelter/<run> --cat 3 --from 0 --to 999
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sellanything_ml.sketch import brushtokens_l as L
from sellanything_ml.sketch import eyegpt_l8 as EG
from sellanything_ml.sketch import shelter as S

E, ELL, ER = S._tools()
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def latest() -> Path:
    runs = sorted(p for p in S.RECORD_ROOT.glob("*_run*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no shelter record in {S.RECORD_ROOT}")
    return runs[-1]


class Replay:
    """The cat's canvases stroke by stroke: the face, and from the eyes line on the eye canvas too."""

    def __init__(self, lines: list[dict]) -> None:
        ys, xs = torch.meshgrid(torch.arange(256) + 0.5, torch.arange(256) + 0.5, indexing="ij")
        self.grid = torch.stack([xs, ys], -1)
        self.lines, self.i = lines, 0
        self.face = S.Layers(torch.ones(256, 256, 3), lambda c, j, rows: S.face_strokes(E, c, j, rows, self.grid),
                             lambda j: L.brush(j)["blend"])
        self.rows: dict[int, list] = {}
        self.eyes = self.zoom = None
        self.canvas = torch.ones(256, 256, 3).numpy()
        self.crop = None

    def until(self, t: float) -> list[dict]:
        """Lays every stroke up to the cat's second t; returns the strokes laid."""
        laid = []
        while self.i < len(self.lines) and self.lines[self.i].get("t", 0.0) <= t:
            line = self.lines[self.i]
            self.i += 1
            if "eyes" in line:
                self._start_eyes(line["eyes"])
            elif line.get("phase") == "face":
                row = np.array([line["row"]])
                self.rows.setdefault(line["layer"], []).append(row)
                self.canvas = self.face.add(line["layer"], row).numpy()
                laid.append(line)
            elif line.get("phase") == "eyes":
                bins = np.array([line["bins"]])
                self.canvas = self.eyes.add(line["layer"], bins)
                self.crop = self.zoom.add(line["layer"], bins)
                laid.append(line)
        return laid

    def _start_eyes(self, info: dict) -> None:
        brushes = json.loads((EG.RUN_DIR / info["run"] / "brushes.json").read_text())
        self.box = tuple(info["box"])
        tokens = [L.SP["BOS"]]                 # L8's strokes as a sequence: what the eye canvas is painted from
        for k in sorted(self.rows):
            tokens += [L.SP[f"LAYER{k}"]] + np.concatenate(self.rows[k]).reshape(-1).tolist() + [L.SP["LAYER_END"]]
        base = ELL.l8_on_crop(np.array(tokens + [L.SP["EOS"]]), EG.FACE_BOX, self.box)
        blend = lambda j: bool(brushes[j].get("blend"))      # noqa: E731
        self.eyes = S.Layers(self.canvas, lambda c, j, b: ER.layers_on_face(c, [np.concatenate(b)], [brushes[j]], self.box, EG.FACE_BOX), blend)
        self.zoom = S.Layers(base, lambda c, j, b: ER.layers_on_crop(c, [np.concatenate(b)], [brushes[j]]), blend)
        self.crop = base


def u8(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def ringed(a: np.ndarray, pts: list[tuple[float, float]], side: int) -> Image.Image:
    im = u8(a).resize((side, side), Image.LANCZOS)
    d, k = ImageDraw.Draw(im), side / 256
    for x, y in pts:
        d.ellipse((x * k - 5, y * k - 5, x * k + 5, y * k + 5), outline=(255, 40, 120), width=2)
    return im


def centre(line: dict) -> tuple[float, float]:
    if line["phase"] == "face":
        return tuple(L.stroke_values(np.array([line["row"]]), line["layer"])["xy"][0])
    return line["bins"][0] + 0.5, line["bins"][1] + 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", type=Path, default=None, help="a run's folder in ml/logs/shelter (default: the latest)")
    ap.add_argument("--cat", type=int, required=True, help="1-8, as the game counts")
    ap.add_argument("--from", dest="t0", type=float, required=True, help="the cat's second, as the game shows it")
    ap.add_argument("--to", dest="t1", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.5, help="GIF: seconds between frames")
    ap.add_argument("--side", type=int, default=384)
    a = ap.parse_args()
    rec = a.record or latest()
    lines = [json.loads(x) for x in open(rec / f"cat{a.cat}.jsonl")]
    r = Replay(lines)
    r.until(a.t0)
    start = r.canvas.copy()
    frames, window = [u8(start).resize((a.side, a.side), Image.LANCZOS)], []
    t = a.t0
    while t < a.t1:
        t = min(t + a.step, a.t1)
        before = r.canvas.copy()
        laid = r.until(t)
        for line in laid:                       # how far the colour is from the paint under it, for face strokes
            if line["phase"] == "face" and not L.brush(line["layer"])["blend"]:     # blending strokes carry no colour
                v = L.stroke_values(np.array([line["row"]]), line["layer"])
                x, y = np.clip(v["xy"][0].astype(int), 1, 254)
                line["off"] = float(np.abs(v["colour"][0] - before[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(0)).mean())
        window += laid
        frames.append(ringed(r.crop if laid and laid[-1]["phase"] == "eyes" else r.canvas,
                             [centre(x) for x in laid if (x["phase"] == "eyes") == (bool(laid) and laid[-1]["phase"] == "eyes")],
                             a.side))
    tag = f"cat{a.cat}_{a.t0:g}-{a.t1:g}"
    face_pts = [centre(x) for x in window if x["phase"] == "face"]
    eye_pts = [centre(x) for x in window if x["phase"] == "eyes"]
    tiles = [("開始 " + f"{a.t0:g} s", u8(start).resize((a.side, a.side), Image.LANCZOS)),
             ("結束 " + f"{a.t1:g} s", u8(r.canvas).resize((a.side, a.side), Image.LANCZOS)),
             (f"這段畫的 {len(face_pts)} 筆（圈起來）", ringed(r.canvas, face_pts, a.side))]
    if eye_pts:
        tiles.append((f"眼睛畫布：這段的 {len(eye_pts)} 筆", ringed(r.crop, eye_pts, a.side)))
    font = ImageFont.truetype(FONT, 18)
    out = Image.new("RGB", (len(tiles) * (a.side + 8) + 8, a.side + 40), "white")
    d = ImageDraw.Draw(out)
    for i, (cap, im) in enumerate(tiles):
        d.text((8 + i * (a.side + 8), 8), cap, fill=(0, 0, 0), font=font)
        out.paste(im, (8 + i * (a.side + 8), 34))
    out.save(rec / f"replay_{tag}.png")
    frames[0].save(rec / f"replay_{tag}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    by: dict = {}
    for x in window:
        name = S.FACE_LAYERS[x["layer"]] if x["phase"] == "face" else f"eyes layer {x['layer'] + 1}"
        g = by.setdefault(name, {"strokes": 0, "off": []})
        g["strokes"] += 1
        if "off" in x:
            g["off"].append(x["off"])
    print(json.dumps({"record": str(rec), "cat": a.cat, "from": a.t0, "to": a.t1, "strokes": len(window),
                      "by_layer": {k: {"strokes": v["strokes"], "colour_off_paint_under": round(float(np.mean(v["off"])), 3) if v["off"] else None}
                                   for k, v in by.items()},
                      "png": str(rec / f"replay_{tag}.png"), "gif": str(rec / f"replay_{tag}.gif")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
