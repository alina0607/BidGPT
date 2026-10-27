"""Placeholder buyer sprite sheets, drawn by code, in the format docs/art.md specifies.

    python scripts/make_placeholder_art.py          # writes game/assets/characters/

Stand-ins until real art exists: same cell size, anchor, states, frame counts and metadata as the
spec, so swapping in drawn sheets later means replacing PNGs and nothing else. Each look is a
small pixel figure (head, hair, body, arms) whose pose changes per frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "game" / "assets" / "characters"
CELL = 64
ANCHOR = (32, 62)                    # bottom center: where the feet stand

# state: (frames, seconds per frame, loops)  — docs/art.md
STATES = {
    "idle": (4, 0.20, True),
    "talk": (4, 0.12, True),
    "happy": (6, 0.08, False),
    "unhappy": (6, 0.08, False),
    "leave": (8, 0.05, False),
    "win": (6, 0.08, False),
}

LOOKS = {
    "look01": {"skin": (241, 194, 160), "hair": (70, 45, 30), "shirt": (66, 133, 244), "legs": (60, 60, 80)},
    "look02": {"skin": (198, 134, 96), "hair": (25, 20, 20), "shirt": (219, 68, 55), "legs": (90, 70, 50)},
    "look03": {"skin": (255, 219, 172), "hair": (230, 190, 90), "shirt": (15, 157, 88), "legs": (50, 50, 60)},
    "look04": {"skin": (141, 85, 54), "hair": (40, 30, 25), "shirt": (244, 180, 0), "legs": (70, 50, 90)},
    "look05": {"skin": (224, 172, 105), "hair": (110, 60, 30), "shirt": (156, 39, 176), "legs": (40, 40, 40)},
    "look06": {"skin": (255, 205, 148), "hair": (200, 200, 200), "shirt": (96, 125, 139), "legs": (80, 60, 40)},
    "look07": {"skin": (115, 70, 45), "hair": (15, 15, 15), "shirt": (0, 150, 136), "legs": (60, 40, 70)},
    "look08": {"skin": (234, 192, 134), "hair": (170, 70, 40), "shirt": (255, 112, 67), "legs": (45, 55, 75)},
}
INK = (30, 30, 35)


def pose(state: str, i: int, n: int) -> dict:
    """Offsets for frame i of n: body bob/hop, arm raise, mouth, x shift, head tilt."""
    t = i / n
    p = {"dy": 0, "dx": 0, "arms": 0, "mouth": "line", "lean": 0}
    if state == "idle":
        p["dy"] = -round(max(0.0, math.sin(t * 2 * math.pi)))    # breathing; feet never below the anchor
    elif state == "talk":
        p["mouth"] = "open" if i % 2 == 0 else "line"
    elif state == "happy":
        p["dy"] = -round(6 * math.sin(t * math.pi))               # a hop
        p["arms"], p["mouth"] = 1, "smile"
    elif state == "unhappy":
        p["dx"] = (-2, 2, -2, 2, -1, 0)[i]                        # head shake
        p["mouth"], p["lean"] = "frown", 1
    elif state == "leave":
        p["dx"] = round(1.5 * i)                                  # steps right; the game fades it out
        p["dy"] = -(i % 2)
        p["mouth"] = "frown"
    elif state == "win":
        p["dy"] = -round(4 * abs(math.sin(t * 2 * math.pi)))
        p["arms"], p["mouth"] = 2, "smile"
    return p


def draw_figure(d: ImageDraw.ImageDraw, ox: int, look: dict, p: dict) -> None:
    """One figure in the cell starting at x = ox, feet on the anchor."""
    cx, feet = ox + ANCHOR[0] + p["dx"], ANCHOR[1] + p["dy"]
    # legs
    d.rectangle([cx - 7, feet - 12, cx - 2, feet], fill=look["legs"])
    d.rectangle([cx + 2, feet - 12, cx + 7, feet], fill=look["legs"])
    # body
    body_top = feet - 30
    d.rectangle([cx - 10, body_top, cx + 10, feet - 11], fill=look["shirt"], outline=INK)
    # arms: 0 down, 1 up on one side, 2 both up
    left_up, right_up = p["arms"] >= 2, p["arms"] >= 1
    for side, up in ((-1, left_up), (1, right_up)):
        xa, xb = sorted([cx + side * 11, cx + side * 15])
        top, bottom = (body_top - 10, body_top + 4) if up else (body_top + 2, body_top + 18)
        d.rectangle([xa, top, xb, bottom], fill=look["shirt"], outline=INK)
    # head (facing right: eye toward the right)
    hx, hy = cx + p["lean"], body_top - 11
    d.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=look["skin"], outline=INK)
    d.chord([hx - 11, hy - 12, hx + 11, hy + 4], 180, 360, fill=look["hair"])
    d.rectangle([hx + 3, hy - 2, hx + 5, hy + 1], fill=INK)                   # eye
    mouth = {"line": [(hx + 1, hy + 6), (hx + 7, hy + 6)],
             "open": None, "smile": [(hx + 1, hy + 5), (hx + 4, hy + 7), (hx + 8, hy + 5)],
             "frown": [(hx + 1, hy + 7), (hx + 4, hy + 5), (hx + 8, hy + 7)]}[p["mouth"]]
    if mouth is None:
        d.ellipse([hx + 2, hy + 4, hx + 7, hy + 9], fill=INK)
    else:
        d.line(mouth, fill=INK, width=1)


def frame_image(look: dict, state: str, i: int, frames: int) -> Image.Image:
    """One cell. Drawn on its own canvas and checked, so a pose that reaches the edge fails here
    instead of being cut off (or bleeding into the next cell) in the sheet."""
    cell = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    draw_figure(ImageDraw.Draw(cell), 0, look, pose(state, i, frames))
    left, top, right, bottom = cell.getbbox()
    if left < 1 or top < 1 or right > CELL - 1 or bottom > CELL - 1:
        raise ValueError(f"{state} frame {i} reaches the 1 px margin: bbox {(left, top, right, bottom)}")
    return cell


def sheet(look: dict, state: str) -> Image.Image:
    frames = STATES[state][0]
    img = Image.new("RGBA", (CELL * frames, CELL), (0, 0, 0, 0))
    for i in range(frames):
        img.alpha_composite(frame_image(look, state, i, frames), (i * CELL, 0))
    return img


def main() -> None:
    for name, look in LOOKS.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        states = {}
        for state, (frames, frame_time, loop) in STATES.items():
            file = f"{name}_{state}.png"
            sheet(look, state).save(folder / file)
            states[state] = {"file": file, "cols": frames, "rows": 1, "frames": frames,
                             "frame_time": frame_time, "loop": loop}
        meta = {"look": name, "cell": [CELL, CELL], "anchor": list(ANCHOR), "states": states,
                "placeholder": "drawn by scripts/make_placeholder_art.py"}
        (folder / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(LOOKS)} looks × {len(STATES)} states → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
