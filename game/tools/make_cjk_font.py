"""Chinese glyph atlas for the game's Chinese text: game/assets/fonts/cjk.png and cjk.txt.

Only the characters the game's own strings use (every non-ASCII character inside a string literal in game/src), so
the atlas stays small. The glyphs are Source Han Sans (Adobe, SIL Open Font License 1.1, see cjk_OFL.txt), drawn at
the same pixel sizes and written in the same metrics format as the Nunito atlas (scripts/make_stage_art.py, font()),
which the engine loads as its fallback font (sa::LoadFallbackFont).

Run it again after adding Chinese text to the game:

    python3 game/tools/make_cjk_font.py --font /path/to/SourceHanSans*.otf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "src"
FONTS = ROOT / "assets" / "fonts"
FONT_SIZES = (14, 18, 22, 28, 36, 48)       # the Nunito atlas's sizes, so both fonts pick the same one
EM = 0.92                                   # Chinese drawn a little under the pixel size, to sit with Nunito's capitals
LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def characters() -> list[str]:
    """Every non-ASCII character inside a string literal of game/src, sorted by code point."""
    found: set[str] = set()
    for path in sorted(SOURCES.glob("*.[ch]*")):
        for literal in LITERAL.findall(path.read_text(encoding="utf-8")):
            found |= {c for c in literal if ord(c) > 126}
    return sorted(found)


def build(font_path: Path) -> None:
    chars = characters()
    lines, placed, x, y, row = [], [], 0, 0, 0
    atlas_w = 1024
    for px in FONT_SIZES:
        f = ImageFont.truetype(str(font_path), round(px * EM))
        ascent, descent = f.getmetrics()
        lines.append(f"size {px} {ascent} {ascent + descent}")
        for ch in chars:
            gl, gt, gr, gb = f.getbbox(ch, anchor="ls")
            w, h = max(gr - gl, 0) + 2, max(gb - gt, 0) + 2
            if x + w + 2 > atlas_w:
                x, y, row = 0, y + row + 2, 0
            placed.append((f, ch, x + 1, y + 1, gl, gt))
            lines.append(f"{ord(ch)} {x} {y} {w} {h} {gl - 1} {gt - 1} {f.getlength(ch):.2f}")
            x, row = x + w + 2, max(row, h)
        x, y, row = 0, y + row + 2, 0
    atlas = Image.new("L", (atlas_w, y))
    d = ImageDraw.Draw(atlas)
    for f, ch, gx, gy, gl, gt in placed:
        d.text((gx - gl, gy - gt), ch, font=f, fill=255, anchor="ls")
    out = Image.new("RGBA", atlas.size, (255, 255, 255, 0))
    out.putalpha(atlas)
    out.save(FONTS / "cjk.png", optimize=True)
    (FONTS / "cjk.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(chars)} characters, atlas {atlas.size[0]}x{atlas.size[1]}: {''.join(chars)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", type=Path, required=True, help="a Source Han Sans .otf (any region)")
    build(parser.parse_args().font)


if __name__ == "__main__":
    main()
