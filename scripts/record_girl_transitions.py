"""Record actual OpenGL gameplay for turn and action-to-idle review (not an atlas mockup)."""
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality/girl-transitions-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = range(10, 441, 2)
    env = {**os.environ,
           "SA_KEYS": "d@20:65,a@100:145,space@185:186,shift@245:246,j@295:296,k@340:341,w@380:408",
           "SA_EXIT_AFTER": "445",
           "SA_CAPTURE": ",".join(f"{OUT}/frame-{i:03}.png@{i}" for i in frames)}
    subprocess.run([str(ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                   cwd=ROOT, env=env, check=True, timeout=240)
    groups = {"turns": (10, 180), "actions": (180, 440)}
    for name, (start, end) in groups.items():
        selected = [i for i in frames if start <= i <= end]
        images = [Image.open(OUT / f"frame-{i:03}.png").convert("RGB").resize((960, 540)) for i in selected]
        images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:],
                       duration=80, disposal=2, loop=0)
    # Keep full gameplay PNGs as the source of these contact sheets.
    for name, selected in {
        "turn-review": list(range(18, 44, 4)) + list(range(64, 94, 4)),
        "action-review": list(range(212, 234, 4)) + list(range(260, 282, 4)) +
                         list(range(306, 328, 4)) + list(range(348, 370, 4)),
    }.items():
        sheet = Image.new("RGB", (960, ((len(selected) + 3) // 4) * 155), "#fff4eb")
        draw = ImageDraw.Draw(sheet)
        for n, frame in enumerate(selected):
            tile = Image.open(OUT / f"frame-{frame:03}.png").convert("RGB").resize((240, 135))
            x, y = n % 4 * 240, n // 4 * 155
            sheet.paste(tile, (x, y))
            draw.text((x + 6, y + 137), f"game frame {frame}", fill="#442d2b")
        sheet.save(OUT / f"{name}.jpg")
    print(f"Recorded {len(frames)} actual gameplay screenshots to {OUT}")


if __name__ == "__main__":
    main()
