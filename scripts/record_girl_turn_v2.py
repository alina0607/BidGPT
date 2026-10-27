"""Real game renderer/input, deterministic 60 Hz for recording unaffected by PNG encoding latency."""
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality/girl-turn-v2"


def review():
    for name, start, end, crop in [
        ("turn-right", 232, 384, None),
        ("turn-left-and-cast", 396, 660, None),
    ]:
        images = []
        for frame in range(start, end + 1, 2):
            im = Image.open(OUT / f"frame-{frame:03}.png").convert("RGB")
            images.append(im.crop(crop).resize((600, 540)) if crop else im.resize((960, 540)))
        images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:],
                       duration=[30, 30, 40] * (len(images) // 3) + [30] * (len(images) % 3),
                       disposal=2, loop=0)
    for name, frames, x in [
        ("start-contact", range(238, 286, 4), 835),
        ("return-contact", range(328, 376, 4), 835),
        ("cast-return-contact", range(592, 640, 4), 590),
    ]:
        sheet = Image.new("RGB", (1200, 870), "#fff4eb")
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(frames):
            im = Image.open(OUT / f"frame-{frame:03}.png").convert("RGB").crop((x, 300, x + 300, 570))
            px, py = i % 4 * 300, i // 4 * 290
            sheet.paste(im, (px, py))
            draw.text((px + 5, py + 273), f"game frame {frame}", fill="#442d2b")
        sheet.save(OUT / f"{name}.jpg")


def record_idle():
    frames = range(4, 437, 4)
    env = {**os.environ, "SA_KEYS": "", "SA_CAPTURE_DT": str(1 / 60), "SA_EXIT_AFTER": "440",
           "SA_CAPTURE": ",".join(f"{OUT}/idle-{i:03}.png@{i}" for i in frames)}
    subprocess.run([str(ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                   cwd=ROOT, env=env, check=True, timeout=180)
    images = [Image.open(OUT / f"idle-{i:03}.png").convert("RGB").crop((830, 300, 1130, 570)).resize((600, 540))
              for i in list(frames)[:-1]]  # last frame duplicates the first after one complete 7.2 s period
    images[0].save(OUT / "idle-breath.gif", save_all=True, append_images=images[1:],
                   duration=[60, 70, 70] * 36, disposal=2, loop=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--idle-only" in sys.argv:
        record_idle()
        return
    first = 396 if "--left-only" in sys.argv else 232 if "--turns-only" in sys.argv else 10
    frames = range(first, 661, 2)
    env = {**os.environ, "SA_KEYS": "d@240:330,a@400:490,k@560:561",
           "SA_CAPTURE_DT": str(1 / 60), "SA_EXIT_AFTER": "665",
           "SA_CAPTURE": ",".join(f"{OUT}/frame-{i:03}.png@{i}" for i in frames)}
    subprocess.run([str(ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                   cwd=ROOT, env=env, check=True, timeout=300)
    review()
    if len(sys.argv) == 1:
        record_idle()
    print(f"Recorded {len(frames)} game screenshots at a deterministic 60 Hz; review GIFs use 30 fps.")


if __name__ == "__main__":
    main()
