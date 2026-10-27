"""Record the safe whole-frame fallback. This is NOT a finished turn animation."""
import sys
import os
import subprocess
import record_girl_turn_v2 as capture
from PIL import Image

capture.OUT = capture.ROOT / "docs/quality/girl-frame-playback-safety"
if __name__ == "__main__":
    if "--review-only" not in sys.argv:
        capture.OUT.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "SA_KEYS": "d@240:330,a@400:490", "SA_CAPTURE_DT": str(1/60),
               "SA_EXIT_AFTER": "553", "SA_CAPTURE": ",".join(
                   f"{capture.OUT}/frame-{i:03}.png@{i}" for i in range(232, 551, 2))}
        subprocess.run([str(capture.ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                       cwd=capture.ROOT, env=env, check=True, timeout=240)
    images = [Image.open(capture.OUT / f"frame-{i:03}.png").convert("RGB")
              .crop((580, 280, 1220, 580)) for i in range(232, 551, 2)]
    palette_source = Image.new("RGB", (640, 1200))
    for row, index in enumerate([0, 20, 90, 150]):
        palette_source.paste(images[index], (0, row*300))
    palette = palette_source.quantize(colors=256)
    images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    durations = [30, 30, 40] * (len(images)//3) + [30] * (len(images)%3)
    images[0].save(capture.OUT / "whole-frame-safety.gif", save_all=True, append_images=images[1:],
                   duration=durations, disposal=1, loop=0, optimize=True)
