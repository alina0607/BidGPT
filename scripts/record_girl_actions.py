"""Capture the real OpenGL game's action keys in both directions (no ML server needed)."""
import os
from pathlib import Path
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality/girl-actions-v1/gameplay"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = range(10, 341, 2)
    keys = ("d@15:25,j@40:41,shift@80:81,space@120:121,"
            "a@180:190,j@200:201,shift@240:241,space@280:281")
    env = {**os.environ, "SA_KEYS": keys, "SA_EXIT_AFTER": "345",
           "SA_CAPTURE": ",".join(f"{OUT}/frame-{i:03}.png@{i}" for i in frames)}
    subprocess.run([str(ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                   cwd=ROOT, env=env, check=True, timeout=240)
    # A review GIF sampled from gameplay, not a frame-time benchmark: PNG capture adds overhead.
    for name, first, last in [("right", 10, 178), ("left", 180, 340)]:
        images = [Image.open(OUT / f"frame-{i:03}.png").convert("RGB").resize((960, 540))
                  for i in frames if first <= i <= last]
        images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:],
                       duration=80, disposal=2, loop=0)
    print(f"Recorded {len(frames)} real-game frames to {OUT}")


if __name__ == "__main__":
    main()
