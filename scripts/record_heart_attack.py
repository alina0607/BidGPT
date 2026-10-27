"""Exercise K through the real OpenGL input path, facing both ways; save an in-game review GIF."""
import os
from pathlib import Path
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality/heart-attack-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Holding K cannot autorepeat. A second press deliberately fires after the previous action ends.
    frames = range(10, 141, 2)
    env = {**os.environ, "SA_KEYS": "k@20:42,a@65:67,k@78:98,k@116:117", "SA_EXIT_AFTER": "145",
           "SA_CAPTURE": ",".join(f"{OUT}/frame-{i:03}.png@{i}" for i in frames)}
    subprocess.run([str(ROOT / "build/game/sell_anything"), "--backend", "opengl"],
                   cwd=ROOT, env=env, check=True, timeout=180)
    images = [Image.open(OUT / f"frame-{i:03}.png").convert("RGB").resize((960, 540)) for i in frames]
    images[0].save(OUT / "gameplay.gif", save_all=True, append_images=images[1:],
                   duration=80, disposal=2, loop=0)
    print(f"Recorded {len(frames)} actual game frames to {OUT}")


if __name__ == "__main__":
    main()
