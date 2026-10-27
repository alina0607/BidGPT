"""The game's drawing GPT: turn what the player typed into one of the 345 classes, then draw it stroke
by stroke (design-v2.md §5.3, modes "named" and "random").

    python -m sellanything_ml.sketch.drawer export v2 0.6 2   # version, temperature, guidance → checkpoints/
    python -m sellanything_ml.sketch.drawer dog        # draw one, print the strokes

Strokes come back in Quick, Draw!'s simplified form (0..255, same as `classify` takes), so the game
can scale them into its canvas and the classifier sees them exactly as it would a player's drawing.
Name matching needs no torch, so the server's test mode can use it.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from .data import ML_DIR
from .raster import Stroke, simplify

DRAW_DIR = ML_DIR / "checkpoints" / "drawgpt_v2"     # the version the game uses (v2 since 2026-10-07)
ACCEPT = 0.85          # a typo this close to one class name is taken as that class ("bicyle" → bicycle)
SUGGEST = 0.6          # names at least this close are offered when nothing is accepted
ARTICLES = re.compile(r"^(a|an|the|some|my)\s+")


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", ARTICLES.sub("", text))


def match(text: str, categories: list[str]) -> tuple[str | None, list[str]]:
    """The class the player meant, or None plus up to 3 close names to offer instead."""
    names = {c.lower(): c for c in categories}
    t = _norm(text)
    for cand in (t, t[:-2] if t.endswith("es") else "", t[:-1] if t.endswith("s") else ""):
        if cand in names:
            return names[cand], []
    close = difflib.get_close_matches(t, list(names), n=3, cutoff=SUGGEST)
    if close and difflib.SequenceMatcher(None, t, close[0]).ratio() >= ACCEPT:
        return names[close[0]], []
    if not close:
        close = [n for n in names if t and (t in n or n in t)][:3]
    return None, [names[c] for c in close]


class Drawer:
    def __init__(self, folder: Path = DRAW_DIR, device: str = "cpu"):
        import torch

        from ..llm.model import GPT, GPTConfig
        meta = json.loads((folder / "meta.json").read_text())
        self.categories: list[str] = meta["categories"]
        self.temperature: float = meta["temperature"]
        self.guidance: float = meta.get("guidance", 1.0)        # classifier-free guidance (v2 and later)
        self.no_class: int | None = meta.get("no_class")
        self.centers = np.load(folder / "dict.npy")
        self.model = GPT(GPTConfig(**meta["model"])).to(device)
        self.model.load_state_dict(torch.load(folder / "model.pt", map_location=device)["model"])
        self.model.eval()

    def match(self, text: str) -> tuple[str | None, list[str]]:
        return match(text, self.categories)

    def random_label(self, rng: random.Random) -> str:
        return rng.choice(self.categories)

    def candidates(self, label: str, seed: int, n: int) -> list[list[Stroke]]:
        """n drawings of `label` sampled in one batch."""
        from . import drawgpt
        out = []
        for seq in drawgpt.draw(self.model, self.categories.index(label), n, self.temperature, seed=seed,
                                guidance=self.guidance, no_class=self.no_class):
            strokes = drawgpt.tidy(drawgpt.decode(seq, self.centers), bool(len(seq) and seq[-1] == drawgpt.END))
            out.append(simplify(strokes))
        return out

    def draw(self, label: str, seed: int) -> list[Stroke]:
        """The model's own drawing for this seed; nothing is selected or re-drawn afterwards."""
        return self.candidates(label, seed, 1)[0]


def export(version: str, temperature: float, guidance: float = 1.0) -> Path:
    """Copy a version's best weights, the pen-move dictionary and its sampling settings to
    checkpoints/<run name>/. The game uses the folder DRAW_DIR names."""
    from dataclasses import asdict

    import torch

    from . import drawgpt
    cfg = drawgpt.version(version)
    if guidance != 1.0 and drawgpt.no_class_token(cfg) is None:
        raise SystemExit(f"{version} was trained without class dropout, so it cannot use guidance")
    src = drawgpt.run_dir(cfg) / "drawgpt_best.pt"
    folder = ML_DIR / "checkpoints" / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    st = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model": st["model"]}, folder / "model.pt")
    shutil.copy(drawgpt._paths(cfg)["dict"], folder / "dict.npy")
    meta = {"categories": drawgpt.categories(cfg), "temperature": temperature, "guidance": guidance,
            "no_class": drawgpt.no_class_token(cfg), "version": version,
            "model": asdict(drawgpt.model_config(cfg)), "source": str(src)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"exported {version} to {folder} (temperature {temperature}, guidance {guidance})")
    return folder


def load_default(device: str = "cpu") -> Drawer | None:
    return Drawer(device=device) if (DRAW_DIR / "meta.json").exists() else None


def main() -> None:
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
        return
    drawer = Drawer()
    label, suggestions = drawer.match(" ".join(sys.argv[1:]) or "cat")
    if label is None:
        print("no match; closest:", suggestions)
        return
    strokes = drawer.draw(label, seed=0)
    print(label, len(strokes), "strokes", sum(len(xs) for xs, _ in strokes), "points")


if __name__ == "__main__":
    main()
