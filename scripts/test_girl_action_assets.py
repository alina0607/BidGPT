"""Approved action registration must survive packing without per-pose resizing or recoloring."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import make_stage_art as art


class ActionAssets(unittest.TestCase):
    def test_atlases_and_registration(self):
        manifest = json.loads((art.SRC / "girl_actions_v1/manifest.json").read_text())
        for name, action in manifest["actions"].items():
            with self.subTest(action=name):
                meta = json.loads((art.STAGE / f"girl_{name}.json").read_text())
                atlas = Image.open(art.STAGE / f"girl_{name}.png").convert("RGBA")
                self.assertEqual(meta["frames"], 8)
                self.assertEqual(meta["standing"], 320)
                self.assertFalse(meta["loop"])
                self.assertEqual(meta["parts"]["action"], [1, 7])
                self.assertEqual(meta["frame_ms"], [entry["duration_ms"] for entry in action["frames"]])
                cw, ch = meta["cell"]
                self.assertEqual(atlas.size, (cw * 8, ch))
                first = Image.open(art.SRC / "girl_actions_v1" / action["frames"][0]["file"]).convert("RGBA")
                _, top, _, bottom = art.visible_bbox(first)
                factor = 320 / (bottom - top)
                self.assertEqual(meta["anchor"], [round(270 * factor) + 2, round(460 * factor) + 2])
                for i, entry in enumerate(action["frames"]):
                    frame = atlas.crop((i*cw, 0, (i+1)*cw, ch))
                    alpha = np.array(frame)[:, :, 3]
                    self.assertTrue(np.any(alpha > 128))
                    self.assertFalse(np.any(alpha[:2]) or np.any(alpha[-2:]))
                    self.assertFalse(np.any(alpha[:, :2]) or np.any(alpha[:, -2:]))
                    source = Image.open(art.SRC / "girl_actions_v1" / entry["file"]).convert("RGBA")
                    if name == "jump":
                        registered = Image.new("RGBA", (512, 512))
                        registered.paste(source, (0, 460 - [460, 460, 438, 400, 375, 430, 460, 460][i]))
                        source = registered
                    expected = source.resize((cw - 4, ch - 4), Image.Resampling.LANCZOS)
                    self.assertTrue(np.array_equal(np.array(expected), np.array(frame.crop((2, 2, cw-2, ch-2)))))

    def test_reproducible_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(art, "STAGE", Path(directory)):
                art.girl_actions()
            for name in ("jump", "slide", "kick"):
                for extension in ("png", "json"):
                    file = f"girl_{name}.{extension}"
                    self.assertEqual((Path(directory) / file).read_bytes(), (art.STAGE / file).read_bytes())

    def test_heart_cast_and_projectile(self):
        meta = json.loads((art.STAGE / "girl_cast.json").read_text())
        self.assertEqual(meta["frames"], 6)
        self.assertEqual(meta["frame_ms"], [80, 100, 100, 140, 120, 100])
        self.assertEqual(meta["release_ms"], sum(meta["frame_ms"][1:3]))
        self.assertEqual(meta["standing"], 320)
        self.assertFalse(meta["loop"])
        atlas = Image.open(art.STAGE / "girl_cast.png").convert("RGBA")
        w, h = meta["cell"]
        for i in range(6):
            alpha = np.array(atlas.crop((i*w, 0, (i+1)*w, h)))[:, :, 3]
            self.assertTrue(np.any(alpha > 128))
            self.assertFalse(np.any(alpha[:2]) or np.any(alpha[-2:]))
            self.assertFalse(np.any(alpha[:, :2]) or np.any(alpha[:, -2:]))
        wave = np.array(Image.open(art.STAGE / "heart_wave.png").convert("RGBA"))[:, :, 3]
        self.assertEqual(int(wave.min()), 0)
        self.assertEqual(int(wave.max()), 255)
        self.assertGreater(np.count_nonzero((wave > 0) & (wave < 255)), 1000)
        self.assertEqual(int(wave[0, 0]), 0)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(art, "STAGE", Path(directory)):
                art.girl_heart_attack()
            for file in ("girl_cast.png", "girl_cast.json", "heart_wave.png"):
                self.assertEqual((Path(directory) / file).read_bytes(), (art.STAGE / file).read_bytes())


if __name__ == "__main__":
    unittest.main()
