"""The idle-matched turn is separate from the approved run loop and reproducibly packed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import make_stage_art as art


class TransitionAssets(unittest.TestCase):
    def test_idle_matched_drawings_and_reproducible_pack(self):
        meta = json.loads((art.STAGE / "girl_transition.json").read_text())
        self.assertEqual(meta["frames"], 12)
        self.assertEqual(meta["source_frames"], list(range(12)))
        self.assertEqual(meta["parts"], {"start": [0, 11], "stop": [0, 11], "settle": [0, 11]})
        self.assertTrue(meta["reverse_return"])
        self.assertEqual(meta["frame_ms"], [55] * 12)
        self.assertEqual(meta["standing"], 320)
        atlas = Image.open(art.STAGE / "girl_transition.png").convert("RGBA")
        w, h = meta["cell"]
        self.assertEqual(atlas.size, (w * 12, h))
        for i in range(12):
            alpha = np.array(atlas.crop((i*w, 0, (i+1)*w, h)))[:, :, 3]
            self.assertTrue(np.any(alpha > 128))
            self.assertFalse(np.any(alpha[:2]) or np.any(alpha[-2:]))
            self.assertFalse(np.any(alpha[:, :2]) or np.any(alpha[:, -2:]))
        # The endpoint is the current runtime idle, pixel-for-pixel, not a paler generated face.
        endpoint = atlas.crop((0, 0, w, h))
        endpoint = endpoint.crop(endpoint.getchannel("A").getbbox())
        idle_meta = json.loads((art.STAGE / "girl_idle.json").read_text())
        idle = Image.open(art.STAGE / "girl_idle.png").convert("RGBA").crop((0, 0, *idle_meta["cell"]))
        idle = idle.crop(idle.getchannel("A").getbbox())
        self.assertTrue(np.array_equal(np.array(endpoint), np.array(idle)))
        # Coarse warm-skin swatch audit in the upper 56% (not a perceptual/style substitute).
        # Catch the previous white-face flash without including green dress or saturated pink hair.
        colors = []
        for i in range(12):
            a = np.array(atlas.crop((i*w, 0, (i+1)*w, int(h*.56))))
            r, g, b = a[:, :, :3].astype(float).transpose(2, 0, 1)
            skin = (a[:, :, 3] > 240) & (r-g > 9) & (r-g < 40) & (g-b > 9) & (g-b < 35) & (g > 150)
            self.assertGreater(int(skin.sum()), 1000)
            colors.append(a[:, :, :3][skin].mean(axis=0))
        self.assertLess(float(np.abs(np.array(colors[1:]) - colors[0]).max()), 8)
        approved = (art.STAGE / "girl_run.png").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(art, "STAGE", Path(directory)):
                art.girl_transitions()
            for extension in ("png", "json"):
                file = f"girl_transition.{extension}"
                self.assertEqual((Path(directory) / file).read_bytes(), (art.STAGE / file).read_bytes())
        self.assertEqual((art.STAGE / "girl_run.png").read_bytes(), approved)


if __name__ == "__main__":
    unittest.main()
