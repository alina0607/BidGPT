"""Verify runtime sprites reproduce the approved source without recoloring or pose reordering."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import make_stage_art as art


class ApprovedRunAssets(unittest.TestCase):
    def test_runtime_contract(self):
        meta = json.loads((art.STAGE / "girl_run.json").read_text())
        self.assertEqual(meta["parts"], {"start": [0, 0], "run": [1, 8], "stop": [9, 9]})
        self.assertEqual(meta["frame_ms"], [120] * 10)
        self.assertEqual(sum(meta["frame_ms"][1:9]), 960)
        self.assertEqual(meta["jump"], {"rise": 1, "top": 3, "fall": 5, "land": 9})
        self.assertEqual(meta["gait"]["stride_b"]["frame"], 5)
        atlas = Image.open(art.STAGE / "girl_run.png").convert("RGBA")
        w, h = meta["cell"]
        self.assertEqual(atlas.size, (10 * w, h))
        frames = [np.array(atlas.crop((i*w, 0, (i+1)*w, h))) for i in range(10)]
        self.assertTrue(np.array_equal(frames[0], frames[4]))
        self.assertTrue(np.array_equal(frames[9], frames[4]))
        for frame in frames:
            alpha = frame[:, :, 3]
            self.assertTrue(np.any(alpha > 128))
            self.assertFalse(np.any(alpha[:2]) or np.any(alpha[-2:]))
            self.assertFalse(np.any(alpha[:, :2]) or np.any(alpha[:, -2:]))

    def test_approved_colors_and_order(self):
        cycle = [Image.open(art.SRC / "run_idle_style_v1/frames" / f"frame-{i}.png").convert("RGBA")
                 for i in range(8)]
        # Independent expected pack: one shared crop/scale, no grade_to and no per-frame fit.
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(art, "STAGE", Path(directory)):
                art.figure_sheet([cycle[3], *cycle, cycle[3]], "expected", [120]*10, 320)
            expected = np.array(Image.open(Path(directory) / "expected.png").convert("RGBA"))
        actual = np.array(Image.open(art.STAGE / "girl_run.png").convert("RGBA"))
        self.assertTrue(np.array_equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
