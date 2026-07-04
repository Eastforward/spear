"""Unit tests for tools/transfer_uv_texture.py.

Run in the hunyuan3d env (has trimesh, scipy, cv2):
  cd /data/jzy/code/SPEAR && \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest \\
    tests.test_transfer_uv_texture -v
"""
import os
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

REPO = "/data/jzy/code/SPEAR"
FIX = os.path.join(REPO, "tests/fixtures/tiny_uv_transfer")


class TransferUvTextureTest(unittest.TestCase):

    def test_transfer_full_uv_square(self):
        """mesh_a (UV = xy) with mesh_b (UV = 0.25 + 0.5*xy) as source,
        source diffuse has red L / green R in the [8..24]^2 window ->
        transferred should have red L / green R across ~all of a 32x32 output.
        """
        out = os.path.join(tempfile.mkdtemp(), "transferred.png")
        result = subprocess.run(
            [
                "/data/jzy/miniconda3/envs/hunyuan3d/bin/python",
                os.path.join(REPO, "tools/transfer_uv_texture.py"),
                "--orig-mesh", os.path.join(FIX, "mesh_a.obj"),
                "--hy3d-mesh", os.path.join(FIX, "mesh_b.obj"),
                "--hy3d-diffuse", os.path.join(FIX, "hunyuan_diffuse.png"),
                "--output", out,
                "--size", "32",
                "--dilate", "0",   # keep the test deterministic
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("UV_TRANSFER_OK", result.stdout)
        self.assertTrue(os.path.exists(out))

        img = cv2.imread(out)
        self.assertEqual(img.shape, (32, 32, 3))
        # Left half should be predominantly RED (BGR channel 2 dominates)
        left = img[:, :16]
        right = img[:, 16:]
        left_red_dominant = int((left[..., 2] > left[..., 1]).sum())
        right_green_dominant = int((right[..., 1] > right[..., 2]).sum())
        self.assertGreater(left_red_dominant, 200, f"left half not red: {left_red_dominant}/{16*32}")
        self.assertGreater(right_green_dominant, 200, f"right half not green: {right_green_dominant}/{16*32}")


if __name__ == "__main__":
    unittest.main()
