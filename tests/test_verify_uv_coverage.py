"""Unit tests for tools/verify_uv_coverage.py.

Run in the hunyuan3d env (needs trimesh, cv2, numpy):
  cd /data/jzy/code/SPEAR && \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest \\
    tests.test_verify_uv_coverage -v
"""
import os
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

REPO = "/data/jzy/code/SPEAR"
FIX = os.path.join(REPO, "tests/fixtures/tiny_uv_transfer")


class VerifyUvCoverageTest(unittest.TestCase):

    def _run(self, diffuse, mesh):
        return subprocess.run(
            [
                "/data/jzy/miniconda3/envs/hunyuan3d/bin/python",
                os.path.join(REPO, "tools/verify_uv_coverage.py"),
                "--diffuse", diffuse,
                "--original-mesh", mesh,
            ],
            capture_output=True, text=True,
        )

    def test_fully_painted_passes(self):
        """A 32x32 image fully painted red on mesh_a (UV=[0,1]^2) -> painted_fraction=1.0, uv_area_fraction=1.0, ratio=1.0 -> pass."""
        tmp = tempfile.mkdtemp()
        img = np.full((32, 32, 3), 100, dtype=np.uint8)   # non-zero everywhere
        p = os.path.join(tmp, "full.png")
        cv2.imwrite(p, img)
        r = self._run(p, os.path.join(FIX, "mesh_a.obj"))
        self.assertEqual(r.returncode, 0, msg=f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("UV_COVERAGE_OK", r.stdout)
        # Extract ratio
        for tok in r.stdout.split():
            if tok.startswith("ratio="):
                ratio = float(tok.split("=")[1])
                self.assertGreaterEqual(ratio, 0.85)

    def test_mostly_empty_fails(self):
        """A 32x32 image with almost nothing painted on mesh_a (which
        has UV covering the full [0,1]^2) -> painted_fraction << uv_area_fraction
        -> ratio << 0.85 -> fail."""
        tmp = tempfile.mkdtemp()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[0:4, 0:4] = 200   # 16 painted pixels out of 1024
        p = os.path.join(tmp, "sparse.png")
        cv2.imwrite(p, img)
        r = self._run(p, os.path.join(FIX, "mesh_a.obj"))
        self.assertEqual(r.returncode, 1, msg=f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("UV_COVERAGE_FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main()
