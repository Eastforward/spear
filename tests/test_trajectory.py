"""Unit tests for examples/trajectory.py.

TDD: this file is written BEFORE examples/trajectory.py exists. Steps 1-2
of Task 3 confirm the tests fail with ModuleNotFoundError / AttributeError,
proving the tests actually run (they'd otherwise silently no-op).
"""

import os
import sys
import unittest

import numpy as np

# examples/ is a plain script dir, not a package
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.abspath(os.path.join(_HERE, "..", "examples"))
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)


class GpurirTrajectoryTests(unittest.TestCase):
    def _call(self, **kwargs):
        import trajectory  # imported inside so a missing module fails per-test
        defaults = dict(
            room_size_m=(5.2, 4.4, 2.8),
            n_frames=36,
            speed_bucket="B",
            source_height_m=0.45,
            traj_aug=True,
            seed=42,
            traj_pts_full=200,
        )
        defaults.update(kwargs)
        return trajectory.gpurir_trajectory(**defaults)

    def test_gpurir_returns_correct_shape(self):
        traj = self._call(n_frames=36)
        self.assertEqual(traj.shape, (36, 3))

    def test_gpurir_seed_reproducible(self):
        a = self._call(seed=42)
        b = self._call(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_gpurir_seed_different(self):
        a = self._call(seed=42)
        b = self._call(seed=43)
        self.assertFalse(np.allclose(a, b), "different seeds must give different trajectories")

    def test_gpurir_matches_v77(self):
        """★ KEY CONTRACT: same seed + same params vs v77's get_pos_traj → identical."""
        # v77's module unconditionally does `import gpuRIR` at top-level, which
        # spear-env does not have (audio backend lives in a different env). We
        # only need get_pos_traj (pure numpy + scipy), so stub gpuRIR so the
        # module can import. This keeps the test byte-identical to v77's real
        # function body — we import the same code path v77 audio pipeline uses.
        import types as _types
        _fake = _types.ModuleType("gpuRIR")
        _fake.activateMixedPrecision = lambda *a, **k: None
        sys.modules.setdefault("gpuRIR", _fake)

        sys.path.insert(0, "/data/jzy/code/Spatial/v77_4ch_S2L/data_gen")
        from gen_rir_multiscene_v77 import get_pos_traj  # noqa

        # v77 uses GLOBAL np.random.seed — replicate the exact same call.
        np.random.seed(42)
        pos_audio, _, _ = get_pos_traj(
            room_sz=[5.2, 4.4, 2.8],
            traj_pts=200,
            large_angle=360,
            traj_aug=True,
            speed_bucket="B",
            el_range=None,
            source_height=0.45,
        )
        # gpurir_trajectory uses save/restore of global state so it does NOT
        # care about the caller's global state. Full-res (n_frames=200) call:
        pos_video_full = self._call(n_frames=200, seed=42)
        np.testing.assert_allclose(pos_video_full, pos_audio, atol=1e-6)

    def test_gpurir_downsample_matches_c_choice(self):
        """Q13=C: video subsamples full-res grid by (i * traj_pts_full // n_frames)."""
        full = self._call(n_frames=200, seed=42, traj_pts_full=200)
        sub = self._call(n_frames=36, seed=42, traj_pts_full=200)
        expected = np.stack([full[i * 200 // 36] for i in range(36)])
        np.testing.assert_array_equal(sub, expected)


class WaypointTrajectoryTests(unittest.TestCase):
    def test_waypoint_endpoints(self):
        import trajectory
        traj = trajectory.waypoint_trajectory(
            waypoints_m=[(0.5, 0.5), (5.0, 4.0)],
            n_frames=36,
            room_size_m=(5.2, 4.4, 2.8),
            kind="linear",
        )
        self.assertEqual(traj.shape, (36, 3))
        np.testing.assert_allclose(traj[0, :2], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(traj[-1, :2], [5.0, 4.0], atol=1e-6)

    def test_waypoint_min_points(self):
        import trajectory
        with self.assertRaises(ValueError):
            trajectory.waypoint_trajectory(
                waypoints_m=[(1.0, 1.0)],  # only 1 point
                n_frames=36,
                room_size_m=(5.2, 4.4, 2.8),
            )

    def test_waypoint_clip_to_room(self):
        import trajectory
        with self.assertLogs("trajectory", level="WARNING"):
            traj = trajectory.waypoint_trajectory(
                waypoints_m=[(-1.0, -1.0), (999.0, 999.0)],
                n_frames=10,
                room_size_m=(5.2, 4.4, 2.8),
                wall_margin_m=0.1,
                kind="linear",
            )
        # Endpoints should be clipped to [margin, room_dim - margin]
        self.assertGreaterEqual(traj[0, 0], 0.1 - 1e-9)
        self.assertLessEqual(traj[-1, 0], 5.2 - 0.1 + 1e-9)
        self.assertLessEqual(traj[-1, 1], 4.4 - 0.1 + 1e-9)


class YawTests(unittest.TestCase):
    def test_yaw_straight_line(self):
        import trajectory
        positions = np.stack([
            np.linspace(0.0, 5.0, 20),  # x
            np.zeros(20),                # y
            np.full(20, 0.45),           # z
        ], axis=1)
        yaw = trajectory.compute_yaw_from_positions(positions)
        # +x direction is 0 degrees
        np.testing.assert_allclose(yaw, np.zeros(20), atol=1.0)  # 1 deg tolerance

    def test_yaw_curve(self):
        import trajectory
        # Half-circle in xy at radius 1 centered at (0,0), from (1,0) to (-1,0)
        t = np.linspace(0.0, np.pi, 40)
        positions = np.stack([np.cos(t), np.sin(t), np.full(40, 0.45)], axis=1)
        yaw = trajectory.compute_yaw_from_positions(positions)
        # Yaw of forward tangent goes from ~+90 (moving +y) around to ~-90/+270
        # (moving -y). It should be monotonically increasing modulo 360.
        unwrapped = np.unwrap(np.deg2rad(yaw))
        diffs = np.diff(unwrapped)
        self.assertGreater(diffs.mean(), 0.0, "yaw should sweep in one direction")


if __name__ == "__main__":
    unittest.main()
