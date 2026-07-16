import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "render_in_gpurir_room.py"
    spec = importlib.util.spec_from_file_location("render_in_gpurir_room", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShoeboxRoomLayoutTests(unittest.TestCase):
    def test_layout_returns_six_pieces_in_stable_order(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))

        self.assertEqual(len(pieces), 6)
        self.assertEqual(
            [p["name"] for p in pieces],
            ["floor", "ceiling", "wall_x0", "wall_x1", "wall_y0", "wall_y1"],
        )

    def test_floor_uses_floor_400_mesh_covering_room_footprint(self):
        """Floor now uses the Floor_400x400 starter mesh, whose LOCAL bounds are
        (0,0,-20) to (400,400,0) — pivot in the (-,-) corner, thickness 20cm
        extending DOWN from z=0, top face at z=0. Placing pivot at world (0,0,0)
        with scale (rx/4, ry/4, 1) yields a floor covering exactly the room
        footprint (0,0) to (rx_cm, ry_cm), with the top face still at z=0. This
        preserves the wall-attachment contract while giving realistic wood-plank
        UV density (a room-wide Cube.Cube stretch would show one plank spanning
        the entire 5.2m room)."""
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        floor = pieces[0]

        self.assertEqual(floor["name"], "floor")
        # Explicitly requests Floor_400x400
        self.assertEqual(floor.get("mesh"), mod.FLOOR_MESH)
        # Pivot at world origin (matches Floor_400x400 local pivot at bounds min)
        self.assertEqual(floor["location_cm"], (0.0, 0.0, 0.0))
        # Scale = room_size / 4m so the 400x400 mesh spans room footprint exactly
        sx, sy, sz = floor["scale"]
        self.assertAlmostEqual(sx, 5.2 / 4.0, places=6)
        self.assertAlmostEqual(sy, 4.4 / 4.0, places=6)
        self.assertAlmostEqual(sz, 1.0, places=6)

    def test_ceiling_bottom_face_sits_at_room_height(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        ceiling = pieces[1]

        cx, cy, cz = ceiling["location_cm"]
        sx, sy, sz = ceiling["scale"]
        thickness_cm = sz * 100.0
        bottom_face_z = cz - thickness_cm / 2.0

        self.assertAlmostEqual(bottom_face_z, 280.0, places=6)

    def test_walls_have_inner_faces_flush_with_room_bounds(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        pieces_by_name = {p["name"]: p for p in pieces}

        w = pieces_by_name["wall_x0"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(inner_x, 0.0, places=6)

        w = pieces_by_name["wall_x1"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_x = cx - sx * 100.0 / 2.0
        self.assertAlmostEqual(inner_x, 520.0, places=6)

        w = pieces_by_name["wall_y0"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_y = cy + sy * 100.0 / 2.0
        self.assertAlmostEqual(inner_y, 0.0, places=6)

        w = pieces_by_name["wall_y1"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_y = cy - sy * 100.0 / 2.0
        self.assertAlmostEqual(inner_y, 440.0, places=6)

    def test_wall_heights_match_room_z(self):
        """Walls stay CENTERED on the room mid-height but extend past floor and
        ceiling by WALL_JOINT_OVERLAP_M on each end. This turns the coplanar
        wall/floor and wall/ceiling seams into solid overlaps, so Lumen no
        longer leaks a dark shadow stripe along them. The visible interior
        height is still room_z."""
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        overlap_cm = mod.WALL_JOINT_OVERLAP_M * 100.0
        expected_wall_span_cm = 280.0 + 2.0 * overlap_cm
        for name in ("wall_x0", "wall_x1", "wall_y0", "wall_y1"):
            w = [p for p in pieces if p["name"] == name][0]
            _, _, cz = w["location_cm"]
            _, _, sz = w["scale"]
            self.assertAlmostEqual(cz, 280.0 / 2.0, places=6)
            self.assertAlmostEqual(sz * 100.0, expected_wall_span_cm, places=6)


class GpurirCliTests(unittest.TestCase):
    def test_defaults_match_grill_decisions(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertEqual(args.animal, "dog")
        self.assertEqual(args.room_size_m, [5.2, 4.4, 2.8])
        self.assertEqual(args.window_w_m, 1.4)
        self.assertEqual(args.window_h_m, 1.4)
        self.assertEqual(args.window_z_bottom_m, 0.9)
        self.assertEqual(args.source_offset_m, [0.0, 1.7, 0.0])
        self.assertEqual(args.extra_animal, [])
        self.assertEqual(args.orbit_radius_cm, 200.0)
        self.assertEqual(args.frames, 36)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.framerate, 12)
        self.assertEqual(args.directional_light_intensity_lux, 10.0)
        self.assertEqual(args.run_name, "dog_default")

    def test_room_size_arg_parses_three_floats(self):
        mod = load_module()

        args = mod.parse_args(["--room-size-m", "6.0", "5.0", "2.9"])

        self.assertEqual(args.room_size_m, [6.0, 5.0, 2.9])

    def test_animal_choice_restricted_to_imported_set(self):
        mod = load_module()

        for name in ("cat", "dog", "goose", "yak"):
            args = mod.parse_args(["--animal", name])
            self.assertEqual(args.animal, name)

        with self.assertRaises(SystemExit):
            mod.parse_args(["--animal", "unicorn"])


class GpurirLayoutTests(unittest.TestCase):
    def test_write_gpurir_layout_creates_png(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = mod.write_gpurir_layout(
                tmp,
                room_size_m=(5.2, 4.4, 2.8),
                mic_pos_cm=(260.0, 220.0, 120.0),
                source_pos_cm=(260.0, 390.0, 120.0),
                window_bounds_cm={
                    "left_x": 160.0,
                    "right_x": 360.0,
                    "bottom_z": 20.0,
                    "top_z": 260.0,
                    "y": 440.0,
                },
                orbit_radius_cm=200.0,
            )
            self.assertEqual(path, os.path.join(tmp, "layout.png"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)


class RoomChecklistTests(unittest.TestCase):
    def test_room_checklist_extends_solo_checklist_with_room_fields(self):
        mod = load_module()
        solo = {
            "name": "dog",
            "frames": 36,
            "target_cm": 80.0,
            "scale": 0.4015,
            "radius_cm": 200.0,
            "ground_z_cm": 0.5,
            "bounds_bottom_z_cm": 1.0,
            "lift_applied_cm": 0.0,
            "penetration_after_lift_cm": 0.0,
            "clearance_cm": 0.5,
            "tolerance_cm": 0.5,
            "ground_ok": True,
        }

        checklist = mod.build_room_checklist(
            solo_checklist=solo,
            room_size_m=(5.2, 4.4, 2.8),
            mic_pos_cm=(260.0, 220.0, 120.0),
            source_pos_cm=(260.0, 390.0, 120.0),
            window_bounds_cm={
                "left_x": 160.0,
                "right_x": 360.0,
                "bottom_z": 20.0,
                "top_z": 260.0,
                "y": 440.0,
            },
            directional_light_intensity_lux=10.0,
            ceiling_casts_shadow=True,
            window_top_wall_casts_shadow=True,
            window_wall_casts_shadow=True,
            wall_material="/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls",
            floor_material="/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor",
        )

        self.assertEqual(checklist["name"], "dog")
        self.assertEqual(checklist["scale"], 0.4015)
        self.assertTrue(checklist["ground_ok"])

        self.assertEqual(checklist["room_size_m"], (5.2, 4.4, 2.8))
        self.assertEqual(checklist["mic_pos_cm"], (260.0, 220.0, 120.0))
        self.assertEqual(checklist["source_pos_cm"], (260.0, 390.0, 120.0))
        self.assertEqual(checklist["window_bounds_cm"]["left_x"], 160.0)
        self.assertEqual(checklist["directional_light_intensity_lux"], 10.0)
        self.assertIn("MI_Walls", checklist["wall_material"])
        self.assertIn("MI_Floor", checklist["floor_material"])

        self.assertIn("human_review", checklist)
        self.assertIsInstance(checklist["human_review"], list)
        self.assertEqual(len(checklist["human_review"]), 4)
        joined = " ".join(checklist["human_review"]).lower()
        self.assertIn("wall", joined)
        self.assertIn("ceiling", joined)
        self.assertIn("window", joined)
        self.assertIn("shadow", joined)


class MicSourcePositionTests(unittest.TestCase):
    def test_mic_at_room_center_height_1_2m(self):
        mod = load_module()

        pos = mod.compute_mic_position_cm(room_size_m=(5.2, 4.4, 2.8))

        self.assertAlmostEqual(pos[0], 260.0, places=6)
        self.assertAlmostEqual(pos[1], 220.0, places=6)
        self.assertAlmostEqual(pos[2], 120.0, places=6)

    def test_mic_uses_gpurir_canonical_1_2m_height(self):
        mod = load_module()

        self.assertEqual(mod.MIC_HEIGHT_M, 1.2)

    def test_source_position_defaults_to_mic_plus_1_7m_along_y(self):
        mod = load_module()

        pos = mod.compute_source_position_cm(room_size_m=(5.2, 4.4, 2.8))

        self.assertAlmostEqual(pos[0], 260.0, places=6)
        self.assertAlmostEqual(pos[1], 390.0, places=6)
        self.assertAlmostEqual(pos[2], 120.0, places=6)

    def test_source_position_respects_custom_offset(self):
        mod = load_module()

        pos = mod.compute_source_position_cm(
            room_size_m=(5.2, 4.4, 2.8),
            source_offset_m=(1.0, 0.5, -0.7),
        )

        self.assertAlmostEqual(pos[0], 360.0, places=6)
        self.assertAlmostEqual(pos[1], 270.0, places=6)
        self.assertAlmostEqual(pos[2], 50.0, places=6)


class WindowWallLayoutTests(unittest.TestCase):
    def test_four_pieces_leave_exact_window_hole(self):
        mod = load_module()

        pieces = mod.compute_window_wall_layout(
            room_size_m=(5.2, 4.4, 2.8),
            window_w_m=2.0,
            window_h_m=2.4,
            window_cx_m=2.6,
            window_z_bottom_m=0.2,
        )

        self.assertEqual(len(pieces), 4)
        names = [p["name"] for p in pieces]
        self.assertEqual(
            sorted(names),
            sorted(["wall_y1_bottom", "wall_y1_top", "wall_y1_left", "wall_y1_right"]),
        )

        by_name = {p["name"]: p for p in pieces}

        overlap_cm = mod.WALL_JOINT_OVERLAP_M * 100.0

        # Sill sinks INTO the floor by overlap_cm at the bottom, so its bottom
        # face sits below z=0. Its top face still meets the window opening at
        # window_z_bottom (20cm here).
        b = by_name["wall_y1_bottom"]
        cx, cy, cz = b["location_cm"]
        sx, sy, sz = b["scale"]
        top_z = cz + sz * 100.0 / 2.0
        bottom_z = cz - sz * 100.0 / 2.0
        self.assertAlmostEqual(bottom_z, -overlap_cm, places=6)
        self.assertAlmostEqual(top_z, 20.0, places=6)
        self.assertAlmostEqual(sx * 100.0, 520.0, places=6)

        # Lintel sticks UP INTO the ceiling by overlap_cm at the top; its bottom
        # face still meets the window opening at window_z_top (260cm here).
        t = by_name["wall_y1_top"]
        cx, cy, cz = t["location_cm"]
        sx, sy, sz = t["scale"]
        bottom_z = cz - sz * 100.0 / 2.0
        top_z = cz + sz * 100.0 / 2.0
        self.assertAlmostEqual(bottom_z, 260.0, places=6)
        self.assertAlmostEqual(top_z, 280.0 + overlap_cm, places=6)
        self.assertAlmostEqual(sx * 100.0, 520.0, places=6)

        l = by_name["wall_y1_left"]
        cx, cy, cz = l["location_cm"]
        sx, sy, sz = l["scale"]
        left_x = cx - sx * 100.0 / 2.0
        right_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(left_x, 0.0, places=6)
        self.assertAlmostEqual(right_x, 160.0, places=6)
        self.assertAlmostEqual(sz * 100.0, 240.0, places=6)
        self.assertAlmostEqual(cz, 140.0, places=6)

        r = by_name["wall_y1_right"]
        cx, cy, cz = r["location_cm"]
        sx, sy, sz = r["scale"]
        left_x = cx - sx * 100.0 / 2.0
        right_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(left_x, 360.0, places=6)
        self.assertAlmostEqual(right_x, 520.0, places=6)
        self.assertAlmostEqual(sz * 100.0, 240.0, places=6)

    def test_all_pieces_share_wall_y_and_thickness(self):
        mod = load_module()

        pieces = mod.compute_window_wall_layout(
            room_size_m=(5.2, 4.4, 2.8),
            window_w_m=2.0,
            window_h_m=2.4,
            window_cx_m=2.6,
            window_z_bottom_m=0.2,
        )
        ys = {p["location_cm"][1] for p in pieces}
        scale_ys = {p["scale"][1] for p in pieces}
        self.assertEqual(len(ys), 1)
        self.assertEqual(len(scale_ys), 1)
        self.assertAlmostEqual(list(ys)[0], 440.0 + 10.0 / 2.0, places=6)
        self.assertAlmostEqual(list(scale_ys)[0], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
