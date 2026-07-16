import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "render_in_apartment.py"
    spec = importlib.util.spec_from_file_location("render_in_apartment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenderInApartmentTests(unittest.TestCase):
    def test_parallel_instance_settings_isolate_rpc_temp_log_and_shared_memory(self):
        mod = load_module()

        settings = mod.parallel_instance_settings(39102, graphics_adapter=2)

        self.assertEqual(settings["rpc_port"], 39102)
        self.assertEqual(settings["graphics_adapter"], 2)
        self.assertEqual(settings["temp_dir"], "tmp/spear_instance_39102")
        self.assertEqual(settings["log"], "SpearSim_rpc_39102.log")
        self.assertEqual(settings["shared_memory_initial_unique_id"], 391020000)

    def test_parallel_instance_settings_reject_invalid_port_or_adapter(self):
        mod = load_module()

        with self.assertRaises(ValueError):
            mod.parallel_instance_settings(80)
        with self.assertRaises(ValueError):
            mod.parallel_instance_settings(39102, graphics_adapter=-1)

    def test_compute_asset_fit_places_mesh_bottom_on_floor(self):
        mod = load_module()
        meta = {"ext": 200.0, "bmin_z": -100.0, "height": 200.0}

        fit = mod.compute_asset_fit(meta=meta, target_cm=50.0, floor_z=0.0)

        self.assertEqual(fit["scale"], 0.25)
        self.assertEqual(fit["actor_z"], 25.0)
        self.assertEqual(fit["center_z"], 25.0)
        self.assertEqual(fit["actor_z"] + meta["bmin_z"] * fit["scale"], 0.0)

    def test_compute_asset_fit_uses_measured_apartment_floor_height(self):
        mod = load_module()
        meta = {"ext": 200.0, "bmin_z": -100.0, "height": 200.0}

        fit = mod.compute_asset_fit(meta=meta, target_cm=50.0, floor_z=29.0)

        self.assertEqual(fit["actor_z"], 54.0)
        self.assertEqual(fit["center_z"], 54.0)
        self.assertEqual(fit["actor_z"] + meta["bmin_z"] * fit["scale"], 29.0)

    def test_compute_bounds_lift_raises_actor_above_measured_floor(self):
        mod = load_module()

        lift = mod.compute_bounds_lift(
            bounds_bottom_z=0.0,
            ground_z=27.1,
            clearance_cm=2.0,
            tolerance_cm=0.5,
        )

        self.assertTrue(math.isclose(lift, 29.1))

    def test_compute_bounds_lift_ignores_tiny_error_inside_tolerance(self):
        mod = load_module()

        lift = mod.compute_bounds_lift(
            bounds_bottom_z=28.8,
            ground_z=27.1,
            clearance_cm=2.0,
            tolerance_cm=0.5,
        )

        self.assertEqual(lift, 0.0)

    def test_orbit_pose_clamps_radius_and_points_at_center(self):
        mod = load_module()

        pose = mod.compute_orbit_pose(
            frame_index=0,
            total_frames=36,
            center_x=0.0,
            center_y=0.0,
            center_z=25.0,
            target_cm=80.0,
            r_factor=4.0,
            max_radius_cm=200.0,
            cam_z_offset_cm=40.0,
        )

        self.assertEqual(pose["location"], {"X": 200.0, "Y": 0.0, "Z": 65.0})
        self.assertEqual(pose["rotation"]["Yaw"], 180.0)
        self.assertTrue(
            math.isclose(
                pose["rotation"]["Pitch"],
                -math.degrees(math.atan2(40.0, 200.0)),
            )
        )

    def test_orbit_pose_quarter_turn_uses_unreal_yaw_convention(self):
        mod = load_module()

        pose = mod.compute_orbit_pose(
            frame_index=9,
            total_frames=36,
            center_x=0.0,
            center_y=0.0,
            center_z=25.0,
            target_cm=80.0,
            r_factor=4.0,
            max_radius_cm=200.0,
            cam_z_offset_cm=40.0,
        )

        self.assertTrue(math.isclose(pose["location"]["X"], 0.0, abs_tol=1e-9))
        self.assertEqual(pose["location"]["Y"], 200.0)
        self.assertTrue(math.isclose(pose["rotation"]["Yaw"], -90.0))

    def test_should_remove_furniture_keeps_structure_and_windows(self):
        mod = load_module()

        self.assertTrue(mod.should_remove_actor("Meshes/06_sofa/LivingRoom_Sofa"))
        self.assertTrue(mod.should_remove_actor("Meshes/07_table/LivingRoom_Table"))
        self.assertTrue(mod.should_remove_actor("Meshes/05_chair/Dining_Chair"))
        self.assertTrue(mod.should_remove_actor("Meshes/18_pillow/Sofa_Pillow"))
        self.assertTrue(mod.should_remove_actor("Meshes/35_lamp/Table_Lamp"))
        self.assertFalse(mod.should_remove_actor("Meshes/01_wall/Wall_01"))
        self.assertFalse(mod.should_remove_actor("Meshes/02_floor/Floor"))
        self.assertFalse(mod.should_remove_actor("Meshes/09_window/LivingRoom_Window"))
        self.assertFalse(mod.should_remove_actor(""))

    def test_output_dir_uses_apartment_prefix(self):
        mod = load_module()

        self.assertEqual(
            mod.build_output_dir("/tmp/spear", "Clock"),
            "/tmp/spear/render_apartment_Clock",
        )

    def test_cli_defaults_use_validated_apartment_turntable_setup(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertEqual(args.spawn_x, -120.0)
        self.assertEqual(args.spawn_y, 80.0)
        self.assertEqual(args.max_radius_cm, 130.0)
        self.assertEqual(args.target_cm, 80.0)
        self.assertEqual(args.ground_clearance_cm, 0.5)
        self.assertEqual(args.ground_tolerance_cm, 0.5)
        self.assertTrue(args.clear_furniture)


class ChecklistTests(unittest.TestCase):
    def test_build_solo_checklist_captures_all_deterministic_fields(self):
        mod = load_module()

        checklist = mod.build_solo_checklist(
            name="cat",
            ground_z=27.11,
            bounds_bottom_z=27.6,
            lift_cm=0.0,
            penetration_after_lift=0.01,
            scale=0.401,
            target_cm=80.0,
            radius=130.0,
            frames=36,
            clearance_cm=0.5,
            tolerance_cm=0.5,
        )

        self.assertEqual(checklist["name"], "cat")
        self.assertEqual(checklist["frames"], 36)
        self.assertEqual(checklist["target_cm"], 80.0)
        self.assertEqual(checklist["scale"], 0.401)
        self.assertEqual(checklist["radius_cm"], 130.0)
        self.assertEqual(checklist["ground_z_cm"], 27.11)
        self.assertEqual(checklist["bounds_bottom_z_cm"], 27.6)
        self.assertEqual(checklist["lift_applied_cm"], 0.0)
        self.assertEqual(checklist["penetration_after_lift_cm"], 0.01)
        self.assertTrue(checklist["ground_ok"])
        self.assertIn("clearance_cm", checklist)
        self.assertIn("tolerance_cm", checklist)

    def test_build_solo_checklist_flags_bad_ground(self):
        mod = load_module()

        checklist = mod.build_solo_checklist(
            name="yak",
            ground_z=27.11,
            bounds_bottom_z=10.0,
            lift_cm=0.0,
            penetration_after_lift=17.11,
            scale=1.0,
            target_cm=80.0,
            radius=130.0,
            frames=36,
            clearance_cm=0.5,
            tolerance_cm=0.5,
        )

        self.assertFalse(checklist["ground_ok"])

    def test_write_checklist_roundtrips_json(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            data = {"name": "cat", "frames": 36}

            path = mod.write_checklist(tmp, data)

            self.assertEqual(path, os.path.join(tmp, "checklist.json"))
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), data)


class GroupLayoutTests(unittest.TestCase):
    def test_write_group_layout_creates_png(self):
        mod = load_module()
        positions = [
            {"name": "cat", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "yak", "x": 45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = mod.write_group_layout(
                tmp, positions, radius_cm=485.0, center_x=-120.0, center_y=80.0
            )
            self.assertEqual(path, os.path.join(tmp, "layout.png"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)


class GroupModeCliTests(unittest.TestCase):
    def test_group_mode_parses_animals_list_and_defaults(self):
        mod = load_module()

        args = mod.parse_args(
            ["--mode", "group", "--animals", "cat,dog,goose,yak"]
        )

        self.assertEqual(args.mode, "group")
        self.assertEqual(args.animals, ["cat", "dog", "goose", "yak"])
        self.assertEqual(args.gap_cm, 30.0)
        self.assertEqual(args.group_name, "group")

    def test_group_mode_rejects_unknown_animal_in_list(self):
        mod = load_module()

        with self.assertRaises(SystemExit):
            mod.parse_args(["--mode", "group", "--animals", "cat,unicorn"])

    def test_animals_group_output_root_default(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertEqual(args.output_root, mod.DEFAULT_TMP_ROOT)
        self.assertEqual(
            mod.ANIMALS_OUTPUT_SUBDIR, "render_animals_apartment"
        )


class LineupPositionTests(unittest.TestCase):
    def test_lineup_centers_group_at_given_point(self):
        mod = load_module()
        metas = {
            "cat": {"ext": 199.5, "bmin_z": -65.0, "height": 130.0},
            "dog": {"ext": 199.25, "bmin_z": -80.9, "height": 164.1},
            "goose": {"ext": 199.2, "bmin_z": -95.1, "height": 189.9},
            "yak": {"ext": 198.9, "bmin_z": -45.2, "height": 86.6},
        }

        positions = mod.compute_lineup_positions(
            animals=["cat", "dog", "goose", "yak"],
            metas=metas,
            target_cm=80.0,
            gap_cm=30.0,
            center_x=-120.0,
            center_y=80.0,
        )

        self.assertEqual([p["name"] for p in positions], ["cat", "dog", "goose", "yak"])
        for p in positions:
            self.assertEqual(p["y"], 80.0)
            self.assertEqual(p["half_extent_cm"], 40.0)

        xs = [p["x"] for p in positions]
        for a, b in zip(xs, xs[1:]):
            self.assertAlmostEqual(b - a, 110.0)
        self.assertAlmostEqual(sum(xs) / len(xs), -120.0)

    def test_lineup_single_animal_lands_at_center(self):
        mod = load_module()
        metas = {"cat": {"ext": 199.5, "bmin_z": -65.0, "height": 130.0}}

        positions = mod.compute_lineup_positions(
            animals=["cat"],
            metas=metas,
            target_cm=80.0,
            gap_cm=30.0,
            center_x=-120.0,
            center_y=80.0,
        )

        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["x"], -120.0)


class GroupOrbitRadiusTests(unittest.TestCase):
    def test_orbit_radius_expands_by_half_span_but_clamps(self):
        mod = load_module()
        positions = [
            {"name": "a", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "b", "x":  45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]

        r = mod.compute_group_orbit_radius(
            positions=positions,
            target_cm=80.0,
            base_r_factor=4.0,
            max_radius_cm=1000.0,
        )
        self.assertAlmostEqual(r, 485.0)

    def test_orbit_radius_clamps_to_max(self):
        mod = load_module()
        positions = [
            {"name": "a", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "b", "x":  45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]

        r = mod.compute_group_orbit_radius(
            positions=positions,
            target_cm=80.0,
            base_r_factor=4.0,
            max_radius_cm=300.0,
        )
        self.assertEqual(r, 300.0)


class AnimalCliShortcutTests(unittest.TestCase):
    def test_animal_shortcut_fills_asset_bp_and_name(self):
        mod = load_module()

        args = mod.parse_args(["--mode", "turntable", "--animal", "cat"])

        self.assertEqual(
            args.asset_bp,
            "/Game/MyAssets/Audioset/Blueprints/cat/BP_cat.BP_cat_C",
        )
        self.assertEqual(args.name, "cat")

    def test_animal_shortcut_defaults_to_none_and_keeps_clock_defaults(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertIsNone(args.animal)
        self.assertEqual(args.asset_bp, mod.DEFAULT_ASSET_BP)
        self.assertEqual(args.name, mod.DEFAULT_NAME)

    def test_animal_shortcut_respects_explicit_name_override(self):
        mod = load_module()

        args = mod.parse_args(
            ["--mode", "turntable", "--animal", "dog", "--name", "MyDog"]
        )

        self.assertEqual(args.name, "MyDog")
        self.assertEqual(
            args.asset_bp,
            "/Game/MyAssets/Audioset/Blueprints/dog/BP_dog.BP_dog_C",
        )

    def test_animal_shortcut_rejects_unsupported_animal(self):
        mod = load_module()

        with self.assertRaises(SystemExit):
            mod.parse_args(["--animal", "unicorn"])


class AnimalResolutionTests(unittest.TestCase):
    def test_animal_bp_path_uses_audioset_blueprints_folder(self):
        mod = load_module()

        self.assertEqual(
            mod.animal_bp_path("cat"),
            "/Game/MyAssets/Audioset/Blueprints/cat/BP_cat.BP_cat_C",
        )
        self.assertEqual(
            mod.animal_bp_path("yak"),
            "/Game/MyAssets/Audioset/Blueprints/yak/BP_yak.BP_yak_C",
        )

    def test_animal_bp_path_rejects_unknown_animal(self):
        mod = load_module()

        with self.assertRaises(ValueError):
            mod.animal_bp_path("unicorn")

    def test_supported_animals_matches_imported_blueprints(self):
        mod = load_module()

        self.assertEqual(mod.SUPPORTED_ANIMALS, ("cat", "dog", "goose", "yak"))

    def test_animal_meta_path_returns_lowercase_json(self):
        mod = load_module()

        self.assertEqual(
            mod.animal_meta_path("/tmp/meta", "cat"),
            "/tmp/meta/cat.json",
        )


if __name__ == "__main__":
    unittest.main()
