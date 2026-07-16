from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_render_controlled_geometry_pixal_review_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "blender_render_controlled_geometry_pixal_review_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_review_views_use_canonical_positive_y_front_and_positive_z_up():
    views = renderer.view_positions((1.0, 2.0, 3.0), 10.0)

    assert views == {
        "front": (1.0, 12.0, 3.0),
        "back": (1.0, -8.0, 3.0),
        "side": (11.0, 2.0, 3.0),
        "quarter": (8.2, 9.2, 3.8),
        "top": (1.0, 2.0, 13.0),
    }


def test_contact_sheet_contains_reference_and_all_five_views():
    from PIL import Image

    reference = Image.new("RGB", (64, 64), (255, 0, 0))
    views = {
        name: Image.new("RGB", (64, 64), color)
        for name, color in {
            "front": (0, 255, 0),
            "back": (0, 0, 255),
            "side": (255, 255, 0),
            "top": (255, 0, 255),
            "quarter": (0, 255, 255),
        }.items()
    }

    sheet = renderer.make_contact_sheet(reference, views)

    assert sheet.size == (900, 800)
    assert sheet.mode == "RGB"


def test_cli_requires_exact_canonical_review_inputs():
    args = renderer.parse_args(
        [
            "--asset-id",
            "route2_v3_female_shorts",
            "--input-glb",
            "/tmp/input.glb",
            "--pixal-manifest",
            "/tmp/input.manifest.json",
            "--output-dir",
            "/tmp/static_review_v1",
        ]
    )

    assert args.asset_id == "route2_v3_female_shorts"
    assert args.input_glb == Path("/tmp/input.glb")
    assert args.pixal_manifest == Path("/tmp/input.manifest.json")
    assert args.output_dir == Path("/tmp/static_review_v1")


def test_renderer_has_no_overwrite_path_and_records_material_texture_checks():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "renameat2" in source
    assert "_RENAME_NOREPLACE" in source
    assert "os.path.lexists(output_dir)" in source
    assert '"materials_present": bool(materials)' in source
    assert '"images_present": len(bpy.data.images) > 0' in source
    assert '"front_axis": "positive-y"' in source
    assert '"up_axis": "positive-z"' in source
    assert '"top": (x, y, z + radius)' in source


def test_private_reference_path_is_excluded_from_json_payload():
    authenticated = {
        "manifest_record": {"path": "/tmp/manifest.json"},
        "glb_record": {"path": "/tmp/model.glb"},
        "reference": Path("/tmp/reference.png"),
        "reference_record": {"path": "/tmp/reference.png"},
    }

    public_input = {
        key: value for key, value in authenticated.items() if key != "reference"
    }

    assert "reference" not in public_input
    assert json.loads(json.dumps(public_input)) == public_input
