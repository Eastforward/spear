import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_render_i23d_review.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("blender_render_i23d_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_view_positions_cover_front_side_back_and_quarter():
    renderer = _load_renderer()

    views = renderer.view_positions(center=(1.0, 2.0, 3.0), radius=4.0)

    assert views == {
        "front": (1.0, -2.0, 3.0),
        "side": (5.0, 2.0, 3.0),
        "back": (1.0, 6.0, 3.0),
        "quarter": (3.88, -0.8799999999999999, 3.32),
    }

    flipped = renderer.view_positions(
        center=(1.0, 2.0, 3.0), radius=4.0, front_sign=1
    )
    assert flipped["front"] == (1.0, 6.0, 3.0)
    assert flipped["back"] == (1.0, -2.0, 3.0)
    assert flipped["quarter"] == (3.88, 4.88, 3.32)

    positive_x = renderer.view_positions_for_axis(
        center=(1.0, 2.0, 3.0), radius=4.0, front_axis="positive-x"
    )
    assert positive_x == {
        "front": (5.0, 2.0, 3.0),
        "side": (1.0, 6.0, 3.0),
        "back": (-3.0, 2.0, 3.0),
        "quarter": (3.88, 4.88, 3.32),
    }


def test_parse_args_accepts_static_glb_review_contract(tmp_path):
    renderer = _load_renderer()
    args = renderer.parse_args(
        [
            "--input",
            str(tmp_path / "asset.glb"),
            "--output-dir",
            str(tmp_path / "renders"),
            "--width",
            "600",
            "--height",
            "800",
            "--front-axis",
            "positive-y",
        ]
    )

    assert args.input == tmp_path / "asset.glb"
    assert args.output_dir == tmp_path / "renders"
    assert (args.width, args.height) == (600, 800)
    assert args.front_axis == "positive-y"
    assert args.include_top is False
    assert args.animal_material_preview is False

    top_args = renderer.parse_args(
        [
            "--input",
            str(tmp_path / "asset.glb"),
            "--output-dir",
            str(tmp_path / "renders"),
            "--include-top",
            "--animal-material-preview",
        ]
    )
    assert top_args.include_top is True
    assert top_args.animal_material_preview is True


def test_renderer_exposes_blender_main_without_importing_bpy_at_module_load():
    renderer = _load_renderer()

    assert callable(renderer.main)
