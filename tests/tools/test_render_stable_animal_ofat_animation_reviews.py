from pathlib import Path

import pytest

from tools import render_stable_animal_ofat_animation_reviews as review


def test_render_command_is_fixed_direction_pbr_walk_idle_evidence(tmp_path: Path):
    command = review.build_render_command(
        blender=Path("/blender"),
        glb=Path("/instance.glb"),
        action="Walking",
        output_dir=tmp_path,
        camera_reference_diagonal=8.5,
        frames=20,
        width=640,
        height=480,
        samples=8,
    )
    assert "--review-clay-color" not in command
    assert command[command.index("--action") + 1] == "Walking"
    assert command[command.index("--camera-reference-diagonal") + 1] == "8.500000000"
    assert "--ground-plane" in command
    assert "--orthographic" in command
    assert command[command.index("--view") + 1] == "side"


def test_render_command_rejects_noncanonical_action(tmp_path: Path):
    with pytest.raises(review.AnimationReviewError, match="unsupported action"):
        review.build_render_command(
            blender=Path("/blender"),
            glb=Path("/instance.glb"),
            action="Running",
            output_dir=tmp_path,
            camera_reference_diagonal=8.5,
            frames=20,
            width=640,
            height=480,
            samples=8,
        )
