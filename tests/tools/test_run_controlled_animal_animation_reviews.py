from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import run_controlled_animal_animation_reviews as reviews


def test_review_specs_cover_walk_side_front_and_idle_side():
    assert reviews.REVIEW_SPECS == (
        ("walking_side", "Walking", "side"),
        ("walking_front", "Walking", "front"),
        ("idle_side", "Idle", "side"),
    )


def test_render_command_enables_ground_contact_review(tmp_path):
    command = reviews.build_render_command(
        tmp_path / "animal.glb",
        tmp_path / "frames",
        action="Walking",
        view="side",
    )

    assert command[command.index("--action") + 1] == "Walking"
    assert command[command.index("--view") + 1] == "side"
    assert command[command.index("--n-frames") + 1] == "24"
    assert "--ground-plane" in command


def test_encode_command_pins_dataset_video_contract(tmp_path):
    command = reviews.build_encode_command(tmp_path / "frames", tmp_path / "video.mp4")

    assert command[command.index("-framerate") + 1] == "12"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "+faststart" in command


def test_deformation_gate_runs_before_animation_media(tmp_path):
    command = reviews.build_deformation_command(
        tmp_path / "animal.glb", tmp_path / "deformation.json"
    )

    assert str(reviews.DEFORMATION_AUDITOR) in command
    assert command[command.index("--input") + 1].endswith("animal.glb")
    assert command[command.index("--output") + 1].endswith("deformation.json")
    source = reviews.Path(reviews.__file__).read_text(encoding="utf-8")
    assert 'if deformation_overall == "rejected"' in source
    assert '"status": "rejected_deformation_gate"' in source
    assert '"deformation_rejections_rendered": False' in source


@pytest.mark.parametrize("workers", [0, 17])
def test_review_runner_rejects_unsafe_worker_counts(tmp_path, workers):
    with pytest.raises(contracts.ContractError, match="workers"):
        reviews.run_reviews(
            tmp_path / "missing.json", tmp_path / "output", workers=workers
        )
