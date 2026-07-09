import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from human_mixamo_runtime import prepare_human_mixamo_runtime  # noqa: E402


def _write_fbx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00")


def test_prepare_human_mixamo_runtime_writes_legacy_and_per_animation_files(tmp_path):
    tag_dir = tmp_path / "approved" / "human_male_blue_hoodie_v1"
    tag_dir.mkdir(parents=True)
    walking = tmp_path / "smoke" / "walking.glb"
    idle = tmp_path / "smoke" / "idle.glb"
    walking.parent.mkdir()
    walking.write_bytes(b"walking rigged glb")
    idle.write_bytes(b"idle rigged glb")
    mixamo_root = tmp_path / "mixamo"
    _write_fbx(mixamo_root / "raw" / "Walking.fbx")
    _write_fbx(mixamo_root / "raw" / "Standing_Idle.fbx")

    metadata_path = prepare_human_mixamo_runtime(
        tag_dir,
        walking_glb=walking,
        idle_glb=idle,
        mixamo_root=mixamo_root,
    )

    assert metadata_path == tag_dir / "mesh_runtime.json"
    assert (tag_dir / "mesh_runtime.glb").read_bytes() == b"walking rigged glb"
    assert (tag_dir / "mesh_runtime_walking.glb").read_bytes() == b"walking rigged glb"
    assert (
        (tag_dir / "mesh_runtime_standing_idle.glb").read_bytes()
        == b"idle rigged glb"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema_version"] == "human_mixamo_runtime_v1"
    assert metadata["runtime_type"] == "mixamo_humanoid_nearest_skin_transfer"
    assert metadata["default_animation"] == "Walking"
    assert metadata["legacy_runtime"] == metadata["animations"]["Walking"]["glb_path"]
    assert metadata["animations"]["Walking"]["role"] == "walk"
    assert metadata["animations"]["Walking"]["loop"] is True
    assert metadata["animations"]["Standing_Idle"]["role"] == "idle"
    assert metadata["animations"]["Standing_Idle"]["motion_style"] == "stationary"
