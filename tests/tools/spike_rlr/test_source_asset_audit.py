import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from source_asset_audit import audit_approved_assets  # noqa: E402


def _write_registry(root: Path):
    asset_path = root / "dog" / "beagle" / "dog_beagle_0002" / "asset.json"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text(json.dumps({
        "schema_version": "source_asset_v1",
        "asset_id": "dog_beagle_0002",
        "legacy_tag": "dog_beagle_v2",
        "asset_class": "animal",
        "category": "dog",
        "family": "beagle",
        "variant": {"variant_index": 2},
        "generation": {"text_description": "beagle dog"},
        "appearance": {"dominant_colors": []},
        "visual_assets": {},
        "rig": {"skeleton_family": "quaternius_dog", "animations": [], "loop_required": True},
        "audio": {"default_lookup": "dog_bark", "allowed_lookups": ["dog_bark"]},
        "review": {"overall_status": "approved"},
    }), encoding="utf-8")
    (root / "registry.json").write_text(json.dumps({
        "schema_version": "source_assets_v1",
        "assets": [{
            "asset_id": "dog_beagle_0002",
            "asset_class": "animal",
            "category": "dog",
            "family": "beagle",
            "path": "dog/beagle/dog_beagle_0002/asset.json",
            "overall_status": "approved",
        }],
    }), encoding="utf-8")


def _write_approved_tag(root: Path, tag: str, *, complete: bool):
    tag_dir = root / tag
    tag_dir.mkdir(parents=True)
    (tag_dir / "direction.json").write_text(json.dumps({
        "human_approved": True,
        "human_approved_by": "jzy",
        "human_approved_at": "2026-07-09T00:00:00+00:00",
    }), encoding="utf-8")
    (tag_dir / "mesh_oriented.glb").write_bytes(b"mesh")
    if complete:
        (tag_dir / "mesh_runtime.glb").write_bytes(b"runtime")
        (tag_dir / "mesh_runtime.json").write_text("{}", encoding="utf-8")
        (tag_dir / "hy3d_diffuse.jpg").write_bytes(b"diffuse")
    return tag_dir


def test_audit_approved_assets_reports_registered_and_incomplete_tags(tmp_path):
    registry_root = tmp_path / "registry"
    approved_root = tmp_path / "approved"
    registry_root.mkdir()
    approved_root.mkdir()
    _write_registry(registry_root)
    _write_approved_tag(approved_root, "dog_beagle_v2", complete=True)
    _write_approved_tag(approved_root, "dog_old", complete=False)

    rows = audit_approved_assets(
        approved_dir=approved_root,
        registry_root=registry_root,
    )

    by_tag = {row["legacy_tag"]: row for row in rows}
    assert by_tag["dog_beagle_v2"]["registered"] is True
    assert by_tag["dog_beagle_v2"]["classification_complete"] is True
    assert by_tag["dog_old"]["registered"] is False
    assert by_tag["dog_old"]["classification_complete"] is False
    assert "mesh_runtime.glb" in by_tag["dog_old"]["missing_files"]
