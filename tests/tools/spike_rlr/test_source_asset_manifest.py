import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from source_asset_manifest import (  # noqa: E402
    CANDIDATE_MANIFEST_NAME,
    asset_id_for_tag,
    build_hy3d_candidate_manifest,
    sync_candidate_manifest_review,
    write_hy3d_candidate_manifest,
)


def test_asset_id_for_tag_uses_variant_suffix():
    assert asset_id_for_tag("dog_pug", "dog", "pug") == "dog_pug_0001"
    assert asset_id_for_tag("dog_beagle_v2", "dog", "beagle") == "dog_beagle_0002"
    assert (
        asset_id_for_tag("cat_british_shorthair_v2", "cat", "british_shorthair")
        == "cat_british_shorthair_0002"
    )


def test_build_hy3d_candidate_manifest_separates_intended_and_measured_fields(tmp_path):
    tag_dir = tmp_path / "pending" / "dog_beagle_v2"
    tag_dir.mkdir(parents=True)
    for name in (
        "mesh.obj",
        "mesh_oriented.glb",
        "hy3d_diffuse.jpg",
        "hy3d_roughness.jpg",
        "hy3d_metallic.jpg",
        "reference.png",
        "direction.json",
    ):
        (tag_dir / name).write_bytes(b"x")

    manifest = build_hy3d_candidate_manifest(
        tag_dir,
        tag="dog_beagle_v2",
        species="dog",
        breed="beagle",
        seed=4101,
        positive_prompt="a beagle dog in perfect side profile view",
        flux_model="flux_dev",
        created_at="2026-07-09T00:00:00+00:00",
    )

    assert manifest["schema_version"] == "source_asset_v1"
    assert manifest["asset_id"] == "dog_beagle_0002"
    assert manifest["legacy_tag"] == "dog_beagle_v2"
    assert manifest["category"] == "dog"
    assert manifest["family"] == "beagle"
    assert manifest["variant"]["intended_color_label"] == "beagle"
    assert manifest["generation"]["positive_prompt"].startswith("a beagle dog")
    assert manifest["generation"]["text_description"] == "beagle dog"
    assert manifest["appearance"]["dominant_colors"] == []
    assert manifest["appearance"]["color_measurement_status"] == "pending"
    assert manifest["review"]["overall_status"] == "needs_review"
    assert manifest["review"]["direction_status"] == "needs_review"
    assert manifest["audio"]["default_lookup"] == "dog_bark"
    assert manifest["visual_assets"]["mesh_original"].endswith("mesh.obj")


def test_write_hy3d_candidate_manifest(tmp_path):
    tag_dir = tmp_path / "pending" / "cat_british_shorthair_v2"
    tag_dir.mkdir(parents=True)
    (tag_dir / "mesh.obj").write_text("v 0 0 0\n")

    path = write_hy3d_candidate_manifest(
        tag_dir,
        tag="cat_british_shorthair_v2",
        species="cat",
        breed="british shorthair",
        seed=4102,
        positive_prompt="a british shorthair cat",
        created_at="2026-07-09T00:00:00+00:00",
    )

    assert path == tag_dir / CANDIDATE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    assert manifest["asset_id"] == "cat_british_shorthair_0002"
    assert manifest["family"] == "british_shorthair"
    assert manifest["audio"]["default_lookup"] == "cat_purring"


def test_sync_candidate_manifest_review_updates_direction_status(tmp_path):
    tag_dir = tmp_path / "approved" / "dog_beagle_v2"
    tag_dir.mkdir(parents=True)
    path = write_hy3d_candidate_manifest(
        tag_dir,
        tag="dog_beagle_v2",
        species="dog",
        breed="beagle",
        seed=4101,
        positive_prompt="a beagle dog",
        created_at="2026-07-09T00:00:00+00:00",
    )
    direction = {
        "human_approved": True,
        "human_approved_by": "jzy",
        "human_approved_at": "2026-07-09T01:02:03+00:00",
        "mesh_oriented": str((tag_dir / "mesh_oriented.glb").resolve()),
    }

    updated = sync_candidate_manifest_review(tag_dir, direction)

    assert updated == path
    manifest = json.loads(path.read_text())
    assert manifest["review"]["direction_status"] == "approved"
    assert manifest["review"]["overall_status"] == "needs_runtime_gate"
    assert manifest["review"]["approved_by"] == "jzy"
    assert manifest["visual_assets"]["mesh_oriented"].endswith("mesh_oriented.glb")
