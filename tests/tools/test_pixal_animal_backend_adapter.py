import json
from pathlib import Path

from tools.pixal_animal_backend_adapter import (
    DINO_REVISION,
    FLUX1_REVISION,
    PIXAL_REVISION,
    build_candidate_manifest,
    build_pixal_command,
    validate_reference_lineage,
)


def _old_candidate(tmp_path: Path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"rgba-reference")
    manifest = {
        "schema_version": "source_asset_v1",
        "asset_id": "dog_pug_0001",
        "legacy_tag": "dog_pug_v1",
        "asset_class": "animal",
        "category": "dog",
        "family": "pug",
        "generation": {
            "source_pipeline": "flux+hunyuan3d",
            "model": "flux_dev+hunyuan3d-2.1",
            "seed": 5101,
            "positive_prompt": "a pug dog in perfect side profile view",
            "negative_prompt": None,
            "text_description": "pug dog",
            "created_at": "2026-07-09T09:34:42+00:00",
        },
        "visual_assets": {"reference_image": str(reference)},
        "rig": {
            "skeleton_family": "quaternius_dog",
            "animations": ["Idle", "Walking"],
            "loop_required": True,
        },
        "audio": {
            "default_lookup": "dog_bark",
            "allowed_lookups": ["dog_bark", "silent"],
        },
    }
    path = tmp_path / "source_asset_candidate.json"
    path.write_text(json.dumps(manifest))
    return path, reference, manifest


def test_pixal_command_reuses_pinned_wrapper_and_never_invokes_hunyuan(tmp_path):
    reference = tmp_path / "reference.png"
    output = tmp_path / "pixal_raw.glb"

    command = build_pixal_command(
        reference=reference,
        output=output,
        gpu=2,
        seed=5101,
    )

    assert command[0].endswith("envs/avengine-3dgen/bin/python3.10")
    assert command[1].endswith("tools/i23d_human_bakeoff.py")
    assert command[2:4] == ["--backend", "pixal3d"]
    assert "--low-vram" in command
    assert command[command.index("--manual-fov") + 1] == "0.2"
    assert command[command.index("--seed") + 1] == "5101"
    assert not any("hunyuan" in item.lower() for item in command)


def test_reference_lineage_resolves_historical_flux1_revision(tmp_path):
    path, reference, _manifest = _old_candidate(tmp_path)

    lineage = validate_reference_lineage(path, reference)

    assert lineage["generator"] == "black-forest-labs/FLUX.1-dev"
    assert lineage["revision"] == FLUX1_REVISION
    assert lineage["seed"] == 5101
    assert lineage["reference"]["sha256"]


def test_pixal_candidate_keeps_embedded_pbr_and_quaternius_downstream(tmp_path):
    old_path, reference, old = _old_candidate(tmp_path)
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"packed-pbr-glb")
    generated = tmp_path / "pixal_raw.manifest.json"
    generated.write_text("{}")
    lineage = validate_reference_lineage(old_path, reference)

    candidate = build_candidate_manifest(
        tag="dog_pug_pixal_canary_v1",
        tag_dir=tmp_path,
        old_candidate=old,
        reference_lineage=lineage,
        mesh=mesh,
        generated_manifest=generated,
        pbr_readback={
            "passed": True,
            "mesh_count": 1,
            "material_count": 1,
            "texture_count": 2,
            "image_count": 2,
        },
    )

    assert candidate["usage_scope"] == "research_candidate"
    assert candidate["formal_registration_authorized"] is False
    assert candidate["generation"]["source_pipeline"] == (
        "historical_flux1_reference+pixal3d"
    )
    assert candidate["generation"]["models"]["pixal3d"]["revision"] == PIXAL_REVISION
    assert candidate["generation"]["models"]["dino"]["revision"] == DINO_REVISION
    assert candidate["generation"]["models"]["reference_generator"]["revision"] == FLUX1_REVISION
    assert candidate["visual_assets"]["mesh_original"].endswith("mesh.glb")
    assert candidate["visual_assets"]["diffuse"] is None
    assert candidate["appearance"]["color_measurement_status"] == (
        "pending_pixal_pbr_measurement"
    )
    assert candidate["appearance"]["historical_hunyuan_texture_measurements_reused"] is False
    assert candidate["material_contract"]["embedded_pbr_preserved"] is True
    assert candidate["material_contract"]["new_diffuse_override_allowed"] is False
    assert candidate["rig"] == old["rig"]
    assert candidate["audio"] == old["audio"]
    assert candidate["review"]["overall_status"] == "needs_review"
