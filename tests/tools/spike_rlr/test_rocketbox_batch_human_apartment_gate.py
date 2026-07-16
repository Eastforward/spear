import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ready_batch_avatar(
    tmp_path: Path,
    *,
    avatar_id: str = "rocketbox_children_female_child_01",
    legacy_asset_id: str = "rocketbox_female_child_01",
    category: str = "Children",
    demographic: str = "child",
    gender: str = "female",
    authored_height_cm: float = 143.3028,
    ue_height_cm: float = 142.9338,
    height_range_cm: list[float] | None = None,
    ceiling_cm: float = 280.0,
    minimum_headroom_cm: float = 25.0,
):
    height_range_cm = height_range_cm or (
        [80.0, 170.0] if demographic == "child" else [140.0, 215.0]
    )
    tag = f"{avatar_id}_original_ue_v1"
    runtime_root = tmp_path / "rocketbox_batch_native_runtime_ue_v1"
    runtime_dir = runtime_root / tag
    runtime_dir.mkdir(parents=True)
    runtime_glb = runtime_dir / "runtime.glb"
    runtime_glb.write_bytes(b"batch-native-runtime")
    source_manifest = {
        "schema": "rocketbox_batch_native_ue_runtime_v1",
        "tag": tag,
        "base_avatar_id": avatar_id,
        "asset_id": legacy_asset_id,
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "demographic": demographic,
        "gender": gender,
        "runtime_glb": {
            "filename": "runtime.glb",
            "size_bytes": runtime_glb.stat().st_size,
            "sha256": _sha256(runtime_glb),
        },
        "normalization": {
            "schema": "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1",
            "normalized_joint_count": 80,
            "in_place_actions": ["Walking"],
            "root_motion": {
                "Walking": {
                    "maximum_horizontal_deviation_after_m": 1.0e-8,
                    "maximum_vertical_world_error_m": 1.0e-8,
                }
            },
        },
        "runtime_motion_contract": {
            "walking_embedded_horizontal_root_motion": "removed",
            "walking_vertical_motion": "preserved",
            "dynamic_ground_snap_to_floor_required": True,
        },
        "expected_ue_qa": {
            "actor_scale": 1.0,
            "authored_height_cm": authored_height_cm,
            "authored_height_preserved": True,
            "height_range_cm": height_range_cm,
            "apartment_ceiling_cm": ceiling_cm,
            "ceiling_headroom_cm": ceiling_cm - authored_height_cm,
            "demographic": demographic,
            "mouth_audio_height_cm": (
                authored_height_cm * (0.88 if demographic == "child" else 0.90)
            ),
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": 15.0,
        },
        "automatic_checks": {"overall": "passed"},
    }
    source_manifest_path = _write_json(
        runtime_dir / "normalization_manifest.json", source_manifest
    )

    import_root = tmp_path / "rocketbox_batch_native_ue_import_v1"
    import_dir = import_root / tag
    import_manifest = {
        "schema": "rocketbox_batch_native_ue_import_v1",
        "tag": tag,
        "base_avatar_id": avatar_id,
        "asset_id": legacy_asset_id,
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "source_glb": str(runtime_glb.resolve()),
        "source_glb_sha256": _sha256(runtime_glb),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "reload_verification": {"status": "passed"},
        "runtime_contract": {
            "actor_scale": 1.0,
            "bone_count": 80,
            "bounds": {
                "height_cm": ue_height_cm,
                "authored_height_cm": authored_height_cm,
                "authored_height_delta_cm": abs(ue_height_cm - authored_height_cm),
                "authored_height_tolerance_cm": max(3.0, authored_height_cm * 0.02),
                "authored_height_preserved": True,
                "height_range_cm": height_range_cm,
                "height_passed": True,
                "ground_passed": True,
            },
        },
        "content": {
            "animations": {
                "Walking": f"/Game/Meshes/gate_{tag}/Walking.Walking",
                "Standing_Idle": f"/Game/Meshes/gate_{tag}/Standing_Idle.Standing_Idle",
            },
            "blueprint": f"/Game/Blueprints/gate_{tag}/BP_gate_{tag}",
        },
    }
    _write_json(import_dir / "ue_import_manifest.json", import_manifest)

    inventory_path = _write_json(
        tmp_path / "rocketbox_route1_inventory_v1" / "inventory.json",
        {
            "schema_version": "rocketbox_human_inventory_v1",
            "population": {"total": 1},
            "automatic_checks": {"overall": "passed"},
            "apartment_height_policy": {
                "actor_scale": 1.0,
                "ceiling_cm": ceiling_cm,
                "minimum_headroom_cm": minimum_headroom_cm,
                "authored_height_preserved": True,
            },
            "avatars": [
                {
                    "base_avatar_id": avatar_id,
                    "legacy_asset_id": legacy_asset_id,
                    "category": category,
                    "demographic": demographic,
                    "gender": gender,
                    "inventory_status": "passed",
                    "height_contract": {
                        "status": "passed",
                        "actor_scale": 1.0,
                        "authored_height_cm": authored_height_cm,
                        "allowed_height_cm": height_range_cm,
                        "apartment_ceiling_cm": ceiling_cm,
                        "minimum_ceiling_headroom_cm": minimum_headroom_cm,
                        "ceiling_headroom_cm": ceiling_cm - authored_height_cm,
                        "mouth_audio_height_cm": source_manifest["expected_ue_qa"][
                            "mouth_audio_height_cm"
                        ],
                    },
                }
            ],
        },
    )
    formal_registry_root = tmp_path / "source_assets_v1"
    formal_registry_root.mkdir()
    return (
        tag,
        runtime_root,
        import_root,
        inventory_path,
        formal_registry_root,
    )


def test_batch_gate_accepts_child_at_authored_height_and_exports_room_contract(tmp_path):
    from human_apartment_gate import assert_batch_native_rocketbox_apartment_ready

    tag, runtime_root, import_root, inventory, registry = _ready_batch_avatar(
        tmp_path
    )

    evidence = assert_batch_native_rocketbox_apartment_ready(
        tag,
        runtime_root=runtime_root,
        ue_import_root=import_root,
        inventory_path=inventory,
        formal_registry_root=registry,
    )

    assert evidence["tag"] == tag
    assert evidence["demographic"] == "child"
    assert evidence["gender"] == "female"
    assert evidence["actor_scale"] == 1.0
    assert evidence["authored_height_cm"] == pytest.approx(143.3028)
    assert evidence["height_cm"] == pytest.approx(142.9338)
    assert evidence["apartment_ceiling_cm"] == 280.0
    assert evidence["ceiling_headroom_cm"] > 25.0
    assert evidence["audio_source_height_m"] == pytest.approx(1.26106464)
    assert set(evidence["animations"]) == {"Walking", "Standing_Idle"}


def test_batch_gate_rejects_avatar_without_apartment_headroom(tmp_path):
    from human_apartment_gate import (
        HumanApartmentGateError,
        assert_batch_native_rocketbox_apartment_ready,
    )

    tag, runtime_root, import_root, inventory, registry = _ready_batch_avatar(
        tmp_path,
        avatar_id="rocketbox_adults_male_too_tall",
        legacy_asset_id="rocketbox_male_too_tall",
        category="Adults",
        demographic="adult",
        gender="male",
        authored_height_cm=265.0,
        ue_height_cm=264.5,
        height_range_cm=[140.0, 300.0],
    )

    with pytest.raises(HumanApartmentGateError, match="headroom"):
        assert_batch_native_rocketbox_apartment_ready(
            tag,
            runtime_root=runtime_root,
            ue_import_root=import_root,
            inventory_path=inventory,
            formal_registry_root=registry,
        )


def test_batch_tag_detection_is_structural_but_gate_remains_inventory_locked():
    from human_apartment_gate import is_batch_native_rocketbox_human_candidate

    assert is_batch_native_rocketbox_human_candidate(
        "rocketbox_adults_female_adult_01_original_ue_v1"
    )
    assert is_batch_native_rocketbox_human_candidate(
        "rocketbox_professions_military_male_01_original_ue_v1"
    )
    assert not is_batch_native_rocketbox_human_candidate(
        "rocketbox_male_adult_01_original_ue_v3"
    )
    assert not is_batch_native_rocketbox_human_candidate("../../escape")


def test_apartment_runner_routes_batch_tags_through_inventory_gate():
    runner = (
        REPO / "tools" / "spike_rlr" / "run_render_pass_apartment.py"
    ).read_text(encoding="utf-8")

    assert "is_batch_native_rocketbox_human_candidate" in runner
    assert "assert_batch_native_rocketbox_apartment_ready" in runner
    assert "elif is_batch_native_rocketbox_human_candidate(tag):" in runner
