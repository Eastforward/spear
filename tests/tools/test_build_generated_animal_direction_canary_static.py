from pathlib import Path

import pytest

from tools import build_generated_animal_direction_canary as canary


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_generated_animal_direction_canary.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_canary_disables_inference_and_never_authorizes_binding():
    text = source()
    assert '"automatic_orientation_inference": "disabled"' in text
    assert '"allowed_yaw_degrees": [-90, 0, 90, 180]' in text
    assert '"binding_pretransform": "not_authorized_by_this_visual_gate"' in text
    assert '"direction_approval_does_not_authorize_binding": True' in text


def test_canary_authenticates_every_artifact_and_refuses_overwrite():
    text = source()
    assert '"prebind_lod_glb": record(args.direction_preview_glb)' in text
    assert '"static_contact_sheet": record(args.static_contact_sheet)' in text
    assert '"static_top_view": record(args.static_top_view)' in text
    assert "refusing to replace output" in text
    assert "manifest hash readback failed" in text


def test_canary_marks_preview_mesh_as_preview_only():
    text = source()
    assert '"direction_preview_only": True' in text
    assert '"runtime_topology": "separate_strict_gate_in_progress"' in text
    assert '"target_animation_generated_for_this_decision": False' in text


def test_canary_can_opt_into_two_stage_manual_yaw_without_auto_inference():
    text = source()
    assert '"controlled_animal_pose_direction_manual_review_manifest_v3"' in text
    assert '"--manual-residual-yaw"' in text
    assert '"step_degrees": [1, 5, 15]' in text
    assert '"range_degrees": [-45, 45]' in text
    assert '"visual_authority": "torso_and_spine_longitudinal_axis"' in text
    assert '"must_not_follow": "turned_head_or_tail"' in text


def test_declared_view_alignment_uses_profile_yaw_and_geometry_only_as_gate(
    tmp_path, monkeypatch
):
    mesh = tmp_path / "one_shot.glb"
    mesh.write_bytes(b"immutable one-shot mesh")
    monkeypatch.setattr(
        canary.audit_quadruped_i23d_geometry,
        "audit",
        lambda *_args, **_kwargs: {
            "torso_midline": {
                "sensitivity_global_axis_yaw_degrees": [31.2, 32.1, 31.3]
            }
        },
    )

    result = canary.declared_view_canonicalization_audit(
        mesh,
        canonicalization_yaw_deg=30,
        maximum_residual_deg=3,
    )

    assert result["declared_canonicalization_yaw_deg"] == 30
    assert result["postcanonical_residual_yaw_degrees"] == pytest.approx(
        [1.2, 2.1, 1.3]
    )
    assert result["status"] == "passed_declared_view_canonicalization"
    assert result["applied_yaw_was_inferred_from_geometry"] is False
    assert result["single_attempt_failure_requires_new_profile_not_seed_retry"] is True


def test_declared_view_alignment_rejects_residual_without_seed_retry(
    tmp_path, monkeypatch
):
    mesh = tmp_path / "one_shot.glb"
    mesh.write_bytes(b"immutable one-shot mesh")
    monkeypatch.setattr(
        canary.audit_quadruped_i23d_geometry,
        "audit",
        lambda *_args, **_kwargs: {
            "torso_midline": {
                "sensitivity_global_axis_yaw_degrees": [30.5, 36.0, 31.0]
            }
        },
    )

    with pytest.raises(canary.CanaryError, match="single-attempt|one-shot"):
        canary.declared_view_canonicalization_audit(
            mesh,
            canonicalization_yaw_deg=30,
            maximum_residual_deg=3,
        )
