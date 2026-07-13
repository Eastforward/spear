"""Static contract for the new-canary manual cardinal review manifest."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/build_controlled_animal_pose_canary_review.py"
)


def test_canary_builder_requires_authenticated_parallel_evidence_bundles():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--registry", action="append"' in text
    assert 'parser.add_argument("--lod-binding-batch", action="append"' in text
    assert 'parser.add_argument("--animation-review-batch", action="append"' in text
    assert 'parser.add_argument("--animation-decisions", action="append"' in text
    assert 'parser.add_argument("--apartment-spec-manifest", action="append"' in text
    assert "passed_lod_binding_glb_readback" in text
    assert "rendered_pending_visual_qa" in text
    assert 'APARTMENT_SPEC_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"' in text
    assert 'APARTMENT_STATUS_SCHEMA = "controlled_animal_apartment_render_status_v1"' in text
    assert 'status.get("current_failures") != []' in text
    assert 'status.get("incomplete_jobs") != []' in text


def test_canary_builder_keeps_manual_cardinal_gate_and_agent_status_separate():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"automatic_orientation_inference": "disabled"' in text
    assert '"hidden_reflection_or_mirror": "forbidden"' in text
    assert '"allowed_yaw_degrees": [-90, 0, 90, 180]' in text
    assert "new_canary_animation_agent_approved_pending_human_review" in text
    assert "new_canary_animation_rejected" in text
    assert '"browser_decisions_are_transform_overlays_only": True' in text


def test_canary_builder_attaches_authenticated_apartment_walk_idle_media():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"side_by_side_review_annotated.mp4"' in text
    assert '"apartment_v1_view0.mp4"' in text
    assert '"topdown_review.mp4"' in text
    assert 'media[f"apartment_{action.lower()}_{view}"]' in text
    assert 'f"Apartment media attached to rejected animation: {asset_id}"' in text
