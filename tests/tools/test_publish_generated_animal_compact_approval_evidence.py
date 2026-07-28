import hashlib
import json

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import prepare_user_approved_generated_animal_ue_imports as bridge
from tools import publish_generated_animal_compact_approval_evidence as subject


ASSET_ID = "dog_fixture_pixel3d_v1"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path, **metadata):
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        **metadata,
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exact_v4_review(tmp_path, *, asset_id=ASSET_ID):
    animated_glb = tmp_path / "animated.glb"
    animated_glb.write_bytes(b"current Pixel3D animated GLB")
    idle_side = tmp_path / "idle_side.mp4"
    idle_side.write_bytes(b"current Pixel3D Idle side readback")
    walking_side = tmp_path / "walking_side.mp4"
    walking_side.write_bytes(b"current Pixel3D Walking side readback")
    lineage = {"asset_id": asset_id}
    payload = {
        "schema": subject.EXACT_V4_REVIEW_SCHEMA,
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "automatic_admission_gates": {
            "all_automatic_gates_passed": True,
        },
        "inputs": {"target_rig_lineage": lineage},
        "forward_contract": {"target_rig_lineage": lineage},
        "outputs": {
            "animated_glb": _record(animated_glb),
            "media": {
                "idle_side": _record(
                    idle_side,
                    codec="h264",
                    human_review_repeated=False,
                ),
                "walking_side": _record(
                    walking_side,
                    codec="h264",
                    human_review_repeated=False,
                ),
            },
        },
    }
    review = tmp_path / "review_run.json"
    _write_json(review, payload)
    return {
        "path": review,
        "payload": payload,
        "animated_glb": animated_glb,
        "idle_side": idle_side,
        "walking_side": walking_side,
    }


def test_publishes_canonical_motion_style_and_current_readback(tmp_path):
    style_video = tmp_path / "approved_style.mp4"
    style_video.write_bytes(b"owner-approved reusable Idle/Walking style")
    style_output = tmp_path / "motion_style_approval.json"
    subject.publish_motion_style_approval(
        style_video=style_video,
        output=style_output,
    )
    style = contracts.load_json(style_output)
    assert set(style) == {
        "schema",
        "status",
        "actions",
        "evidence_video",
        "approval_sha256",
    }
    assert style == {
        "schema": subject.MOTION_STYLE_APPROVAL_SCHEMA,
        "status": "approved_for_idle_walking_motion_style",
        "actions": ["Idle", "Walking"],
        "evidence_video": _record(style_video),
        "approval_sha256": style["approval_sha256"],
    }
    assert style["approval_sha256"] == subject._hash_without(
        style,
        "approval_sha256",
    )
    _style_path, loaded_style, _style_descriptor, loaded_video = (
        bridge.load_motion_style_approval(
            style_output,
            expected_file_sha256=_sha256(style_output),
        )
    )
    assert loaded_style == style
    assert loaded_video == style_video.resolve()

    review = _exact_v4_review(tmp_path)
    readback_output = tmp_path / "current_asset_short_readback.json"
    subject.publish_current_asset_short_readback(
        review=review["path"],
        asset_id=ASSET_ID,
        output=readback_output,
    )
    readback = contracts.load_json(readback_output)
    assert set(readback) == {
        "schema",
        "status",
        "asset_id",
        "animation_review",
        "reviewed_animated_glb",
        "actions",
        "action_readbacks",
        "checks",
        "receipt_sha256",
    }
    assert readback["schema"] == subject.CURRENT_ASSET_SHORT_READBACK_SCHEMA
    assert readback["status"] == "passed_current_asset_geometry_and_actions"
    assert readback["asset_id"] == ASSET_ID
    assert readback["animation_review"] == _record(review["path"])
    assert readback["reviewed_animated_glb"] == _record(
        review["animated_glb"]
    )
    assert readback["actions"] == ["Idle", "Walking"]
    assert readback["action_readbacks"] == {
        "Idle": _record(review["idle_side"]),
        "Walking": _record(review["walking_side"]),
    }
    assert readback["checks"] == subject.CURRENT_ASSET_SHORT_READBACK_CHECKS
    assert readback["receipt_sha256"] == subject._hash_without(
        readback,
        "receipt_sha256",
    )
    _receipt_path, loaded_readback, _receipt_descriptor, loaded_videos = (
        bridge.load_current_asset_short_readback(
            readback_output,
            expected_file_sha256=_sha256(readback_output),
            expected_asset_id=ASSET_ID,
            review_path=review["path"],
            review_payload=review["payload"],
            reviewed_animated_glb=review["animated_glb"],
        )
    )
    assert loaded_readback == readback
    assert loaded_videos == {
        "Idle": review["idle_side"].resolve(),
        "Walking": review["walking_side"].resolve(),
    }


def test_current_readback_rejects_tampered_reviewed_media(tmp_path):
    review = _exact_v4_review(tmp_path)
    review["idle_side"].write_bytes(b"tampered after exact v4 review")
    output = tmp_path / "current_asset_short_readback.json"
    with pytest.raises(
        subject.EvidenceError,
        match=r"idle_side readback changed",
    ):
        subject.publish_current_asset_short_readback(
            review=review["path"],
            asset_id=ASSET_ID,
            output=output,
        )
    assert not output.exists()


def test_current_readback_rejects_wrong_asset_id(tmp_path):
    review = _exact_v4_review(tmp_path)
    output = tmp_path / "current_asset_short_readback.json"
    with pytest.raises(
        subject.EvidenceError,
        match=r"asset_id does not match --asset-id",
    ):
        subject.publish_current_asset_short_readback(
            review=review["path"],
            asset_id="dog_other_pixel3d_v1",
            output=output,
        )
    assert not output.exists()


def test_publish_refuses_to_replace_existing_artifact(tmp_path):
    style_video = tmp_path / "approved_style.mp4"
    style_video.write_bytes(b"owner-approved style")
    output = tmp_path / "motion_style_approval.json"
    output.write_bytes(b"preserve me")
    with pytest.raises(contracts.ContractError, match=r"refusing to replace"):
        subject.publish_motion_style_approval(
            style_video=style_video,
            output=output,
        )
    assert output.read_bytes() == b"preserve me"
