from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[3] / "tools" / "spike_rlr"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _module():
    return importlib.import_module("second_retarget_facing_review")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, external: bool = False) -> dict:
    result = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    result["path" if external else "filename"] = (
        str(path.resolve()) if external else path.name
    )
    return result


def _write_glb(path: Path, action_name: str = "Walking") -> None:
    document = {
        "asset": {"version": "2.0"},
        "animations": [{"name": action_name, "channels": [], "samplers": []}],
        "meshes": [{"primitives": [{}]}],
        "skins": [{"joints": [0]}],
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, 12 + 8 + len(payload))
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


@pytest.fixture()
def second_attempt(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    static_qa = upstream / "static_qa.json"
    semantic = {
        "pelvis": "bone_0",
        "left_clavicle": "bone_6",
        "right_clavicle": "bone_25",
        "left_thigh": "bone_44",
        "right_thigh": "bone_48",
    }
    static_qa.write_text(
        json.dumps(
            {
                "schema": "tokenrig_human_static_qa_v1",
                "asset_id": "rocketbox_male_adult_01",
                "decision": "automatic_static_checks_passed",
                "checks": {
                    "axis_canonicalization": {
                        "canonical_front": "negative-y",
                        "transform_count": 1,
                    },
                    "semantic_mapping": {"semantic_bones": semantic},
                },
            }
        ),
        encoding="utf-8",
    )
    failure = upstream / "retarget_failure.json"
    failure.write_text(
        json.dumps(
            {
                "schema": "tokenrig_rocketbox_retarget_attempt_v1",
                "asset_id": "rocketbox_male_adult_01",
                "decision": "rejected",
                "readiness_bundle_published": False,
                "preserved_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "second_attempt_rotation_only_diagnostic_reconstruction_v1"
    diagnostic.mkdir()
    glb = diagnostic / "walking_rotation_only_reconstruction.glb"
    _write_glb(glb)
    artifacts = {glb.name: _record(glb)}
    for view in ("front", "side", "feet"):
        for suffix in ("mp4", "png"):
            path = diagnostic / f"walking_{view}.{suffix}"
            path.write_bytes(f"{view}:{suffix}".encode("ascii"))
            artifacts[path.name] = _record(path)
    manifest = {
        "schema": "second_attempt_rotation_only_diagnostic_reconstruction_v1",
        "asset_id": "rocketbox_male_adult_01",
        "classification": "technical_diagnostic_only",
        "decision": "rejected_attempt_visualized_by_nonformal_reconstruction",
        "formal_dataset_asset": False,
        "readiness_bundle_published": False,
        "automatic_checks": "diagnostic_reconstruction_integrity_passed",
        "user_approval": "not_requested_for_diagnostic_reconstruction",
        "reconstruction_notice": {"is_original_second_attempt_artifact": False},
        "bound_second_failure": _record(failure, external=True),
        "authenticated_inputs": {
            "static": {
                "canonical_front": "negative-y",
                "static_qa": _record(static_qa, external=True),
                "semantic_mapping": {"semantic_bones": semantic},
            }
        },
        "motion": {"action_name": "Walking", "fps": 30, "frame_count": 33},
        "artifacts": artifacts,
    }
    (diagnostic / "diagnostic_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return diagnostic


def _bind_points() -> dict[str, tuple[float, float, float]]:
    return {
        "pelvis": (0.0, 0.0, 1.0),
        "left_clavicle": (-0.3, 0.0, 1.7),
        "right_clavicle": (0.3, 0.0, 1.7),
        "left_thigh": (-0.15, 0.0, 0.9),
        "right_thigh": (0.15, 0.0, 0.9),
    }


def _frame(y: float = 0.0, x: float = 0.0) -> dict[str, tuple[float, float, float]]:
    return {
        role: (point[0] + x, point[1] + y, point[2])
        for role, point in _bind_points().items()
    }


def test_alignment_classes_are_independent_of_travel_sign_authentication():
    review = _module()
    assert review.classify_alignment(0.99) == "aligned"
    assert review.classify_alignment(0.05) == "sideways"
    assert review.classify_alignment(-0.99) == "reversed"
    assert review.classify_alignment(None) == "travel_undefined"


@pytest.mark.parametrize(
    ("frames", "expected", "expected_dot"),
    [
        ([_frame(y=0.0), _frame(y=-0.1), _frame(y=-0.2)], "aligned", 1.0),
        ([_frame(x=0.0), _frame(x=0.1), _frame(x=0.2)], "sideways", 0.0),
        ([_frame(y=0.0), _frame(y=0.1), _frame(y=0.2)], "reversed", -1.0),
    ],
)
def test_facing_is_signed_from_bind_front_not_from_travel(frames, expected, expected_dot):
    result = _module().compute_facing_samples(_bind_points(), frames, fps=30)
    assert result["bind_authentication"]["canonical_front"] == [0.0, -1.0, 0.0]
    assert result["bind_authentication"]["dot"] == pytest.approx(1.0)
    assert {sample["classification"] for sample in result["frames"]} == {expected}
    assert result["summary"]["median_body_travel_dot"] == pytest.approx(expected_dot)


def test_zero_root_displacement_is_undefined_not_fabricated_alignment():
    result = _module().compute_facing_samples(
        _bind_points(), [_frame(), _frame(), _frame()], fps=30
    )
    assert all(sample["travel_direction"] is None for sample in result["frames"])
    assert all(sample["body_travel_dot"] is None for sample in result["frames"])
    assert result["summary"] == {
        "valid_travel_frame_count": 0,
        "undefined_travel_frame_count": 3,
        "median_body_travel_dot": None,
        "worst_body_travel_dot": None,
        "reversed_frame_ratio": None,
        "sideways_frame_ratio": None,
        "overall_classification": "travel_undefined",
    }


def test_authenticates_exact_rejected_second_attempt(second_attempt: Path):
    result = _module().authenticate_second_attempt(second_attempt)
    assert result["asset_id"] == "rocketbox_male_adult_01"
    assert result["canonical_front"] == "negative-y"
    assert result["motion"] == {"action_name": "Walking", "fps": 30, "frame_count": 33}
    assert result["semantic_bones"]["pelvis"] == "bone_0"
    assert set(result["media"]) == {"front", "side", "feet"}


def test_stage_specific_semantic_audit_metadata_may_differ_when_core_bones_match(
    second_attempt: Path,
):
    manifest_path = second_attempt / "diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["authenticated_inputs"]["static"]["semantic_mapping"][
        "runtime_only_audit"
    ] = ["bone_6", "bone_25"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _module().authenticate_second_attempt(second_attempt)
    assert result["semantic_bones"]["left_clavicle"] == "bone_6"


def test_rejects_changed_source_media(second_attempt: Path):
    (second_attempt / "walking_side.mp4").write_bytes(b"tampered")
    with pytest.raises(Exception, match="walking_side.mp4.*changed"):
        _module().authenticate_second_attempt(second_attempt)


def test_rejects_formal_or_user_approved_claim(second_attempt: Path):
    manifest_path = second_attempt / "diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["formal_dataset_asset"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="formal dataset asset"):
        _module().authenticate_second_attempt(second_attempt)


def test_rejects_extra_or_wrong_glb_action(second_attempt: Path):
    glb = second_attempt / "walking_rotation_only_reconstruction.glb"
    _write_glb(glb, "Standing_Idle")
    manifest_path = second_attempt / "diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][glb.name] = _record(glb)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="exactly one Walking animation"):
        _module().authenticate_second_attempt(second_attempt)


def _metrics_payload() -> dict:
    frames = []
    for frame in range(1, 34):
        frames.append(
            {
                "frame": frame,
                "body_travel_dot": 0.95,
                "body_travel_signed_angle_deg": 18.0,
                "classification": "aligned",
                "travel_speed_mps": 0.8,
            }
        )
    return {
        "schema": "second_retarget_facing_metrics_v1",
        "fps": 30,
        "frame_count": 33,
        "bind_authentication": {
            "dot": 1.0,
            "sign_selected_without_travel": True,
        },
        "frames": frames,
        "summary": {
            "median_body_travel_dot": 0.95,
            "worst_body_travel_dot": 0.90,
            "overall_classification": "aligned",
        },
    }


def test_review_html_is_four_view_synchronized_and_human_authoritative():
    html = _module().build_review_html(_metrics_payload()).decode("utf-8")
    for label in ("正面 Front", "侧面 Side", "脚部 Feet", "俯视朝向 Top + arrows"):
        assert label in html
    for route in ("/media/front", "/media/side", "/media/feet", "/media/top"):
        assert route in html
    for token in (
        'id="master-toggle"',
        'data-step="-1"',
        'data-step="1"',
        'id="playback-rate"',
        "currentTime",
        "0.5 / FPS",
        "body_travel_signed_angle_deg",
        "body_travel_dot",
        "localStorage",
        "sideways",
        "reversed",
        "aligned_but_deformed",
        "第二次 retarget 已拒绝",
        "最终视觉判断由你决定",
        "蓝色：身体前向",
        "红色：root 位移方向",
        "灰色：FRONT -Y",
    ):
        assert token in html
    assert "<form" not in html
    assert "method=\"post\"" not in html.lower()
    assert "Approve" not in html
    assert ".fbx" not in html.lower()


def test_review_html_rejects_malformed_or_non_33_frame_metrics():
    metrics = _metrics_payload()
    metrics["frame_count"] = 32
    with pytest.raises(Exception, match="exactly 33 frames"):
        _module().build_review_html(metrics)


def _gait_frame(*, phase: float, sideways: bool) -> dict:
    value = _frame(y=-phase * 0.1)
    for side, hip_x in (("left", -0.15), ("right", 0.15)):
        sign = -1.0 if side == "left" else 1.0
        hip = (hip_x, -phase * 0.1, 0.9)
        if sideways:
            knee = (hip_x + sign * (0.10 + phase * 0.08), -phase * 0.1, 0.5)
            ankle = (hip_x + sign * phase * 0.24, -phase * 0.1, 0.1)
        else:
            knee = (hip_x, -phase * 0.1 - 0.10 - phase * 0.08, 0.5)
            ankle = (hip_x, -phase * 0.1 - phase * 0.24, 0.1)
        value[f"{side}_thigh"] = hip
        value[f"{side}_calf"] = knee
        value[f"{side}_foot"] = ankle
    return value


def test_gait_plane_distinguishes_forward_walk_from_sideways_leg_swing():
    review = _module()
    forward = review.compute_gait_plane_samples(
        [_gait_frame(phase=value, sideways=False) for value in (0.0, 0.5, 1.0)],
        fps=30,
    )
    sideways = review.compute_gait_plane_samples(
        [_gait_frame(phase=value, sideways=True) for value in (0.0, 0.5, 1.0)],
        fps=30,
    )
    for side in ("left", "right"):
        assert forward["legs"][side]["lateral_to_forward_excursion_ratio"] < 0.2
        assert forward["legs"][side]["mean_knee_normal_dot_lateral_abs"] > 0.95
        assert sideways["legs"][side]["lateral_to_forward_excursion_ratio"] > 4.0
        assert sideways["legs"][side]["mean_knee_normal_dot_forward_abs"] > 0.95
    assert forward["overall_classification"] == "sagittal_forward_gait"
    assert sideways["overall_classification"] == "sideways_leg_swing"


def test_gait_plane_requires_bilateral_hip_knee_ankle_semantics():
    frame = _gait_frame(phase=0.0, sideways=False)
    del frame["left_calf"]
    with pytest.raises(Exception, match="left_calf"):
        _module().compute_gait_plane_samples([frame, frame], fps=30)
