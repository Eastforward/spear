from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import generated_asset_emitter_contract as contract


REPO = Path(__file__).resolve().parents[2]
BLENDER = Path("/data/jzy/.local/bin/blender")
FIXTURE = REPO / "tests/fixtures/blender_make_asymmetric_static_fixture.py"
FINALIZER = REPO / "tools/blender_finalize_generated_static_object.py"


def _record(path: Path):
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


@pytest.mark.skipif(not BLENDER.is_file(), reason="pinned Blender is unavailable")
def test_finalizer_rotates_scales_grounds_and_reimports_static_fixture(tmp_path):
    watertight_glb = tmp_path / "watertight.glb"
    fixture = subprocess.run(
        [
            str(BLENDER),
            "-b",
            "--python-exit-code",
            "2",
            "--python",
            str(FIXTURE),
            "--",
            "--output",
            str(watertight_glb),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert fixture.returncode == 0, fixture.stdout + fixture.stderr

    source_record = _record(watertight_glb)
    watertight_manifest = tmp_path / "watertight_manifest.json"
    watertight_manifest.write_text(
        json.dumps(
            {
                "schema": "avengine_watertight_textured_runtime_proxy_v1",
                "status": "research_candidate_pending_static_and_animation_qa",
                "input": source_record,
                "output": source_record,
                "topology": {
                    "final": {
                        "boundary_edges": 0,
                        "wire_edges": 0,
                        "nonmanifold_edges_over_two_faces": 0,
                    }
                },
                "authority_contract": {
                    "approved_skeleton_or_animation_touched": False
                },
                "formal_dataset_registration_authorized": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    decision_path = tmp_path / "static_decision.json"
    decision = {
        "schema": "avengine_controlled_static_object_decision_v1",
        "instance_id": "asymmetric_static_fixture_0001",
        "request_sha256": "1" * 64,
        "profile_sha256": "2" * 64,
        "target_physical_profile": {
            "control_attribute": None,
            "measurement": "height_cm",
            "target_value_cm": 200.0,
            "tolerance_cm": 0.1,
        },
        "pixal_output": source_record,
        "decision": "approved_for_watertight_finalization",
        "next_gate": "watertight_then_static_finalization",
        "formal_dataset_registration_authorized": False,
    }
    decision["decision_sha256"] = contract.json_sha256(decision)
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    review = tmp_path / "heading_review.png"
    review.write_bytes(b"reviewed heading fixture\n")
    heading_path = tmp_path / "heading.json"
    heading = {
        "schema": "avengine_static_heading_review_v1",
        "instance_id": decision["instance_id"],
        "request_sha256": decision["request_sha256"],
        "profile_sha256": decision["profile_sha256"],
        "input_glb_sha256": source_record["sha256"],
        "review_artifact": _record(review),
        "reviewed_source_front_yaw_deg": 90.0,
        "target_front_axis": "positive-x",
        "decision": "approved_for_positive_x_normalization",
        "formal_dataset_registration_authorized": False,
    }
    heading_path.write_text(json.dumps(heading, indent=2) + "\n", encoding="utf-8")

    output = tmp_path / "final.glb"
    manifest = tmp_path / "finalization.json"
    command = [
        str(BLENDER),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(FINALIZER),
        "--",
        "--input-glb",
        str(watertight_glb),
        "--watertight-manifest",
        str(watertight_manifest),
        "--static-decision",
        str(decision_path),
        "--heading-evidence",
        str(heading_path),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "passed_final_scaled_grounded_canonical_glb"
    assert payload["coordinate_system"] == contract.COORDINATE_SYSTEM
    assert payload["heading"]["applied_world_z_yaw_deg"] == -90.0
    assert payload["physical_scale"]["uniform_scale"] == pytest.approx(2.0)
    assert payload["physical_scale"]["readback_height_m"] == pytest.approx(
        2.0, abs=1.0e-5
    )
    assert payload["grounding"]["minimum_up_after_export_readback_m"] == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert payload["scene_readback"]["no_rig_or_animation"] is True
    contract.validate_static_finalization(manifest, output)

    retry = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert retry.returncode != 0
    assert "refusing to replace" in retry.stdout + retry.stderr
