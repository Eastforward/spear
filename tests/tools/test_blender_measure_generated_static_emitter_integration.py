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
MEASURER = REPO / "tools/blender_measure_generated_static_emitter.py"


def _record(path: Path):
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


@pytest.mark.skipif(not BLENDER.is_file(), reason="pinned Blender is unavailable")
def test_asymmetric_surface_measurement_preserves_nonzero_positive_z_right(tmp_path):
    final_glb = tmp_path / "final.glb"
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
            str(final_glb),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert fixture.returncode == 0, fixture.stdout + fixture.stderr

    finalization_path = tmp_path / "finalization.json"
    finalization = {
        "schema": contract.STATIC_FINALIZATION_SCHEMA,
        "created_at": "2026-07-27T00:00:00+00:00",
        "status": "passed_final_scaled_grounded_canonical_glb",
        "asset_class": "static_object",
        "instance_id": "asymmetric_static_fixture_0001",
        "request_sha256": "1" * 64,
        "profile_sha256": "2" * 64,
        "input": _record(final_glb),
        "output": _record(final_glb),
        "coordinate_system": contract.COORDINATE_SYSTEM,
        "heading": {"passed": True, "target_front_axis": "positive-x"},
        "physical_scale": {"passed": True},
        "grounding": {"passed": True},
        "scene_readback": {
            "mesh_count": 1,
            "skin_count": 0,
            "animation_count": 0,
        },
        "formal_dataset_registration_authorized": False,
    }
    finalization_path.write_text(
        json.dumps(finalization, indent=2) + "\n",
        encoding="utf-8",
    )
    review_path = tmp_path / "review.txt"
    review_path.write_text("fixture reviewed semantic feature\n", encoding="utf-8")
    spec_path = tmp_path / "anchor_spec.json"
    spec = {
        "schema": contract.STATIC_ANCHOR_SPEC_SCHEMA,
        "instance_id": finalization["instance_id"],
        "request_sha256": finalization["request_sha256"],
        "profile_sha256": finalization["profile_sha256"],
        "finalized_glb_sha256": _record(final_glb)["sha256"],
        "finalization_manifest_sha256": hashlib.sha256(
            finalization_path.read_bytes()
        ).hexdigest(),
        "anchor_id": "reviewed_surface_feature",
        "anchor_type": "object_speaker",
        "semantic_role": "reviewed_surface_feature",
        "selection": {
            "method": "reviewed_bbox_fraction_nearest_surface_v1",
            "aggregation": "weighted_centroid",
            "maximum_search_distance_fraction": 0.25,
            "samples": [
                {
                    "target_fraction_xyz": [1.0, 0.5, 0.5],
                    "weight": 1.0,
                }
            ],
        },
        "review_evidence": _record(review_path),
        "formal_dataset_registration_authorized": False,
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "emitter.json"
    marker = tmp_path / "marker.glb"
    command = [
        str(BLENDER),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(MEASURER),
        "--",
        "--input-glb",
        str(final_glb),
        "--finalization-manifest",
        str(finalization_path),
        "--anchor-spec",
        str(spec_path),
        "--output",
        str(output),
        "--marker-glb",
        str(marker),
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
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == contract.MEASUREMENT_SCHEMA
    assert payload["coordinate_system"] == contract.COORDINATE_SYSTEM
    assert payload["status"] == "measured_pending_marker_visual_review"
    assert payload["formal_dataset_registration_authorized"] is False
    forward, up, right = payload["emitter_anchor"]["offset_m"]
    assert forward == pytest.approx(2.0, abs=1.0e-5)
    assert up == pytest.approx(0.5, abs=1.0e-5)
    assert right == pytest.approx(1.5, abs=1.0e-5)
    assert right > 0.0
    assert marker.is_file() and marker.stat().st_size > 0

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
