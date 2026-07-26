"""Unit tests for the local forward-review page generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_forward_review_page import main


@pytest.fixture
def workspace(tmp_path):
    turntable = tmp_path / "turntable"
    turntable.mkdir()
    frames = []
    for index in range(8):
        name = f"turntable_{index:03d}.png"
        (turntable / name).write_bytes(b"png")
        frames.append({"index": index, "azimuth_deg": 45.0*index, "path": name})
    candidates = []
    for yaw in (51.0, 231.0):
        name = f"candidate_{int(yaw):03d}.png"
        (turntable / name).write_bytes(b"png")
        candidates.append({"candidate_yaw_deg": yaw, "path": name})
    (turntable / "turntable_manifest.json").write_text(
        json.dumps(
            {
                "schema": "avengine_forward_turntable_render_v1",
                "frames": frames,
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    estimate = tmp_path / "estimate.json"
    estimate.write_text(
        json.dumps(
            {
                "input": {"path": "/x/rig.glb", "sha256": "0"*64},
                "estimate": {
                    "unsigned_axis_yaw_deg": 51.0,
                    "estimated_front_yaw_deg": 51.0,
                    "candidate_front_yaw_degrees": [51.0, -129.0],
                    "head_end_vote": {
                        "confidence": 0.17,
                        "signals": {"leg_spacing_vote": 0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    gait = tmp_path / "gait.json"
    gait.write_text(
        json.dumps(
            {
                "status": "pass",
                "result": {
                    "classification": "forward",
                    "stance_drift_yaw_deg": 172.6,
                    "stance_drift": {"mean_drift_ratio_of_diagonal": 1.28e-3},
                },
            }
        ),
        encoding="utf-8",
    )
    video = tmp_path / "walking_side.mp4"
    video.write_bytes(b"mp4")
    return tmp_path, turntable, estimate, gait, video


def test_page_embeds_data_and_relative_media(workspace):
    tmp_path, turntable, estimate, gait, video = workspace
    output = tmp_path / "pages" / "review.html"
    assert main(
        [
            "--asset-workspace", "collie_test_v1",
            "--turntable-dir", str(turntable),
            "--estimate-json", str(estimate),
            "--gait-audit", str(gait),
            "--preview-video", str(video),
            "--output-html", str(output),
        ]
    ) == 0
    text = output.read_text(encoding="utf-8")
    assert "collie_test_v1" in text
    assert "../turntable/turntable_000.png" in text
    assert "../walking_side.mp4" in text
    payload = json.loads(
        text.split('<script id="data" type="application/json">')[1].split(
            "</script>"
        )[0]
    )
    assert len(payload["frames"]) == 8
    assert payload["gait"]["result"]["classification"] == "forward"
    assert payload["check_fields"][0] == "walking_direction"


def test_page_refuses_replacement(workspace):
    tmp_path, turntable, estimate, _gait, _video = workspace
    output = tmp_path / "review.html"
    argv = [
        "--asset-workspace", "collie_test_v1",
        "--turntable-dir", str(turntable),
        "--estimate-json", str(estimate),
        "--output-html", str(output),
    ]
    assert main(argv) == 0
    with pytest.raises(SystemExit, match="refusing to replace"):
        main(argv)
