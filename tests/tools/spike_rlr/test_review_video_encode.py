import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from tools.spike_rlr.review_video_encode import encode_rgb_frames


def test_raw_rgb_encoder_streams_frames_without_png_staging(tmp_path):
    frames = [
        np.full((48, 64, 3), (index * 80, 20, 200), dtype=np.uint8)
        for index in range(3)
    ]
    output = tmp_path / "streamed.mp4"

    count = encode_rgb_frames(iter(frames), output, fps=3)

    assert count == 3
    assert output.is_file() and output.stat().st_size > 0
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,pix_fmt,nb_read_frames",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )["streams"][0]
    assert probe == {
        "width": 64,
        "height": 48,
        "pix_fmt": "yuv420p",
        "nb_read_frames": "3",
    }
    assert list(tmp_path.glob("*.png")) == []


def test_raw_rgb_encoder_rejects_shape_drift(tmp_path):
    frames = iter(
        [
            np.zeros((48, 64, 3), dtype=np.uint8),
            np.zeros((50, 64, 3), dtype=np.uint8),
        ]
    )

    try:
        encode_rgb_frames(frames, tmp_path / "bad.mp4", fps=3)
    except ValueError as error:
        assert "shape changed" in str(error)
    else:
        raise AssertionError("raw RGB shape drift was accepted")


def test_ue_marker_video_streams_without_marker_png_staging(tmp_path, monkeypatch):
    from tools.spike_rlr import build_review_videos as review

    monkeypatch.setattr(review, "REPO_ROOT", tmp_path)
    clip = tmp_path / "clip"
    frames = clip / "videos" / "apartment_v1_view0"
    frames.mkdir(parents=True)
    (clip / "spec.json").write_text(
        json.dumps(
            {
                "camera_configs": [{"fov_deg": 90.0, "fov_v_deg": 60.0}],
                "render_config": {"fps": 3},
            }
        ),
        encoding="utf-8",
    )
    (clip / "apartment_v1_metadata.json").write_text(
        json.dumps(
            {
                "n_frames": 3,
                "mic_pose_6DoF": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
                "sources": [
                    {
                        "tag": "rocketbox_adults_female_adult_01_original_ue_v1",
                        "source_in_fov_per_frame": [True, True, True],
                        "source_world_xyz_per_frame": [
                            [2.0, 0.0, 1.2],
                            [2.0, 0.1, 1.2],
                            [2.0, 0.2, 1.2],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for index in range(3):
        Image.new("RGB", (64, 48), (20 + index * 20, 40, 60)).save(
            frames / f"frame_{index:04d}.png"
        )

    output = clip / "videos" / "markers.mp4"
    review.write_ue_marker_video(clip, output)

    assert output.is_file() and output.stat().st_size > 0
    staging = (
        tmp_path
        / "tmp/spike_rlr/ue_marker_frames"
        / review.review_work_key(clip)
    )
    assert not staging.exists()


def test_topdown_renderer_can_return_even_sized_rgb_frame_without_png():
    from tools.spike_rlr.render_topdown_2d import _load_scene, render_frame_rgb

    repo = Path(__file__).resolve().parents[3]
    spec_path = repo / "data/apartment_v1_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scene = _load_scene(spec_path)

    frame = render_frame_rgb(0, scene, spec)

    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] % 2 == 0
    assert frame.shape[1] % 2 == 0


def test_review_builder_direct_script_import_matches_runtime_launcher():
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "/data/jzy/miniconda3/envs/spear-env/bin/python",
            str(repo / "tools/spike_rlr/build_review_videos.py"),
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--clip-dir" in result.stdout
