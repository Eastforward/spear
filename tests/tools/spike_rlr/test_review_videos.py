import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_overlay_lines_count_per_source_visibility(tmp_path):
    from build_review_videos import build_overlay_lines

    clip = tmp_path / "clip_0007"
    clip.mkdir()
    (clip / "spec.json").write_text(json.dumps({
        "sources": [
            {"tag": "dog_beagle_v2", "audio_lookup": "dog_bark",
             "motion_style": "steady"},
            {"tag": "cat_british_shorthair_v2", "audio_lookup": "cat_purring",
             "motion_style": "stationary"},
        ],
    }))
    (clip / "flags.json").write_text(json.dumps({
        "leaves_camera_fov": True,
        "never_occluded": False,
        "stationary": True,
    }))
    (clip / "apartment_v1_metadata.json").write_text(json.dumps({
        "n_frames": 4,
        "sources": [
            {
                "tag": "dog_beagle_v2",
                "category": "dog_bark",
                "source_in_fov_per_frame": [True, True, False, False],
                "source_visible_from_camera_per_frame": [True, False, False, False],
                "source_occluded_by_furniture_per_frame": [False, True, False, False],
            },
            {
                "tag": "cat_british_shorthair_v2",
                "category": "cat_purring",
                "source_in_fov_per_frame": [False, False, False, False],
                "source_visible_from_camera_per_frame": [False, False, False, False],
                "source_occluded_by_furniture_per_frame": [False, False, False, False],
            },
        ],
    }))

    lines = build_overlay_lines(clip)

    assert lines[0] == (
        "clip_0007 | n_src=2 | flags=leaves_camera_fov,stationary"
    )
    assert lines[1] == (
        "dog_beagle_v2 dog_bark steady | center-FOV 2/4 | "
        "center-visible 1/4 (0) | occ 1/4"
    )
    assert lines[2] == (
        "cat_british_shorthair_v2 cat_purring stationary | "
        "center-FOV 0/4 | center-visible 0/4 (none) | occ 0/4"
    )


def test_topdown_tag_style_names_review_animals():
    from render_topdown_2d import _style_for_tag

    assert _style_for_tag("dog_golden")["label"] == "GOLDEN"
    assert _style_for_tag("dog_beagle_v2")["label"] == "BEAGLE"
    assert _style_for_tag("cat_british_shorthair_v2")["label"] == "BRITISH SHORTHAIR"


def test_project_source_to_frame_uses_camera_image_handedness():
    from build_review_videos import project_source_to_frame

    center = project_source_to_frame(
        src_xyz=(2.0, 0.0, 1.0),
        mic_pos=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        fov_h_deg=90.0,
        fov_v_deg=60.0,
        width=640,
        height=480,
    )
    mic_left = project_source_to_frame(
        src_xyz=(2.0, 1.0, 1.0),
        mic_pos=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        fov_h_deg=90.0,
        fov_v_deg=60.0,
        width=640,
        height=480,
    )
    mic_right = project_source_to_frame(
        src_xyz=(2.0, -1.0, 1.0),
        mic_pos=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        fov_h_deg=90.0,
        fov_v_deg=60.0,
        width=640,
        height=480,
    )
    behind = project_source_to_frame(
        src_xyz=(-2.0, 0.0, 1.0),
        mic_pos=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        fov_h_deg=90.0,
        fov_v_deg=60.0,
        width=640,
        height=480,
    )

    assert center == (320, 240)
    assert mic_left[0] < center[0]
    assert mic_left[1] == center[1]
    assert mic_right[0] > center[0]
    assert mic_right[1] == center[1]
    assert behind is None


def test_marker_xyz_prefers_ue_visual_sidecar_over_acoustic_point(tmp_path):
    from build_review_videos import load_actor_visual_metadata, marker_xyz_for_source_frame

    clip = tmp_path / "clip_0001"
    videos = clip / "videos"
    videos.mkdir(parents=True)
    (videos / "actor_visual_metadata.json").write_text(json.dumps({
        "sources": [
            {
                "tag": "dog_beagle_v2",
                "visual_center_world_xyz_per_frame": [
                    [2.0, 1.0, 0.9],
                    [2.0, 1.5, 0.9],
                ],
            },
        ],
    }))
    src = {
        "tag": "dog_beagle_v2",
        "source_world_xyz_per_frame": [
            [2.0, 0.0, 0.45],
            [2.0, 0.0, 0.45],
        ],
    }

    visual = load_actor_visual_metadata(clip)

    assert marker_xyz_for_source_frame(src, 1, visual) == [2.0, 1.5, 0.9]


def test_marker_xyz_falls_back_to_acoustic_point_when_sidecar_missing():
    from build_review_videos import marker_xyz_for_source_frame

    src = {
        "tag": "dog_golden",
        "source_world_xyz_per_frame": [[2.0, 0.0, 0.45]],
    }

    assert marker_xyz_for_source_frame(src, 0, {}) == [2.0, 0.0, 0.45]
