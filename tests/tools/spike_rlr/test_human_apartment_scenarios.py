import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_wav(path: Path, frequency_hz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24000
    t = np.arange(6 * sample_rate, dtype=np.float32) / sample_rate
    sf.write(path, 0.1 * np.sin(2 * np.pi * frequency_hz * t), sample_rate)


def _speech_root(tmp_path: Path) -> Path:
    root = tmp_path / "LibriTTS"
    root.mkdir()
    (root / "speakers.tsv").write_text(
        "READER\tGENDER\tSUBSET\tNAME\n"
        "1000\tM\ttrain-clean-100\tTest Man\n"
        "1001\tF\ttrain-clean-100\tTest Woman\n",
        encoding="utf-8",
    )
    (root / "LICENSE.txt").write_text("CC BY 4.0 test fixture\n", encoding="utf-8")
    male = root / "train-clean-100" / "1000" / "1" / "1000_1_000001_000000.wav"
    female = root / "train-clean-100" / "1001" / "1" / "1001_1_000001_000000.wav"
    _write_wav(male, 180.0)
    _write_wav(female, 240.0)
    male.with_suffix(".normalized.txt").write_text("Male test sentence.\n", encoding="utf-8")
    female.with_suffix(".normalized.txt").write_text("Female test sentence.\n", encoding="utf-8")
    return root


def test_human_scenario_bundle_is_deterministic_collision_gated_and_traceable(tmp_path):
    from human_apartment_scenarios import write_human_scenario_bundle

    root = _speech_root(tmp_path)
    out_root = tmp_path / "human_examples"
    manifest_path = write_human_scenario_bundle(
        out_root=out_root,
        base_spec_path=REPO / "data" / "apartment_v1_spec.json",
        speech_root=root,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "human_apartment_scenario_bundle_v1"
    assert manifest["usage_scope"] == "technical_spike_only"
    assert manifest["speech_license"]["spdx"] == "CC-BY-4.0"
    assert set(manifest["scenarios"]) == {
        "male_walk_female_idle",
        "female_walk_male_idle",
        "dual_walk_pass",
    }

    expected_actions = {
        "male_walk_female_idle": ["Walking", "Standing_Idle"],
        "female_walk_male_idle": ["Standing_Idle", "Walking"],
        "dual_walk_pass": ["Walking", "Walking"],
    }
    for scenario_id, descriptor in manifest["scenarios"].items():
        spec_path = Path(descriptor["spec_path"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert descriptor["spec_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
        assert spec["usage_scope"] == "technical_spike_only"
        assert spec["minimum_source_separation_m"] >= 0.75
        assert spec["event_constraint_report"]["passed"] is True
        assert spec["render_config"]["streaming_warmup_frames"] >= 120
        assert spec["render_config"]["camera_warmup_frames"] >= 40
        assert [source["wanted_anim"] for source in spec["sources"]] == expected_actions[scenario_id]
        assert all(
            source["walking_forward_yaw_offset_deg"] == 90.0
            for source in spec["sources"]
        )
        assert all(
            np.asarray(source["trajectory_m"], dtype=float).shape == (75, 3)
            for source in spec["sources"]
        )
        assert all(
            np.allclose(np.asarray(source["trajectory_m"], dtype=float)[:, 2], 0.0)
            for source in spec["sources"]
        )

        for source in spec["sources"]:
            expected_rate = 0.45 if source["wanted_anim"] == "Walking" else 1.0
            assert source["animation_play_rate"] == expected_rate
            trajectory = np.asarray(source["trajectory_m"], dtype=float)
            if source["wanted_anim"] != "Walking":
                continue
            chord = trajectory[-1, :2] - trajectory[0, :2]
            relative = trajectory[:, :2] - trajectory[0, :2]
            signed_offset = (
                chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
            ) / np.linalg.norm(chord)
            assert np.ptp(signed_offset) >= 0.12
            tangent_yaw = np.unwrap(np.arctan2(
                np.gradient(trajectory[:, 1]),
                np.gradient(trajectory[:, 0]),
            ))
            assert np.degrees(np.ptp(tangent_yaw)) >= 12.0

        voiced = [source for source in spec["sources"] if not source.get("mute_audio")]
        assert voiced
        for source in voiced:
            provenance = source["speech_provenance"]
            assert source["strict_audio"] is True
            assert provenance["speaker_gender"] == source["identity_gender"]
            assert provenance["transcript"]
            audio = Path(source["audio_path"])
            assert provenance["audio_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()

    dual = json.loads(
        Path(manifest["scenarios"]["dual_walk_pass"]["spec_path"]).read_text()
    )
    trajectories = [np.asarray(source["trajectory_m"], dtype=float) for source in dual["sources"]]
    distances = np.linalg.norm(trajectories[0][:, :2] - trajectories[1][:, :2], axis=1)
    assert distances.min() >= dual["minimum_source_separation_m"]


def test_turnaround_bundle_has_one_slow_actor_and_one_explicit_reversal(tmp_path):
    from human_apartment_scenarios import write_human_scenario_bundle
    from scene_two_dogs_apartment import compose_two_dog_scene_apartment

    root = _speech_root(tmp_path)
    out_root = tmp_path / "turnaround"
    manifest_path = write_human_scenario_bundle(
        out_root=out_root,
        base_spec_path=REPO / "data" / "apartment_v1_spec.json",
        speech_root=root,
        scenario_set="turnaround",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scenario_set"] == "turnaround"
    assert set(manifest["scenarios"]) == {"male_walk_turnaround"}

    descriptor = manifest["scenarios"]["male_walk_turnaround"]
    spec_path = Path(descriptor["spec_path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["event_constraint_report"]["passed"] is True
    assert spec["rig_direction_check_windows"] == [
        {"label": "left_to_right", "frame_a": 12, "frame_b": 28},
        {"label": "right_to_left", "frame_a": 48, "frame_b": 64},
    ]
    assert len(spec["sources"]) == 1

    source = spec["sources"][0]
    assert source["identity_gender"] == "M"
    assert source["wanted_anim"] == "Walking"
    assert source["animation_play_rate"] == 0.45
    assert source["motion"] == "piecewise_linear_raw"
    assert source["speech_provenance"]["speaker_gender"] == "M"

    trajectory = np.asarray(source["trajectory_m"], dtype=float)
    semantic_yaw = np.asarray(source["facing_yaw_deg_per_frame"], dtype=float)
    assert trajectory.shape == (75, 3)
    assert semantic_yaw.shape == (75,)
    assert np.allclose(trajectory[37], trajectory[38])
    assert np.allclose(semantic_yaw[:38], semantic_yaw[0])
    assert np.allclose(semantic_yaw[38:], semantic_yaw[38])
    reversal_deg = ((semantic_yaw[38] - semantic_yaw[37] + 180.0) % 360.0) - 180.0
    assert abs(reversal_deg) == pytest.approx(180.0)

    camera_yaw_rad = np.deg2rad(float(spec["camera_configs"][0]["yaw_deg"]))
    camera_right = np.asarray(
        [np.sin(camera_yaw_rad), -np.cos(camera_yaw_rad)], dtype=float
    )
    screen_x = (
        trajectory[:, :2] - np.asarray(spec["mic"]["pos_m"][:2], dtype=float)
    ) @ camera_right
    assert screen_x[37] - screen_x[0] >= 1.4
    assert screen_x[38] - screen_x[-1] >= 1.4

    scene = compose_two_dog_scene_apartment(spec_path)
    assert len(scene.animals) == 1
    assert np.allclose(
        scene.animals[0].yaw_deg,
        (semantic_yaw + source["walking_forward_yaw_offset_deg"]) % 360.0,
    )


def test_example_runner_orders_ue_rlr_and_non_registry_finalization(tmp_path, monkeypatch):
    import run_human_apartment_example as runner

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"scenario_id": "example"}), encoding="utf-8")
    mesh = tmp_path / "apartment.glb"
    materials = tmp_path / "materials.json"
    mesh.write_bytes(b"mesh")
    materials.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "clip"
    calls = []

    def fake_render(spec_path, rendered_out, csv_path, clip_id):
        calls.append(("ue", spec_path, rendered_out, csv_path, clip_id))
        (rendered_out / "videos").mkdir(parents=True)
        (rendered_out / "videos" / "apartment_v1_view0.mp4").write_bytes(b"video")

    def fake_subprocess(command, check, cwd, env):
        calls.append(("rlr", command, check, cwd, env))
        Path(command[command.index("--out") + 1]).write_bytes(b"audio")
        return SimpleNamespace(returncode=0)

    def fake_finalize(**kwargs):
        calls.append(("finalize", kwargs))
        return {"annotated": kwargs["out_dir"] / "videos" / "review.mp4"}

    monkeypatch.setattr(runner, "render_apartment", fake_render)
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(runner, "finalize_human_apartment_clip", fake_finalize)

    result = runner.run_human_apartment_example(
        spec_path=spec,
        out_dir=out_dir,
        clip_id="example",
        mesh_path=mesh,
        materials_path=materials,
        ss2_python=Path("/fake/ss2/python"),
        quality="low",
    )

    assert [call[0] for call in calls] == ["ue", "rlr", "finalize"]
    audio_command = calls[1][1]
    assert audio_command[0] == "/fake/ss2/python"
    assert audio_command[audio_command.index("--channel-layout") + 1] == "binaural"
    assert audio_command[audio_command.index("--quality") + 1] == "low"
    assert calls[1][4]["LD_PRELOAD"] == (
        "/usr/lib/x86_64-linux-gnu/libEGL.so.1:"
        "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    )
    assert calls[2][1]["publish_registry"] is False
    assert result["audio"] == out_dir / "binaural.wav"


def test_bundle_runner_honors_selected_scenario_order(tmp_path, monkeypatch):
    import run_human_apartment_example as runner

    bundle = {
        "schema_version": "human_apartment_scenario_bundle_v1",
        "scenarios": {
            "first": {
                "clip_id": "first",
                "spec_path": str(tmp_path / "first.json"),
                "output_dir": str(tmp_path / "clips" / "first"),
            },
            "second": {
                "clip_id": "second",
                "spec_path": str(tmp_path / "second.json"),
                "output_dir": str(tmp_path / "clips" / "second"),
            },
        },
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        runner,
        "run_human_apartment_example",
        lambda **kwargs: calls.append(kwargs) or {"clip_id": kwargs["clip_id"]},
    )

    results = runner.run_human_apartment_bundle(
        bundle_path=bundle_path,
        scenario_ids=["second", "first"],
        mesh_path=tmp_path / "mesh.glb",
        materials_path=tmp_path / "materials.json",
        ss2_python=Path("/fake/ss2/python"),
    )

    assert [call["clip_id"] for call in calls] == ["second", "first"]
    assert [result["clip_id"] for result in results] == ["second", "first"]
