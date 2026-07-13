import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _spec():
    return {
        "spec_version": "apartment_v1",
        "mic": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
        "camera_configs": [{"fov_deg": 75.0, "fov_v_deg": 60.0}],
        "render_config": {
            "fps": 2,
            "n_frames": 4,
            "duration_s": 2.0,
        },
        "sources": [
            {"tag": "human_walk", "wanted_anim": "Walking"},
            {"tag": "human_idle", "wanted_anim": "Standing_Idle"},
        ],
    }


def test_build_flag_details_uses_runtime_scene_trajectories():
    from human_apartment_evidence import build_flag_details

    scene = SimpleNamespace(animals=[
        SimpleNamespace(
            tag="human_walk",
            trajectory_m=np.asarray([[2.0, -0.2, 0.0], [2.2, -0.1, 0.0], [2.4, 0.0, 0.0], [2.6, 0.1, 0.0]]),
        ),
        SimpleNamespace(
            tag="human_idle",
            trajectory_m=np.asarray([[1.5, 0.4, 0.0]] * 4),
        ),
    ])

    details = build_flag_details(
        _spec(),
        scene,
        furniture_bboxes=[],
        wall_bboxes=[],
    )

    assert details["per_source"]["human_walk"]["stationary"] is False
    assert details["per_source"]["human_idle"]["stationary"] is True
    assert details["aggregate"]["stationary"] is True
    assert set(details["per_source"]) == {"human_walk", "human_idle"}


def test_write_silent_wav_has_exact_duration_channels_and_zero_samples(tmp_path):
    from human_apartment_evidence import write_silent_wav

    path = write_silent_wav(
        tmp_path / "binaural.wav",
        duration_s=2.0,
        sample_rate_hz=8000,
        channels=2,
    )

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getframerate() == 8000
        assert wav.getnframes() == 16000
        assert set(wav.readframes(wav.getnframes())) == {0}


def _write(path, data=b"artifact"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def test_publish_technical_registry_merges_walk_and_idle_without_formal_promotion(tmp_path):
    from human_apartment_evidence import publish_technical_registry_clip

    registry_root = tmp_path / "ue_apartment_smoke" / "registry"
    asset_dir = tmp_path / "hy3d_rocketbox_template_fit_v1" / "rocketbox_male_adult_01"
    ue_manifest = _write(asset_dir / "ue_import_manifest.json", {
        "schema": "hy3d_rocketbox_ue_import_v1",
        "tag": "hy3d_rocketbox_male_adult_01_spike",
        "asset_id": "rocketbox_male_adult_01",
        "usage_scope": "technical_spike_only",
        "content": {"blueprint": "/Game/BP_human"},
    })

    for action, clip_id in (("Walking", "male_walk_final"), ("Standing_Idle", "male_idle_final")):
        clip_dir = tmp_path / "ue_apartment_smoke" / clip_id
        _write(clip_dir / "spec.json", {"sources": [{"wanted_anim": action}]})
        _write(clip_dir / "runtime_gate.json", {"human_gate_evidence": [{"tag": "hy3d_rocketbox_male_adult_01_spike"}]})
        _write(clip_dir / "videos" / "actor_visual_metadata.json", {"automatic_checks": {"overall": "passed"}})
        _write(clip_dir / "videos" / "apartment_v1_view0.mp4")
        _write(clip_dir / "videos" / "side_by_side_review_annotated.mp4")
        publish_technical_registry_clip(
            registry_root=registry_root,
            tag="hy3d_rocketbox_male_adult_01_spike",
            asset_id="rocketbox_male_adult_01",
            action_name=action,
            clip_id=clip_id,
            clip_dir=clip_dir,
            ue_import_manifest=ue_manifest,
        )

    registry_path = registry_root / "hy3d_rocketbox_male_adult_01_spike.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "human_apartment_technical_registry_v1"
    assert payload["usage_scope"] == "technical_spike_only"
    assert payload["formal_registry_promotion"] is False
    assert set(payload["clips"]) == {"Walking", "Standing_Idle"}
    assert payload["clips"]["Walking"]["clip_id"] == "male_walk_final"
    assert payload["clips"]["Standing_Idle"]["clip_id"] == "male_idle_final"


def test_finalize_clip_publishes_review_inputs_metadata_and_registry(tmp_path, monkeypatch):
    import human_apartment_evidence as evidence

    tag = "hy3d_rocketbox_male_adult_01_spike"
    asset_id = "rocketbox_male_adult_01"
    source_spec = _spec()
    source_spec["sources"] = [{
        "tag": tag,
        "wanted_anim": "Walking",
        "audio_lookup": "silent",
        "mute_audio": True,
    }]
    spec_path = _write(tmp_path / "source_spec.json", source_spec)
    clip_dir = tmp_path / "ue_apartment_smoke" / "male_walk_final"
    _write(clip_dir / "runtime_gate.json", {
        "human_gate_evidence": [{
            "tag": tag,
            "asset_id": asset_id,
            "asset_dir": str(tmp_path / "stable" / asset_id),
        }],
    })
    _write(
        clip_dir / "videos" / "actor_visual_metadata.json",
        {"automatic_checks": {"overall": "passed"}},
    )
    _write(clip_dir / "videos" / "apartment_v1_view0.mp4")
    ue_manifest = _write(tmp_path / "stable" / asset_id / "ue_import_manifest.json", {
        "schema": "hy3d_rocketbox_ue_import_v1",
        "tag": tag,
        "asset_id": asset_id,
        "usage_scope": "technical_spike_only",
        "content": {"blueprint": "/Game/BP_human"},
    })

    scene = SimpleNamespace(animals=[SimpleNamespace(
        tag=tag,
        trajectory_m=np.asarray([[2.0, 0.0, 0.0]] * 4),
    )])
    monkeypatch.setattr(evidence, "_compose_scene", lambda _path: scene)
    monkeypatch.setattr(evidence, "_apartment_obstacles", lambda _spec: ([], []))

    def fake_metadata(spec_path, out_dir, clip_id):
        assert spec_path == clip_dir / "spec.json"
        _write(out_dir / "apartment_v1_metadata.json", {"clip_id": clip_id})

    def fake_reviews(out_dir):
        annotated = _write(
            out_dir / "videos" / "side_by_side_review_annotated.mp4"
        )
        return {"annotated": annotated}

    monkeypatch.setattr(evidence, "_compute_metadata", fake_metadata)
    monkeypatch.setattr(evidence, "_build_reviews", fake_reviews)

    result = evidence.finalize_human_apartment_clip(
        spec_path=spec_path,
        out_dir=clip_dir,
        clip_id="male_walk_final",
    )

    assert json.loads((clip_dir / "spec.json").read_text()) == source_spec
    assert (clip_dir / "flags.json").is_file()
    assert (clip_dir / "flag_details.json").is_file()
    assert (clip_dir / "apartment_v1_metadata.json").is_file()
    assert (clip_dir / "binaural.wav").is_file()
    assert result["annotated"].is_file()
    registry = clip_dir.parent / "registry" / f"{tag}.json"
    assert registry.is_file()
    assert json.loads(registry.read_text())["clips"]["Walking"]["clip_id"] == "male_walk_final"
    assert ue_manifest.is_file()


def test_finalize_native_v3_clip_uses_manifest_path_without_technical_asset_dir(
    tmp_path, monkeypatch
):
    import human_apartment_evidence as evidence

    tag = "rocketbox_male_adult_01_original_ue_v3"
    asset_id = "rocketbox_male_adult_01"
    source_spec = _spec()
    source_spec["usage_scope"] = "research_candidate"
    source_spec["sources"] = [{
        "tag": tag,
        "wanted_anim": "Walking",
        "audio_lookup": "silent",
        "mute_audio": True,
    }]
    spec_path = _write(tmp_path / "source_spec.json", source_spec)
    clip_dir = tmp_path / "ue_apartment_smoke" / "native_walk_final"
    ue_manifest = _write(
        tmp_path / "native_import" / tag / "ue_import_manifest.json",
        {
            "schema": "rocketbox_native_ue_import_v3",
            "tag": tag,
            "asset_id": asset_id,
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "content": {"blueprint": f"/Game/BP_gate_{tag}"},
        },
    )
    _write(clip_dir / "runtime_gate.json", {
        "human_gate_evidence": [{
            "tag": tag,
            "asset_id": asset_id,
            "usage_scope": "research_candidate",
            "ue_import_manifest_path": str(ue_manifest.resolve()),
        }],
    })
    _write(
        clip_dir / "videos" / "actor_visual_metadata.json",
        {"automatic_checks": {"overall": "passed"}},
    )
    _write(clip_dir / "videos" / "apartment_v1_view0.mp4")

    scene = SimpleNamespace(animals=[SimpleNamespace(
        tag=tag,
        trajectory_m=np.asarray([[2.0, 0.0, 0.0]] * 4),
    )])
    monkeypatch.setattr(evidence, "_compose_scene", lambda _path: scene)
    monkeypatch.setattr(evidence, "_apartment_obstacles", lambda _spec: ([], []))
    monkeypatch.setattr(
        evidence,
        "_compute_metadata",
        lambda _spec, out_dir, clip_id: _write(
            out_dir / "apartment_v1_metadata.json", {"clip_id": clip_id}
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_build_reviews",
        lambda out_dir: {
            "annotated": _write(
                out_dir / "videos" / "side_by_side_review_annotated.mp4"
            )
        },
    )

    result = evidence.finalize_human_apartment_clip(
        spec_path=spec_path,
        out_dir=clip_dir,
        clip_id="native_walk_final",
    )

    registry = result["registries"][0]
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "human_apartment_research_candidate_registry_v1"
    )
    assert payload["usage_scope"] == "research_candidate"
    assert payload["formal_registry_promotion"] is False
    assert payload["clips"]["Walking"]["clip_id"] == "native_walk_final"


def test_finalize_controlled_animal_clip_publishes_authenticated_registry(
    tmp_path, monkeypatch
):
    import human_apartment_evidence as evidence

    asset_id = "dog_golden_retriever_example"
    tag = f"pixal_{asset_id}"
    decision = _write(
        tmp_path / "evidence" / "animation_decision.json",
        {
            "asset_id": asset_id,
            "decision": "approved_for_ue_apartment",
            "decision_sha256": "decision-sha",
        },
    )
    imported = _write(
        tmp_path / "evidence" / "ue_import_result.json",
        {
            "schema": "pixal_animal_ue_import_result_v1",
            "results": [
                {
                    "legacy_tag": asset_id,
                    "tag": tag,
                    "source_sha256": "source-sha",
                    "actions": ["Idle", "Walking"],
                    "blueprint": f"/Game/BP_{tag}",
                }
            ],
        },
    )

    def descriptor(path):
        return {
            "path": str(path.resolve()),
            "sha256": evidence.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    source_spec = _spec()
    source_spec["sources"] = [
        {
            "tag": tag,
            "asset_id": asset_id,
            "asset_class": "animal",
            "species": "dog",
            "breed": "golden_retriever",
            "sampled_attributes": {"size": "medium", "coat_color": "light"},
            "wanted_anim": "Walking",
            "audio_lookup": "silent",
            "mute_audio": True,
            "controlled_animal_gate": {
                "schema": "controlled_animal_apartment_gate_v1",
                "status": "approved_for_research_candidate_apartment",
                "asset_id": asset_id,
                "tag": tag,
                "animation_decision": {
                    **descriptor(decision),
                    "decision_sha256": "decision-sha",
                },
                "ue_import_result": descriptor(imported),
                "ue_source_sha256": "source-sha",
                "formal_dataset_registration_authorized": False,
            },
        }
    ]
    spec_path = _write(tmp_path / "source_spec.json", source_spec)
    clip_dir = tmp_path / "controlled_animals" / "walk"
    _write(clip_dir / "runtime_gate.json", {"human_gate_evidence": []})
    _write(
        clip_dir / "videos" / "actor_visual_metadata.json",
        {"automatic_checks": {"overall": "passed"}},
    )
    _write(clip_dir / "videos" / "apartment_v1_view0.mp4")
    _write(clip_dir / "videos" / "topdown_review.mp4")

    scene = SimpleNamespace(
        animals=[
            SimpleNamespace(
                tag=tag,
                trajectory_m=np.asarray([[2.0, 0.0, 0.0]] * 4),
            )
        ]
    )
    monkeypatch.setattr(evidence, "_compose_scene", lambda _path: scene)
    monkeypatch.setattr(evidence, "_apartment_obstacles", lambda _spec: ([], []))
    monkeypatch.setattr(
        evidence,
        "_compute_metadata",
        lambda _spec, out_dir, clip_id: _write(
            out_dir / "apartment_v1_metadata.json", {"clip_id": clip_id}
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_build_reviews",
        lambda out_dir: {
            "annotated": _write(
                out_dir / "videos" / "side_by_side_review_annotated.mp4"
            )
        },
    )

    result = evidence.finalize_human_apartment_clip(
        spec_path=spec_path,
        out_dir=clip_dir,
        clip_id="controlled_dog_walk",
    )

    assert len(result["registries"]) == 1
    payload = json.loads(result["registries"][0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "controlled_animal_apartment_research_candidate_registry_v1"
    )
    assert payload["usage_scope"] == "research_candidate"
    assert payload["formal_registry_promotion"] is False
    assert payload["species"] == "dog"
    assert payload["clips"]["Walking"]["clip_id"] == "controlled_dog_walk"


def test_finalize_stable_animal_clip_publishes_pending_human_registry(
    tmp_path, monkeypatch
):
    import human_apartment_evidence as evidence

    asset_id = "quaternius_ultimate_husky_v1"
    template_id = asset_id
    tag = "stable_dog_husky_quaternius_ultimate_husky_v1"
    source_sha256 = "stable-source-sha"

    def descriptor(path):
        return {
            "path": str(path.resolve()),
            "sha256": evidence.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    deformation = _write(
        tmp_path / "evidence" / "deformation.json",
        {
            "schema": "avengine_skinned_deformation_audit_v1",
            "overall": "passed",
            "input_sha256": source_sha256,
            "formal_dataset_registration_authorized": False,
        },
    )
    direction = {
        "authored_front_axis": "negative_y",
        "runtime_front_axis": "positive_x",
        "cardinal_yaw_deg": 90,
        "automatic_fine_yaw_inference": False,
        "review_status": "agent_selected_pending_human_review",
    }
    template_registry = _write(
        tmp_path / "evidence" / "template_registry.json",
        {
            "schema": "avengine_quaternius_stable_template_registry_v1",
            "entries": [
                {
                    "template_id": template_id,
                    "runtime_glb": {"sha256": source_sha256},
                    "deformation_audit": descriptor(deformation),
                    "direction": direction,
                    "qa": {
                        "walking_deformation": (
                            "passed_automatic_deformation_measurements"
                        ),
                        "idle_deformation": (
                            "passed_automatic_deformation_measurements"
                        ),
                    },
                }
            ],
        },
    )
    imported = _write(
        tmp_path / "evidence" / "ue_import_result.json",
        {
            "schema": "stable_animal_ue_import_result_v1",
            "results": [
                {
                    "template_id": template_id,
                    "asset_id": asset_id,
                    "tag": tag,
                    "source_sha256": source_sha256,
                    "actions": ["Idle", "Walking"],
                    "blueprint": f"/Game/BP_{tag}",
                    "human_review_status": (
                        "agent_selected_pending_human_review"
                    ),
                    "formal_dataset_registration_authorized": False,
                }
            ],
        },
    )
    gate = {
        "schema": "stable_animal_apartment_gate_v1",
        "status": "approved_for_automated_research_candidate_apartment",
        "asset_id": asset_id,
        "template_id": template_id,
        "tag": tag,
        "species": "dog",
        "breed": "husky",
        "template_registry": descriptor(template_registry),
        "ue_import_result": descriptor(imported),
        "source_sha256": source_sha256,
        "deformation_audit": descriptor(deformation),
        "direction": direction,
        "human_visual_review": "pending",
        "formal_dataset_registration_authorized": False,
    }
    source_spec = _spec()
    source_spec["usage_scope"] = "research_candidate"
    source_spec["sources"] = [
        {
            "tag": tag,
            "asset_id": asset_id,
            "template_id": template_id,
            "asset_class": "animal",
            "species": "dog",
            "breed": "husky",
            "wanted_anim": "Walking",
            "walking_forward_yaw_offset_deg": 90,
            "audio_lookup": "silent",
            "mute_audio": True,
            "stable_animal_gate": gate,
        }
    ]
    spec_path = _write(tmp_path / "source_spec.json", source_spec)
    clip_dir = tmp_path / "stable_animals" / "walk"
    _write(clip_dir / "runtime_gate.json", {"human_gate_evidence": []})
    _write(
        clip_dir / "videos" / "actor_visual_metadata.json",
        {"automatic_checks": {"overall": "passed"}},
    )
    _write(clip_dir / "videos" / "apartment_v1_view0.mp4")
    _write(clip_dir / "videos" / "topdown_review.mp4")

    scene = SimpleNamespace(
        animals=[
            SimpleNamespace(
                tag=tag,
                trajectory_m=np.asarray([[2.0, 0.0, 0.0]] * 4),
            )
        ]
    )
    monkeypatch.setattr(evidence, "_compose_scene", lambda _path: scene)
    monkeypatch.setattr(evidence, "_apartment_obstacles", lambda _spec: ([], []))
    monkeypatch.setattr(
        evidence,
        "_compute_metadata",
        lambda _spec, out_dir, clip_id: _write(
            out_dir / "apartment_v1_metadata.json", {"clip_id": clip_id}
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_build_reviews",
        lambda out_dir: {
            "annotated": _write(
                out_dir / "videos" / "side_by_side_review_annotated.mp4"
            )
        },
    )

    result = evidence.finalize_human_apartment_clip(
        spec_path=spec_path,
        out_dir=clip_dir,
        clip_id="stable_husky_walk",
    )

    assert len(result["registries"]) == 1
    payload = json.loads(result["registries"][0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "stable_animal_apartment_research_candidate_registry_v1"
    )
    assert payload["usage_scope"] == "research_candidate"
    assert payload["formal_registry_promotion"] is False
    assert payload["human_visual_review"] == "pending"
    assert payload["clips"]["Walking"]["clip_id"] == "stable_husky_walk"


def test_finalize_example_can_preserve_frozen_baseline_registry(tmp_path, monkeypatch):
    import human_apartment_evidence as evidence

    tag = "hy3d_rocketbox_male_adult_01_spike"
    source_spec = _spec()
    source_spec["sources"] = [{
        "tag": tag,
        "wanted_anim": "Walking",
        "audio_lookup": "silent",
        "mute_audio": True,
    }]
    spec_path = _write(tmp_path / "source_spec.json", source_spec)
    clip_dir = tmp_path / "ue_apartment_smoke" / "multi_human_example"
    _write(clip_dir / "runtime_gate.json", {
        "human_gate_evidence": [{
            "tag": tag,
            "asset_id": "rocketbox_male_adult_01",
            "asset_dir": str(tmp_path / "stable" / "rocketbox_male_adult_01"),
        }],
    })
    _write(
        clip_dir / "videos" / "actor_visual_metadata.json",
        {"automatic_checks": {"overall": "passed"}},
    )
    _write(clip_dir / "videos" / "apartment_v1_view0.mp4")

    scene = SimpleNamespace(animals=[SimpleNamespace(
        tag=tag,
        trajectory_m=np.asarray([[2.0, 0.0, 0.0]] * 4),
    )])
    monkeypatch.setattr(evidence, "_compose_scene", lambda _path: scene)
    monkeypatch.setattr(evidence, "_apartment_obstacles", lambda _spec: ([], []))
    monkeypatch.setattr(
        evidence,
        "_compute_metadata",
        lambda _spec, out_dir, clip_id: _write(
            out_dir / "apartment_v1_metadata.json", {"clip_id": clip_id}
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_build_reviews",
        lambda out_dir: {
            "annotated": _write(
                out_dir / "videos" / "side_by_side_review_annotated.mp4"
            )
        },
    )

    result = evidence.finalize_human_apartment_clip(
        spec_path=spec_path,
        out_dir=clip_dir,
        clip_id="multi_human_example",
        publish_registry=False,
    )

    assert result["registries"] == []
    assert not (clip_dir.parent / "registry").exists()


def test_registry_merge_uses_cross_process_lock_for_parallel_walk_idle():
    source = (
        REPO / "tools" / "spike_rlr" / "human_apartment_evidence.py"
    ).read_text(encoding="utf-8")

    assert "import fcntl" in source
    assert "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)" in source
    assert "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)" in source
