import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np
import pytest
import soundfile as sf

from tools.spike_rlr.acoustic_scene_contract import (
    approved_acoustic_scene_contract,
    approved_rlr_renderer_contract,
)
from tools.spike_rlr.active_frame_rir_evidence import (
    active_frame_indices,
    serialize_active_frame_rir_evidence,
)
from tools.spike_rlr.rlr_materials import build_rlr_materials_payload

from tools.run_rocketbox_batch_apartment_reviews import (
    _stable_animal_source_gate_is_valid,
    assign_unique_rpc_ports,
    audio_is_complete,
    build_jobs,
    build_render_command,
    finalize_environment,
    incomplete_jobs,
    job_is_complete,
    raw_render_is_complete,
    worker_environment,
)
from tools.spike_rlr.animal_audio import pinned_animal_audio_contract
from tools.spike_rlr.run_audio_pass_rlr import _load_dry_source


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    tag = "rocketbox_children_female_child_01_original_ue_v1"
    spec = root / "specs" / tag / "walking.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "tag": tag,
                        "wanted_anim": "Walking",
                        "actor_scale": 1.0,
                        "mute_audio": True,
                        "audio_lookup": "silent",
                    }
                ]
            }
        )
    )
    out = root / "clips" / tag / "walking"
    payload = {
        "schema": "rocketbox_batch_apartment_specs_v1",
        "avatar_count": 1,
        "clip_count": 2,
        "records": [
            {
                "base_avatar_id": "rocketbox_children_female_child_01",
                "tag": tag,
                "actions": {
                    "Walking": {
                        "spec": str(spec),
                        "clip_id": f"{tag}_walking",
                        "output_dir": str(out),
                    },
                    "Standing_Idle": {
                        "spec": str(root / "specs" / tag / "idle.json"),
                        "clip_id": f"{tag}_idle",
                        "output_dir": str(root / "clips" / tag / "idle"),
                    },
                },
            }
        ],
    }
    idle_spec = root / "specs" / tag / "idle.json"
    idle_spec.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "tag": tag,
                        "wanted_anim": "Standing_Idle",
                        "actor_scale": 1.0,
                        "mute_audio": True,
                        "audio_lookup": "silent",
                    }
                ]
            }
        )
    )
    path = root / "batch_spec_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def _representative_manifest(tmp_path: Path) -> Path:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    record = payload["records"][0]
    record["base_avatar_id"] = "recolored_adult_male"
    record["tag"] = "rocketbox_male_adult_01_shirt_blue_ue_v3"
    walking = record["actions"]["Walking"]
    spec_path = Path(walking["spec"])
    spec = json.loads(spec_path.read_text())
    spec["sources"][0]["tag"] = record["tag"]
    spec_path.write_text(json.dumps(spec))
    record["actions"] = {"Walking": walking}
    payload.update(
        {
            "schema": "rocketbox_representative_table_loop_specs_v1",
            "avatar_count": 1,
            "clip_count": 1,
        }
    )
    path.write_text(json.dumps(payload))
    return path


def _controlled_animal_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "controlled_animals"
    asset_id = "cat_siamese_bindpose_example"
    tag = f"pixal_{asset_id}"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    decision = evidence / "animation_decision.json"
    decision.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "decision": "approved_for_ue_apartment",
                "decision_sha256": "decision-sha",
            }
        )
    )
    imported = evidence / "ue_import_result.json"
    imported.write_text(
        json.dumps(
            {
                "schema": "pixal_animal_ue_import_result_v1",
                "results": [
                    {
                        "legacy_tag": asset_id,
                        "tag": tag,
                        "source_sha256": "source-sha",
                        "actions": ["Idle", "Walking"],
                    }
                ],
            }
        )
    )

    def artifact(path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    actions = {}
    for action in ("Walking", "Idle"):
        motion = action.lower()
        spec = root / "specs" / tag / f"{motion}.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "tag": tag,
                            "asset_id": asset_id,
                            "asset_class": "animal",
                            "species": "cat",
                            "wanted_anim": action,
                            "actor_scale": 0.081,
                            "controlled_animal_gate": {
                                "schema": "controlled_animal_apartment_gate_v1",
                                "status": "approved_for_research_candidate_apartment",
                                "asset_id": asset_id,
                                "tag": tag,
                                "animation_decision": {
                                    **artifact(decision),
                                    "decision_sha256": "decision-sha",
                                },
                                "ue_import_result": artifact(imported),
                                "ue_source_sha256": "source-sha",
                                "formal_dataset_registration_authorized": False,
                            },
                        }
                    ]
                }
            )
        )
        actions[action] = {
            "spec": str(spec),
            "clip_id": f"{tag}_{motion}",
            "output_dir": str(root / "clips" / tag / motion),
        }
    manifest = root / "spec_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "controlled_animal_walk_idle_apartment_specs_v1",
                "avatar_count": 1,
                "clip_count": 2,
                "records": [
                    {
                        "base_avatar_id": asset_id,
                        "asset_id": asset_id,
                        "tag": tag,
                        "actions": actions,
                    }
                ],
            }
        )
    )
    return manifest


def _stable_animal_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "stable_animals"
    asset_id = "quaternius_ultimate_husky_v1"
    tag = "stable_dog_husky_quaternius_ultimate_husky_v1"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)

    def artifact(path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    deformation = evidence / "deformation.json"
    deformation.write_text(json.dumps({"overall": "passed"}))
    source_sha = "stable-source-sha"
    registry = evidence / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "avengine_quaternius_stable_template_registry_v1",
                "entries": [
                    {
                        "template_id": asset_id,
                        "runtime_glb": {"sha256": source_sha},
                        "deformation_audit": artifact(deformation),
                        "qa": {
                            "walking_deformation": "passed_automatic_deformation_measurements",
                            "idle_deformation": "passed_automatic_deformation_measurements",
                        },
                        "direction": {
                            "cardinal_yaw_deg": 90,
                            "automatic_fine_yaw_inference": False,
                            "review_status": "agent_selected_pending_human_review",
                        },
                    }
                ],
            }
        )
    )
    imported = evidence / "ue_import_result.json"
    imported.write_text(
        json.dumps(
            {
                "schema": "stable_animal_ue_import_result_v1",
                "results": [
                    {
                        "template_id": asset_id,
                        "tag": tag,
                        "source_sha256": source_sha,
                        "actions": ["Idle", "Walking"],
                        "formal_dataset_registration_authorized": False,
                    }
                ],
            }
        )
    )

    actions = {}
    for action in ("Walking", "Idle"):
        motion = action.lower()
        spec = root / "specs" / tag / f"{motion}.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        source = {
            "tag": tag,
            "asset_id": asset_id,
            "template_id": asset_id,
            "asset_class": "animal",
            "species": "dog",
            "breed": "husky",
            "wanted_anim": action,
            "walking_forward_yaw_offset_deg": 90,
            "actor_scale": 0.15,
        }
        source["stable_animal_gate"] = {
            "schema": "stable_animal_apartment_gate_v1",
            "status": "approved_for_automated_research_candidate_apartment",
            "asset_id": asset_id,
            "template_id": asset_id,
            "tag": tag,
            "species": "dog",
            "breed": "husky",
            "template_registry": artifact(registry),
            "ue_import_result": artifact(imported),
            "source_sha256": source_sha,
            "deformation_audit": artifact(deformation),
            "human_visual_review": "pending",
            "formal_dataset_registration_authorized": False,
        }
        spec.write_text(json.dumps({"sources": [source]}))
        actions[action] = {
            "spec": str(spec),
            "clip_id": f"{tag}_{motion}",
            "output_dir": str(root / "clips" / tag / motion),
        }
    manifest = root / "spec_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "stable_animal_walk_idle_apartment_specs_v1",
                "avatar_count": 1,
                "clip_count": 2,
                "records": [
                    {
                        "base_avatar_id": asset_id,
                        "asset_id": asset_id,
                        "template_id": asset_id,
                        "tag": tag,
                        "actions": actions,
                    }
                ],
            }
        )
    )
    return manifest


def test_build_jobs_is_action_filtered_and_manifest_locked(tmp_path):
    manifest = _manifest(tmp_path)

    jobs = build_jobs(manifest, actions={"Walking"})

    assert len(jobs) == 1
    assert jobs[0].action == "Walking"
    assert jobs[0].tag.endswith("_original_ue_v1")
    assert jobs[0].spec_path.is_file()
    assert jobs[0].output_dir.name == "walking"

    excluded = build_jobs(
        manifest,
        actions={"Walking"},
        exclude_avatar_ids={"rocketbox_children_female_child_01"},
    )
    assert excluded == []


def test_build_jobs_accepts_single_action_representative_and_recolor_tag(tmp_path):
    jobs = build_jobs(_representative_manifest(tmp_path), actions={"Walking"})

    assert len(jobs) == 1
    assert jobs[0].base_avatar_id == "recolored_adult_male"
    assert jobs[0].tag == "rocketbox_male_adult_01_shirt_blue_ue_v3"
    assert jobs[0].action == "Walking"


def test_build_jobs_accepts_controlled_animal_walk_idle_and_instance_scale(tmp_path):
    manifest = _controlled_animal_manifest(tmp_path)

    jobs = build_jobs(manifest)

    assert [(job.base_avatar_id, job.action) for job in jobs] == [
        ("cat_siamese_bindpose_example", "Idle"),
        ("cat_siamese_bindpose_example", "Walking"),
    ]
    assert build_jobs(manifest, actions={"Idle"})[0].action == "Idle"


def test_build_jobs_accepts_stable_animal_without_claiming_human_approval(tmp_path):
    manifest = _stable_animal_manifest(tmp_path)

    jobs = build_jobs(manifest)

    assert [(job.base_avatar_id, job.action) for job in jobs] == [
        ("quaternius_ultimate_husky_v1", "Idle"),
        ("quaternius_ultimate_husky_v1", "Walking"),
    ]
    source = json.loads(jobs[0].spec_path.read_text())["sources"][0]
    assert _stable_animal_source_gate_is_valid(source)
    assert source["stable_animal_gate"]["human_visual_review"] == "pending"


def test_stable_gate_rejects_symlinked_artifact_path(tmp_path):
    manifest = _stable_animal_manifest(tmp_path)
    job = build_jobs(manifest, actions={"Walking"})[0]
    source = json.loads(job.spec_path.read_text())["sources"][0]
    registry = Path(
        source["stable_animal_gate"]["template_registry"]["path"]
    )
    link = registry.with_name("registry_link.json")
    link.symlink_to(registry)
    source["stable_animal_gate"]["template_registry"]["path"] = str(link)

    assert not _stable_animal_source_gate_is_valid(source)


def test_stage_commands_use_stable_launcher_and_do_not_mix_gpu_and_cpu_work(tmp_path):
    job = build_jobs(_manifest(tmp_path), actions={"Walking"})[0]

    render = build_render_command(
        job,
        stage="render",
        python_executable=Path("/env/bin/python"),
    )
    finalize = build_render_command(
        job,
        stage="finalize",
        python_executable=Path("/env/bin/python"),
    )

    assert render[0] == "/env/bin/python"
    assert render[1].endswith("tools/spike_rlr/run_human_apartment_smoke.py")
    assert render[-2:] == ["--stage", "render"]
    assert finalize[-2:] == ["--stage", "finalize"]
    assert "--finalize-evidence" not in render
    assert "--finalize-evidence" not in finalize
    assert str(job.spec_path) in render
    assert str(job.output_dir) in render


def test_worker_environment_isolates_rpc_gpu_and_matplotlib(tmp_path):
    environment = worker_environment(
        base_environment={"PATH": "/bin"},
        rpc_port=39120,
        graphics_adapter=0,
        render_offscreen=True,
    )

    assert environment["SPEAR_APARTMENT_RPC_PORT"] == "39120"
    assert environment["SPEAR_GRAPHICS_ADAPTER"] == "0"
    assert environment["SPEAR_RIG_ASSERT"] == "1"
    assert environment["SPEAR_RENDER_OFFSCREEN"] == "1"
    assert environment["MPLCONFIGDIR"] == "/tmp/avengine-matplotlib-gpu-0"
    assert environment["PATH"] == "/bin"

    same_gpu_new_port = worker_environment(
        base_environment={},
        rpc_port=40123,
        graphics_adapter=0,
        render_offscreen=True,
    )
    other_gpu = worker_environment(
        base_environment={},
        rpc_port=40124,
        graphics_adapter=1,
        render_offscreen=True,
    )
    assert same_gpu_new_port["MPLCONFIGDIR"] == environment["MPLCONFIGDIR"]
    assert other_gpu["MPLCONFIGDIR"] != environment["MPLCONFIGDIR"]


def test_finalize_environment_uses_cpu_cache_and_drops_gpu_worker_state():
    environment = finalize_environment(
        base_environment={
            "PATH": "/bin",
            "SPEAR_APARTMENT_RPC_PORT": "40123",
            "SPEAR_GRAPHICS_ADAPTER": "2",
            "SPEAR_RENDER_OFFSCREEN": "1",
            "SPEAR_RIG_ASSERT": "1",
        }
    )

    assert environment["PATH"] == "/bin"
    assert environment["MPLCONFIGDIR"] == "/tmp/avengine-matplotlib-finalize"
    assert "SPEAR_APARTMENT_RPC_PORT" not in environment
    assert "SPEAR_GRAPHICS_ADAPTER" not in environment
    assert "SPEAR_RENDER_OFFSCREEN" not in environment
    assert "SPEAR_RIG_ASSERT" not in environment


def test_resume_requires_all_primary_topdown_metadata_and_registry_evidence(tmp_path):
    job = build_jobs(_manifest(tmp_path), actions={"Walking"})[0]
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "command.log").write_text(
        json.dumps({"event": "finish", "status": "passed"}) + "\n"
    )
    (job.output_dir / "runtime_gate.json").write_text(
        json.dumps({"human_gate_evidence": [{"tag": job.tag}]})
    )
    videos = job.output_dir / "videos"
    videos.mkdir()
    (videos / "actor_visual_metadata.json").write_text(
        json.dumps({"automatic_checks": {"overall": "passed"}})
    )
    for name in (
        "apartment_v1_view0.mp4",
        "topdown_review.mp4",
        "side_by_side_review_annotated.mp4",
    ):
        (videos / name).write_bytes(b"video")
    registry = job.output_dir.parent / "registry" / f"{job.tag}.json"
    registry.parent.mkdir()
    registry.write_text(
        json.dumps(
            {
                "tag": job.tag,
                "usage_scope": "research_candidate",
                "clips": {"Walking": {"clip_id": job.clip_id}},
            }
        )
    )

    assert job_is_complete(job)

    (videos / "topdown_review.mp4").unlink()
    assert not job_is_complete(job)


def test_render_only_finish_is_raw_ready_but_not_final_evidence(tmp_path):
    job = build_jobs(_manifest(tmp_path), actions={"Walking"})[0]
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "command.log").write_text(
        json.dumps({"event": "finish", "stage": "render", "status": "passed"})
        + "\n"
    )
    (job.output_dir / "runtime_gate.json").write_text(
        json.dumps({"human_gate_evidence": [{"tag": job.tag}]})
    )
    videos = job.output_dir / "videos"
    frames = videos / "apartment_v1_view0"
    frames.mkdir(parents=True)
    (videos / "actor_visual_metadata.json").write_text(
        json.dumps({"automatic_checks": {"overall": "passed"}})
    )
    (videos / "apartment_v1_view0.mp4").write_bytes(b"video")
    spec = json.loads(job.spec_path.read_text())
    spec["render_config"] = {"n_frames": 2}
    job.spec_path.write_text(json.dumps(spec))
    for index in range(2):
        (frames / f"frame_{index:04d}.png").write_bytes(b"png")
    (job.output_dir / "profile_per_clip.csv").write_text("stage,seconds\n")

    assert raw_render_is_complete(job)
    assert not job_is_complete(job)


def test_rpc_ports_are_unique_per_job_instead_of_reused_per_worker(tmp_path):
    jobs = build_jobs(_manifest(tmp_path))

    assignments = assign_unique_rpc_ports(jobs, base_rpc_port=40100)

    assert len(assignments) == 2
    assert set(assignments.values()) == {40100, 40101}
    assert assignments[(jobs[0].base_avatar_id, jobs[0].action)] != assignments[
        (jobs[1].base_avatar_id, jobs[1].action)
    ]


def test_incomplete_jobs_remain_a_batch_failure_even_without_current_exception(tmp_path):
    jobs = build_jobs(_manifest(tmp_path))

    missing = incomplete_jobs(jobs)

    assert [(job.base_avatar_id, job.action) for job in missing] == [
        (job.base_avatar_id, job.action) for job in jobs
    ]


def _write_review_audio_evidence(job, *, silent=False):
    spec = json.loads(job.spec_path.read_text())
    source_spec = spec["sources"][0]
    contract = pinned_animal_audio_contract("dog_bark")
    source_spec.update(
        {
            "audio_lookup": "dog_bark",
            "audio_contract": contract,
            "audio_sha256": contract["sha256"],
            "audio_source_size_bytes": contract["size_bytes"],
            "audio_source_codec": contract["codec"],
            "audio_source_channels": contract["channels"],
            "audio_source_sample_width_bytes": contract["sample_width_bytes"],
            "audio_source_sample_rate_hz": contract["sample_rate_hz"],
            "audio_source_frame_count": contract["frame_count"],
            "audio_source_duration_s": contract["duration_s"],
            "audio_source_species": contract["species"],
            "audio_dry_source_policy": contract["dry_source_policy"],
            "audio_spatialization_status": contract["spatialization_status"],
            "audio_known_spatialized_derivative_sha256": contract[
                "known_spatialized_derivative_sha256"
            ],
            "audio_item_origin": contract["item_origin"],
            "audio_objective_content_qa_status": contract[
                "objective_audio_content_qa_status"
            ],
            "audio_item_level_license_status": contract[
                "item_level_license_status"
            ],
            "audio_item_level_license_snapshot": contract[
                "item_level_license_snapshot"
            ],
            "audio_formal_registration_authorized": contract[
                "formal_registration_authorized"
            ],
            "strict_audio": True,
            "mute_audio": False,
            "trajectory_m": [[1.0, 1.0, 0.45]] * 45,
        }
    )
    spec["mic"] = {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0}
    spec["spec_version"] = "apartment_v1"
    spec["audio_config"] = {"sample_rate_hz": 16000, "duration_s": 3.0}
    spec["render_config"] = {"n_frames": 45, "fps": 15}
    job.spec_path.write_text(json.dumps(spec))

    tag = source_spec["tag"]
    source = {}
    dry = _load_dry_source(
        tag,
        sample_rate=16000,
        duration_s=3.0,
        source_spec=source_spec,
        schedule_metadata_out=source,
    )
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "binaural_source_schedule.json").write_text(
        json.dumps(
            {
                "schema": "rlr_audio_source_schedules_v1",
                "sources": {tag: source},
            }
        )
    )
    scheduled_dry = job.output_dir / f"binaural_{tag}_scheduled_dry.wav"
    sf.write(str(scheduled_dry), dry, 16000, subtype="PCM_16")
    raw_wet = np.column_stack((dry, dry * 0.65)).astype(np.float32)
    pre_normalization_peak = float(np.max(np.abs(raw_wet)))
    normalized_wet = raw_wet * (0.9 / pre_normalization_peak)
    if silent:
        normalized_wet[:] = 0.0
    sf.write(
        str(job.output_dir / "binaural.wav"),
        normalized_wet,
        16000,
        subtype="PCM_16",
    )
    solo_path = job.output_dir / f"binaural_{tag}_binaural.wav"
    solo_path.write_bytes((job.output_dir / "binaural.wav").read_bytes())
    staged = job.output_dir / ".authenticated_inputs"
    staged.mkdir()
    mesh = staged / "acoustic_mesh.glb"
    materials = staged / "acoustic_materials.json"
    derived_materials = staged / "rlr_materials.json"
    mesh.write_bytes(Path("tmp/spike_rlr/apartment_v1_mesh.glb").read_bytes())
    materials.write_bytes(
        Path("tmp/spike_rlr/apartment_v1_materials.json").read_bytes()
    )
    derived_materials.write_bytes(
        build_rlr_materials_payload(
            json.loads(materials.read_text(encoding="utf-8"))
        )
    )
    for path in (mesh, materials, derived_materials):
        path.chmod(0o400)
    staged.chmod(0o500)
    frame_indices = active_frame_indices(
        dry,
        n_frames=45,
        samples_per_frame=round(16000 / 15),
    )
    rir_path = job.output_dir / f"{tag}_active_frame_binaural_rir.npz"
    rir_path.write_bytes(
        serialize_active_frame_rir_evidence(
            source_tag=tag,
            frame_indices=frame_indices,
            rirs=[
                np.asarray([[1.0], [0.65]], dtype=np.float32)
                for _ in frame_indices
            ],
            source_positions_scene_m=np.asarray(
                source_spec["trajectory_m"],
                dtype=np.float64,
            )[frame_indices]
            + np.asarray(
                [
                    0.0,
                    0.0,
                    source_spec.get("audio_source_height_offset_m", 0.0),
                ]
            ),
            mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
            mic_yaw_deg=0.0,
            sample_rate_hz=16000,
            n_samples_total=len(dry),
            n_frames=45,
            fps=15.0,
            samples_per_frame=round(16000 / 15),
        )
    )

    def descriptor(path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    (job.output_dir / "binaural_audio_render_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rlr_audio_render_manifest_v3",
                "backend": "habitat_sim_rlr_audio_sensor",
                "channel_layout": "binaural_native",
                "sample_rate_hz": 16000,
                "duration_s": 3.0,
                "n_frames": 45,
                "fps": 15.0,
                "quality_mode": "high",
                "indirect_ray_count": 500,
                "native_binaural_channel_order": [0, 1],
                "source_tags": [tag],
                "spec": descriptor(job.spec_path),
                "acoustic_mesh": descriptor(mesh),
                "acoustic_materials": descriptor(materials),
                "derived_rlr_materials": descriptor(derived_materials),
                "acoustic_scene_contract": approved_acoustic_scene_contract(spec),
                "renderer_contract": approved_rlr_renderer_contract(
                    sample_rate_hz=16000,
                    quality_mode="high",
                ),
                "source_schedule": descriptor(
                    job.output_dir / "binaural_source_schedule.json"
                ),
                "output_wav": descriptor(job.output_dir / "binaural.wav"),
                "per_source_outputs": {
                    tag: {
                        "binaural": descriptor(solo_path),
                        "scheduled_dry": descriptor(scheduled_dry),
                        "active_frame_rir": descriptor(rir_path),
                        "pre_normalization_peak": pre_normalization_peak,
                        "mic_local_azimuth_deg_per_frame": [45.0] * 45,
                    }
                },
                "mix_pre_normalization_peak": pre_normalization_peak,
                "behavior_gates": {
                    "per_source_event_alignment": "validate_on_readback",
                    "binaural_spatial_ild": "validate_on_readback",
                    "mixture_reconstruction": "validate_on_readback",
                    "active_frame_rir_replay": "validate_on_readback",
                },
                "technical_render_status": "passed",
                "formal_registration_authorized": False,
            }
        )
    )
    return job.output_dir / "binaural_source_schedule.json", tag


def test_audio_resume_requires_strict_pinned_schedule_and_non_silent_waveform(
    tmp_path,
):
    job = build_jobs(_stable_animal_manifest(tmp_path), actions={"Walking"})[0]
    _write_review_audio_evidence(job)

    assert audio_is_complete(job)


def test_audio_resume_rejects_old_weak_schedule_without_source_contract(
    tmp_path,
):
    job = build_jobs(_stable_animal_manifest(tmp_path), actions={"Walking"})[0]
    schedule_path, tag = _write_review_audio_evidence(job)
    schedule = json.loads(schedule_path.read_text())
    schedule["sources"][tag].pop("source_contract")
    schedule_path.write_text(json.dumps(schedule))

    assert not audio_is_complete(job)


def test_audio_resume_rejects_stale_license_or_zero_output(tmp_path):
    manifest = _stable_animal_manifest(tmp_path)
    stale_job = build_jobs(manifest, actions={"Walking"})[0]
    schedule_path, tag = _write_review_audio_evidence(stale_job)
    schedule = json.loads(schedule_path.read_text())
    schedule["sources"][tag]["formal_registration_authorized"] = True
    schedule_path.write_text(json.dumps(schedule))
    assert not audio_is_complete(stale_job)

    silent_job = build_jobs(manifest, actions={"Idle"})[0]
    _write_review_audio_evidence(silent_job, silent=True)
    assert not audio_is_complete(silent_job)


def test_audio_resume_rejects_wrong_but_self_consistent_acoustic_scene(tmp_path):
    job = build_jobs(
        _stable_animal_manifest(tmp_path),
        actions={"Walking"},
    )[0]
    _write_review_audio_evidence(job)
    manifest_path = job.output_dir / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    wrong_mesh = job.output_dir / "wrong_scene.glb"
    wrong_mesh.write_bytes(b"self-consistent-but-unapproved")
    payload = wrong_mesh.read_bytes()
    manifest["acoustic_mesh"] = {
        "path": str(wrong_mesh.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    manifest_path.write_text(json.dumps(manifest))

    assert not audio_is_complete(job)


def test_audio_resume_rejects_renderer_contract_tamper(tmp_path):
    job = build_jobs(
        _stable_animal_manifest(tmp_path),
        actions={"Walking"},
    )[0]
    _write_review_audio_evidence(job)
    manifest_path = job.output_dir / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["renderer_contract"]["acoustics"]["direct"] = False
    manifest_path.write_text(json.dumps(manifest))

    assert not audio_is_complete(job)


def test_audio_resume_rejects_pre_rir_manifest_v1(tmp_path):
    job = build_jobs(
        _stable_animal_manifest(tmp_path),
        actions={"Walking"},
    )[0]
    _write_review_audio_evidence(job)
    manifest_path = job.output_dir / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "rlr_audio_render_manifest_v1"
    manifest["behavior_gates"].pop("active_frame_rir_replay")
    manifest["behavior_gates"][
        "pinned_dry_reconstruction_and_wet_causality"
    ] = "validate_on_readback"
    manifest_path.write_text(json.dumps(manifest))

    assert not audio_is_complete(job)


def test_audio_resume_rejects_missing_manifest_and_silent_or_malformed_dog(
    tmp_path,
):
    manifest = _stable_animal_manifest(tmp_path)
    missing_manifest_job = build_jobs(
        manifest,
        actions={"Walking"},
    )[0]
    _write_review_audio_evidence(missing_manifest_job)
    (
        missing_manifest_job.output_dir
        / "binaural_audio_render_manifest.json"
    ).unlink()
    assert not audio_is_complete(missing_manifest_job)

    silent_dog_job = build_jobs(manifest, actions={"Idle"})[0]
    spec = json.loads(silent_dog_job.spec_path.read_text())
    spec["sources"][0].update(
        {"audio_lookup": "silent", "mute_audio": True}
    )
    silent_dog_job.spec_path.write_text(json.dumps(spec))
    assert not audio_is_complete(silent_dog_job)

    malformed_job = build_jobs(manifest, actions={"Walking"})[0]
    spec = json.loads(malformed_job.spec_path.read_text())
    spec["sources"][0].update(
        {
            "audio_lookup": "dog_bark",
            "strict_audio": True,
            "mute_audio": "false",
        }
    )
    malformed_job.spec_path.write_text(json.dumps(spec))
    assert not audio_is_complete(malformed_job)


def test_silent_alpaca_requires_exact_authenticated_contract(tmp_path):
    from tools.run_rocketbox_batch_apartment_reviews import _audible_source_tags
    from tools.spike_rlr.animal_audio import bind_animal_silence_contract

    tag = "stable_alpaca_quaternius"
    registry_path = tmp_path / "registry.json"
    import_path = tmp_path / "import.json"
    registry_path.write_text("{}")
    import_path.write_text("{}")

    def gate_descriptor(path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    source = {
        "tag": tag,
        "asset_id": "alpaca_asset",
        "template_id": "alpaca_asset",
        "asset_class": "animal",
        "species": "alpaca",
        "audio_lookup": "silent",
        "stable_animal_gate": {
            "schema": "stable_animal_apartment_gate_v1",
            "status": "approved_for_automated_research_candidate_apartment",
            "asset_id": "alpaca_asset",
            "template_id": "alpaca_asset",
            "tag": tag,
            "species": "alpaca",
            "source_sha256": "a" * 64,
            "template_registry": gate_descriptor(registry_path),
            "ue_import_result": gate_descriptor(import_path),
            "formal_dataset_registration_authorized": False,
        },
    }
    bind_animal_silence_contract(source)
    spec = {"sources": [source]}
    assert _audible_source_tags(spec, expected_job_tag=tag) == set()

    source["audio_silence_contract"]["species"] = "donkey_ass"
    with pytest.raises(ValueError, match="contract changed"):
        _audible_source_tags(spec, expected_job_tag=tag)
