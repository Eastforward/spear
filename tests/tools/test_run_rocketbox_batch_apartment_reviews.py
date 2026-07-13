import hashlib
import json
from pathlib import Path

from tools.run_rocketbox_batch_apartment_reviews import (
    _stable_animal_source_gate_is_valid,
    assign_unique_rpc_ports,
    build_jobs,
    build_render_command,
    finalize_environment,
    incomplete_jobs,
    job_is_complete,
    raw_render_is_complete,
    worker_environment,
)


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    tag = "rocketbox_children_female_child_01_original_ue_v1"
    spec = root / "specs" / tag / "walking.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        json.dumps(
            {
                "sources": [
                    {"tag": tag, "wanted_anim": "Walking", "actor_scale": 1.0}
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
