"""Run resumable Rocketbox Apartment Walk/Idle review jobs from a spec manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = SPEAR_ROOT / "tools/spike_rlr/run_human_apartment_smoke.py"
DEFAULT_PYTHON = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
DEFAULT_SS2_PYTHON = Path("/data/jzy/miniconda3/envs/ss2/bin/python")
DEFAULT_AUDIO_LAUNCHER = SPEAR_ROOT / "tools/spike_rlr/run_audio_pass_rlr.py"
DEFAULT_AUDIO_MESH = SPEAR_ROOT / "tmp/spike_rlr/apartment_v1_mesh.glb"
DEFAULT_AUDIO_MATERIALS = SPEAR_ROOT / "tmp/spike_rlr/apartment_v1_materials.json"
STABLE_TEMPLATE_REGISTRY_SCHEMAS = {
    "avengine_quaternius_stable_template_registry_v1",
    "avengine_stable_animal_template_registry_v2",
}
STABLE_PENDING_REVIEW_STATUSES = {
    "agent_selected_pending_human_review",
    "local_ofat_visual_review_pending",
}


@dataclass(frozen=True)
class ReviewJob:
    base_avatar_id: str
    tag: str
    action: str
    spec_path: Path
    output_dir: Path
    clip_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _controlled_animal_source_gate_is_valid(source: dict) -> bool:
    gate = source.get("controlled_animal_gate", {})
    if (
        source.get("asset_class") != "animal"
        or gate.get("schema") != "controlled_animal_apartment_gate_v1"
        or gate.get("status") != "approved_for_research_candidate_apartment"
        or gate.get("asset_id") != source.get("asset_id")
        or gate.get("tag") != source.get("tag")
        or gate.get("formal_dataset_registration_authorized") is not False
    ):
        return False
    decision_artifact = gate.get("animation_decision", {})
    import_artifact = gate.get("ue_import_result", {})
    try:
        decision_path = Path(decision_artifact["path"]).resolve()
        import_path = Path(import_artifact["path"]).resolve()
        for path, artifact in (
            (decision_path, decision_artifact),
            (import_path, import_artifact),
        ):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                return False
        decision = _read_json(decision_path)
        imported = _read_json(import_path)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    result = {
        item.get("legacy_tag"): item for item in imported.get("results", [])
    }.get(source.get("asset_id"))
    return bool(
        decision.get("asset_id") == source.get("asset_id")
        and decision.get("decision") == "approved_for_ue_apartment"
        and decision.get("decision_sha256")
        == decision_artifact.get("decision_sha256")
        and imported.get("schema") == "pixal_animal_ue_import_result_v1"
        and result
        and result.get("tag") == source.get("tag")
        and result.get("source_sha256") == gate.get("ue_source_sha256")
        and set(result.get("actions", [])) == {"Idle", "Walking"}
    )


def _stable_animal_source_gate_is_valid(source: dict) -> bool:
    gate = source.get("stable_animal_gate", {})
    if (
        source.get("asset_class") != "animal"
        or gate.get("schema") != "stable_animal_apartment_gate_v1"
        or gate.get("status")
        != "approved_for_automated_research_candidate_apartment"
        or gate.get("asset_id") != source.get("asset_id")
        or gate.get("template_id") != source.get("template_id")
        or gate.get("tag") != source.get("tag")
        or gate.get("species") != source.get("species")
        or gate.get("breed") != source.get("breed")
        or gate.get("human_visual_review") != "pending"
        or gate.get("formal_dataset_registration_authorized") is not False
    ):
        return False
    registry_artifact = gate.get("template_registry", {})
    import_artifact = gate.get("ue_import_result", {})
    deformation_artifact = gate.get("deformation_audit", {})
    try:
        for artifact in (
            registry_artifact,
            import_artifact,
            deformation_artifact,
        ):
            path = Path(artifact["path"]).resolve()
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                return False
        registry = _read_json(Path(registry_artifact["path"]))
        imported = _read_json(Path(import_artifact["path"]))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    entry = {
        item.get("template_id"): item for item in registry.get("entries", [])
    }.get(source.get("template_id"))
    result = {
        item.get("template_id"): item for item in imported.get("results", [])
    }.get(source.get("template_id"))
    return bool(
        registry.get("schema") in STABLE_TEMPLATE_REGISTRY_SCHEMAS
        and imported.get("schema") == "stable_animal_ue_import_result_v1"
        and entry
        and result
        and entry.get("runtime_glb", {}).get("sha256")
        == gate.get("source_sha256")
        and entry.get("deformation_audit", {}).get("sha256")
        == deformation_artifact.get("sha256")
        and str(entry.get("qa", {}).get("walking_deformation", "")).startswith(
            "passed_"
        )
        and str(entry.get("qa", {}).get("idle_deformation", "")).startswith(
            "passed_"
        )
        and entry.get("direction", {}).get("automatic_fine_yaw_inference")
        is False
        and entry.get("direction", {}).get("review_status")
        in STABLE_PENDING_REVIEW_STATUSES
        and float(source.get("walking_forward_yaw_offset_deg"))
        == float(entry.get("direction", {}).get("cardinal_yaw_deg"))
        and result.get("tag") == source.get("tag")
        and result.get("source_sha256") == gate.get("source_sha256")
        and set(result.get("actions", [])) == {"Idle", "Walking"}
        and result.get("formal_dataset_registration_authorized") is False
    )


def _job_uses_controlled_animal_gate(job: ReviewJob) -> bool:
    try:
        sources = _read_json(job.spec_path).get("sources", [])
    except (OSError, json.JSONDecodeError):
        return False
    return len(sources) == 1 and _controlled_animal_source_gate_is_valid(sources[0])


def _job_uses_stable_animal_gate(job: ReviewJob) -> bool:
    try:
        sources = _read_json(job.spec_path).get("sources", [])
    except (OSError, json.JSONDecodeError):
        return False
    return len(sources) == 1 and _stable_animal_source_gate_is_valid(sources[0])


def _runtime_gate_accepts_job(job: ReviewJob, runtime_gate: dict) -> bool:
    gate_tags = {
        item.get("tag") for item in runtime_gate.get("human_gate_evidence", [])
    }
    return (
        job.tag in gate_tags
        or _job_uses_controlled_animal_gate(job)
        or _job_uses_stable_animal_gate(job)
    )


def build_jobs(
    manifest_path: Path,
    *,
    actions: set[str] | None = None,
    avatar_ids: set[str] | None = None,
    exclude_avatar_ids: set[str] | None = None,
) -> list[ReviewJob]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    payload = _read_json(manifest_path)
    records = payload.get("records")
    schema = payload.get("schema")
    if not isinstance(records, list) or payload.get("avatar_count") != len(records):
        raise RuntimeError("Rocketbox Apartment spec manifest is invalid")
    controlled_animal = False
    stable_animal = False
    if schema == "rocketbox_batch_apartment_specs_v1":
        expected_action_set = {"Walking", "Standing_Idle"}
        expected_clip_count = len(records) * 2
        require_original_tag = True
    elif schema in {
        "rocketbox_representative_table_loop_specs_v1",
        "rocketbox_camera_pass_table_loop_specs_v2",
    }:
        expected_action_set = {"Walking"}
        expected_clip_count = len(records)
        require_original_tag = False
    elif schema == "controlled_animal_walk_idle_apartment_specs_v1":
        expected_action_set = {"Walking", "Idle"}
        expected_clip_count = len(records) * 2
        require_original_tag = False
        controlled_animal = True
    elif schema == "stable_animal_walk_idle_apartment_specs_v1":
        expected_action_set = {"Walking", "Idle"}
        expected_clip_count = len(records) * 2
        require_original_tag = False
        stable_animal = True
    else:
        raise RuntimeError("Apartment spec manifest schema is invalid")
    if payload.get("clip_count") != expected_clip_count:
        raise RuntimeError("Rocketbox Apartment spec clip count is invalid")
    actions = actions or expected_action_set
    if not actions or not actions <= {"Walking", "Standing_Idle", "Idle"}:
        raise ValueError(f"unsupported action filter: {sorted(actions)}")
    if not actions <= expected_action_set:
        raise ValueError(
            f"actions {sorted(actions)} are incompatible with manifest schema {schema}"
        )
    avatar_ids = avatar_ids or set()
    exclude_avatar_ids = exclude_avatar_ids or set()
    if avatar_ids & exclude_avatar_ids:
        raise ValueError(
            f"avatar include/exclude overlap: {sorted(avatar_ids & exclude_avatar_ids)}"
        )

    jobs = []
    seen = set()
    for record in sorted(records, key=lambda item: item["base_avatar_id"]):
        avatar_id = record["base_avatar_id"]
        tag = record["tag"]
        if avatar_id in exclude_avatar_ids:
            continue
        if avatar_ids and avatar_id not in avatar_ids:
            continue
        if require_original_tag and tag != f"{avatar_id}_original_ue_v1":
            raise RuntimeError(f"spec manifest tag mismatch: {avatar_id}")
        action_records = record.get("actions", {})
        if set(action_records) != expected_action_set:
            raise RuntimeError(f"spec manifest action pair changed: {avatar_id}")
        for action in sorted(actions & set(action_records)):
            item = action_records[action]
            spec_path = Path(item["spec"]).resolve()
            output_dir = Path(item["output_dir"]).resolve()
            clip_id = str(item["clip_id"])
            if (
                not spec_path.is_file()
                or not spec_path.is_relative_to(root)
                or not output_dir.is_relative_to(root)
            ):
                raise RuntimeError(f"review job escaped spec root: {avatar_id}/{action}")
            spec = _read_json(spec_path)
            sources = spec.get("sources", [])
            actor_scale = sources[0].get("actor_scale") if sources else None
            scale_is_valid = (
                isinstance(actor_scale, (int, float))
                and not isinstance(actor_scale, bool)
                and 0.0 < float(actor_scale) <= 2.0
                if controlled_animal or stable_animal
                else actor_scale == 1.0
            )
            if (
                len(sources) != 1
                or sources[0].get("tag") != tag
                or sources[0].get("wanted_anim") != action
                or not scale_is_valid
                or (
                    controlled_animal
                    and (
                        sources[0].get("asset_class") != "animal"
                        or not sources[0].get("species")
                        or sources[0].get("asset_id") != avatar_id
                        or not _controlled_animal_source_gate_is_valid(sources[0])
                    )
                )
                or (
                    stable_animal
                    and (
                        sources[0].get("asset_class") != "animal"
                        or not sources[0].get("species")
                        or sources[0].get("asset_id") != avatar_id
                        or sources[0].get("template_id") != avatar_id
                        or not _stable_animal_source_gate_is_valid(sources[0])
                    )
                )
            ):
                raise RuntimeError(f"review spec identity changed: {avatar_id}/{action}")
            identity = (avatar_id, action)
            if identity in seen:
                raise RuntimeError(f"duplicate review job: {identity}")
            seen.add(identity)
            jobs.append(
                ReviewJob(
                    base_avatar_id=avatar_id,
                    tag=tag,
                    action=action,
                    spec_path=spec_path,
                    output_dir=output_dir,
                    clip_id=clip_id,
                )
            )
    if avatar_ids - {job.base_avatar_id for job in jobs}:
        raise ValueError(
            f"unknown avatar ids: {sorted(avatar_ids - {job.base_avatar_id for job in jobs})}"
        )
    return jobs


def build_render_command(
    job: ReviewJob,
    *,
    stage: str,
    python_executable: Path = DEFAULT_PYTHON,
    launcher: Path = DEFAULT_LAUNCHER,
) -> list[str]:
    if stage not in {"render", "finalize", "all"}:
        raise ValueError(f"unsupported review stage: {stage!r}")
    return [
        str(python_executable),
        str(launcher),
        "--spec",
        str(job.spec_path),
        "--out-dir",
        str(job.output_dir),
        "--clip-id",
        job.clip_id,
        "--stage",
        stage,
    ]


def worker_environment(
    *,
    base_environment: dict[str, str],
    rpc_port: int,
    graphics_adapter: int,
    render_offscreen: bool = False,
) -> dict[str, str]:
    if not 1024 <= int(rpc_port) <= 65535:
        raise ValueError(f"invalid RPC port: {rpc_port}")
    if int(graphics_adapter) < 0:
        raise ValueError(f"invalid graphics adapter: {graphics_adapter}")
    environment = dict(base_environment)
    environment.update(
        {
            "SPEAR_APARTMENT_RPC_PORT": str(int(rpc_port)),
            "SPEAR_GRAPHICS_ADAPTER": str(int(graphics_adapter)),
            "SPEAR_RIG_ASSERT": "1",
            # One worker owns an adapter at a time, so a per-adapter cache is
            # concurrency-safe and avoids rebuilding Matplotlib's font cache
            # for every unique per-job RPC port.
            "MPLCONFIGDIR": (
                f"/tmp/avengine-matplotlib-gpu-{int(graphics_adapter)}"
            ),
        }
    )
    if render_offscreen:
        environment["SPEAR_RENDER_OFFSCREEN"] = "1"
    else:
        environment.pop("SPEAR_RENDER_OFFSCREEN", None)
    return environment


def finalize_environment(*, base_environment: dict[str, str]) -> dict[str, str]:
    """Return a CPU-only finalizer environment with one prewarmed font cache."""
    environment = dict(base_environment)
    for name in (
        "SPEAR_APARTMENT_RPC_PORT",
        "SPEAR_GRAPHICS_ADAPTER",
        "SPEAR_RENDER_OFFSCREEN",
        "SPEAR_RIG_ASSERT",
    ):
        environment.pop(name, None)
    environment["MPLCONFIGDIR"] = "/tmp/avengine-matplotlib-finalize"
    return environment


def _command_log_records(command_log: Path) -> list[dict]:
    try:
        lines = command_log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _last_finish_record(command_log: Path) -> dict | None:
    records = [
        payload
        for payload in _command_log_records(command_log)
        if payload.get("event") == "finish"
    ]
    return records[-1] if records else None


def raw_render_is_complete(job: ReviewJob) -> bool:
    """Check the exact artifacts needed by the independent CPU finalizer."""
    records = _command_log_records(job.output_dir / "command.log")
    explicit_render_records = [
        record for record in records if record.get("stage") == "render"
    ]
    if explicit_render_records:
        latest = explicit_render_records[-1]
        if latest.get("event") != "finish" or latest.get("status") != "passed":
            return False
    try:
        spec = _read_json(job.spec_path)
        runtime_gate = _read_json(job.output_dir / "runtime_gate.json")
        visual = _read_json(
            job.output_dir / "videos" / "actor_visual_metadata.json"
        )
        n_frames = int(spec["render_config"]["n_frames"])
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        n_frames <= 0
        or not _runtime_gate_accepts_job(job, runtime_gate)
        or visual.get("automatic_checks", {}).get("overall") != "passed"
    ):
        return False
    videos = job.output_dir / "videos"
    apartment_video = videos / "apartment_v1_view0.mp4"
    profile = job.output_dir / "profile_per_clip.csv"
    if (
        not apartment_video.is_file()
        or apartment_video.stat().st_size <= 0
        or not profile.is_file()
        or profile.stat().st_size <= 0
    ):
        return False
    frames_dir = videos / "apartment_v1_view0"
    for index in range(n_frames):
        frame = frames_dir / f"frame_{index:04d}.png"
        if not frame.is_file() or frame.stat().st_size <= 0:
            return False
    return True


def audio_is_required(job: ReviewJob) -> bool:
    spec = _read_json(job.spec_path)
    return any(
        not source.get("mute_audio") and source.get("audio_lookup") != "silent"
        for source in spec.get("sources", [])
    )


def audio_is_complete(job: ReviewJob) -> bool:
    if not audio_is_required(job):
        return True
    audio_path = job.output_dir / "binaural.wav"
    schedule_path = job.output_dir / "binaural_source_schedule.json"
    try:
        spec = _read_json(job.spec_path)
        schedule = _read_json(schedule_path)
        expected_duration_s = float(spec["audio_config"]["duration_s"])
        with wave.open(str(audio_path), "rb") as stream:
            channels = int(stream.getnchannels())
            sample_rate = int(stream.getframerate())
            duration_s = float(stream.getnframes()) / sample_rate
    except (KeyError, TypeError, ValueError, OSError, EOFError, wave.Error):
        return False
    expected_tags = {
        source["tag"]
        for source in spec.get("sources", [])
        if not source.get("mute_audio") and source.get("audio_lookup") != "silent"
    }
    return (
        channels == 2
        and sample_rate == int(spec["audio_config"]["sample_rate_hz"])
        and abs(duration_s - expected_duration_s) <= 1.0 / sample_rate
        and set(schedule.get("sources", {})) == expected_tags
    )


def build_audio_command(job: ReviewJob, *, quality: str) -> list[str]:
    if quality not in {"low", "high", "max"}:
        raise ValueError(f"unsupported RLR audio quality: {quality}")
    return [
        str(DEFAULT_SS2_PYTHON),
        str(DEFAULT_AUDIO_LAUNCHER),
        "--spec",
        str(job.spec_path),
        "--mesh",
        str(DEFAULT_AUDIO_MESH),
        "--materials",
        str(DEFAULT_AUDIO_MATERIALS),
        "--out",
        str(job.output_dir / "binaural.wav"),
        "--channel-layout",
        "binaural",
        "--quality",
        quality,
    ]


def job_is_complete(job: ReviewJob) -> bool:
    finish = _last_finish_record(job.output_dir / "command.log")
    if (
        finish is None
        or finish.get("status") != "passed"
        or finish.get("stage") not in {None, "all", "finalize"}
    ):
        return False
    try:
        runtime_gate = _read_json(job.output_dir / "runtime_gate.json")
        visual = _read_json(
            job.output_dir / "videos" / "actor_visual_metadata.json"
        )
        registry = _read_json(
            job.output_dir.parent / "registry" / f"{job.tag}.json"
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    registry_clip = registry.get("clips", {}).get(job.action, {})
    if (
        not _runtime_gate_accepts_job(job, runtime_gate)
        or visual.get("automatic_checks", {}).get("overall") != "passed"
        or registry.get("tag") != job.tag
        or registry.get("usage_scope") != "research_candidate"
        or registry_clip.get("clip_id") != job.clip_id
    ):
        return False
    for name in (
        "apartment_v1_view0.mp4",
        "topdown_review.mp4",
        "side_by_side_review_annotated.mp4",
    ):
        path = job.output_dir / "videos" / name
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def incomplete_jobs(jobs: list[ReviewJob]) -> list[ReviewJob]:
    """Return every job whose complete evidence contract is not readable."""
    return [job for job in jobs if not job_is_complete(job)]


def assign_unique_rpc_ports(
    jobs: list[ReviewJob], *, base_rpc_port: int
) -> dict[tuple[str, str], int]:
    """Assign one non-reused RPC port to every job in a runner invocation."""
    if not 1024 <= int(base_rpc_port) <= 65535:
        raise ValueError(f"invalid base RPC port: {base_rpc_port}")
    if jobs and int(base_rpc_port) + len(jobs) - 1 > 65535:
        raise ValueError("Rocketbox batch RPC port range exceeds 65535")
    assignments = {}
    for offset, job in enumerate(jobs):
        identity = (job.base_avatar_id, job.action)
        if identity in assignments:
            raise ValueError(f"duplicate Rocketbox job identity: {identity}")
        assignments[identity] = int(base_rpc_port) + offset
    return assignments


def _run_with_timeout(
    command: list[str],
    *,
    environment: dict[str, str],
    console_log: Path,
    timeout_seconds: float,
    launch_lock: threading.Lock | None = None,
    launch_state: dict | None = None,
    minimum_launch_interval_seconds: float = 0.0,
) -> int:
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("ab") as stream:
        def launch():
            interval = max(0.0, float(minimum_launch_interval_seconds))
            if launch_state is not None and interval > 0.0:
                previous = launch_state.get("last_launch_monotonic")
                if previous is not None:
                    time.sleep(max(0.0, interval - (time.monotonic() - previous)))
            launched = subprocess.Popen(
                command,
                cwd=SPEAR_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if launch_state is not None:
                launch_state["last_launch_monotonic"] = time.monotonic()
            return launched

        if launch_lock is None:
            process = launch()
        else:
            # Four packaged UE processes can render concurrently, but starting
            # all four Vulkan swapchains on one X server in the same instant
            # leaves some processes before RPC initialization.  Serialize only
            # Popen and keep a small interval; the expensive render loops still
            # overlap on their assigned GPUs.
            with launch_lock:
                process = launch()
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise TimeoutError(
                f"review timed out after {timeout_seconds} seconds"
            ) from error


def terminate_orphaned_ue_processes(rpc_port: int) -> list[int]:
    """Terminate packaged UE children that outlived a failed RPC client.

    Matching the unique per-job config path avoids touching unrelated Unreal
    jobs on the shared host.
    """
    marker = f"spear_instance_{int(rpc_port)}/config.yaml"
    matches = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"SpearSim" not in command or marker.encode() not in command:
            continue
        pid = int(proc_dir.name)
        if pid != os.getpid():
            matches.append(pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in matches:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM and matches:
            time.sleep(1.0)
    return matches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--action", action="append", choices=["Walking", "Standing_Idle", "Idle"]
    )
    parser.add_argument("--avatar-id", action="append", default=[])
    parser.add_argument("--exclude-avatar-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--finalize-workers",
        type=int,
        default=8,
        help="Independent CPU workers for metadata, top-down, and FFmpeg review.",
    )
    parser.add_argument("--graphics-adapter", action="append", type=int, default=[])
    parser.add_argument("--base-rpc-port", type=int, default=39200)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--finalize-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--audio-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--audio-quality", choices=("low", "high", "max"), default="low")
    parser.add_argument(
        "--ue-launch-stagger-seconds",
        type=float,
        default=10.0,
        help=(
            "Minimum interval between packaged UE process starts. Rendering "
            "still overlaps after each process reaches its assigned GPU."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-offscreen", action="store_true")
    parser.add_argument("--status", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    if not 1 <= args.finalize_workers <= 32:
        raise ValueError("finalize-workers must be in [1, 32]")
    adapters = args.graphics_adapter or [0]
    if len(adapters) < args.workers:
        raise ValueError("provide at least one --graphics-adapter per worker")
    resources = queue.Queue()
    for index in range(args.workers):
        resources.put(adapters[index])
    jobs = build_jobs(
        args.manifest,
        actions=set(args.action) if args.action else None,
        avatar_ids=set(args.avatar_id),
        exclude_avatar_ids=set(args.exclude_avatar_id),
    )
    rpc_ports = assign_unique_rpc_ports(jobs, base_rpc_port=args.base_rpc_port)
    selected = [job for job in jobs if not (args.resume and job_is_complete(job))]
    if args.resume:
        finalize_ready = [
            job
            for job in selected
            if raw_render_is_complete(job) and audio_is_complete(job)
        ]
        render_pending = [job for job in selected if job not in finalize_ready]
    else:
        finalize_ready = []
        render_pending = list(selected)
    started_at = _utc_now()
    ue_launch_lock = threading.Lock()
    ue_launch_state = {}

    def run_render(job: ReviewJob) -> dict:
        adapter = resources.get()
        rpc_port = rpc_ports[(job.base_avatar_id, job.action)]
        try:
            if not (args.resume and raw_render_is_complete(job)):
                environment = worker_environment(
                    base_environment=os.environ.copy(),
                    rpc_port=rpc_port,
                    graphics_adapter=adapter,
                    render_offscreen=args.render_offscreen,
                )
                if _job_uses_controlled_animal_gate(
                    job
                ) or _job_uses_stable_animal_gate(job):
                    environment["SPEAR_SKIP_REVIEW_GATE"] = "1"
                command = build_render_command(job, stage="render")
                try:
                    return_code = _run_with_timeout(
                        command,
                        environment=environment,
                        console_log=job.output_dir / "batch_console.log",
                        timeout_seconds=args.timeout_seconds,
                        launch_lock=ue_launch_lock,
                        launch_state=ue_launch_state,
                        minimum_launch_interval_seconds=(
                            args.ue_launch_stagger_seconds
                        ),
                    )
                except BaseException:
                    terminate_orphaned_ue_processes(rpc_port)
                    raise
                if return_code != 0:
                    terminate_orphaned_ue_processes(rpc_port)
                    raise RuntimeError(f"render process returned {return_code}")
            if not raw_render_is_complete(job):
                raise RuntimeError(
                    "render returned zero but raw render contract is incomplete"
                )
        finally:
            resources.put(adapter)

        if audio_is_required(job) and not (args.resume and audio_is_complete(job)):
            audio_environment = finalize_environment(
                base_environment=os.environ.copy()
            )
            audio_environment["LD_PRELOAD"] = (
                "/usr/lib/x86_64-linux-gnu/libEGL.so.1:"
                "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"
            )
            return_code = _run_with_timeout(
                build_audio_command(job, quality=args.audio_quality),
                environment=audio_environment,
                console_log=job.output_dir / "batch_console.log",
                timeout_seconds=args.audio_timeout_seconds,
            )
            if return_code != 0:
                raise RuntimeError(f"audio process returned {return_code}")
        if not audio_is_complete(job):
            raise RuntimeError("audio returned zero but binaural contract is incomplete")
        return {
            "base_avatar_id": job.base_avatar_id,
            "tag": job.tag,
            "action": job.action,
            "clip_id": job.clip_id,
            "output_dir": str(job.output_dir),
            "rpc_port": rpc_port,
            "graphics_adapter": adapter,
            "stage": "render_and_audio",
            "audio_required": audio_is_required(job),
            "status": "passed",
        }

    def run_finalize(job: ReviewJob) -> dict:
        environment = finalize_environment(base_environment=os.environ.copy())
        command = build_render_command(job, stage="finalize")
        return_code = _run_with_timeout(
            command,
            environment=environment,
            console_log=job.output_dir / "batch_console.log",
            timeout_seconds=args.finalize_timeout_seconds,
        )
        if return_code != 0:
            raise RuntimeError(f"finalize process returned {return_code}")
        if not job_is_complete(job):
            raise RuntimeError(
                "finalize returned zero but evidence contract is incomplete"
            )
        return {
            "base_avatar_id": job.base_avatar_id,
            "tag": job.tag,
            "action": job.action,
            "clip_id": job.clip_id,
            "output_dir": str(job.output_dir),
            "stage": "finalize",
            "status": "passed",
        }

    results = []
    render_results = []
    failures = []
    base_finalize_environment = finalize_environment(
        base_environment=os.environ.copy()
    )
    Path(base_finalize_environment["MPLCONFIGDIR"]).mkdir(
        parents=True, exist_ok=True
    )
    subprocess.run(
        [str(DEFAULT_PYTHON), "-c", "import matplotlib.pyplot"],
        cwd=SPEAR_ROOT,
        env=base_finalize_environment,
        check=True,
    )

    def record_failure(job: ReviewJob, stage: str, error: BaseException) -> None:
        failures.append(
            {
                "base_avatar_id": job.base_avatar_id,
                "tag": job.tag,
                "action": job.action,
                "clip_id": job.clip_id,
                "output_dir": str(job.output_dir),
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        print(
            f"ROCKETBOX_APARTMENT_REVIEW_FAILED {job.base_avatar_id} "
            f"{job.action} stage={stage}: {type(error).__name__}: {error}",
            flush=True,
        )

    with (
        ThreadPoolExecutor(max_workers=args.workers) as render_executor,
        ThreadPoolExecutor(max_workers=args.finalize_workers) as finalize_executor,
    ):
        finalize_futures = {
            finalize_executor.submit(run_finalize, job): job
            for job in finalize_ready
        }
        render_futures = {
            render_executor.submit(run_render, job): job
            for job in render_pending
        }
        for future in as_completed(render_futures):
            job = render_futures[future]
            try:
                result = future.result()
            except BaseException as error:
                record_failure(job, "render", error)
            else:
                render_results.append(result)
                print(
                    f"ROCKETBOX_APARTMENT_RENDER_OK {job.base_avatar_id} "
                    f"{job.action}",
                    flush=True,
                )
                finalize_futures[
                    finalize_executor.submit(run_finalize, job)
                ] = job

        for future in as_completed(list(finalize_futures)):
            job = finalize_futures[future]
            try:
                result = future.result()
            except BaseException as error:
                record_failure(job, "finalize", error)
            else:
                results.append(result)
                print(
                    f"ROCKETBOX_APARTMENT_REVIEW_OK {job.base_avatar_id} "
                    f"{job.action}",
                    flush=True,
                )

    missing = incomplete_jobs(jobs)
    all_passed = [job for job in jobs if job not in missing]
    status_path = (args.status or args.manifest.parent / "batch_render_status.json").resolve()
    manifest_schema = _read_json(args.manifest.resolve()).get("schema")
    status = {
        "schema": (
            "controlled_animal_apartment_render_status_v1"
            if manifest_schema == "controlled_animal_walk_idle_apartment_specs_v1"
            else (
                "stable_animal_apartment_render_status_v1"
                if manifest_schema == "stable_animal_walk_idle_apartment_specs_v1"
                else "rocketbox_batch_apartment_render_status_v1"
            )
        ),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "manifest": str(args.manifest.resolve()),
        "job_count": len(jobs),
        "selected_job_count": len(selected),
        "raw_ready_at_start_count": len(finalize_ready),
        "render_pending_at_start_count": len(render_pending),
        "passed_job_count": len(all_passed),
        "failed_job_count": len(failures),
        "incomplete_job_count": len(missing),
        "workers": args.workers,
        "finalize_workers": args.finalize_workers,
        "graphics_adapters": adapters[: args.workers],
        "rpc_port_policy": "unique_per_job",
        "rpc_port_range": (
            [min(rpc_ports.values()), max(rpc_ports.values())]
            if rpc_ports
            else []
        ),
        "resume": bool(args.resume),
        "render_offscreen": bool(args.render_offscreen),
        "ue_launch_stagger_seconds": float(args.ue_launch_stagger_seconds),
        "current_results": sorted(results, key=lambda item: (item["tag"], item["action"])),
        "current_render_results": sorted(
            render_results, key=lambda item: (item["tag"], item["action"])
        ),
        "current_failures": sorted(
            failures, key=lambda item: (item["tag"], item["action"])
        ),
        "incomplete_jobs": [
            {
                "base_avatar_id": job.base_avatar_id,
                "tag": job.tag,
                "action": job.action,
                "clip_id": job.clip_id,
                "output_dir": str(job.output_dir),
            }
            for job in missing
        ],
    }
    _atomic_json(status_path, status)
    if failures or missing:
        return 1
    print(
        f"ROCKETBOX_APARTMENT_BATCH_OK jobs={len(jobs)} "
        f"passed={len(all_passed)} status={status_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
