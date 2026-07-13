"""Evidence helpers for technical-spike humans rendered in apartment_0000."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import wave
from pathlib import Path

from flag_verifier import verify_flag_details
from source_trajectory import acoustic_trajectory


REGISTRY_SCHEMA = "human_apartment_technical_registry_v1"
USAGE_SCOPE = "technical_spike_only"
RESEARCH_CANDIDATE_REGISTRY_SCHEMA = (
    "human_apartment_research_candidate_registry_v1"
)
RESEARCH_CANDIDATE_USAGE_SCOPE = "research_candidate"
CONTROLLED_ANIMAL_REGISTRY_SCHEMA = (
    "controlled_animal_apartment_research_candidate_registry_v1"
)
CONTROLLED_ANIMAL_GATE_SCHEMA = "controlled_animal_apartment_gate_v1"
STABLE_ANIMAL_REGISTRY_SCHEMA = (
    "stable_animal_apartment_research_candidate_registry_v1"
)
STABLE_ANIMAL_GATE_SCHEMA = "stable_animal_apartment_gate_v1"
_SAFE_TAG = re.compile(r"[a-z0-9_]+")
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"evidence must be a direct regular file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def build_flag_details(spec, scene, *, furniture_bboxes, wall_bboxes):
    """Compute standard apartment flags from the exact composed trajectories."""
    placements = list(scene.animals)
    specs_by_tag = {
        str(source["tag"]): source
        for source in spec.get("sources", [])
        if source.get("tag")
    }
    return verify_flag_details(
        spec_dict=spec,
        trajectories=[
            acoustic_trajectory(
                placement.trajectory_m,
                specs_by_tag.get(placement.tag, {}),
            )
            for placement in placements
        ],
        furniture_bboxes=furniture_bboxes,
        wall_bboxes=wall_bboxes,
        source_tags=[placement.tag for placement in placements],
    )


def write_silent_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate_hz: int = 16000,
    channels: int = 2,
) -> Path:
    """Write deterministic 16-bit silence for visual-only baseline reviews."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(float(duration_s) * int(sample_rate_hz)))
    bytes_per_frame = int(channels) * 2
    zero_chunk = b"\x00" * (min(frame_count, 4096) * bytes_per_frame)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate_hz))
        remaining = frame_count
        while remaining:
            count = min(remaining, 4096)
            wav.writeframesraw(zero_chunk[: count * bytes_per_frame])
            remaining -= count
        wav.writeframes(b"")
    return path


def _compose_scene(spec_path: Path):
    from scene_two_dogs_apartment import compose_two_dog_scene_apartment

    return compose_two_dog_scene_apartment(spec_path)


def _apartment_obstacles(spec: dict):
    from apartment_builtin_obstacles import (
        apartment_builtin_visual_obstacle_bboxes_xyz,
    )
    from scene_two_dogs_apartment import (
        _kept_furniture_bboxes,
        _shell_wall_bboxes,
    )

    categories = json.loads(
        (
            REPO_ROOT
            / "tools"
            / "spike_rlr"
            / "apartment_furniture_categories.json"
        ).read_text(encoding="utf-8")
    )
    furniture = [
        ((x0, y0, 0.0), (x1, y1, 1.5))
        for x0, y0, x1, y1 in _kept_furniture_bboxes(spec, categories)
    ]
    furniture.extend(apartment_builtin_visual_obstacle_bboxes_xyz(spec))
    walls = [
        ((x0, y0, 0.0), (x1, y1, 2.8))
        for x0, y0, x1, y1 in _shell_wall_bboxes(spec)
    ]
    return furniture, walls


def _compute_metadata(spec_path: Path, out_dir: Path, clip_id: str) -> None:
    from compute_acoustic_metadata import compute

    compute(
        spec_path,
        out_dir,
        out_dir / "profile_per_clip.csv",
        clip_id,
    )


def _build_reviews(out_dir: Path) -> dict:
    from build_review_videos import build_review_videos

    return build_review_videos(out_dir)


def _publish_registry_clip(
    *,
    registry_root: Path,
    tag: str,
    asset_id: str,
    action_name: str,
    clip_id: str,
    clip_dir: Path,
    ue_import_manifest: Path,
    usage_scope: str,
    registry_schema: str,
    require_formal_registration_false: bool,
) -> Path:
    """Atomically add one final clip to an isolated non-formal registry."""
    if _SAFE_TAG.fullmatch(str(tag)) is None:
        raise ValueError(f"unsafe technical registry tag: {tag!r}")
    if action_name not in {"Walking", "Standing_Idle"}:
        raise ValueError(f"unsupported human baseline action: {action_name!r}")

    clip_dir = Path(clip_dir).resolve()
    manifest_path = Path(ue_import_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag or manifest.get("asset_id") != asset_id:
        raise ValueError("UE import manifest identity does not match registry clip")
    if manifest.get("usage_scope") != usage_scope:
        raise ValueError(
            f"UE import manifest must remain {usage_scope}"
        )
    if (
        require_formal_registration_false
        and manifest.get("formal_registration_authorized") is not False
    ):
        raise ValueError("UE import manifest improperly authorizes registration")

    evidence = {
        "clip_id": str(clip_id),
        "spec": file_descriptor(clip_dir / "spec.json"),
        "runtime_gate": file_descriptor(clip_dir / "runtime_gate.json"),
        "actor_visual_metadata": file_descriptor(
            clip_dir / "videos" / "actor_visual_metadata.json"
        ),
        "apartment_video": file_descriptor(
            clip_dir / "videos" / "apartment_v1_view0.mp4"
        ),
        "annotated_review_video": file_descriptor(
            clip_dir / "videos" / "side_by_side_review_annotated.mp4"
        ),
    }

    registry_root = Path(registry_root).resolve()
    registry_root.mkdir(parents=True, exist_ok=True)
    registry_path = registry_root / f"{tag}.json"
    lock_path = registry_root / f".{tag}.registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if registry_path.exists():
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != registry_schema:
                    raise ValueError("existing human registry schema is not supported")
                if payload.get("tag") != tag or payload.get("asset_id") != asset_id:
                    raise ValueError("existing technical registry identity does not match")
                clips = dict(payload.get("clips", {}))
            else:
                clips = {}

            clips[action_name] = evidence
            payload = {
                "schema_version": registry_schema,
                "usage_scope": usage_scope,
                "formal_registry_promotion": False,
                "tag": tag,
                "asset_id": asset_id,
                "blueprint": manifest.get("content", {}).get("blueprint"),
                "ue_import_manifest": file_descriptor(manifest_path),
                "clips": clips,
            }
            return atomic_write_json(registry_path, payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def publish_technical_registry_clip(
    *,
    registry_root: Path,
    tag: str,
    asset_id: str,
    action_name: str,
    clip_id: str,
    clip_dir: Path,
    ue_import_manifest: Path,
) -> Path:
    """Add one stable-template technical-spike clip to its registry."""
    return _publish_registry_clip(
        registry_root=registry_root,
        tag=tag,
        asset_id=asset_id,
        action_name=action_name,
        clip_id=clip_id,
        clip_dir=clip_dir,
        ue_import_manifest=ue_import_manifest,
        usage_scope=USAGE_SCOPE,
        registry_schema=REGISTRY_SCHEMA,
        require_formal_registration_false=False,
    )


def publish_research_candidate_registry_clip(
    *,
    registry_root: Path,
    tag: str,
    asset_id: str,
    action_name: str,
    clip_id: str,
    clip_dir: Path,
    ue_import_manifest: Path,
) -> Path:
    """Add one native Rocketbox research-candidate clip to its registry."""
    return _publish_registry_clip(
        registry_root=registry_root,
        tag=tag,
        asset_id=asset_id,
        action_name=action_name,
        clip_id=clip_id,
        clip_dir=clip_dir,
        ue_import_manifest=ue_import_manifest,
        usage_scope=RESEARCH_CANDIDATE_USAGE_SCOPE,
        registry_schema=RESEARCH_CANDIDATE_REGISTRY_SCHEMA,
        require_formal_registration_false=True,
    )


def _verified_artifact_descriptor(artifact: dict, *, label: str) -> tuple[Path, dict]:
    try:
        path = Path(artifact["path"]).resolve()
        expected_sha256 = str(artifact["sha256"])
        expected_size = int(artifact["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} descriptor") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a direct regular file: {path}")
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} descriptor no longer matches: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    return path, payload


def publish_controlled_animal_registry_clip(
    *,
    registry_root: Path,
    source: dict,
    action_name: str,
    clip_id: str,
    clip_dir: Path,
) -> Path:
    """Publish one authenticated controlled-animal Apartment review clip."""
    tag = str(source.get("tag") or "")
    asset_id = str(source.get("asset_id") or "")
    species = str(source.get("species") or "")
    gate = source.get("controlled_animal_gate", {})
    if _SAFE_TAG.fullmatch(tag) is None:
        raise ValueError(f"unsafe controlled-animal registry tag: {tag!r}")
    if action_name not in {"Walking", "Idle"}:
        raise ValueError(f"unsupported controlled-animal action: {action_name!r}")
    if (
        source.get("asset_class") != "animal"
        or not asset_id
        or not species
        or gate.get("schema") != CONTROLLED_ANIMAL_GATE_SCHEMA
        or gate.get("status") != "approved_for_research_candidate_apartment"
        or gate.get("asset_id") != asset_id
        or gate.get("tag") != tag
        or gate.get("formal_dataset_registration_authorized") is not False
    ):
        raise ValueError("controlled-animal source gate identity is invalid")

    decision_path, decision = _verified_artifact_descriptor(
        gate.get("animation_decision", {}), label="animation decision"
    )
    import_path, imported = _verified_artifact_descriptor(
        gate.get("ue_import_result", {}), label="UE import result"
    )
    imported_result = {
        item.get("legacy_tag"): item for item in imported.get("results", [])
    }.get(asset_id)
    if (
        decision.get("asset_id") != asset_id
        or decision.get("decision") != "approved_for_ue_apartment"
        or decision.get("decision_sha256")
        != gate.get("animation_decision", {}).get("decision_sha256")
        or imported.get("schema") != "pixal_animal_ue_import_result_v1"
        or not imported_result
        or imported_result.get("tag") != tag
        or imported_result.get("source_sha256") != gate.get("ue_source_sha256")
        or set(imported_result.get("actions", [])) != {"Idle", "Walking"}
    ):
        raise ValueError("controlled-animal approval/import evidence is inconsistent")

    clip_dir = Path(clip_dir).resolve()
    evidence = {
        "clip_id": str(clip_id),
        "spec": file_descriptor(clip_dir / "spec.json"),
        "runtime_gate": file_descriptor(clip_dir / "runtime_gate.json"),
        "actor_visual_metadata": file_descriptor(
            clip_dir / "videos" / "actor_visual_metadata.json"
        ),
        "apartment_video": file_descriptor(
            clip_dir / "videos" / "apartment_v1_view0.mp4"
        ),
        "topdown_review_video": file_descriptor(
            clip_dir / "videos" / "topdown_review.mp4"
        ),
        "annotated_review_video": file_descriptor(
            clip_dir / "videos" / "side_by_side_review_annotated.mp4"
        ),
    }

    registry_root = Path(registry_root).resolve()
    registry_root.mkdir(parents=True, exist_ok=True)
    registry_path = registry_root / f"{tag}.json"
    lock_path = registry_root / f".{tag}.registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if registry_path.exists():
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != CONTROLLED_ANIMAL_REGISTRY_SCHEMA:
                    raise ValueError("existing controlled-animal registry schema changed")
                if (
                    payload.get("tag") != tag
                    or payload.get("asset_id") != asset_id
                    or payload.get("species") != species
                ):
                    raise ValueError("existing controlled-animal registry identity changed")
                clips = dict(payload.get("clips", {}))
            else:
                clips = {}
            clips[action_name] = evidence
            payload = {
                "schema_version": CONTROLLED_ANIMAL_REGISTRY_SCHEMA,
                "usage_scope": RESEARCH_CANDIDATE_USAGE_SCOPE,
                "formal_registry_promotion": False,
                "tag": tag,
                "asset_id": asset_id,
                "species": species,
                "breed": source.get("breed"),
                "sampled_attributes": source.get("sampled_attributes", {}),
                "blueprint": imported_result.get("blueprint"),
                "animation_decision": file_descriptor(decision_path),
                "ue_import_result": file_descriptor(import_path),
                "ue_source_sha256": gate.get("ue_source_sha256"),
                "clips": clips,
            }
            return atomic_write_json(registry_path, payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def publish_stable_animal_registry_clip(
    *,
    registry_root: Path,
    source: dict,
    action_name: str,
    clip_id: str,
    clip_dir: Path,
) -> Path:
    """Publish one authenticated native-template animal Apartment clip.

    This registry deliberately preserves the missing human-review state.  A
    successful automatic render is a research candidate, never an implicit
    formal-dataset or human-visual approval.
    """
    tag = str(source.get("tag") or "")
    asset_id = str(source.get("asset_id") or "")
    template_id = str(source.get("template_id") or "")
    species = str(source.get("species") or "")
    breed = str(source.get("breed") or "")
    gate = source.get("stable_animal_gate", {})
    if _SAFE_TAG.fullmatch(tag) is None:
        raise ValueError(f"unsafe stable-animal registry tag: {tag!r}")
    if action_name not in {"Walking", "Idle"}:
        raise ValueError(f"unsupported stable-animal action: {action_name!r}")
    if (
        source.get("asset_class") != "animal"
        or not asset_id
        or not template_id
        or not species
        or not breed
        or gate.get("schema") != STABLE_ANIMAL_GATE_SCHEMA
        or gate.get("status")
        != "approved_for_automated_research_candidate_apartment"
        or gate.get("asset_id") != asset_id
        or gate.get("template_id") != template_id
        or gate.get("tag") != tag
        or gate.get("species") != species
        or gate.get("breed") != breed
        or gate.get("human_visual_review") != "pending"
        or gate.get("formal_dataset_registration_authorized") is not False
    ):
        raise ValueError("stable-animal source gate identity is invalid")

    template_registry_path, template_registry = _verified_artifact_descriptor(
        gate.get("template_registry", {}), label="stable template registry"
    )
    import_path, imported = _verified_artifact_descriptor(
        gate.get("ue_import_result", {}), label="stable UE import result"
    )
    deformation_path, deformation = _verified_artifact_descriptor(
        gate.get("deformation_audit", {}), label="stable deformation audit"
    )
    entry = {
        item.get("template_id"): item
        for item in template_registry.get("entries", [])
    }.get(template_id)
    imported_result = {
        item.get("template_id"): item for item in imported.get("results", [])
    }.get(template_id)
    direction = gate.get("direction", {})
    source_sha256 = str(gate.get("source_sha256") or "")
    try:
        source_yaw = float(source.get("walking_forward_yaw_offset_deg"))
        cardinal_yaw = float(direction.get("cardinal_yaw_deg"))
    except (TypeError, ValueError) as exc:
        raise ValueError("stable-animal cardinal direction is invalid") from exc
    if (
        template_registry.get("schema")
        != "avengine_quaternius_stable_template_registry_v1"
        or imported.get("schema") != "stable_animal_ue_import_result_v1"
        or deformation.get("schema")
        != "avengine_skinned_deformation_audit_v1"
        or deformation.get("overall") != "passed"
        or deformation.get("input_sha256") != source_sha256
        or deformation.get("formal_dataset_registration_authorized") is not False
        or not entry
        or not imported_result
        or entry.get("runtime_glb", {}).get("sha256") != source_sha256
        or entry.get("deformation_audit", {}).get("sha256")
        != gate.get("deformation_audit", {}).get("sha256")
        or entry.get("direction") != direction
        or not str(entry.get("qa", {}).get("walking_deformation", "")).startswith(
            "passed_"
        )
        or not str(entry.get("qa", {}).get("idle_deformation", "")).startswith(
            "passed_"
        )
        or direction.get("automatic_fine_yaw_inference") is not False
        or direction.get("review_status")
        != "agent_selected_pending_human_review"
        or source_yaw != cardinal_yaw
        or imported_result.get("asset_id") != asset_id
        or imported_result.get("tag") != tag
        or imported_result.get("source_sha256") != source_sha256
        or set(imported_result.get("actions", [])) != {"Idle", "Walking"}
        or imported_result.get("status", "passed") != "passed"
        or imported_result.get("human_review_status")
        != "agent_selected_pending_human_review"
        or imported_result.get("formal_dataset_registration_authorized") is not False
        or not imported_result.get("blueprint")
    ):
        raise ValueError("stable-animal template/import evidence is inconsistent")

    clip_dir = Path(clip_dir).resolve()
    evidence = {
        "clip_id": str(clip_id),
        "spec": file_descriptor(clip_dir / "spec.json"),
        "runtime_gate": file_descriptor(clip_dir / "runtime_gate.json"),
        "actor_visual_metadata": file_descriptor(
            clip_dir / "videos" / "actor_visual_metadata.json"
        ),
        "apartment_video": file_descriptor(
            clip_dir / "videos" / "apartment_v1_view0.mp4"
        ),
        "topdown_review_video": file_descriptor(
            clip_dir / "videos" / "topdown_review.mp4"
        ),
        "annotated_review_video": file_descriptor(
            clip_dir / "videos" / "side_by_side_review_annotated.mp4"
        ),
    }
    audio_path = clip_dir / "binaural.wav"
    schedule_path = clip_dir / "binaural_source_schedule.json"
    if audio_path.is_file():
        evidence["binaural_audio"] = file_descriptor(audio_path)
    if schedule_path.is_file():
        evidence["binaural_source_schedule"] = file_descriptor(schedule_path)

    registry_root = Path(registry_root).resolve()
    registry_root.mkdir(parents=True, exist_ok=True)
    registry_path = registry_root / f"{tag}.json"
    lock_path = registry_root / f".{tag}.registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if registry_path.exists():
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != STABLE_ANIMAL_REGISTRY_SCHEMA:
                    raise ValueError("existing stable-animal registry schema changed")
                if (
                    payload.get("tag") != tag
                    or payload.get("asset_id") != asset_id
                    or payload.get("template_id") != template_id
                    or payload.get("species") != species
                    or payload.get("breed") != breed
                    or payload.get("human_visual_review") != "pending"
                    or payload.get("formal_registry_promotion") is not False
                ):
                    raise ValueError("existing stable-animal registry identity changed")
                clips = dict(payload.get("clips", {}))
            else:
                clips = {}
            clips[action_name] = evidence
            payload = {
                "schema_version": STABLE_ANIMAL_REGISTRY_SCHEMA,
                "usage_scope": RESEARCH_CANDIDATE_USAGE_SCOPE,
                "formal_registry_promotion": False,
                "human_visual_review": "pending",
                "tag": tag,
                "asset_id": asset_id,
                "template_id": template_id,
                "species": species,
                "breed": breed,
                "sampled_attributes": source.get("sampled_attributes", {}),
                "blueprint": imported_result.get("blueprint"),
                "source_sha256": source_sha256,
                "direction": direction,
                "template_registry": file_descriptor(template_registry_path),
                "deformation_audit": file_descriptor(deformation_path),
                "ue_import_result": file_descriptor(import_path),
                "clips": clips,
            }
            return atomic_write_json(registry_path, payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def finalize_human_apartment_clip(
    *,
    spec_path: Path,
    out_dir: Path,
    clip_id: str,
    publish_registry: bool = True,
) -> dict:
    """Publish the complete review/evidence package after a successful render."""
    source_spec_path = Path(spec_path).resolve()
    out_dir = Path(out_dir).resolve()
    spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
    copied_spec = atomic_write_bytes(out_dir / "spec.json", source_spec_path.read_bytes())

    runtime_gate_path = out_dir / "runtime_gate.json"
    visual_metadata_path = out_dir / "videos" / "actor_visual_metadata.json"
    apartment_video = out_dir / "videos" / "apartment_v1_view0.mp4"
    for required in (runtime_gate_path, visual_metadata_path, apartment_video):
        if not required.is_file():
            raise ValueError(f"missing rendered evidence: {required}")
    visual_metadata = json.loads(visual_metadata_path.read_text(encoding="utf-8"))
    if visual_metadata.get("automatic_checks", {}).get("overall") != "passed":
        raise ValueError("actor visual metadata automatic checks did not pass")

    scene = _compose_scene(copied_spec)
    furniture, walls = _apartment_obstacles(spec)
    flag_details = build_flag_details(
        spec,
        scene,
        furniture_bboxes=furniture,
        wall_bboxes=walls,
    )
    atomic_write_json(out_dir / "flags.json", flag_details["aggregate"])
    atomic_write_json(out_dir / "flag_details.json", flag_details)

    audio_path = out_dir / "binaural.wav"
    if not audio_path.exists():
        sources = list(spec.get("sources", []))
        all_silent = all(
            bool(source.get("mute_audio"))
            or source.get("audio_lookup") == "silent"
            for source in sources
        )
        if not all_silent:
            raise ValueError(
                "non-silent human clip needs a rendered binaural.wav before review"
            )
        audio_config = spec.get("audio_config", {})
        write_silent_wav(
            audio_path,
            duration_s=float(spec["render_config"]["duration_s"]),
            sample_rate_hz=int(audio_config.get("sample_rate_hz", 16000)),
            channels=2,
        )

    _compute_metadata(copied_spec, out_dir, str(clip_id))
    review_outputs = _build_reviews(out_dir)
    annotated = Path(review_outputs["annotated"])
    if not annotated.is_file():
        raise ValueError("annotated apartment review video was not published")

    runtime_gate = json.loads(runtime_gate_path.read_text(encoding="utf-8"))
    spec_by_tag = {
        str(source["tag"]): source
        for source in spec.get("sources", [])
        if source.get("tag")
    }
    registry_paths = []
    for gate in runtime_gate.get("human_gate_evidence", []) if publish_registry else []:
        tag = str(gate["tag"])
        source = spec_by_tag.get(tag)
        if source is None:
            raise ValueError(f"rendered human gate tag is absent from spec: {tag}")
        usage_scope = gate.get(
            "usage_scope",
            USAGE_SCOPE if "asset_dir" in gate else None,
        )
        common = {
            "registry_root": out_dir.parent / "registry",
            "tag": tag,
            "asset_id": str(gate["asset_id"]),
            "action_name": str(source.get("wanted_anim") or "Walking"),
            "clip_id": str(clip_id),
            "clip_dir": out_dir,
        }
        if usage_scope == USAGE_SCOPE:
            asset_dir = Path(gate["asset_dir"]).resolve()
            registry_paths.append(
                publish_technical_registry_clip(
                    **common,
                    ue_import_manifest=asset_dir / "ue_import_manifest.json",
                )
            )
        elif usage_scope == RESEARCH_CANDIDATE_USAGE_SCOPE:
            manifest_path = Path(gate["ue_import_manifest_path"]).resolve()
            registry_paths.append(
                publish_research_candidate_registry_clip(
                    **common,
                    ue_import_manifest=manifest_path,
                )
            )
        else:
            raise ValueError(
                f"unsupported rendered human usage scope for {tag}: {usage_scope!r}"
            )

    if publish_registry:
        for source in spec.get("sources", []):
            if "controlled_animal_gate" in source:
                registry_paths.append(
                    publish_controlled_animal_registry_clip(
                        registry_root=out_dir.parent / "registry",
                        source=source,
                        action_name=str(source.get("wanted_anim") or "Walking"),
                        clip_id=str(clip_id),
                        clip_dir=out_dir,
                    )
                )
            if "stable_animal_gate" in source:
                registry_paths.append(
                    publish_stable_animal_registry_clip(
                        registry_root=out_dir.parent / "registry",
                        source=source,
                        action_name=str(source.get("wanted_anim") or "Walking"),
                        clip_id=str(clip_id),
                        clip_dir=out_dir,
                    )
                )

    return {
        **review_outputs,
        "spec": copied_spec,
        "flags": out_dir / "flags.json",
        "flag_details": out_dir / "flag_details.json",
        "metadata": out_dir / "apartment_v1_metadata.json",
        "audio": audio_path,
        "registries": registry_paths,
    }
