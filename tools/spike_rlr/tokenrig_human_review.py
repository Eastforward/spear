#!/usr/bin/env python3
"""Hash-locked contract and consolidated review builder for Route-2 humans."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


MOTIONS = {"walking": "Walking", "standing_idle": "Standing_Idle"}
VIEWS = ("front", "side", "top", "feet", "skeleton")
STATIC_EVIDENCE = (
    "bind_front.png",
    "bind_back.png",
    "bind_side.png",
    "bind_top.png",
    "skeleton_overlay.png",
    "weights_contact.png",
    "texture_compare.png",
    "joint_hierarchy.txt",
)
AGENT_VISUAL_CHECKS = (
    "feet_contact_reasonable",
    "penetration_within_one_centimeter",
    "feet_not_inverted",
    "no_visible_hovering",
    "shoulders_and_hips_stable",
    "trousers_intact",
    "attachments_stable",
    "motion_faces_negative_y",
    "loops_visually_clean",
    "pbr_texture_preserved",
    "skeleton_tracks_mesh",
)
PASS_STATUS = "agent_qa_passed_pending_user_acceptance"
REJECTED_STATUS = "rejected"
PENDING_STATUS = "pending_agent_visual_qa"
REVIEW_SCHEMA = "tokenrig_human_dynamic_review_v1"
MEDIA_QA_SCHEMA = "tokenrig_human_media_qa_v1"
DECISION_SCHEMA = "tokenrig_human_agent_visual_qa_v1"
CATALOG_SCHEMA = "tokenrig_route2_review_catalog_v1"
_ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ReviewContractError(RuntimeError):
    """A review snapshot is missing, stale, unsafe, or contradictory."""


class ReviewNotAccepted(ReviewContractError):
    """The current exact snapshot has not passed agent visual QA."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _real_directory(path: Path, description: str) -> Path:
    path = _absolute(path)
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ReviewContractError(f"{description} must be a direct real directory: {path}")
    return path


def _regular_file(path: Path, root: Path, description: str) -> Path:
    path = _absolute(path)
    root = _real_directory(root, f"{description} root")
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ReviewContractError(f"{description} must be a direct regular file: {path}")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReviewContractError(f"{description} is outside its authenticated root") from error
    if path.stat().st_size <= 0:
        raise ReviewContractError(f"{description} is empty")
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewContractError(f"{description} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ReviewContractError(f"{description} must contain an object")
    return value


def _record(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    path = Path(path)
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if filename is not None:
        value["filename"] = filename
    return value


def _validate_record(path: Path, record: Any, description: str, *, filename: str | None = None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ReviewContractError(f"{description} descriptor is missing")
    if filename is not None and record.get("filename") != filename:
        raise ReviewContractError(f"{description} filename is not canonical")
    if record.get("sha256") != sha256_file(path):
        raise ReviewContractError(f"{description} SHA-256 does not match the review snapshot")
    if record.get("size_bytes") != path.stat().st_size:
        raise ReviewContractError(f"{description} size does not match the review snapshot")
    return _record(path, filename=filename)


def _external_path(record: Any, description: str) -> Path:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ReviewContractError(f"{description} has no absolute authenticated path")
    supplied = Path(record["path"])
    if not supplied.is_absolute():
        raise ReviewContractError(f"{description} path is not absolute")
    path = _absolute(supplied)
    return _regular_file(path, path.parent, description)


def agent_decision_path(review_dir: Path) -> Path:
    review_dir = _absolute(review_dir)
    return review_dir.with_name(f"{review_dir.name}.agent_visual_qa.json")


def validated_review_snapshot(review_dir: Path) -> dict[str, Any]:
    """Validate every byte that an agent or user sees on the review page."""

    root = _real_directory(review_dir, "dynamic review bundle")
    manifest_path = _regular_file(root / "review_manifest.json", root, "review_manifest.json")
    manifest = _load_json(manifest_path, "review manifest")
    encoded = json.dumps(manifest, sort_keys=True)
    if manifest.get("schema") != REVIEW_SCHEMA:
        raise ReviewContractError(f"review manifest schema must be {REVIEW_SCHEMA}")
    asset_id = manifest.get("asset_id")
    if not isinstance(asset_id, str) or not _ASSET_RE.fullmatch(asset_id):
        raise ReviewContractError("review manifest asset_id is invalid")
    if manifest.get("automatic_checks") != "passed":
        raise ReviewContractError("review automatic checks are not passed")
    if manifest.get("canonical_front") != "negative-y":
        raise ReviewContractError("review does not prove FRONT -Y")
    if manifest.get("canonical_up") != "positive-z":
        raise ReviewContractError("review does not prove UP +Z")
    if manifest.get("fixed_floor_z_m") != 0.0:
        raise ReviewContractError("review floor is not Z=0")
    if manifest.get("agent_visual_qa") != PENDING_STATUS:
        raise ReviewContractError("render manifest may only start at pending agent visual QA")
    if manifest.get("user_acceptance") != "pending_user_review" or "user_approved" in encoded:
        raise ReviewContractError("review manifest must not claim user approval")
    if not isinstance(manifest.get("display_label"), str) or not manifest["display_label"].strip():
        raise ReviewContractError("review display label is missing")
    if not isinstance(manifest.get("instance_kind"), str) or not _ASSET_RE.fullmatch(manifest["instance_kind"]):
        raise ReviewContractError("review instance kind is invalid")

    upstream = manifest.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("asset_id") != asset_id:
        raise ReviewContractError("review upstream asset identity is stale")
    upstream_paths: dict[str, Path] = {}
    upstream_sha256: dict[str, str] = {}
    for key in ("static_qa", "bind_pose", "retarget_manifest", "retarget_metrics"):
        path = _external_path(upstream.get(key), f"upstream {key}")
        _validate_record(path, upstream[key], f"upstream {key}")
        upstream_paths[key] = path
        upstream_sha256[key] = sha256_file(path)
    glbs = upstream.get("glbs")
    if not isinstance(glbs, Mapping) or set(glbs) != set(MOTIONS):
        raise ReviewContractError("review upstream GLBs do not cover exactly Walk and Idle")
    for motion in MOTIONS:
        path = _external_path(glbs[motion], f"upstream {motion} GLB")
        _validate_record(path, glbs[motion], f"upstream {motion} GLB")
        upstream_paths[f"glb:{motion}"] = path
        upstream_sha256[f"glb:{motion}"] = sha256_file(path)

    asset_root = root.parent
    static_root = upstream_paths["static_qa"].parent
    retarget_root = upstream_paths["retarget_manifest"].parent
    if static_root.parent != asset_root:
        raise ReviewContractError("static QA bundle is not contained by the asset root")
    if retarget_root.parent != asset_root:
        raise ReviewContractError("retarget bundle is not contained by the asset root")
    if upstream_paths["bind_pose"].parent != static_root:
        raise ReviewContractError("bind pose is not contained by the static QA bundle")
    if upstream_paths["retarget_metrics"].parent != retarget_root or any(
        upstream_paths[f"glb:{motion}"].parent != retarget_root for motion in MOTIONS
    ):
        raise ReviewContractError("retarget artifacts are not contained by one bundle")

    evidence = upstream.get("static_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(STATIC_EVIDENCE):
        raise ReviewContractError("static evidence set is incomplete or unexpected")
    static_paths: dict[str, Path] = {}
    static_sha256: dict[str, str] = {}
    for filename in STATIC_EVIDENCE:
        path = _external_path(evidence[filename], filename)
        _validate_record(path, evidence[filename], filename)
        static_paths[filename] = path
        static_sha256[filename] = sha256_file(path)
    if any(path.parent != static_root for path in static_paths.values()):
        raise ReviewContractError("static evidence is not contained by the static QA bundle")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "renderer",
        "ffmpeg",
        "ffprobe",
    }:
        raise ReviewContractError("review execution contract is incomplete")
    execution_paths: dict[str, Path] = {}
    execution_sha256: dict[str, str] = {}
    execution_versions: dict[str, str] = {}
    for name in ("renderer", "ffmpeg", "ffprobe"):
        descriptor = execution[name]
        path = _external_path(descriptor, f"execution {name}")
        _validate_record(path, descriptor, f"execution {name}")
        if name in {"ffmpeg", "ffprobe"}:
            version = descriptor.get("version") if isinstance(descriptor, Mapping) else None
            if not isinstance(version, str) or not version.lower().startswith(name):
                raise ReviewContractError(f"execution {name} version is invalid")
            execution_versions[name] = version
        execution_paths[name] = path
        execution_sha256[name] = sha256_file(path)

    media_qa_path = _regular_file(root / "media_qa.json", root, "media_qa.json")
    _validate_record(media_qa_path, manifest.get("media_qa"), "media_qa.json", filename="media_qa.json")
    media_qa = _load_json(media_qa_path, "media QA")
    if media_qa.get("schema") != MEDIA_QA_SCHEMA or media_qa.get("automatic_checks") != "passed" or media_qa.get("asset_id") != asset_id:
        raise ReviewContractError("media QA schema, asset, or automatic checks are stale")

    actions = manifest.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(MOTIONS):
        raise ReviewContractError("review actions must be exactly Walk and Idle")
    if set(media_qa.get("actions", {})) != set(MOTIONS):
        raise ReviewContractError("media QA actions must be exactly Walk and Idle")
    media_paths: dict[str, Path] = {}
    media_sha256: dict[str, str] = {}
    expected_files = {"review_manifest.json", "media_qa.json"}
    for motion, action_name in MOTIONS.items():
        action = actions[motion]
        if not isinstance(action, Mapping) or action.get("action_name") != action_name or action.get("fps") != 30:
            raise ReviewContractError(f"action contract is stale: {motion}")
        views = action.get("views")
        if not isinstance(views, Mapping) or set(views) != set(VIEWS):
            raise ReviewContractError(f"review view set is incomplete: {motion}")
        qa_views = media_qa["actions"].get(motion)
        if not isinstance(qa_views, Mapping) or set(qa_views) != set(VIEWS):
            raise ReviewContractError(f"media QA view set is incomplete: {motion}")
        for view in VIEWS:
            kinds = views[view]
            if not isinstance(kinds, Mapping) or set(kinds) != {"png", "mp4"}:
                raise ReviewContractError(f"review media kinds are incomplete: {motion}/{view}")
            for kind in ("png", "mp4"):
                filename = f"{motion}_{view}.{kind}"
                expected_files.add(filename)
                path = _regular_file(root / filename, root, filename)
                _validate_record(path, kinds[kind], filename, filename=filename)
                key = f"{motion}:{view}:{kind}"
                media_paths[key] = path
                media_sha256[key] = sha256_file(path)
    actual_files = {item.name for item in root.iterdir()}
    if actual_files != expected_files:
        raise ReviewContractError("dynamic review bundle has missing or unexpected files")
    return {
        "asset_id": asset_id,
        "display_label": manifest["display_label"].strip(),
        "instance_kind": manifest["instance_kind"],
        "review_dir": str(root),
        "review_manifest": manifest,
        "review_manifest_path": manifest_path,
        "review_manifest_sha256": sha256_file(manifest_path),
        "media_qa_path": media_qa_path,
        "media_qa_sha256": sha256_file(media_qa_path),
        "upstream_paths": upstream_paths,
        "upstream_sha256": upstream_sha256,
        "static_evidence_paths": static_paths,
        "static_evidence_sha256": static_sha256,
        "dynamic_media_paths": media_paths,
        "dynamic_media_sha256": media_sha256,
        "execution_paths": execution_paths,
        "execution_sha256": execution_sha256,
        "execution_versions": execution_versions,
        "agent_visual_qa": PENDING_STATUS,
        "user_acceptance": "pending_user_review",
    }


def _decision_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "review_manifest_sha256": snapshot["review_manifest_sha256"],
        "media_qa_sha256": snapshot["media_qa_sha256"],
        "upstream_sha256": dict(snapshot["upstream_sha256"]),
        "static_evidence_sha256": dict(snapshot["static_evidence_sha256"]),
        "dynamic_media_sha256": dict(snapshot["dynamic_media_sha256"]),
        "execution_sha256": dict(snapshot["execution_sha256"]),
        "execution_versions": dict(snapshot["execution_versions"]),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with Path(path).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ReviewContractError(f"agent visual QA decision already exists: {path}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_agent_visual_qa(
    review_dir: Path,
    *,
    status: str,
    reviewer: str,
    notes: str,
    checks: Mapping[str, bool],
) -> Path:
    if status not in {PASS_STATUS, REJECTED_STATUS}:
        raise ReviewContractError("agent visual QA status must be pending-user pass or rejected")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ReviewContractError("agent visual QA reviewer must be non-empty")
    if not isinstance(notes, str) or not notes.strip():
        raise ReviewContractError("agent visual QA notes must be non-empty")
    if not isinstance(checks, Mapping) or set(checks) != set(AGENT_VISUAL_CHECKS) or any(not isinstance(value, bool) for value in checks.values()):
        raise ReviewContractError("agent visual QA checklist is incomplete or unexpected")
    if status == PASS_STATUS and not all(checks.values()):
        raise ReviewContractError("all visual checks must pass before agent acceptance")
    snapshot = validated_review_snapshot(review_dir)
    destination = agent_decision_path(Path(review_dir))
    if os.path.lexists(destination):
        raise ReviewContractError(f"agent visual QA decision already exists: {destination}")
    _real_directory(destination.parent, "agent decision parent")
    payload = {
        "schema": DECISION_SCHEMA,
        "asset_id": snapshot["asset_id"],
        "status": status,
        "reviewer_kind": "agent",
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "checks": dict(checks),
        "snapshot": _decision_snapshot(snapshot),
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise ReviewContractError("agent decision may not claim user approval")
    _write_exclusive(destination, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    destination.chmod(0o444)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)
    return destination


def read_agent_visual_qa(review_dir: Path) -> dict[str, Any]:
    path = agent_decision_path(Path(review_dir))
    try:
        snapshot = validated_review_snapshot(review_dir)
    except ReviewContractError as error:
        if os.path.lexists(path):
            raise ReviewContractError(
                f"review snapshot changed after the agent visual QA decision: {error}"
            ) from error
        raise
    if not os.path.lexists(path):
        return {
            "schema": DECISION_SCHEMA,
            "asset_id": snapshot["asset_id"],
            "status": PENDING_STATUS,
            "reviewer_kind": "agent",
            "user_acceptance": "pending_user_review",
        }
    path = _regular_file(path, path.parent, "agent visual QA decision")
    payload = _load_json(path, "agent visual QA decision")
    if payload.get("schema") != DECISION_SCHEMA or payload.get("asset_id") != snapshot["asset_id"]:
        raise ReviewContractError("agent visual QA decision schema or asset is stale")
    if payload.get("status") not in {PASS_STATUS, REJECTED_STATUS} or payload.get("reviewer_kind") != "agent":
        raise ReviewContractError("agent visual QA decision status or reviewer kind is invalid")
    if payload.get("user_acceptance") != "pending_user_review" or "user_approved" in json.dumps(payload):
        raise ReviewContractError("agent visual QA decision must not claim user approval")
    if payload.get("snapshot") != _decision_snapshot(snapshot):
        raise ReviewContractError("review snapshot changed after the agent visual QA decision")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(AGENT_VISUAL_CHECKS):
        raise ReviewContractError("agent visual QA decision checklist is stale")
    if payload["status"] == PASS_STATUS and not all(value is True for value in checks.values()):
        raise ReviewContractError("agent visual QA pass contains a failed check")
    return payload


def assert_agent_qa_passed(review_dir: Path) -> dict[str, Any]:
    decision = read_agent_visual_qa(review_dir)
    if decision.get("status") != PASS_STATUS:
        raise ReviewNotAccepted(f"agent visual QA is {decision.get('status', PENDING_STATUS)}")
    return decision


def _validate_destination(destination: Path) -> Path:
    destination = _absolute(destination)
    if os.path.lexists(destination):
        raise ReviewContractError(f"consolidated review output already exists: {destination}")
    _real_directory(destination.parent, "consolidated review parent")
    return destination


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise ReviewContractError("atomic renameat2 no-replace is unavailable") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise ReviewContractError(f"consolidated review output already exists: {destination}")
    raise ReviewContractError(f"atomic review publication failed: {os.strerror(number)}")


def _served_media_url(
    *, asset_id: str, motion: str, view: str, kind: str, sha256: str
) -> str:
    return (
        f"/media/{quote(asset_id, safe='')}/{motion}/{view}/{kind}"
        f"?expected_sha256={sha256}"
    )


def _served_static_url(*, asset_id: str, filename: str, sha256: str) -> str:
    return (
        f"/static-evidence/{quote(asset_id, safe='')}/{quote(filename, safe='')}"
        f"?expected_sha256={sha256}"
    )


def _entry_from_snapshot(snapshot: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    media = {}
    for motion in MOTIONS:
        media[motion] = {}
        for view in VIEWS:
            media[motion][view] = {
                kind: {
                    "path": str(snapshot["dynamic_media_paths"][f"{motion}:{view}:{kind}"]),
                    "sha256": snapshot["dynamic_media_sha256"][f"{motion}:{view}:{kind}"],
                    "size_bytes": snapshot["dynamic_media_paths"][
                        f"{motion}:{view}:{kind}"
                    ].stat().st_size,
                }
                for kind in ("png", "mp4")
            }
    return {
        "asset_id": snapshot["asset_id"],
        "display_label": snapshot["display_label"],
        "instance_kind": snapshot["instance_kind"],
        "status": decision["status"],
        "agent_review": {
            "reviewer": decision.get("reviewer", ""),
            "notes": decision.get("notes", ""),
        },
        "user_acceptance": "pending_user_review",
        "review_dir": snapshot["review_dir"],
        "review_manifest": {
            "path": str(snapshot["review_manifest_path"]),
            "sha256": snapshot["review_manifest_sha256"],
            "size_bytes": snapshot["review_manifest_path"].stat().st_size,
        },
        "decision_path": str(agent_decision_path(Path(snapshot["review_dir"]))) if decision["status"] != PENDING_STATUS else None,
        "decision_sha256": sha256_file(agent_decision_path(Path(snapshot["review_dir"]))) if decision["status"] != PENDING_STATUS else None,
        "decision_size_bytes": agent_decision_path(Path(snapshot["review_dir"])).stat().st_size if decision["status"] != PENDING_STATUS else None,
        "snapshot": _decision_snapshot(snapshot),
        "static_evidence": {
            filename: {
                "path": str(snapshot["static_evidence_paths"][filename]),
                "sha256": snapshot["static_evidence_sha256"][filename],
                "size_bytes": snapshot["static_evidence_paths"][filename].stat().st_size,
            }
            for filename in STATIC_EVIDENCE
        },
        "media": media,
    }


def render_review_html(entries: Sequence[Mapping[str, Any]], *, output_dir: Path) -> str:
    cards = []
    rail = []
    for index, entry in enumerate(entries):
        asset_id = str(entry["asset_id"])
        safe_id = html.escape(asset_id, quote=True)
        label = html.escape(str(entry["display_label"]))
        status = html.escape(str(entry["status"]))
        rail.append(f'<a href="#{safe_id}"><strong>{label}</strong><span>{status}</span></a>')
        static_figures = []
        for filename in STATIC_EVIDENCE:
            if filename.endswith(".txt"):
                continue
            record = entry["static_evidence"][filename]
            src = html.escape(
                _served_static_url(
                    asset_id=asset_id,
                    filename=filename,
                    sha256=str(record["sha256"]),
                ),
                quote=True,
            )
            caption = html.escape(filename.replace("_", " ").replace(".png", "").title())
            static_figures.append(f'<figure><img src="{src}" loading="lazy" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        hierarchy_record = entry["static_evidence"]["joint_hierarchy.txt"]
        hierarchy = html.escape(
            _served_static_url(
                asset_id=asset_id,
                filename="joint_hierarchy.txt",
                sha256=str(hierarchy_record["sha256"]),
            ),
            quote=True,
        )
        motion_sections = []
        for motion, action_name in MOTIONS.items():
            figures = []
            for view in VIEWS:
                record = entry["media"][motion][view]
                video = html.escape(
                    _served_media_url(
                        asset_id=asset_id,
                        motion=motion,
                        view=view,
                        kind="mp4",
                        sha256=str(record["mp4"]["sha256"]),
                    ),
                    quote=True,
                )
                poster = html.escape(
                    _served_media_url(
                        asset_id=asset_id,
                        motion=motion,
                        view=view,
                        kind="png",
                        sha256=str(record["png"]["sha256"]),
                    ),
                    quote=True,
                )
                figures.append(
                    f'<figure><video controls loop muted playsinline preload="metadata" poster="{poster}" src="{video}"></video>'
                    f'<figcaption>{html.escape(view.title())}</figcaption></figure>'
                )
            hidden = "" if motion == "walking" else " hidden"
            motion_sections.append(f'<div class="motion-grid" data-motion="{motion}"{hidden}>{"".join(figures)}</div>')
        cards.append(
            f'<article id="{safe_id}" class="asset-card" data-asset-index="{index}">'
            + f'<header><div><p class="eyebrow">{html.escape(str(entry["instance_kind"]))}</p><h2>{label}</h2></div>'
            + f'<span class="status" data-status="{html.escape(str(entry["status"]), quote=True)}">{status}</span></header>'
            + '<p class="contract">FRONT -Y · UP +Z · fixed floor Z=0 · user acceptance pending</p>'
            + (
                '<p class="agent-note"><strong>Agent review:</strong> '
                + html.escape(str(entry.get("agent_review", {}).get("reviewer", "")))
                + " · "
                + html.escape(str(entry.get("agent_review", {}).get("notes", "")))
                + "</p>"
                if entry.get("agent_review", {}).get("reviewer")
                else ""
            )
            + '<details><summary>Static bind evidence</summary><div class="static-grid">'
            + "".join(static_figures)
            + f'</div><a class="hierarchy" href="{hierarchy}">Joint hierarchy</a></details>'
            + '<div class="motion-tabs"><button type="button" data-show-motion="walking" aria-pressed="true">Walk</button>'
            + '<button type="button" data-show-motion="standing_idle" aria-pressed="false">Idle</button></div>'
            + "".join(motion_sections)
            + "</article>"
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route 2 Acceptance Review</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17212b;background:#f4f6f7}*{box-sizing:border-box}body{margin:0}.shell{max-width:1500px;margin:auto;padding:18px}.mast{display:flex;justify-content:space-between;gap:18px;align-items:end;border-bottom:1px solid #c8d0d5;padding-bottom:14px}.mast h1{font-size:22px;margin:0}.mast p{margin:4px 0 0;color:#53616b}.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:18px;padding-top:18px}.rail{display:grid;align-content:start;gap:7px;position:sticky;top:12px}.rail a{display:grid;gap:3px;padding:9px;border:1px solid #c5ced4;border-radius:7px;background:white;color:inherit;text-decoration:none}.rail span{font-size:11px;color:#4e606b;overflow-wrap:anywhere}.asset-card{background:white;border:1px solid #c5ced4;border-radius:9px;padding:14px;margin-bottom:18px}.asset-card header{display:flex;justify-content:space-between;gap:12px;align-items:center}.asset-card h2{font-size:19px;margin:0}.eyebrow{font-size:11px;text-transform:uppercase;color:#61707a;margin:0 0 3px}.status{font-size:12px;color:#17643c;overflow-wrap:anywhere}.status[data-status="rejected"]{color:#9a3b34}.contract{font-size:12px;color:#51616c}.motion-tabs{display:flex;gap:7px;margin:12px 0 9px}.motion-tabs button{border:1px solid #9eacb5;border-radius:6px;background:#fff;padding:7px 13px;font-weight:700}.motion-tabs button[aria-pressed="true"]{background:#dceef3;border-color:#176f8a}.motion-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.motion-grid[hidden]{display:none}.motion-grid figure,.static-grid figure{margin:0;min-width:0}.motion-grid video{display:block;width:100%;aspect-ratio:16/9;background:#10161a}.motion-grid figcaption,.static-grid figcaption{font-size:12px;font-weight:700;margin-top:4px}.static-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0}.static-grid img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#10161a}.hierarchy{font-size:12px}summary{cursor:pointer;font-weight:700}@media(max-width:850px){.layout{grid-template-columns:1fr}.rail{position:static;grid-template-columns:repeat(2,minmax(0,1fr))}.motion-grid{grid-template-columns:1fr}.static-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.mast{align-items:start;flex-direction:column}}
</style></head><body><main class="shell"><header class="mast"><div><h1>Route 2 Acceptance Review</h1><p>Static bind + Walking / Standing Idle · hash-locked agent QA evidence</p></div><p>User acceptance remains pending</p></header><div class="layout"><nav class="rail">""" + "".join(rail) + "</nav><section>" + "".join(cards) + """</section></div></main><script>
document.querySelectorAll('.asset-card').forEach(card=>card.querySelectorAll('[data-show-motion]').forEach(button=>button.addEventListener('click',()=>{const wanted=button.dataset.showMotion;card.querySelectorAll('[data-show-motion]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));card.querySelectorAll('[data-motion]').forEach(grid=>grid.hidden=grid.dataset.motion!==wanted);})));</script></body></html>"""


def build_consolidated_bundle(review_dirs: Sequence[Path], output_dir: Path) -> Path:
    if not review_dirs:
        raise ReviewContractError("at least one dynamic review directory is required")
    destination = _validate_destination(output_dir)
    snapshots = [validated_review_snapshot(Path(item)) for item in review_dirs]
    asset_ids = [snapshot["asset_id"] for snapshot in snapshots]
    if len(asset_ids) != len(set(asset_ids)):
        raise ReviewContractError("duplicate asset_id in consolidated review inputs")
    decisions = [read_agent_visual_qa(Path(item)) for item in review_dirs]
    entries = [_entry_from_snapshot(snapshot, decision) for snapshot, decision in zip(snapshots, decisions)]
    statuses = {entry["status"] for entry in entries}
    overall = PASS_STATUS if statuses == {PASS_STATUS} else ("contains_rejected" if REJECTED_STATUS in statuses else PENDING_STATUS)
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".staging", dir=str(destination.parent)))
        review_html = render_review_html(entries, output_dir=destination)
        html_path = staging / "review.html"
        _write_exclusive(html_path, review_html.encode("utf-8"))
        catalog = {
            "schema": CATALOG_SCHEMA,
            "overall_agent_qa": overall,
            "user_acceptance": "pending_user_review",
            "entries": entries,
            "review_html": {"filename": "review.html", "sha256": sha256_file(html_path), "size_bytes": html_path.stat().st_size},
        }
        if "user_approved" in json.dumps(catalog):
            raise ReviewContractError("catalog may not claim user approval")
        catalog_path = staging / "review_catalog.json"
        _write_exclusive(catalog_path, (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        for path in (html_path, catalog_path):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            path.chmod(0o444)
        _fsync_directory(staging)
        staging.chmod(0o555)
        _rename_noreplace(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return destination / "review_catalog.json"
    finally:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            for path in staging.iterdir():
                path.chmod(0o600)
            shutil.rmtree(staging)


def validate_review_catalog(catalog_dir: Path) -> dict[str, Any]:
    root = _real_directory(catalog_dir, "consolidated review bundle")
    if {item.name for item in root.iterdir()} != {"review.html", "review_catalog.json"}:
        raise ReviewContractError("consolidated review bundle has missing or unexpected files")
    catalog_path = _regular_file(root / "review_catalog.json", root, "review_catalog.json")
    html_path = _regular_file(root / "review.html", root, "review.html")
    catalog = _load_json(catalog_path, "review catalog")
    if catalog.get("schema") != CATALOG_SCHEMA or catalog.get("user_acceptance") != "pending_user_review":
        raise ReviewContractError("review catalog schema or user-acceptance state is invalid")
    if "user_approved" in json.dumps(catalog):
        raise ReviewContractError("review catalog may not claim user approval")
    _validate_record(html_path, catalog.get("review_html"), "review.html", filename="review.html")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReviewContractError("review catalog has no entries")
    seen = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("asset_id") in seen:
            raise ReviewContractError("review catalog contains a duplicate or invalid asset")
        seen.add(entry["asset_id"])
        source = Path(str(entry.get("review_dir", "")))
        try:
            snapshot = validated_review_snapshot(source)
        except ReviewContractError as error:
            raise ReviewContractError(
                f"review source snapshot changed: {entry.get('asset_id')}: {error}"
            ) from error
        if entry.get("snapshot") != _decision_snapshot(snapshot):
            raise ReviewContractError(f"review source snapshot changed: {entry.get('asset_id')}")
        decision = read_agent_visual_qa(source)
        if decision.get("status") != PENDING_STATUS:
            current_decision_path = agent_decision_path(source)
            if (
                entry.get("decision_path") != str(current_decision_path)
                or current_decision_path.is_symlink()
                or not current_decision_path.is_file()
                or entry.get("decision_sha256") != sha256_file(current_decision_path)
            ):
                raise ReviewContractError(
                    f"review source decision changed: {entry.get('asset_id')}"
                )
        if entry.get("status") != decision.get("status"):
            raise ReviewContractError(f"review source decision changed: {entry.get('asset_id')}")
        expected_entry = _entry_from_snapshot(snapshot, decision)
        if dict(entry) != expected_entry:
            raise ReviewContractError(
                f"review catalog entry does not match validated snapshot: {entry.get('asset_id')}"
            )
    rebuilt_html = render_review_html(entries, output_dir=root).encode("utf-8")
    if html_path.read_bytes() != rebuilt_html:
        raise ReviewContractError("review.html does not match rebuilt validated entries")
    statuses = {entry["status"] for entry in entries}
    expected_overall = (
        PASS_STATUS
        if statuses == {PASS_STATUS}
        else ("contains_rejected" if REJECTED_STATUS in statuses else PENDING_STATUS)
    )
    if catalog.get("overall_agent_qa") != expected_overall:
        raise ReviewContractError("review catalog overall agent state is stale")
    return catalog


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-site")
    build.add_argument("--review-dir", type=Path, action="append", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    record = sub.add_parser("record-agent-qa")
    record.add_argument("--review-dir", type=Path, required=True)
    record.add_argument("--status", choices=(PASS_STATUS, REJECTED_STATUS), required=True)
    record.add_argument("--reviewer", required=True)
    record.add_argument("--notes", required=True)
    record.add_argument("--checks-json", type=Path, required=True)
    validate = sub.add_parser("validate-site")
    validate.add_argument("--catalog-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-site":
        result = build_consolidated_bundle(args.review_dir, args.output_dir)
        print(f"TOKENRIG_ROUTE2_REVIEW_SITE_OK {result}")
    elif args.command == "record-agent-qa":
        checks = _load_json(args.checks_json, "agent checks")
        result = record_agent_visual_qa(args.review_dir, status=args.status, reviewer=args.reviewer, notes=args.notes, checks=checks)
        print(f"TOKENRIG_AGENT_VISUAL_QA_RECORDED {result}")
    else:
        validate_review_catalog(args.catalog_dir)
        print(f"TOKENRIG_ROUTE2_REVIEW_SITE_VALID {args.catalog_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
