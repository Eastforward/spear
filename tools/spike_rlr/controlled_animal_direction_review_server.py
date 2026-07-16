#!/usr/bin/env python3
"""Manual source-pose and direction review for controlled animals.

The gate deliberately performs no automatic orientation inference and applies
no hidden mirror.  A reviewer first rejects malformed source poses (internally
twisted torso, turned head, inconsistent leg planes, or floating paws).  The
legacy v2 contract then chooses only a cardinal whole-mesh yaw.  The v3
contract separates a small reviewer-controlled rigid torso-axis alignment from
the cardinal head/tail choice, which is required for otherwise-valid Pixal
meshes whose entire horizontal frame is rotated by a non-cardinal angle.  It
never rewrites a GLB, registry, historical decision, or historical video.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping

import numpy as np
import trimesh
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    send_from_directory,
)


SPEAR_ROOT = Path(__file__).resolve().parents[2]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_MANIFEST = (
    SPEAR_ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "controlled_animal_pose_direction_manual_review_v2_20260713/review_manifest.json"
)
DEFAULT_STATE_ROOT = (
    SPEAR_ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "controlled_animal_pose_direction_review_state_v2_20260713"
)
MANIFEST_SCHEMA = "controlled_animal_pose_direction_manual_review_manifest_v2"
MANIFEST_SCHEMA_V3 = "controlled_animal_pose_direction_manual_review_manifest_v3"
STATE_SCHEMA = "controlled_animal_pose_direction_manual_review_state_v2"
STATE_SCHEMA_V3 = "controlled_animal_pose_direction_manual_review_state_v3"
DECISION_SCHEMA = "controlled_animal_pose_direction_manual_decision_v2"
DECISION_SCHEMA_V3 = "controlled_animal_pose_direction_manual_decision_v3"
ALLOWED_DELTAS = {-90.0, 90.0, 180.0}
CARDINAL_YAWS = {-90.0, 0.0, 90.0, 180.0}
ALLOWED_AXIS_DELTAS = {-15.0, -5.0, -1.0, 1.0, 5.0, 15.0}
MAX_MANUAL_AXIS_ALIGNMENT_DEG = 45.0
POSE_CHECK_HINTS = (
    "spine_is_straight",
    "head_is_aligned_with_torso",
    "front_and_hind_legs_share_consistent_planes",
    "all_paws_share_one_ground_plane",
)
INTERACTIVE_MAX_POINTS = 18_000


class ReviewServerError(RuntimeError):
    """Raised when the review manifest or mutable review state is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewServerError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReviewServerError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            return
        except FileExistsError as error:
            raise ReviewServerError(f"refusing to replace decision: {path}") from error
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_compact_json(path: Path, value: Mapping[str, Any]) -> None:
    """Persist a non-authoritative browser cache without partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-9):
        return 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-9):
        return 0.0
    return round(normalized, 6)


def _yaw_matrix_y_up(yaw_deg: float) -> np.ndarray:
    radians = math.radians(float(yaw_deg))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
    )


def _manual_preview_matrix(yaw_deg: float) -> np.ndarray:
    """Return a rotation-only manual preview transform (never a reflection)."""
    return _yaw_matrix_y_up(yaw_deg)


def _load_preview_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ReviewServerError(f"empty preview scene: {path}")
        if hasattr(loaded, "to_mesh"):
            mesh = loaded.to_mesh()
        else:  # pragma: no cover - compatibility with older trimesh
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ReviewServerError(f"preview mesh has no faces: {path}")
    return mesh


def _sample_surface_points(
    mesh: trimesh.Trimesh,
    *,
    max_points: int,
) -> np.ndarray:
    """Return deterministic area-uniform points from the real mesh surface."""

    if max_points <= 0:
        raise ReviewServerError("max_points must be positive")
    source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.flatnonzero(np.isfinite(face_areas) & (face_areas > 0.0))
    if len(faces) and len(valid_faces):
        probabilities = face_areas[valid_faces]
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(0xA11CE)
        chosen = rng.choice(valid_faces, size=max_points, replace=True, p=probabilities)
        triangles = source_vertices[faces[chosen]]
        root_u = np.sqrt(rng.random(max_points))
        v = rng.random(max_points)
        return (
            (1.0 - root_u)[:, None] * triangles[:, 0]
            + (root_u * (1.0 - v))[:, None] * triangles[:, 1]
            + (root_u * v)[:, None] * triangles[:, 2]
        )
    points = source_vertices
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points.copy()


def _render_orientation_preview(
    mesh: trimesh.Trimesh,
    destination: Path,
    *,
    yaw_deg: float,
    max_points: int = 45_000,
) -> None:
    """Render a side/top fallback image without exporting a scratch mesh.

    The previous Hunyuan review renderer accepted a mesh filename, which made
    every button click serialize a 100k-face GLB before drawing it.  That is
    unnecessarily slow for a yaw-only gate.  Here we transform the certified
    vertices in memory and plot a deterministic point silhouette; the PBR
    contact sheet remains visible next to it for appearance/identity review.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rotation = _manual_preview_matrix(yaw_deg)
    bounds_vertices = source_vertices @ rotation.T

    # Plot deterministic, area-uniform surface samples instead of a slice of
    # the vertex array.  Pixal meshes have highly non-uniform vertex density;
    # vertex-only plots made broad valid belly/chest triangles look like holes
    # when viewed end-on.  Surface samples still leave a real missing face
    # empty, while representing existing triangles faithfully.
    points = _sample_surface_points(mesh, max_points=max_points)
    vertices = points @ rotation.T

    bounds = np.vstack((bounds_vertices.min(axis=0), bounds_vertices.max(axis=0)))
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    center = bounds.mean(axis=0)

    fig, (side, top) = plt.subplots(
        1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [1.8, 1.0]}
    )
    side.scatter(
        vertices[:, 0], vertices[:, 1], s=0.45, c="#286bb3", alpha=0.22,
        linewidths=0, rasterized=True,
    )
    top.scatter(
        vertices[:, 0], vertices[:, 2], s=0.45, c="#1f8a55", alpha=0.22,
        linewidths=0, rasterized=True,
    )

    for axis, vertical_index, title, ylabel in (
        (side, 1, "SIDE: animal must stand upright", "+Y UP"),
        (top, 2, "TOP-DOWN: torso/spine axis must point right", "+Z SIDE"),
    ):
        horizontal_margin = extent[0] * 0.12
        vertical_margin = extent[vertical_index] * 0.12
        axis.set_xlim(bounds[0, 0] - horizontal_margin, bounds[1, 0] + horizontal_margin)
        axis.set_ylim(
            bounds[0, vertical_index] - vertical_margin,
            bounds[1, vertical_index] + vertical_margin,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("world +X  (HEAD / FORWARD ->)", color="#08752d")
        axis.set_ylabel(ylabel)
        arrow_y = center[vertical_index]
        axis.annotate(
            "TORSO FORWARD / +X",
            xy=(bounds[1, 0] + horizontal_margin * 0.78, arrow_y),
            xytext=(center[0] + extent[0] * 0.12, arrow_y),
            color="#08752d",
            fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": "#08752d", "lw": 3},
            ha="left",
            va="bottom",
        )
        axis.text(
            bounds[0, 0] + horizontal_margin * 0.15,
            arrow_y,
            "TORSO BACK",
            color="#a12424",
            fontweight="bold",
            ha="left",
            va="top",
        )

    side.axhline(bounds[0, 1], color="#6b7280", linestyle=":", linewidth=1.2)
    fig.suptitle(
        f"raw mesh + manual cardinal yaw {yaw_deg:+.0f} deg\n"
        "Align the TORSO/SPINE axis; never compensate a turned head with fine yaw",
        fontsize=13,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = path.resolve()
    manifest = _read_json(path)
    if (
        manifest.get("schema") not in {MANIFEST_SCHEMA, MANIFEST_SCHEMA_V3}
        or manifest.get("manifest_sha256") != _hash_without(manifest, "manifest_sha256")
        or manifest.get("asset_count") != len(manifest.get("entries", []))
        or manifest.get("formal_dataset_registration_authorized") is not False
        or manifest.get("safety", {}).get("browser_decisions_are_transform_overlays_only")
        is not True
    ):
        raise ReviewServerError("direction review manifest contract/hash is invalid")
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        asset_id = entry.get("asset_id")
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or asset_id in entries
            or entry.get("species") not in {"cat", "dog", "horse"}
            or entry.get("current_evidence_status", {}).get("walking_direction")
            not in {
                "rejected_by_user_visual_review",
                "new_canary_pending_manual_review",
                "new_canary_animation_rejected",
                "new_canary_animation_agent_approved_pending_human_review",
            }
        ):
            raise ReviewServerError("direction review entry identity/status is invalid")
        required_artifacts = (
            "prebind_lod_glb",
            "static_contact_sheet",
            "static_top_view",
        )
        optional_artifacts = (
            "walking_side",
            "walking_front",
            "idle_side",
            "apartment_walking_review",
            "apartment_walking_main",
            "apartment_walking_topdown",
            "apartment_idle_review",
            "apartment_idle_main",
            "apartment_idle_topdown",
            "apartment_spec_manifest",
            "apartment_batch_status",
            "pixal_raw_glb",
            "i23d_raw_glb",
            "reference_image",
            "generation_manifest",
            "current_bound_glb",
        )
        artifacts = entry.get("artifacts", {})
        if (
            any(name not in artifacts for name in required_artifacts)
            or not ({"pixal_input_rgba", "reference_image"} & set(artifacts))
        ):
            raise ReviewServerError("direction review core evidence is incomplete")
        for name in (
            *required_artifacts,
            "pixal_input_rgba",
            *optional_artifacts,
        ):
            if name not in artifacts:
                continue
            record = entry.get("artifacts", {}).get(name, {})
            artifact_path = Path(str(record.get("absolute_path", ""))).resolve()
            try:
                artifact_path.relative_to(AVENGINE_ROOT.resolve())
            except ValueError as error:
                raise ReviewServerError(f"artifact escaped AVEngine root: {name}") from error
            if (
                artifact_path.is_symlink()
                or not artifact_path.is_file()
                or artifact_path.stat().st_size != record.get("size_bytes")
                or _sha256_file(artifact_path) != record.get("sha256")
            ):
                raise ReviewServerError(f"direction review artifact changed: {name}")
        entries[asset_id] = entry
    return manifest, entries


def _public_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    artifact_names = (
        "pixal_raw_glb",
        "i23d_raw_glb",
        "pixal_input_rgba",
        "reference_image",
        "generation_manifest",
        "prebind_lod_glb",
        "static_contact_sheet",
        "static_top_view",
        "current_bound_glb",
        "walking_side",
        "walking_front",
        "idle_side",
        "apartment_walking_review",
        "apartment_walking_main",
        "apartment_walking_topdown",
        "apartment_idle_review",
        "apartment_idle_main",
        "apartment_idle_topdown",
    )
    return {
        "asset_id": entry["asset_id"],
        "species": entry["species"],
        "breed": entry.get("breed"),
        "profile_schema_id": entry.get("profile_schema_id"),
        "sampled_attributes": entry.get("sampled_attributes", {}),
        "artifacts": {
            name: {
                "url": entry["artifacts"][name]["server_path"],
                "absolute_path": entry["artifacts"][name]["absolute_path"],
            }
            for name in artifact_names
            if name in entry["artifacts"]
        },
        "current_evidence_status": entry["current_evidence_status"],
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>动物源姿势与整90°方向人工审核</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#0b0e14;color:#e7eaf0;height:100vh;overflow:hidden}button,input,select,textarea{font:inherit}
.app{display:grid;grid-template-columns:340px minmax(0,1fr);height:100vh}aside{background:#111620;border-right:1px solid #273043;display:flex;flex-direction:column;min-height:0}
header{padding:16px;border-bottom:1px solid #273043}h1{font-size:19px;margin:0 0 7px}.summary,.muted{font-size:12px;color:#9aa7ba}.links{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
.link,.btn{color:#e7eaf0;background:#182131;border:1px solid #344157;border-radius:8px;padding:7px 10px;text-decoration:none;cursor:pointer}.link:hover,.btn:hover{background:#24466e}
.filters{display:grid;gap:8px;padding:10px;border-bottom:1px solid #273043}input,select,textarea{width:100%;color:#e7eaf0;background:#171e2b;border:1px solid #303b50;border-radius:8px;padding:8px}
#list{overflow:auto;padding:7px}.item{width:100%;text-align:left;color:inherit;background:transparent;border:1px solid transparent;border-radius:9px;padding:9px;cursor:pointer}.item:hover{background:#171e2b}.item.active{background:#1b2940;border-color:#3c6ea8}.item-title{font-size:12px;font-weight:650;overflow-wrap:anywhere}.item-meta{font-size:11px;color:#9aa7ba;margin-top:3px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:#f59e0b}.dot.ok{background:#22c55e}.dot.bad{background:#ef4444}
main{overflow:auto;padding:20px}.stage{max-width:1450px;margin:auto}.danger{background:#441b22;border:1px solid #9f3446;border-radius:10px;padding:12px;margin-bottom:14px;color:#ffd8df}.title-row{display:flex;justify-content:space-between;gap:12px}h2{font-size:20px;margin:0;overflow-wrap:anywhere}.pills{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.pill{font-size:12px;border:1px solid #334155;background:#172131;border-radius:999px;padding:4px 8px}
.grid{display:grid;grid-template-columns:minmax(430px,1.15fr) minmax(400px,1fr);gap:14px}.card{background:#111620;border:1px solid #273043;border-radius:12px;padding:14px}h3{font-size:15px;margin:0 0 8px}.instructions{font-size:13px;color:#ccd6e5;line-height:1.55;margin-bottom:10px}.preview-wrap{position:relative;background:#fff;border-radius:8px;min-height:380px;display:flex;align-items:center;justify-content:center;overflow:hidden}.preview-wrap canvas{display:block;width:100%;height:min(56vh,590px);min-height:380px}.preview-loading{position:absolute;color:#334155;font-size:13px;pointer-events:none}.static img{width:100%;border-radius:8px;background:#05070a}.evidence-pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.evidence figure{margin:0}.evidence figcaption{font-size:11px;color:#b8c6dc;margin:4px 0 10px}.contact{margin-top:8px}
.controls{display:grid;grid-template-columns:repeat(4,minmax(70px,1fr));gap:6px;margin-top:9px}.controls button{padding:8px 3px}.fine-controls{grid-template-columns:repeat(7,minmax(54px,1fr))}.control-label{font-size:12px;color:#8fb3df;margin-top:10px}.yaw{font-family:ui-monospace,monospace;color:#7dd3fc;text-align:center;margin:8px 0}.checks{display:grid;gap:5px;margin-top:10px;padding:9px;border:1px solid #334155;border-radius:8px;background:#0d141f}.checks-title{font-size:12px;color:#8fb3df;margin-bottom:3px}.checks label{display:flex;gap:8px;align-items:flex-start;font-size:12px;color:#d5deeb}.checks input{width:auto;margin-top:2px}.decision{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.approve{background:#126137;border-color:#1f9d57}.reject{background:#7d2631;border-color:#bd4353}
.video-card{margin-top:14px}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.tabs button.active{background:#24466e;border-color:#4e83bd}video{width:100%;max-height:560px;background:#000;border-radius:9px}.path{font-size:11px;color:#b8c6dc;white-space:nowrap;overflow:auto;margin-top:7px;padding:7px;background:#0c111a;border-radius:6px}
.note{margin-top:8px}.nav{display:flex;gap:6px}.status-line{font-size:12px;color:#fbbf24;margin-top:5px}
@media(max-width:950px){body{height:auto;overflow:auto}.app{display:block;height:auto}aside{height:42vh}.grid{grid-template-columns:1fr}main{padding:12px}.controls{grid-template-columns:repeat(4,1fr)}}
</style></head><body>
<script id="assets" type="application/json">{{assets_json|safe}}</script>
<div class="app"><aside><header><h1>{{direction_title}}</h1><div class="summary">{{asset_count}} 个资产 · {{direction_summary}}</div><div class="links"><a class="link" href="/docs/controlled_animal_video_review.html">动物成片</a><a class="link" href="/docs/rocketbox_human_video_review.html">人类视频</a></div></header>
<div class="filters"><input id="search" type="search" placeholder="搜索 asset / breed / 属性"><select id="species"><option value="">全部动物</option><option value="cat">猫</option><option value="dog">狗</option><option value="horse">马</option></select></div><div id="list"></div></aside>
<main><div class="stage"><div class="danger"><b>先审源姿势，再审坐标方向：</b>歪头、躯干内部扭曲、前后脚错轨不能靠 yaw 修复。v3 的小角度只允许纠正<strong>整只模型刚体坐标系</strong>，请沿躯干/脊柱纵轴调平，绝不能追着歪向一边的头或尾巴。页面不自动判断方向、不预置 mirror，也不覆盖旧 GLB 或视频。</div>
<div class="title-row"><div><h2 id="title"></h2><div id="sub" class="muted"></div><div id="status" class="status-line"></div></div><div class="nav"><button class="btn" id="prev">←</button><button class="btn" id="next">→</button></div></div><div id="pills" class="pills"></div>
<div class="grid"><section class="card"><h3>{{gate_heading}}</h3><div class="instructions">{{gate_instructions|safe}}</div><div class="preview-wrap"><canvas id="preview" aria-label="方向实时预览"></canvas><span id="preview-loading" class="preview-loading">正在加载原始 mesh 点云…</span></div><div id="yaw" class="yaw"></div><div id="legacy-controls" class="controls" {% if two_stage %}hidden{% endif %}><button class="btn rot" data-d="-90">↺ −90°</button><button class="btn" id="reset">重置 0°</button><button class="btn rot" data-d="90">↻ +90°</button><button class="btn rot" data-d="180">⇄ 180°</button></div><div id="two-stage-controls" {% if not two_stage %}hidden{% endif %}><div class="control-label">第一步：只看右侧 TOP-DOWN 的躯干/脊柱轴，实时调平</div><div class="controls fine-controls"><button class="btn axis" data-d="-15">−15°</button><button class="btn axis" data-d="-5">−5°</button><button class="btn axis" data-d="-1">−1°</button><button class="btn" id="reset-axis">轴归零</button><button class="btn axis" data-d="1">+1°</button><button class="btn axis" data-d="5">+5°</button><button class="btn axis" data-d="15">+15°</button></div><div class="control-label">第二步：选择头尾/正方向；不会清除第一步的小角度</div><div class="controls"><button class="btn cardinal" data-v="-90">−90°</button><button class="btn cardinal" data-v="0">0°</button><button class="btn cardinal" data-v="90">+90°</button><button class="btn cardinal" data-v="180">180°</button></div></div><div class="checks"><div class="checks-title">可选检查提示（不勾选也能保存）</div><label><input class="pose-check" id="check-spine" type="checkbox">躯干/脊柱本身笔直，没有内部斜扭</label><label><input class="pose-check" id="check-head" type="checkbox">头颈沿躯干方向，没有朝镜头或侧面歪转</label><label><input class="pose-check" id="check-legs" type="checkbox">前后腿轨迹/平面一致，没有错轨</label><label><input class="pose-check" id="check-ground" type="checkbox">四只脚处于同一合理地面，没有悬空</label></div><textarea id="notes" class="note" rows="2" placeholder="可选备注；拒绝时建议写明：歪头 / 躯干内部扭曲 / 前后脚错轨 / 脚未落地"></textarea><div class="decision"><button class="btn approve" id="approve">姿势合格并保存当前方向（0°）</button><button class="btn reject" id="reject">拒绝源姿势，退回重生</button></div></section>
<section class="card static evidence"><h3>绑定前原始证据（不随按钮变化）</h3><div class="instructions">左图是进入 image-to-3D 的当次 FLUX 参考图，右图是当次原始 3D mesh 顶视图。两者用于区分“生成时已经歪”与“后续坐标方向错误”。</div><div class="evidence-pair"><figure><img id="inputref" alt="I2-3D input"><figcaption>当次 FLUX / I2-3D 输入</figcaption></figure><figure><img id="topview" alt="I2-3D static top view"><figcaption>当次原始 3D mesh 顶视图</figcaption></figure></div><img id="contact" class="contact" alt="static contact sheet"><div id="rawpath" class="path"></div></section></div>
<section class="card video-card"><h3>第二道门：绑定后动作与真实移动方向</h3><div class="instructions">页面同时保留隔离动画证据和已通过批状态认证的 UE Apartment Walk/Idle。请以当前资产的状态标签为准：被拒绝结果只用于定位问题；带 Apartment 标签的结果仍需人工确认方向，不能据此自动注册为正式资产。</div><div class="tabs" id="tabs"></div><video id="video" controls preload="metadata"></video><div id="videopath" class="path"></div></section>
</div></main></div>
<script>
const all=JSON.parse(document.getElementById('assets').textContent),twoStage={{two_stage_json|safe}};let filtered=[...all],idx=0,state={};const $=x=>document.getElementById(x);
async function loadState(){state=await (await fetch('/api/state')).json();render()}
function current(){return filtered[idx]}
function apply(){const q=$('search').value.toLowerCase(),s=$('species').value;filtered=all.filter(a=>(!s||a.species===s)&&(!q||(`${a.asset_id} ${a.breed} ${JSON.stringify(a.sampled_attributes)}`).toLowerCase().includes(q)));idx=0;render()}
function itemStatus(id){return state[id]?.decision?.status||'pending'}
function evidenceLabel(a){const e=a.current_evidence_status.walking_direction||'';return e.includes('rejected')?'历史失败动画，仅供定位':e.includes('agent_approved')?'新动画候选，待人工审核':'方向证据待审核'}
function renderList(){const list=$('list');list.replaceChildren();filtered.forEach((a,i)=>{const b=document.createElement('button'),st=itemStatus(a.asset_id),failed=(a.current_evidence_status.walking_direction||'').includes('rejected');b.className='item'+(i===idx?' active':'');b.innerHTML=`<div class="item-title"><span class="dot ${st.includes('approved')?'ok':(st.includes('rejected')||failed)?'bad':''}"></span></div><div class="item-meta"></div>`;b.querySelector('.item-title').append(a.asset_id);b.querySelector('.item-meta').textContent=`${a.species} · ${a.breed} · ${evidenceLabel(a)} · ${st}`;b.onclick=()=>{idx=i;render()};list.append(b)})}
let view='apartment_walking_review';const labels={walking_side:'绑定后 Walk 侧面',walking_front:'绑定后 Walk 正面',idle_side:'绑定后 Idle 侧面',apartment_walking_review:'UE Walk + Top-down',apartment_walking_main:'UE Walk 主视图',apartment_walking_topdown:'Walk Top-down',apartment_idle_review:'UE Idle + Top-down',apartment_idle_main:'UE Idle 主视图',apartment_idle_topdown:'Idle Top-down'};
function poseChecks(){return{spine_is_straight:$('check-spine').checked,head_is_aligned_with_torso:$('check-head').checked,front_and_hind_legs_share_consistent_planes:$('check-legs').checked,all_paws_share_one_ground_plane:$('check-ground').checked}}
const cloudCache=new Map();let prefetchTimer=0,prefetchGeneration=0,yawBusy=false;
function preloadImage(url){return new Promise(resolve=>{const image=new Image();image.onload=image.onerror=resolve;image.src=url})}
async function ensureCloud(a,{quiet=false}={}){if(cloudCache.has(a.asset_id)){if(current()?.asset_id===a.asset_id)drawPreview();return cloudCache.get(a.asset_id)}if(!quiet)$('preview-loading').textContent='正在加载原始 mesh 点云…';const response=await fetch(`/api/preview-points/${encodeURIComponent(a.asset_id)}`);if(!response.ok)throw Error((await response.json()).error||'点云加载失败');const data=await response.json();cloudCache.set(a.asset_id,data);if(current()?.asset_id===a.asset_id)drawPreview();return data}
function arrow(ctx,x1,y1,x2,y2,color,label){const angle=Math.atan2(y2-y1,x2-x1);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=4;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-13*Math.cos(angle-Math.PI/6),y2-13*Math.sin(angle-Math.PI/6));ctx.lineTo(x2-13*Math.cos(angle+Math.PI/6),y2-13*Math.sin(angle+Math.PI/6));ctx.closePath();ctx.fill();ctx.font='700 13px system-ui';ctx.fillText(label,Math.min(x1,x2)+5,Math.min(y1,y2)-8);ctx.restore()}
function drawPreview(){const canvas=$('preview'),a=current();if(!a)return;const cloud=cloudCache.get(a.asset_id),loading=$('preview-loading');if(!cloud){loading.hidden=false;return}loading.hidden=true;const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(430,Math.round(rect.width)),h=Math.max(380,Math.round(rect.height));canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);const yaw=Number((state[a.asset_id]||{}).yaw_deg||0)*Math.PI/180,c=Math.cos(yaw),s=Math.sin(yaw),points=cloud.points.map(p=>[c*p[0]+s*p[2],p[1],-s*p[0]+c*p[2]]);const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let k=0;k<3;k++){if(p[k]<min[k])min[k]=p[k];if(p[k]>max[k])max[k]=p[k]}const gap=26,leftW=(w-gap)*.61,rightX=leftW+gap,rightW=w-rightX,pad=38,topPad=52,bottomPad=58;function projector(kind){const x0=kind==='side'?0:rightX,pw=kind==='side'?leftW:rightW,vi=kind==='side'?1:2,rx=Math.max(1e-9,max[0]-min[0]),ry=Math.max(1e-9,max[vi]-min[vi]),scale=Math.min((pw-pad*2)/rx,(h-topPad-bottomPad)/ry)*.92,cx=(min[0]+max[0])/2,cy=(min[vi]+max[vi])/2;return p=>[x0+pw/2+(p[0]-cx)*scale,topPad+(h-topPad-bottomPad)/2-(p[vi]-cy)*scale]};const side=projector('side'),top=projector('top');ctx.fillStyle='rgba(40,107,179,.24)';for(const p of points){const q=side(p);ctx.fillRect(q[0],q[1],1.25,1.25)}ctx.fillStyle='rgba(31,138,85,.24)';for(const p of points){const q=top(p);ctx.fillRect(q[0],q[1],1.25,1.25)}ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.strokeRect(1,1,leftW-2,h-2);ctx.strokeRect(rightX+1,1,rightW-2,h-2);ctx.fillStyle='#111827';ctx.font='700 14px system-ui';ctx.fillText('SIDE：动物必须直立',18,26);ctx.fillText('TOP-DOWN：躯干/脊柱必须朝右',rightX+18,26);arrow(ctx,leftW*.22,h-28,leftW*.82,h-28,'#08752d','FORWARD +X  →');arrow(ctx,rightX+rightW*.16,h-28,rightX+rightW*.84,h-28,'#08752d','FORWARD +X  →');arrow(ctx,22,h*.72,22,h*.28,'#2563eb','UP +Y  ↑')}
function normalizeYaw(value){let yaw=((Number(value)+180)%360+360)%360-180;if(Math.abs(yaw+180)<1e-9)return 180;if(Math.abs(yaw)<1e-9)return 0;return yaw}
function updateYawUI(){const a=current();if(!a)return;const st=state[a.asset_id]||{yaw_deg:0},yaw=Number(st.yaw_deg||0),axis=Number(st.axis_alignment_yaw_deg||0),cardinal=Number(st.cardinal_yaw_deg||0);$('yaw').textContent=twoStage?`人工轴对齐 ${axis}° + 头尾方向 ${cardinal}° = 绑定总 yaw ${yaw}°`:`raw mesh + manual cardinal yaw = ${yaw}°`;$('approve').textContent=`姿势合格并保存当前方向（${yaw}°）`;const locked=Boolean(st.decision)||yawBusy;document.querySelectorAll('.rot,.axis,.cardinal,#reset,#reset-axis,#approve,#reject,.pose-check').forEach(x=>x.disabled=locked)}
async function mutateYaw(operation,value=0){if(yawBusy)return;const a=current();if(!a)return;const assetId=a.asset_id,before=Number((state[assetId]||{}).yaw_deg||0),next=operation==='reset'?0:normalizeYaw(before+value);state[assetId]={...(state[assetId]||{}),yaw_deg:next};yawBusy=true;updateYawUI();drawPreview();try{const url=operation==='reset'?`/api/reset/${encodeURIComponent(assetId)}`:`/api/rotate/${encodeURIComponent(assetId)}`,payload=operation==='reset'?{}:{delta_deg:Number(value)},response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),result=await response.json();if(!response.ok)throw Error(result.error||'方向保存失败');state[assetId]=result}catch(error){state[assetId]={...(state[assetId]||{}),yaw_deg:before};alert(error.message)}finally{yawBusy=false;if(current()?.asset_id===assetId){updateYawUI();drawPreview()}renderList()}}
async function mutateTwoStage(mode,value){if(yawBusy)return;const a=current();if(!a)return;const assetId=a.asset_id,before={...(state[assetId]||{})},axis=mode==='axis_delta'?Number(before.axis_alignment_yaw_deg||0)+Number(value):mode==='axis_reset'?0:Number(before.axis_alignment_yaw_deg||0),cardinal=mode==='cardinal_set'?Number(value):Number(before.cardinal_yaw_deg||0),next=normalizeYaw(axis+cardinal);state[assetId]={...before,axis_alignment_yaw_deg:axis,cardinal_yaw_deg:cardinal,yaw_deg:next};yawBusy=true;updateYawUI();drawPreview();try{const response=await fetch(`/api/rotate/${encodeURIComponent(assetId)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,value:Number(value||0)})}),result=await response.json();if(!response.ok)throw Error(result.error||'方向保存失败');state[assetId]=result}catch(error){state[assetId]=before;alert(error.message)}finally{yawBusy=false;if(current()?.asset_id===assetId){updateYawUI();drawPreview()}renderList()}}
async function prefetchNeighbors(){const generation=++prefetchGeneration;if(filtered.length<2)return;for(const offset of [1,-1]){if(generation!==prefetchGeneration)return;const a=filtered[(idx+offset+filtered.length)%filtered.length],input=(a.artifacts.reference_image||a.artifacts.pixal_input_rgba).url;await Promise.all([ensureCloud(a,{quiet:true}),...([input,a.artifacts.static_top_view.url,a.artifacts.static_contact_sheet.url].map(preloadImage))])}}
function queueNeighborPrefetch(){clearTimeout(prefetchTimer);prefetchTimer=setTimeout(prefetchNeighbors,120)}
function render(){renderList();const a=current();if(!a)return;const st=state[a.asset_id]||{yaw_deg:0,revision:0};$('title').textContent=a.asset_id;$('sub').textContent=`${a.species} · ${a.breed} · ${a.profile_schema_id}`;$('status').textContent=`${evidenceLabel(a)}；${twoStage?'躯干轴对齐/头尾方向':'源姿势/整90°倍数方向'}：${st.decision?.status||'待人工审核'}`;$('pills').replaceChildren(...Object.entries(a.sampled_attributes).map(([k,v])=>{const x=document.createElement('span');x.className='pill';x.textContent=`${k}=${v}`;return x}));$('inputref').src=(a.artifacts.reference_image||a.artifacts.pixal_input_rgba).url;$('topview').src=a.artifacts.static_top_view.url;$('contact').src=a.artifacts.static_contact_sheet.url;$('rawpath').textContent=(a.artifacts.i23d_raw_glb||a.artifacts.pixal_raw_glb||a.artifacts.prebind_lod_glb).absolute_path;document.querySelectorAll('.pose-check').forEach(x=>x.checked=false);renderVideo();updateYawUI();drawPreview();ensureCloud(a).catch(e=>{$('preview-loading').hidden=false;$('preview-loading').textContent=e.message});queueNeighborPrefetch()}
function renderVideo(){const a=current();const keys=Object.keys(labels).filter(k=>a.artifacts[k]);const tabs=$('tabs');if(!keys.length){tabs.replaceChildren();$('video').removeAttribute('src');$('videopath').textContent='该阶段尚无动画媒体';return}if(!a.artifacts[view])view=keys[0];tabs.replaceChildren(...keys.map(k=>{const b=document.createElement('button');b.className='btn'+(view===k?' active':'');b.textContent=labels[k];b.onclick=()=>{view=k;renderVideo()};return b}));const m=a.artifacts[view];$('video').src=m.url;$('videopath').textContent=m.absolute_path}
async function post(url,payload={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const j=await r.json();if(!r.ok){alert(j.error||'操作失败');throw Error(j.error)}state[j.asset_id]=j;render()}
document.querySelectorAll('.rot').forEach(b=>b.onclick=()=>mutateYaw('rotate',Number(b.dataset.d)));if($('reset'))$('reset').onclick=()=>mutateYaw('reset');document.querySelectorAll('.axis').forEach(b=>b.onclick=()=>mutateTwoStage('axis_delta',Number(b.dataset.d)));document.querySelectorAll('.cardinal').forEach(b=>b.onclick=()=>mutateTwoStage('cardinal_set',Number(b.dataset.v)));if($('reset-axis'))$('reset-axis').onclick=()=>mutateTwoStage('axis_reset',0);$('approve').onclick=()=>{const checks=poseChecks(),yaw=Number((state[current().asset_id]||{}).yaw_deg||0),status=twoStage?'source_pose_and_manual_orientation_approved':'source_pose_and_cardinal_orientation_approved';if(confirm(`确认源姿势合格，并保存当前方向 ${yaw}°？检查项为可选提示；这只保存人工方向，不会自动注册正式资产。`))post(`/api/decision/${encodeURIComponent(current().asset_id)}`,{status,notes:$('notes').value,pose_checks:checks})};$('reject').onclick=()=>{const notes=$('notes').value.trim()||'源姿势失败：头颈、躯干、腿平面或落地至少一项不合格';if(confirm('确认拒绝源姿势并退回 2D/image-to-3D 重生？'))post(`/api/decision/${encodeURIComponent(current().asset_id)}`,{status:'source_pose_rejected',notes,pose_checks:poseChecks()})};
$('prev').onclick=()=>{if(filtered.length){idx=(idx-1+filtered.length)%filtered.length;render()}};$('next').onclick=()=>{if(filtered.length){idx=(idx+1)%filtered.length;render()}};$('search').oninput=apply;$('species').oninput=apply;window.onresize=drawPreview;window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};loadState();
</script></body></html>"""


def create_app(
    manifest_path: Path,
    state_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8102,
) -> Flask:
    manifest_path = Path(manifest_path).resolve()
    state_root = Path(state_root).resolve()
    manifest, entries = _validate_manifest(manifest_path)
    two_stage = manifest["schema"] == MANIFEST_SCHEMA_V3
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / "previews").mkdir(exist_ok=True)
    (state_root / "states").mkdir(exist_ok=True)
    (state_root / "decisions").mkdir(exist_ok=True)
    (state_root / "interactive_points").mkdir(exist_ok=True)
    locks = {asset_id: threading.Lock() for asset_id in entries}
    mesh_cache: OrderedDict[str, trimesh.Trimesh] = OrderedDict()
    mesh_cache_lock = threading.Lock()
    mesh_cache_capacity = min(4, max(1, len(entries)))
    point_cache: dict[str, dict[str, Any]] = {}
    point_cache_lock = threading.Lock()
    preview_render_lock = threading.Lock()

    app = Flask(__name__)
    app.config.update(
        DIRECTION_MANIFEST=manifest,
        DIRECTION_ENTRIES=entries,
        DIRECTION_STATE_ROOT=state_root,
        DIRECTION_HOST=host,
        DIRECTION_PORT=port,
        DIRECTION_TWO_STAGE=two_stage,
    )

    def state_path(asset_id: str) -> Path:
        return state_root / "states" / f"{asset_id}.json"

    def decision_path(asset_id: str) -> Path:
        return state_root / "decisions" / f"{asset_id}.json"

    def read_state(asset_id: str) -> dict[str, Any]:
        path = state_path(asset_id)
        if not path.exists():
            base: dict[str, Any] = {
                "schema": STATE_SCHEMA_V3 if two_stage else STATE_SCHEMA,
                "asset_id": asset_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "yaw_deg": 0.0,
                "history": [],
                "revision": 0,
            }
            if two_stage:
                base.update(
                    axis_alignment_yaw_deg=0.0,
                    cardinal_yaw_deg=0.0,
                )
        else:
            base = _read_json(path)
            common_invalid = (
                base.get("schema") != (STATE_SCHEMA_V3 if two_stage else STATE_SCHEMA)
                or base.get("asset_id") != asset_id
                or base.get("manifest_sha256") != manifest["manifest_sha256"]
                or not isinstance(base.get("history"), list)
            )
            if two_stage:
                try:
                    axis_yaw = float(base["axis_alignment_yaw_deg"])
                    cardinal_yaw = float(base["cardinal_yaw_deg"])
                    total_yaw = float(base["yaw_deg"])
                except (KeyError, TypeError, ValueError):
                    common_invalid = True
                else:
                    common_invalid = common_invalid or (
                        abs(axis_yaw) > MAX_MANUAL_AXIS_ALIGNMENT_DEG
                        or cardinal_yaw not in CARDINAL_YAWS
                        or not math.isclose(
                            total_yaw,
                            _normalize_yaw(axis_yaw + cardinal_yaw),
                            abs_tol=1.0e-6,
                        )
                    )
            else:
                try:
                    common_invalid = common_invalid or (
                        float(base.get("yaw_deg", 1.0)) not in CARDINAL_YAWS
                    )
                except (TypeError, ValueError):
                    common_invalid = True
            if common_invalid:
                raise ReviewServerError(f"invalid review state: {asset_id}")
        decision_file = decision_path(asset_id)
        if decision_file.exists():
            decision = _read_json(decision_file)
            if (
                decision.get("schema")
                != (DECISION_SCHEMA_V3 if two_stage else DECISION_SCHEMA)
                or decision.get("asset_id") != asset_id
                or decision.get("manifest_sha256") != manifest["manifest_sha256"]
                or decision.get("decision_sha256")
                != _hash_without(decision, "decision_sha256")
            ):
                raise ReviewServerError(f"invalid direction decision: {asset_id}")
            base["decision"] = {
                "status": decision["status"],
                "decided_at": decision["decided_at"],
                "absolute_path": str(decision_file),
            }
        return base

    def write_state(asset_id: str, state: Mapping[str, Any]) -> None:
        _atomic_json(state_path(asset_id), state, replace=True)

    def require_asset(asset_id: str) -> dict[str, Any]:
        entry = entries.get(asset_id)
        if entry is None:
            abort(404)
        return entry

    def preview_path(asset_id: str, yaw_deg: float) -> Path:
        token = f"{yaw_deg:+09.3f}".replace("+", "p").replace("-", "m").replace(".", "d")
        return state_root / "previews" / asset_id / f"surface_v2_yaw_{token}.png"

    def interactive_points_path(asset_id: str) -> Path:
        source = entries[asset_id]["artifacts"]["prebind_lod_glb"]
        return (
            state_root
            / "interactive_points"
            / (
                f"{asset_id}_{source['sha256'][:16]}_"
                f"n{INTERACTIVE_MAX_POINTS}.json"
            )
        )

    def cached_preview_mesh(asset_id: str) -> trimesh.Trimesh:
        with mesh_cache_lock:
            cached = mesh_cache.pop(asset_id, None)
            if cached is not None:
                mesh_cache[asset_id] = cached
                return cached
        entry = entries[asset_id]
        source = Path(entry["artifacts"]["prebind_lod_glb"]["absolute_path"])
        loaded = _load_preview_mesh(source)
        with mesh_cache_lock:
            cached = mesh_cache.pop(asset_id, None)
            if cached is not None:
                mesh_cache[asset_id] = cached
                return cached
            mesh_cache[asset_id] = loaded
            while len(mesh_cache) > mesh_cache_capacity:
                mesh_cache.popitem(last=False)
        return loaded

    def render_preview(asset_id: str, yaw_deg: float) -> Path:
        destination = preview_path(asset_id, yaw_deg)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        mesh = cached_preview_mesh(asset_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with preview_render_lock:
            if not destination.is_file() or destination.stat().st_size == 0:
                _render_orientation_preview(mesh, destination, yaw_deg=yaw_deg)
        return destination

    def cached_preview_points(asset_id: str) -> dict[str, Any]:
        with point_cache_lock:
            cached = point_cache.get(asset_id)
            if cached is not None:
                return cached
        entry = entries[asset_id]
        source = entry["artifacts"]["prebind_lod_glb"]
        cache_path = interactive_points_path(asset_id)
        if cache_path.is_file():
            payload = _read_json(cache_path)
            if (
                payload.get("schema")
                != "controlled_animal_direction_interactive_points_v1"
                or payload.get("asset_id") != asset_id
                or payload.get("source_sha256") != source["sha256"]
                or payload.get("source_size_bytes") != source["size_bytes"]
                or payload.get("coordinate_frame") != "gltf_y_up"
                or not isinstance(payload.get("point_count"), int)
                or not 0 < payload["point_count"] <= INTERACTIVE_MAX_POINTS
                or len(payload.get("points", [])) != payload["point_count"]
            ):
                raise ReviewServerError(
                    f"invalid persistent interactive point cache: {asset_id}"
                )
        else:
            mesh = cached_preview_mesh(asset_id)
            points = _sample_surface_points(
                mesh,
                max_points=INTERACTIVE_MAX_POINTS,
            )
            payload = {
                "schema": "controlled_animal_direction_interactive_points_v1",
                "asset_id": asset_id,
                "source_sha256": source["sha256"],
                "source_size_bytes": source["size_bytes"],
                "coordinate_frame": "gltf_y_up",
                "point_count": int(len(points)),
                "points": np.round(points, decimals=6).tolist(),
            }
            _atomic_compact_json(cache_path, payload)
        with point_cache_lock:
            existing = point_cache.setdefault(asset_id, payload)
        return existing

    @app.errorhandler(ReviewServerError)
    def handle_review_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": str(error)}), 409
        return str(error), 409

    @app.get("/")
    def index():
        public = [_public_entry(entry) for entry in entries.values()]
        if two_stage:
            direction_title = "动物源姿势与两步人工方向审核"
            direction_summary = "躯干轴微调 + 头尾方向；不使用自动判断"
            gate_heading = "人工门：先调平刚体躯干轴，再选择头尾方向"
            gate_instructions = (
                "预览从<strong>原始 100k mesh、identity 变换</strong>开始。"
                "第一步只看 TOP-DOWN 躯干/脊柱纵轴，用 ±1°/5°/15° 实时调平；"
                "第二步选择 0°、−90°、+90° 或 180° 头尾方向。两项相加后让"
                "<strong style=\"color:#4ade80\">躯干朝绿色 +X</strong>。"
                "全部操作由审核者完成，代码不估计角度。"
            )
        else:
            direction_title = "动物源姿势 + 整90°方向人工审核"
            direction_summary = "不使用自动方向或细角度补偿"
            gate_heading = "人工门：源姿势合格后，选择一个整90°倍数方向"
            gate_instructions = (
                "预览从<strong>原始 100k mesh、identity 变换</strong>开始。"
                "可保存的绝对方向为 0°、−90°、+90° 或 180°；按钮每次只绕 "
                "glTF UP 轴旋转。请让<strong style=\"color:#4ade80\">"
                "躯干/脊柱纵轴朝绿色 +X</strong>。"
            )
        return render_template_string(
            HTML,
            asset_count=len(public),
            assets_json=json.dumps(public, ensure_ascii=False).replace("</", "<\\/"),
            two_stage=two_stage,
            two_stage_json=json.dumps(two_stage),
            direction_title=direction_title,
            direction_summary=direction_summary,
            gate_heading=gate_heading,
            gate_instructions=gate_instructions,
        )

    @app.get("/api/state")
    def api_state():
        return jsonify({asset_id: read_state(asset_id) for asset_id in entries})

    @app.get("/api/preview-points/<asset_id>")
    def preview_points(asset_id: str):
        require_asset(asset_id)
        cached_preview_points(asset_id)
        return send_file(
            interactive_points_path(asset_id),
            mimetype="application/json",
            conditional=True,
        )

    @app.get("/preview/<asset_id>.png")
    def preview(asset_id: str):
        require_asset(asset_id)
        with locks[asset_id]:
            state = read_state(asset_id)
            path = render_preview(asset_id, float(state["yaw_deg"]))
        return send_file(path, mimetype="image/png", conditional=True)

    @app.post("/api/rotate/<asset_id>")
    def rotate(asset_id: str):
        require_asset(asset_id)
        payload = request.get_json(silent=True) or {}
        with locks[asset_id]:
            if decision_path(asset_id).exists():
                raise ReviewServerError("direction decision is already immutable")
            state = read_state(asset_id)
            if two_stage:
                mode = payload.get("mode")
                try:
                    value = float(payload.get("value"))
                except (TypeError, ValueError) as error:
                    raise ReviewServerError("manual yaw value must be numeric") from error
                if mode == "axis_delta":
                    if value not in ALLOWED_AXIS_DELTAS:
                        raise ReviewServerError(
                            f"unsupported manual torso-axis delta: {value}"
                        )
                    next_axis = float(state["axis_alignment_yaw_deg"]) + value
                    if abs(next_axis) > MAX_MANUAL_AXIS_ALIGNMENT_DEG:
                        raise ReviewServerError(
                            "manual torso-axis alignment must remain within ±45 degrees"
                        )
                    state["axis_alignment_yaw_deg"] = next_axis
                    state["history"].append(
                        {"operation": "manual_axis_yaw_delta_deg", "value": value}
                    )
                elif mode == "axis_reset":
                    state["axis_alignment_yaw_deg"] = 0.0
                    state["history"].append(
                        {"operation": "manual_axis_yaw_reset", "value": 0.0}
                    )
                elif mode == "cardinal_set":
                    if value not in CARDINAL_YAWS:
                        raise ReviewServerError(
                            f"unsupported cardinal head/tail yaw: {value}"
                        )
                    state["cardinal_yaw_deg"] = value
                    state["history"].append(
                        {"operation": "manual_cardinal_yaw_set_deg", "value": value}
                    )
                else:
                    raise ReviewServerError("two-stage yaw mode is invalid")
                state["yaw_deg"] = _normalize_yaw(
                    float(state["axis_alignment_yaw_deg"])
                    + float(state["cardinal_yaw_deg"])
                )
            else:
                try:
                    delta = float(payload.get("delta_deg"))
                except (TypeError, ValueError) as error:
                    raise ReviewServerError("delta_deg must be numeric") from error
                if delta not in ALLOWED_DELTAS:
                    raise ReviewServerError(
                        f"unsupported yaw delta: {delta}; manual cardinal rotations only"
                    )
                state["yaw_deg"] = _normalize_yaw(float(state["yaw_deg"]) + delta)
                state["history"].append(
                    {"operation": "yaw_delta_deg", "value": delta}
                )
            state["revision"] = int(state.get("revision", 0)) + 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_state(asset_id, state)
        return jsonify(state)

    @app.post("/api/reset/<asset_id>")
    def reset(asset_id: str):
        require_asset(asset_id)
        with locks[asset_id]:
            if decision_path(asset_id).exists():
                raise ReviewServerError("direction decision is already immutable")
            previous = read_state(asset_id)
            state: dict[str, Any] = {
                "schema": STATE_SCHEMA_V3 if two_stage else STATE_SCHEMA,
                "asset_id": asset_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "yaw_deg": 0.0,
                "history": [],
                "revision": int(previous.get("revision", 0)) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if two_stage:
                state.update(
                    axis_alignment_yaw_deg=0.0,
                    cardinal_yaw_deg=0.0,
                )
            write_state(asset_id, state)
        return jsonify(state)

    @app.post("/api/decision/<asset_id>")
    def decide(asset_id: str):
        entry = require_asset(asset_id)
        payload = request.get_json(silent=True) or {}
        status = payload.get("status")
        approved_status = (
            "source_pose_and_manual_orientation_approved"
            if two_stage
            else "source_pose_and_cardinal_orientation_approved"
        )
        if status not in {approved_status, "source_pose_rejected"}:
            raise ReviewServerError("invalid source-pose/cardinal-direction decision")
        notes = str(payload.get("notes", "")).strip()
        pose_checks = payload.get("pose_checks", {})
        if not isinstance(pose_checks, dict):
            raise ReviewServerError("pose_checks must be an object")
        normalized_checks = {
            name: pose_checks.get(name) is True for name in POSE_CHECK_HINTS
        }
        if status == "source_pose_rejected" and not notes:
            raise ReviewServerError("a rejection note is required")
        with locks[asset_id]:
            state = read_state(asset_id)
            yaw_deg = float(state["yaw_deg"])
            if not two_stage and yaw_deg not in CARDINAL_YAWS:
                raise ReviewServerError("saved yaw is not cardinal")
            matrix = _manual_preview_matrix(yaw_deg)
            decision: dict[str, Any] = {
                "schema": DECISION_SCHEMA_V3 if two_stage else DECISION_SCHEMA,
                "manifest_sha256": manifest["manifest_sha256"],
                "asset_id": asset_id,
                "species": entry["species"],
                "breed": entry.get("breed"),
                "status": status,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "notes": notes,
                "source_prebind_lod": copy.deepcopy(
                    entry["artifacts"]["prebind_lod_glb"]
                ),
                "source_reference_image": copy.deepcopy(
                    entry["artifacts"].get("reference_image")
                    or entry["artifacts"]["pixal_input_rgba"]
                ),
                "source_static_top_view": copy.deepcopy(
                    entry["artifacts"]["static_top_view"]
                ),
                "automatic_orientation_inference_used": False,
                "initial_preview_pretransform": "identity",
                "manual_rotation_matrix_3x3": matrix.tolist(),
                "determinant": float(np.linalg.det(matrix)),
                "manual_pose_checks": normalized_checks,
                "manual_pose_checks_are_advisory": True,
                "downstream_candidate": {
                    "binding_pretransform": "not_authorized_by_this_visual_gate",
                    "coordinate_mapping_status": (
                        "requires_binding_basis_and_straight_line_ue_canary"
                    ),
                },
                "current_walking_media_status": entry["current_evidence_status"][
                    "walking_direction"
                ],
                "next_gate": (
                    "derive_binding_basis_then_straight_line_and_curve_dynamic_canary"
                    if status == approved_status
                    else "regenerate_strict_profile_reference_then_i23d_static_review"
                ),
                "formal_dataset_registration_authorized": False,
                "source_assets_modified": False,
                "history": copy.deepcopy(state["history"]),
            }
            if two_stage:
                decision.update(
                    manual_axis_alignment_yaw_about_gltf_positive_y_deg=float(
                        state["axis_alignment_yaw_deg"]
                    ),
                    manual_cardinal_head_tail_yaw_about_gltf_positive_y_deg=float(
                        state["cardinal_yaw_deg"]
                    ),
                    manual_total_yaw_about_gltf_positive_y_deg=yaw_deg,
                    axis_alignment_authority="human_visual_torso_spine_axis",
                    head_tail_authority="human_visual_head_tail_direction",
                )
                decision["downstream_candidate"]["manual_total_yaw_deg"] = yaw_deg
            else:
                decision["manual_cardinal_yaw_about_gltf_positive_y_deg"] = yaw_deg
                decision["downstream_candidate"]["manual_cardinal_yaw_deg"] = yaw_deg
            if "pixal_input_rgba" in entry["artifacts"]:
                decision["source_reference_rgba"] = copy.deepcopy(
                    entry["artifacts"]["pixal_input_rgba"]
                )
            decision["decision_sha256"] = _hash_without(decision, "decision_sha256")
            _atomic_json(decision_path(asset_id), decision, replace=False)
            state["decision"] = {
                "status": status,
                "decided_at": decision["decided_at"],
                "absolute_path": str(decision_path(asset_id)),
            }
        return jsonify(state)

    @app.get("/human")
    def human():
        return redirect("/docs/rocketbox_human_video_review.html")

    @app.get("/animal-videos")
    def animal_videos():
        return redirect("/docs/controlled_animal_video_review.html")

    @app.get("/docs/<path:relative>")
    def docs(relative: str):
        return send_from_directory(AVENGINE_ROOT / "docs", relative, conditional=True)

    @app.get("/external/<path:relative>")
    def external(relative: str):
        return send_from_directory(AVENGINE_ROOT / "external", relative, conditional=True)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = create_app(
        args.manifest, args.state_root, host=args.host, port=args.port
    )
    print(
        "CONTROLLED_ANIMAL_DIRECTION_REVIEW_SERVER_OK "
        f"url=http://{args.host}:{args.port}/ manifest={args.manifest.resolve()} "
        f"state={args.state_root.resolve()}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
