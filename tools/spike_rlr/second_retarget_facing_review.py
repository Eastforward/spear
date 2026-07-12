#!/usr/bin/env python3
"""Authenticate and measure the rejected second Route-2 retarget facing.

This module is intentionally independent of Blender so its input and vector
contracts can be tested before any diagnostic rendering is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence


DIAGNOSTIC_SCHEMA = "second_attempt_rotation_only_diagnostic_reconstruction_v1"
STATIC_QA_SCHEMA = "tokenrig_human_static_qa_v1"
FAILURE_SCHEMA = "tokenrig_rocketbox_retarget_attempt_v1"
ASSET_ID = "rocketbox_male_adult_01"
CANONICAL_FRONT_NAME = "negative-y"
CANONICAL_FRONT = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)
SOURCE_ARTIFACTS = (
    "walking_rotation_only_reconstruction.glb",
    "walking_front.mp4",
    "walking_front.png",
    "walking_side.mp4",
    "walking_side.png",
    "walking_feet.mp4",
    "walking_feet.png",
)
REQUIRED_SEMANTIC_ROLES = (
    "pelvis",
    "left_clavicle",
    "right_clavicle",
    "left_thigh",
    "right_thigh",
)


class FacingReviewError(RuntimeError):
    """An immutable input or facing invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise FacingReviewError(f"{description} must be a direct regular file")
    if path.stat().st_size <= 0:
        raise FacingReviewError(f"{description} is empty")
    return path


def _real_directory(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise FacingReviewError(f"{description} must be a direct real directory")
    return path


def _load_object(path: Path, description: str) -> dict[str, Any]:
    path = _real_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FacingReviewError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise FacingReviewError(f"{description} root must be an object")
    return value


def _validate_record(path: Path, record: Any, description: str) -> dict[str, Any]:
    path = _real_file(path, description)
    if not isinstance(record, Mapping):
        raise FacingReviewError(f"{description} record is missing")
    if record.get("size_bytes") != path.stat().st_size:
        raise FacingReviewError(f"{description} size changed")
    actual = sha256_file(path)
    if record.get("sha256") != actual:
        raise FacingReviewError(f"{description} SHA-256 changed")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _validate_local_record(
    root: Path, filename: str, record: Any
) -> dict[str, Any]:
    if not isinstance(record, Mapping) or record.get("filename") != filename:
        raise FacingReviewError(f"{filename} record filename changed")
    path = _real_file(root / filename, filename)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FacingReviewError(f"{filename} escaped the diagnostic directory") from error
    return _validate_record(path, record, filename)


def _validate_external_record(record: Any, description: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise FacingReviewError(f"{description} external path record is missing")
    return _validate_record(Path(record["path"]), record, description)


def read_glb_document(path: Path) -> dict[str, Any]:
    path = _real_file(path, "diagnostic GLB")
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise FacingReviewError("diagnostic GLB header is invalid")
    version, total_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or total_length != len(raw):
        raise FacingReviewError("diagnostic GLB length/version is invalid")
    offset = 12
    documents = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise FacingReviewError("diagnostic GLB chunk header is truncated")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + length
        if end > len(raw):
            raise FacingReviewError("diagnostic GLB chunk is truncated")
        if kind == 0x4E4F534A:
            documents.append(raw[offset:end])
        offset = end
    if len(documents) != 1:
        raise FacingReviewError("diagnostic GLB must contain exactly one JSON chunk")
    try:
        document = json.loads(documents[0].rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FacingReviewError(f"diagnostic GLB JSON is invalid: {error}") from error
    if not isinstance(document, dict):
        raise FacingReviewError("diagnostic GLB JSON root must be an object")
    return document


def authenticate_second_attempt(diagnostic_dir: Path | str) -> dict[str, Any]:
    root = _real_directory(Path(diagnostic_dir), "second-attempt diagnostic directory")
    manifest_path = _real_file(root / "diagnostic_manifest.json", "diagnostic manifest")
    manifest = _load_object(manifest_path, "diagnostic manifest")
    if manifest.get("schema") != DIAGNOSTIC_SCHEMA:
        raise FacingReviewError("diagnostic manifest schema changed")
    if manifest.get("asset_id") != ASSET_ID:
        raise FacingReviewError("diagnostic asset_id changed")
    if manifest.get("classification") != "technical_diagnostic_only":
        raise FacingReviewError("diagnostic classification changed")
    if manifest.get("decision") != "rejected_attempt_visualized_by_nonformal_reconstruction":
        raise FacingReviewError("diagnostic rejection decision changed")
    if manifest.get("formal_dataset_asset") is not False:
        raise FacingReviewError("diagnostic may not be a formal dataset asset")
    if manifest.get("readiness_bundle_published") is not False:
        raise FacingReviewError("diagnostic may not publish a readiness bundle")
    if manifest.get("automatic_checks") != "diagnostic_reconstruction_integrity_passed":
        raise FacingReviewError("diagnostic reconstruction integrity is not passed")
    if manifest.get("user_approval") != "not_requested_for_diagnostic_reconstruction":
        raise FacingReviewError("diagnostic user approval state changed")
    if "user_approved" in json.dumps(manifest, sort_keys=True):
        raise FacingReviewError("diagnostic may not claim user_approved")
    notice = manifest.get("reconstruction_notice")
    if not isinstance(notice, Mapping) or notice.get("is_original_second_attempt_artifact") is not False:
        raise FacingReviewError("diagnostic reconstruction notice changed")

    failure_record = _validate_external_record(
        manifest.get("bound_second_failure"), "bound second failure"
    )
    failure = _load_object(Path(failure_record["path"]), "bound second failure")
    if (
        failure.get("schema") != FAILURE_SCHEMA
        or failure.get("asset_id") != ASSET_ID
        or failure.get("decision") != "rejected"
        or failure.get("readiness_bundle_published") is not False
        or failure.get("preserved_artifacts") != []
    ):
        raise FacingReviewError("bound second failure is not the exact rejected attempt")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(SOURCE_ARTIFACTS):
        raise FacingReviewError("diagnostic artifact inventory changed")
    checked_artifacts = {
        filename: _validate_local_record(root, filename, artifacts.get(filename))
        for filename in SOURCE_ARTIFACTS
    }
    document = read_glb_document(
        Path(checked_artifacts["walking_rotation_only_reconstruction.glb"]["path"])
    )
    animations = document.get("animations")
    names = [entry.get("name") for entry in animations] if isinstance(animations, list) else []
    if names != ["Walking"]:
        raise FacingReviewError("diagnostic GLB must contain exactly one Walking animation")
    if not isinstance(document.get("meshes"), list) or not document["meshes"]:
        raise FacingReviewError("diagnostic GLB has no mesh")
    if not isinstance(document.get("skins"), list) or len(document["skins"]) != 1:
        raise FacingReviewError("diagnostic GLB must contain exactly one skin")

    static_input = manifest.get("authenticated_inputs", {}).get("static")
    if not isinstance(static_input, Mapping):
        raise FacingReviewError("authenticated static input is missing")
    static_record = _validate_external_record(static_input.get("static_qa"), "static QA")
    static = _load_object(Path(static_record["path"]), "static QA")
    checks = static.get("checks")
    axis = checks.get("axis_canonicalization") if isinstance(checks, Mapping) else None
    if (
        static.get("schema") != STATIC_QA_SCHEMA
        or static.get("asset_id") != ASSET_ID
        or static.get("decision") != "automatic_static_checks_passed"
        or not isinstance(axis, Mapping)
        or axis.get("canonical_front") != CANONICAL_FRONT_NAME
        or axis.get("transform_count") != 1
    ):
        raise FacingReviewError("static QA does not authenticate exactly-once FRONT -Y")
    static_mapping = checks.get("semantic_mapping") if isinstance(checks, Mapping) else None
    semantic_bones = (
        static_mapping.get("semantic_bones")
        if isinstance(static_mapping, Mapping)
        else None
    )
    if not isinstance(semantic_bones, Mapping) or any(
        not isinstance(semantic_bones.get(role), str) or not semantic_bones[role]
        for role in REQUIRED_SEMANTIC_ROLES
    ):
        raise FacingReviewError("static semantic mapping is incomplete")
    embedded_mapping = static_input.get("semantic_mapping")
    if embedded_mapping is not None:
        embedded_semantic_bones = (
            embedded_mapping.get("semantic_bones")
            if isinstance(embedded_mapping, Mapping)
            else None
        )
        if embedded_semantic_bones != semantic_bones:
            raise FacingReviewError(
                "embedded semantic bones changed from static QA"
            )

    motion = manifest.get("motion")
    expected_motion = {"action_name": "Walking", "fps": 30, "frame_count": 33}
    if not isinstance(motion, Mapping) or any(
        motion.get(key) != value for key, value in expected_motion.items()
    ):
        raise FacingReviewError("diagnostic motion contract changed")
    return {
        "asset_id": ASSET_ID,
        "canonical_front": CANONICAL_FRONT_NAME,
        "manifest": _validate_record(manifest_path, {
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }, "diagnostic manifest"),
        "failure": failure_record,
        "static_qa": static_record,
        "semantic_bones": dict(semantic_bones),
        "motion": expected_motion,
        "glb": checked_artifacts["walking_rotation_only_reconstruction.glb"],
        "media": {
            view: {
                kind: checked_artifacts[f"walking_{view}.{kind}"]
                for kind in ("mp4", "png")
            }
            for view in ("front", "side", "feet")
        },
    }


def _vector(value: Sequence[float], description: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise FacingReviewError(f"{description} must be a 3D vector")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise FacingReviewError(f"{description} contains a non-finite value")
    return result  # type: ignore[return-value]


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) - float(right[i]) for i in range(3))  # type: ignore[return-value]


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) + float(right[i]) for i in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(float(component) * factor for component in value)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[i]) * float(right[i]) for i in range(3))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _horizontal_normalized(
    value: Sequence[float], description: str, *, epsilon: float = 1.0e-10
) -> tuple[float, float, float]:
    vector = (float(value[0]), float(value[1]), 0.0)
    length = math.hypot(vector[0], vector[1])
    if not math.isfinite(length) or length <= epsilon:
        raise FacingReviewError(f"{description} has no horizontal direction")
    return (vector[0] / length, vector[1] / length, 0.0)


def _point(
    sample: Mapping[str, Sequence[float]], role: str, description: str
) -> tuple[float, float, float]:
    if role not in sample:
        raise FacingReviewError(f"{description} is missing semantic role {role}")
    return _vector(sample[role], f"{description} {role}")


def _body_right(sample: Mapping[str, Sequence[float]], description: str) -> tuple[float, float, float]:
    shoulder = _sub(
        _point(sample, "right_clavicle", description),
        _point(sample, "left_clavicle", description),
    )
    hip = _sub(
        _point(sample, "right_thigh", description),
        _point(sample, "left_thigh", description),
    )
    return _horizontal_normalized(
        _scale(_add(shoulder, hip), 0.5), f"{description} body-right basis"
    )


def classify_alignment(dot: float | None) -> str:
    if dot is None:
        return "travel_undefined"
    if not math.isfinite(dot) or dot < -1.000001 or dot > 1.000001:
        raise FacingReviewError("body/travel dot product is invalid")
    if dot >= 0.5:
        return "aligned"
    if dot <= -0.5:
        return "reversed"
    return "sideways"


def compute_facing_samples(
    bind_points: Mapping[str, Sequence[float]],
    frames: Sequence[Mapping[str, Sequence[float]]],
    *,
    fps: int,
    travel_epsilon_m: float = 1.0e-5,
) -> dict[str, Any]:
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise FacingReviewError("FPS must be a positive integer")
    if len(frames) < 2:
        raise FacingReviewError("facing review needs at least two frames")
    bind_right = _body_right(bind_points, "bind pose")
    bind_forward = _horizontal_normalized(
        _cross(bind_right, UP), "bind body-forward basis"
    )
    bind_dot = _dot(bind_forward, CANONICAL_FRONT)
    sign = 1.0 if bind_dot >= 0.0 else -1.0
    bind_forward = _scale(bind_forward, sign)
    bind_dot = _dot(bind_forward, CANONICAL_FRONT)
    if bind_dot < 0.8:
        raise FacingReviewError("bind body-forward basis does not authenticate FRONT -Y")

    pelvis = [_point(frame, "pelvis", f"frame {index + 1}") for index, frame in enumerate(frames)]
    results = []
    valid_dots = []
    for index, frame in enumerate(frames):
        right = _body_right(frame, f"frame {index + 1}")
        forward = _scale(
            _horizontal_normalized(_cross(right, UP), f"frame {index + 1} body forward"),
            sign,
        )
        if index == 0:
            displacement = _sub(pelvis[1], pelvis[0])
            time_delta = 1.0 / fps
        elif index == len(frames) - 1:
            displacement = _sub(pelvis[-1], pelvis[-2])
            time_delta = 1.0 / fps
        else:
            displacement = _sub(pelvis[index + 1], pelvis[index - 1])
            time_delta = 2.0 / fps
        horizontal_distance = math.hypot(displacement[0], displacement[1])
        if horizontal_distance <= travel_epsilon_m:
            travel = None
            speed = 0.0
            dot = None
            angle = None
        else:
            travel = _horizontal_normalized(displacement, f"frame {index + 1} travel")
            speed = horizontal_distance / time_delta
            dot = max(-1.0, min(1.0, _dot(forward, travel)))
            angle = math.degrees(math.atan2(_cross(forward, travel)[2], dot))
            valid_dots.append(dot)
        results.append(
            {
                "frame": index + 1,
                "pelvis_position": list(pelvis[index]),
                "body_right": list(right),
                "body_forward": list(forward),
                "body_canonical_dot": _dot(forward, CANONICAL_FRONT),
                "travel_direction": None if travel is None else list(travel),
                "travel_speed_mps": speed,
                "body_travel_dot": dot,
                "body_travel_signed_angle_deg": angle,
                "classification": classify_alignment(dot),
            }
        )
    undefined_count = len(results) - len(valid_dots)
    if valid_dots:
        median_dot = statistics.median(valid_dots)
        overall = classify_alignment(median_dot)
        reversed_ratio = sum(value <= -0.5 for value in valid_dots) / len(valid_dots)
        sideways_ratio = sum(-0.5 < value < 0.5 for value in valid_dots) / len(valid_dots)
        worst = min(valid_dots)
    else:
        median_dot = None
        overall = "travel_undefined"
        reversed_ratio = None
        sideways_ratio = None
        worst = None
    return {
        "schema": "second_retarget_facing_metrics_v1",
        "fps": fps,
        "frame_count": len(results),
        "bind_authentication": {
            "method": "bilateral_clavicle_and_thigh_basis_v1",
            "body_right": list(bind_right),
            "body_forward": list(bind_forward),
            "canonical_front": list(CANONICAL_FRONT),
            "dot": bind_dot,
            "sign_selected_without_travel": True,
        },
        "frames": results,
        "summary": {
            "valid_travel_frame_count": len(valid_dots),
            "undefined_travel_frame_count": undefined_count,
            "median_body_travel_dot": median_dot,
            "worst_body_travel_dot": worst,
            "reversed_frame_ratio": reversed_ratio,
            "sideways_frame_ratio": sideways_ratio,
            "overall_classification": overall,
        },
    }


def build_review_html(metrics: Mapping[str, Any]) -> bytes:
    if (
        metrics.get("schema") != "second_retarget_facing_metrics_v1"
        or metrics.get("fps") != 30
        or metrics.get("frame_count") != 33
        or not isinstance(metrics.get("frames"), list)
        or len(metrics["frames"]) != 33
    ):
        raise FacingReviewError("facing review HTML requires exactly 33 frames at 30 fps")
    for index, sample in enumerate(metrics["frames"], start=1):
        if not isinstance(sample, Mapping) or sample.get("frame") != index:
            raise FacingReviewError("facing review frame metrics are not contiguous")
    payload = json.dumps(metrics, separators=(",", ":"), sort_keys=True).replace(
        "<", "\\u003c"
    )
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>第二次 Retarget 朝向审核</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, "Noto Sans SC", system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #e7edf2; background: #0d1217; }
    main { max-width: 1500px; margin: 0 auto; padding: 18px; }
    header { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: start; border-bottom: 1px solid #34404b; padding-bottom: 14px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .warning { margin: 0; color: #ffcb77; font-weight: 650; }
    .authority { max-width: 460px; color: #c9d5de; line-height: 1.5; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0; font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 16px; height: 5px; border-radius: 3px; }
    .blue { background: #207cf2; } .red { background: #ef3b2d; } .grey { background: #888; } .yellow { background: #e7a82b; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    figure { margin: 0; min-width: 0; border: 1px solid #34404b; background: #161d24; }
    figcaption { padding: 8px 10px; color: #c9d5de; font-size: 13px; font-weight: 650; }
    video { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #070a0d; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0; }
    button, select, textarea { color: #e7edf2; background: #18232c; border: 1px solid #51606c; border-radius: 6px; font: inherit; }
    button, select { min-height: 38px; padding: 7px 12px; cursor: pointer; }
    .frame { margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 700; }
    .evidence { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
    .metric { border: 1px solid #34404b; background: #131a20; padding: 10px; }
    .metric small { display: block; color: #91a1ad; margin-bottom: 5px; }
    .metric strong { font-variant-numeric: tabular-nums; }
    .review { border-top: 1px solid #34404b; margin-top: 16px; padding-top: 14px; }
    .choices { display: flex; flex-wrap: wrap; gap: 8px; }
    .choice[aria-pressed="true"] { border-color: #42b7d2; background: #173c47; }
    textarea { width: 100%; min-height: 76px; padding: 9px; margin: 9px 0; resize: vertical; }
    .fine { color: #91a1ad; font-size: 12px; line-height: 1.5; }
    @media (max-width: 850px) {
      header { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .evidence { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .frame { margin-left: 0; width: 100%; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>第二次 Retarget：人物朝向与动作方向审核</h1>
      <p class="warning">第二次 retarget 已拒绝；本页只用于定位“侧着走/反着走”的根因。</p>
    </div>
    <div class="authority">自动箭头只描述骨架数学关系。人物网格是否真的面向箭头、绑定是否从一开始就错位，最终视觉判断由你决定。</div>
  </header>
  <div class="legend" aria-label="方向图例">
    <span><i class="swatch blue"></i>蓝色：身体前向</span>
    <span><i class="swatch red"></i>红色：root 位移方向</span>
    <span><i class="swatch grey"></i>灰色：FRONT -Y</span>
    <span><i class="swatch yellow"></i>黄色：root 完整轨迹</span>
  </div>
  <section class="grid" aria-label="同步视频">
    <figure><figcaption>正面 Front</figcaption><video data-view="front" src="/media/front" muted loop playsinline preload="auto"></video></figure>
    <figure><figcaption>侧面 Side</figcaption><video data-view="side" src="/media/side" muted loop playsinline preload="auto"></video></figure>
    <figure><figcaption>脚部 Feet</figcaption><video data-view="feet" src="/media/feet" muted loop playsinline preload="auto"></video></figure>
    <figure><figcaption>俯视朝向 Top + arrows</figcaption><video data-view="top" src="/media/top" muted loop playsinline preload="auto"></video></figure>
  </section>
  <div class="controls">
    <button id="master-toggle" type="button">播放 / Play</button>
    <button type="button" data-step="-1">上一帧</button>
    <button type="button" data-step="1">下一帧</button>
    <label>速度 <select id="playback-rate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label>
    <span class="frame" id="frame-label">Frame 1 / 33</span>
  </div>
  <section class="evidence" aria-label="当前帧指标">
    <div class="metric"><small>人工状态</small><strong id="classification">—</strong></div>
    <div class="metric"><small>身体/位移点积</small><strong id="dot">—</strong></div>
    <div class="metric"><small>有符号夹角</small><strong id="angle">—</strong></div>
    <div class="metric"><small>root 速度</small><strong id="speed">—</strong></div>
    <div class="metric"><small>自动整段摘要</small><strong id="summary">—</strong></div>
  </section>
  <section class="review">
    <h2>你的观察（仅保存在本浏览器）</h2>
    <div class="choices">
      <button class="choice" type="button" data-observation="sideways">人物侧着走 sideways</button>
      <button class="choice" type="button" data-observation="reversed">人物反着走 reversed</button>
      <button class="choice" type="button" data-observation="aligned_but_deformed">方向一致但绑定/变形错误</button>
      <button class="choice" type="button" data-observation="camera_misleading">相机造成误导</button>
    </div>
    <textarea id="notes" placeholder="可记录具体帧号、身体朝向、腿部动作或绑定问题"></textarea>
    <button id="export" type="button">导出我的观察 JSON</button>
    <p class="fine">本页没有正式数据审批接口，也不会改写第二次失败记录。你的选择用于接管视觉诊断。</p>
  </section>
</main>
<script id="metrics-data" type="application/json">__METRICS__</script>
<script>
  const metrics = JSON.parse(document.getElementById("metrics-data").textContent);
  const FPS = 30;
  const videos = Array.from(document.querySelectorAll("video"));
  const master = videos.find((video) => video.dataset.view === "top");
  const toggle = document.getElementById("master-toggle");
  const storageKey = "second-retarget-facing-review-v1";
  let observation = null;
  function frameIndex() { return Math.max(0, Math.min(32, Math.round(master.currentTime * FPS))); }
  function syncFollowers() {
    videos.filter((video) => video !== master).forEach((video) => {
      if (Math.abs(video.currentTime - master.currentTime) > 0.5 / FPS) video.currentTime = master.currentTime;
    });
  }
  function updateEvidence() {
    const index = frameIndex();
    const sample = metrics.frames[index];
    document.getElementById("frame-label").textContent = `Frame ${index + 1} / 33`;
    document.getElementById("classification").textContent = sample.classification;
    document.getElementById("dot").textContent = sample.body_travel_dot == null ? "undefined" : sample.body_travel_dot.toFixed(4);
    document.getElementById("angle").textContent = sample.body_travel_signed_angle_deg == null ? "undefined" : `${sample.body_travel_signed_angle_deg.toFixed(2)}°`;
    document.getElementById("speed").textContent = `${sample.travel_speed_mps.toFixed(3)} m/s`;
    document.getElementById("summary").textContent = metrics.summary.overall_classification;
    syncFollowers();
  }
  async function playAll() { await Promise.all(videos.map((video) => video.play().catch(() => null))); toggle.textContent = "暂停 / Pause"; }
  function pauseAll() { videos.forEach((video) => video.pause()); toggle.textContent = "播放 / Play"; }
  toggle.addEventListener("click", () => master.paused ? playAll() : pauseAll());
  document.querySelectorAll("[data-step]").forEach((button) => button.addEventListener("click", () => {
    pauseAll(); master.currentTime = Math.max(0, Math.min((33 - 1) / FPS, master.currentTime + Number(button.dataset.step) / FPS)); syncFollowers(); updateEvidence();
  }));
  document.getElementById("playback-rate").addEventListener("change", (event) => videos.forEach((video) => video.playbackRate = Number(event.target.value)));
  master.addEventListener("timeupdate", updateEvidence);
  master.addEventListener("seeked", updateEvidence);
  function saveObservation() { localStorage.setItem(storageKey, JSON.stringify({observation, notes: document.getElementById("notes").value})); }
  document.querySelectorAll(".choice").forEach((button) => button.addEventListener("click", () => {
    observation = button.dataset.observation;
    document.querySelectorAll(".choice").forEach((candidate) => candidate.setAttribute("aria-pressed", String(candidate === button)));
    saveObservation();
  }));
  document.getElementById("notes").addEventListener("input", saveObservation);
  const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
  if (saved) { observation = saved.observation; document.getElementById("notes").value = saved.notes || ""; document.querySelectorAll(".choice").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.observation === observation))); }
  document.getElementById("export").addEventListener("click", () => {
    const value = {schema: "second_retarget_facing_human_observation_v1", asset_id: "rocketbox_male_adult_01", rejected_attempt: 2, observation, notes: document.getElementById("notes").value, automatic_summary: metrics.summary};
    const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"})); link.download = "second_retarget_facing_observation.json"; link.click(); URL.revokeObjectURL(link.href);
  });
  updateEvidence();
</script>
</body>
</html>
'''
    return template.replace("__METRICS__", payload).encode("utf-8")
