#!/usr/bin/env python3
"""Build a hash-locked pre-bind versus second-retarget gait-plane review."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_DIR = SCRIPT_PATH.parent
SPIKE_DIR = TOOLS_DIR / "spike_rlr"
SPEAR_ROOT = TOOLS_DIR.parent
for directory in (SPEAR_ROOT, TOOLS_DIR, SPIKE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import blender_render_tokenrig_human_review as dynamic_review
from tools import blender_retarget_rocketbox_to_tokenrig as runner
from second_retarget_facing_review import (
    FacingReviewError,
    authenticate_second_attempt,
    compute_gait_plane_samples,
    sha256_file,
    validate_facing_bundle,
)


FPS = 30
BUNDLE_SCHEMA = "prebind_vs_second_retarget_gait_review_v1"


class PrebindReviewError(RuntimeError):
    """The pre-bind gait comparison failed an immutable invariant."""


def _record(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    return {
        "filename" if filename is not None else "path": (
            filename if filename is not None else str(path.resolve())
        ),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrebindReviewError(f"{description} is invalid: {error}") from error
    if not isinstance(value, dict):
        raise PrebindReviewError(f"{description} root must be an object")
    return value


def _sample_pose_frames(
    bpy: Any,
    armature: Any,
    semantic: Mapping[str, Any],
    *,
    frame_start: int,
    frame_end: int,
) -> list[dict[str, tuple[float, float, float]]]:
    names = {
        "pelvis": semantic["pelvis"],
        "left_clavicle": semantic["left_clavicle"],
        "right_clavicle": semantic["right_clavicle"],
        "left_thigh": semantic["left_thigh"],
        "left_calf": semantic["left_calf"],
        "left_foot": semantic["left_foot"],
        "right_thigh": semantic["right_thigh"],
        "right_calf": semantic["right_calf"],
        "right_foot": semantic["right_foot"],
    }
    for role, name in names.items():
        if not isinstance(name, str) or armature.pose.bones.get(name) is None:
            raise PrebindReviewError(f"semantic pose bone is missing: {role}={name}")
    result = []
    scene = bpy.context.scene
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        result.append(
            {
                role: tuple(armature.matrix_world @ armature.pose.bones[name].head)
                for role, name in names.items()
            }
        )
    return result


def _baseline_media(baseline_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    media = manifest.get("media")
    validation = (
        manifest.get("automatic_checks", {})
        .get("media_validation", {})
        .get("media", {})
    )
    expected = ("front", "side", "top", "source_target")
    if not isinstance(media, Mapping) or not isinstance(validation, Mapping):
        raise PrebindReviewError("sealed baseline media records are missing")
    result = {}
    for view in expected:
        filename = media.get(view)
        record = validation.get(view)
        if not isinstance(filename, str) or not isinstance(record, Mapping):
            raise PrebindReviewError(f"sealed baseline {view} media is missing")
        path = baseline_dir / filename
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise PrebindReviewError(f"sealed baseline {view} is not a direct file")
        actual = _record(path)
        if record.get("sha256") != actual["sha256"]:
            raise PrebindReviewError(f"sealed baseline {view} hash changed")
        result[view] = actual
    return result


def build_prebind_html(metrics: Mapping[str, Any]) -> bytes:
    source = metrics.get("source_prebind")
    target = metrics.get("target_second_retarget")
    if metrics.get("schema") != "prebind_vs_second_retarget_gait_plane_v1":
        raise PrebindReviewError("prebind comparison metrics schema is invalid")
    if not isinstance(source, Mapping) or source.get("overall_classification") != "sagittal_forward_gait":
        raise PrebindReviewError("source prebind gait is not authenticated sagittal forward gait")
    if not isinstance(target, Mapping) or target.get("frame_count") != 33:
        raise PrebindReviewError("target second-retarget gait metrics are incomplete")
    payload = json.dumps(metrics, separators=(",", ":"), sort_keys=True).replace(
        "<", "\\u003c"
    )
    html = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>绑定前 vs 第二次 Retarget 腿部动作审核</title>
<style>
:root{color-scheme:dark;font-family:Inter,"Noto Sans SC",system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#0d1217;color:#e8eef3}main{max-width:1600px;margin:auto;padding:16px}h1{margin:0 0 6px;font-size:24px}.warning{color:#ffc86c;margin:0 0 12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}.stage{border:1px solid #394651;background:#121a21;padding:10px}.stage h2{margin:0 0 4px;font-size:18px}.stage p{margin:0 0 10px;color:#aebbc5}.videos{display:grid;grid-template-columns:1fr 1fr;gap:8px}figure{margin:0;background:#080c10;border:1px solid #2c3740}figcaption{padding:6px 8px;color:#b9c6cf;font-size:12px}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:14px 0}button,select,textarea{background:#18242d;color:#e8eef3;border:1px solid #53636f;border-radius:6px;font:inherit}button,select{padding:7px 11px;min-height:38px}.frame{margin-left:auto;font-variant-numeric:tabular-nums}.metrics{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px}.metrics th,.metrics td{border:1px solid #394651;padding:7px;text-align:right}.metrics th:first-child,.metrics td:first-child{text-align:left}.good{color:#7bdba1}.bad{color:#ff8b81}.review{border-top:1px solid #394651;padding-top:12px}.choices{display:flex;gap:8px;flex-wrap:wrap}.choice[aria-pressed=true]{background:#16404b;border-color:#3cc1df}textarea{display:block;width:100%;min-height:72px;margin:8px 0;padding:8px}.fine{color:#97a6b1;font-size:12px}@media(max-width:950px){.columns{grid-template-columns:1fr}.videos{grid-template-columns:1fr}.frame{margin-left:0;width:100%}}
</style></head><body><main>
<h1>绑定前源动作 vs 第二次 TokenRig：腿部动作平面审核</h1>
<p class="warning">人物/root 朝向一致并不足够。本页专门判断髋—膝—踝是否沿人物前后方向弯曲；人工判断优先。</p>
<section class="columns">
 <article class="stage"><h2>绑定前 Rocketbox 源动作</h2><p>这是实际送入 TokenRig retarget 之前、已封存的 Walking。</p><div class="videos">
  <figure><figcaption>Source Front</figcaption><video data-stage="source" src="/source/front" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Source Side</figcaption><video data-stage="source" src="/source/side" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Source Top</figcaption><video data-stage="source" src="/source/top" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Source skeleton + approved Rocketbox</figcaption><video data-stage="source" src="/source/source_target" muted playsinline preload="auto"></video></figure>
 </div></article>
 <article class="stage"><h2>第二次 TokenRig 绑定后</h2><p>相同 Walking 经过第二次 rotation-only retarget 后的 Pixal 网格。</p><div class="videos">
  <figure><figcaption>Target Front</figcaption><video data-stage="target" src="/target/front" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Target Side</figcaption><video data-stage="target" src="/target/side" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Target Feet</figcaption><video data-stage="target" src="/target/feet" muted playsinline preload="auto"></video></figure>
  <figure><figcaption>Target Top + body/root arrows</figcaption><video id="master" data-stage="target" src="/target/top" muted loop playsinline preload="auto"></video></figure>
 </div></article>
</section>
<div class="controls"><button id="toggle" type="button">播放 / Play</button><button data-step="-1" type="button">上一帧</button><button data-step="1" type="button">下一帧</button><label>速度 <select id="rate"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label><strong class="frame" id="frame">Frame 1 / 33</strong></div>
<table class="metrics"><thead><tr><th>阶段 / 腿</th><th>脚横向/前后摆幅比</th><th>膝平面法向·身体横轴</th><th>膝平面法向·身体前向</th><th>自动描述</th></tr></thead><tbody id="metric-body"></tbody></table>
<section class="review"><h2>你的视觉结论</h2><div class="choices"><button class="choice" data-value="retarget_introduced_sideways_plane">retarget 引入横向腿平面</button><button class="choice" data-value="source_animation_wrong">绑定前源动画已错误</button><button class="choice" data-value="target_bind_basis_wrong">TokenRig bind/骨轴基底错误</button><button class="choice" data-value="ambiguous">仍需进一步对照</button></div><textarea id="notes" placeholder="记录帧号、哪条腿、Front/Side 中看到的弯曲方向"></textarea><button id="export" type="button">导出我的观察 JSON</button><p class="fine">只保存在浏览器，不写入 formal approval；第二次结果仍为 rejected。</p></section>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById("payload").textContent),FPS=30,videos=Array.from(document.querySelectorAll("video")),master=document.getElementById("master"),toggle=document.getElementById("toggle");
function sync(){videos.filter(v=>v!==master).forEach(v=>{if(Math.abs(v.currentTime-master.currentTime)>0.5 / FPS)v.currentTime=master.currentTime});document.getElementById("frame").textContent=`Frame ${Math.max(1,Math.min(33,Math.round(master.currentTime*FPS)+1))} / 33`}
async function play(){await Promise.all(videos.map(v=>v.play().catch(()=>null)));toggle.textContent="暂停 / Pause"}function pause(){videos.forEach(v=>v.pause());toggle.textContent="播放 / Play"}toggle.onclick=()=>master.paused?play():pause();master.ontimeupdate=sync;master.onseeked=sync;document.querySelectorAll("[data-step]").forEach(b=>b.onclick=()=>{pause();master.currentTime=Math.max(0,Math.min(32/FPS,master.currentTime+Number(b.dataset.step)/FPS));sync()});document.getElementById("rate").onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
const tbody=document.getElementById("metric-body");for(const [key,label] of [["source_prebind","绑定前"],["target_second_retarget","绑定后"]])for(const side of ["left","right"]){const v=data[key].legs[side],tr=document.createElement("tr"),bad=key==="target_second_retarget";tr.innerHTML=`<td>${label} ${side}</td><td>${v.lateral_to_forward_excursion_ratio.toFixed(3)}</td><td>${v.mean_knee_normal_dot_lateral_abs.toFixed(3)}</td><td>${v.mean_knee_normal_dot_forward_abs.toFixed(3)}</td><td class="${bad?'bad':'good'}">${data[key].overall_classification}</td>`;tbody.appendChild(tr)}
const store="prebind-vs-second-gait-review-v1";let decision=null;function save(){localStorage.setItem(store,JSON.stringify({decision,notes:document.getElementById("notes").value}))}document.querySelectorAll(".choice").forEach(b=>b.onclick=()=>{decision=b.dataset.value;document.querySelectorAll(".choice").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));save()});document.getElementById("notes").oninput=save;const old=JSON.parse(localStorage.getItem(store)||"null");if(old){decision=old.decision;document.getElementById("notes").value=old.notes||"";document.querySelectorAll(".choice").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.value===decision)))}document.getElementById("export").onclick=()=>{const a=document.createElement("a"),value={schema:"prebind_vs_second_gait_human_observation_v1",decision,notes:document.getElementById("notes").value,metrics:{source:data.source_prebind.overall_classification,target:data.target_second_retarget.overall_classification}};a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}));a.download="prebind_vs_second_gait_observation.json";a.click();URL.revokeObjectURL(a.href)};sync();
</script></body></html>'''
    return html.replace("__PAYLOAD__", payload).encode("utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--second-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--second-facing-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(
    *,
    baseline_dir: Path,
    second_diagnostic_dir: Path,
    second_facing_dir: Path,
    output_dir: Path,
    command: Sequence[str],
) -> Path:
    destination = dynamic_review.validate_destination(output_dir)
    baseline_dir = baseline_dir.resolve()
    baseline_manifest_path = baseline_dir / "retarget_manifest.json"
    baseline_blend = baseline_dir / "retarget.blend"
    baseline_auth = runner.authenticate_sealed_walk(
        base_avatar_id="rocketbox_male_adult_01",
        baseline_retarget_blend=baseline_blend,
        baseline_retarget_manifest=baseline_manifest_path,
    )
    baseline_manifest = _load_object(baseline_manifest_path, "baseline retarget manifest")
    source_media = _baseline_media(baseline_dir, baseline_manifest)
    second_auth = authenticate_second_attempt(second_diagnostic_dir)
    facing_manifest = validate_facing_bundle(second_facing_dir)
    try:
        import bpy
    except ImportError as error:
        raise PrebindReviewError("prebind gait builder must run inside Blender") from error
    if tuple(bpy.app.version) != (4, 2, 1):
        raise PrebindReviewError("prebind gait builder requires Blender 4.2.1")

    bpy.ops.wm.open_mainfile(filepath=str(baseline_blend))
    source = runner._identify_walk_source(bpy, baseline_auth["source_animation"])
    source_frames = _sample_pose_frames(
        bpy,
        source.armature,
        runner.ROCKETBOX_ROLE_TO_BONE,
        frame_start=source.frame_start,
        frame_end=source.frame_end,
    )
    source_metrics = compute_gait_plane_samples(source_frames, fps=FPS)

    dynamic_review._clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(second_auth["glb"]["path"]))
    armatures = [value for value in scene.objects if value.type == "ARMATURE"]
    if len(armatures) != 1:
        raise PrebindReviewError("second retarget import must contain one armature")
    target = armatures[0]
    action = target.animation_data.action if target.animation_data else None
    if action is None:
        raise PrebindReviewError("second retarget import has no Walking action")
    frame_start, frame_end = runner._integer_frame_range(action)
    target_semantic = second_auth["semantic_bones"]
    for required in (
        target_semantic["left_calf"],
        target_semantic["right_calf"],
    ):
        if target.pose.bones.get(required) is None:
            raise PrebindReviewError(f"target semantic calf is missing: {required}")
    target_frames = _sample_pose_frames(
        bpy,
        target,
        target_semantic,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    target_metrics = compute_gait_plane_samples(target_frames, fps=FPS)
    metrics = {
        "schema": "prebind_vs_second_retarget_gait_plane_v1",
        "asset_id": "rocketbox_male_adult_01",
        "source_prebind": source_metrics,
        "target_second_retarget": target_metrics,
    }
    html = build_prebind_html(metrics)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=str(destination.parent)
        )
    )
    try:
        _write_json_exclusive(staging / "prebind_gait_metrics.json", metrics)
        _write_exclusive(staging / "review.html", html)
        current_baseline = runner.authenticate_sealed_walk(
            base_avatar_id="rocketbox_male_adult_01",
            baseline_retarget_blend=baseline_blend,
            baseline_retarget_manifest=baseline_manifest_path,
        )
        current_second = authenticate_second_attempt(second_diagnostic_dir)
        current_facing = validate_facing_bundle(second_facing_dir)
        if current_baseline != baseline_auth or current_second != second_auth or current_facing != facing_manifest:
            raise PrebindReviewError("an authenticated source changed during comparison")
        target_media = {
            view: second_auth["media"][view]["mp4"] for view in ("front", "side", "feet")
        }
        top_record = facing_manifest["derived_artifacts"]["top_facing.mp4"]
        target_media["top"] = {
            "path": str((second_facing_dir / top_record["filename"]).resolve()),
            "sha256": top_record["sha256"],
            "size_bytes": top_record["size_bytes"],
        }
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "asset_id": "rocketbox_male_adult_01",
            "classification": "technical_diagnostic_only",
            "decision": "rejected",
            "formal_dataset_asset": False,
            "user_authority": "human_visual_review_required",
            "source_prebind": {"authentication": baseline_auth, "media": source_media},
            "target_second_retarget": {
                "authentication": second_auth,
                "facing_bundle_manifest_sha256": sha256_file(second_facing_dir / "facing_review_manifest.json"),
                "media": target_media,
            },
            "automatic_gait_plane_comparison": {
                "source": source_metrics["overall_classification"],
                "target": target_metrics["overall_classification"],
            },
            "local_artifacts": {
                "prebind_gait_metrics.json": _record(staging / "prebind_gait_metrics.json", filename="prebind_gait_metrics.json"),
                "review.html": _record(staging / "review.html", filename="review.html"),
            },
            "environment": {"blender_version": "4.2.1", "fps": FPS},
            "execution": {"builder": _record(SCRIPT_PATH), "command": list(command)},
        }
        if "user_approved" in json.dumps(manifest, sort_keys=True):
            raise PrebindReviewError("prebind comparison may not claim user approval")
        _write_json_exclusive(staging / "prebind_gait_review_manifest.json", manifest)
        expected = {"prebind_gait_metrics.json", "review.html", "prebind_gait_review_manifest.json"}
        if {path.name for path in staging.iterdir()} != expected:
            raise PrebindReviewError("prebind comparison staging inventory is invalid")
        for path in staging.iterdir():
            dynamic_review._fsync_file(path)
            path.chmod(0o444)
        dynamic_review._fsync_directory(staging)
        staging.chmod(0o555)
        dynamic_review.rename_directory_noreplace(staging, destination)
        staging = None
        return destination / "prebind_gait_review_manifest.json"
    finally:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            for path in staging.iterdir():
                path.chmod(0o600)
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        baseline_dir=args.baseline_dir,
        second_diagnostic_dir=args.second_diagnostic_dir,
        second_facing_dir=args.second_facing_dir,
        output_dir=args.output_dir,
        command=sys.argv,
    )
    print(f"PREBIND_GAIT_REVIEW_OK {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
