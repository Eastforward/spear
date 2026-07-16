#!/usr/bin/env python3
"""Interactive localhost server for shared limb motion-basis correction."""

from __future__ import annotations

import argparse
import getpass
import json
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

from flask import Flask, abort, jsonify, render_template_string, request, send_file, session, url_for

from retarget_motion_basis_review import (
    CANDIDATE_ANGLES,
    VIEWS,
    MotionBasisReviewError,
    record_selection,
    sha256_file,
    validate_review_bundle,
)


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route 2 四肢动作基底人工纠正</title>
<style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#e8edf2;background:#111820}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1480px;margin:auto;padding:18px}.head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.head h1{font-size:23px;margin:0 0 7px}.sub{color:#aebbc7;line-height:1.5;font-size:14px}.warning{padding:10px 13px;background:#38281d;border-left:4px solid #ef9d35;border-radius:5px;color:#ffd7a5;margin:14px 0}.axis{display:flex;gap:16px;align-items:center;justify-content:center;padding:10px;background:#18222c;border:1px solid #2e3b47;border-radius:8px}.front{color:#43dc73;font-weight:800}.up{color:#55aaff;font-weight:800}.stage{display:grid;grid-template-columns:84px minmax(0,1fr) 84px;grid-template-rows:72px auto 72px;gap:8px;align-items:center}.videos{grid-column:2;grid-row:2;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.videos figure{margin:0;min-width:0}.videos figcaption{font-size:13px;font-weight:750;margin:0 0 5px;color:#cbd7e2}.videos video{width:100%;display:block;background:#050709;border:1px solid #354452;border-radius:6px;aspect-ratio:16/9}.videos figure:nth-child(4),.videos figure:nth-child(5){grid-column:span 1}.rot{border:0;border-radius:10px;background:#d9e4ee;color:#17212a;font-size:24px;font-weight:800;padding:9px;cursor:pointer;min-height:56px}.rot small{display:block;font-size:10px;font-weight:650}.left{grid-column:1;grid-row:2}.right{grid-column:3;grid-row:2}.flip{grid-column:2;grid-row:1;justify-self:center}.reset{grid-column:2;grid-row:3;justify-self:center}.rot[aria-pressed=true]{background:#43dc73}.panel{display:grid;grid-template-columns:1fr 310px;gap:14px;margin-top:14px}.metrics,.decision{background:#18222c;border:1px solid #2e3b47;border-radius:8px;padding:13px}.metrics h2,.decision h2{font-size:16px;margin:0 0 10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #2e3b47;text-align:left}.good{color:#43dc73}.bad{color:#ff7575}.controls{display:flex;gap:8px;align-items:center;margin:11px 0}.controls button,.decision button{border:1px solid #526372;background:#253440;color:#eef4f8;border-radius:6px;padding:8px 10px;cursor:pointer}.decision input{width:100%;padding:8px;background:#10171e;color:#fff;border:1px solid #526372;border-radius:6px;margin:5px 0 9px}.decision .use{background:#165f35;border-color:#2b9d59}.decision .none{background:#68302d;border-color:#a95149}.status{font-size:13px;line-height:1.45;margin-top:9px;color:#b8c5cf}.path{font:12px ui-monospace,monospace;word-break:break-all;color:#91a3b2}@media(max-width:900px){.videos{grid-template-columns:1fr}.panel{grid-template-columns:1fr}.stage{grid-template-columns:62px minmax(0,1fr) 62px}.head{display:block}}
</style></head><body><main class="wrap">
<div class="head"><div><h1>Route 2：四肢共享动作基底人工纠正</h1><div class="sub">保持 Pixal mesh、身体朝向、root 轨迹和躯干不动；按钮只切换双臂与双腿共同使用的 canonical 动作基底。</div></div><div class="axis"><span class="front">绿色 FRONT −Y →</span><span class="up">蓝色 UP +Z ↑</span></div></div>
<div class="warning">这是第三次正式 retarget 之前的参数纠正页，不是正式资产批准。第二次结果仍保持 rejected。</div>
<section class="stage">
 <button class="rot flip" data-id="yaw_180">⇄<small>共享基底 180°</small></button>
 <button class="rot left" data-id="yaw_m090">↺<small>共享基底 −90°</small></button>
 <div class="videos">
 {% for view,label in labels.items() %}<figure><figcaption>{{label}}</figcaption><video id="video-{{view}}" controls muted loop playsinline></video></figure>{% endfor %}
 </div>
 <button class="rot right" data-id="yaw_p090">↻<small>共享基底 +90°</small></button>
 <button class="rot reset" data-id="yaw_000">⟲<small>共享基底 0° / reset</small></button>
</section>
<div class="controls"><button id="play">播放/暂停全部</button><button id="back">−1 帧</button><button id="forward">+1 帧</button><label>速度 <select id="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><strong id="active"></strong></div>
<section class="panel"><div class="metrics"><h2>当前候选四肢平面指标</h2><table><thead><tr><th>链</th><th>横向/前后摆幅</th><th>平面法线·横轴</th><th>平面法线·前轴</th><th>结论</th></tr></thead><tbody id="metric-body"></tbody></table><p id="overall"></p><h2>轴向身体门禁</h2><table><thead><tr><th>指标</th><th>最大绝对角</th><th>限值</th><th>结论</th></tr></thead><tbody id="axial-body"></tbody></table><p id="axial-overall"></p></div>
<div class="decision"><h2>保存人工纠正参数</h2><label>Reviewer<input id="reviewer" value="{{reviewer}}"></label><button class="use" id="select">将当前基底用于下一次 retarget</button><button class="none" id="none">四个候选都不对</button><div class="status" id="status">尚未写入选择。</div><p class="path">Bundle: {{bundle_dir}}</p></div></section>
</main><script>
const candidates={{candidates|tojson}}, csrf={{csrf|tojson}}, snapshot={{snapshot|tojson}}, fps=30;
const views={{views|tojson}}, videos=views.map(v=>document.getElementById(`video-${v}`)); let active="yaw_000", switching=false;
function metricClass(v){return v==="sagittal"?"good":v==="sideways"?"bad":""}
function updateMetrics(){const c=candidates[active],tbody=document.getElementById("metric-body");tbody.innerHTML="";for(const [name,v] of Object.entries(c.metrics.limbs)){const tr=document.createElement("tr");tr.innerHTML=`<td>${name}</td><td>${v.lateral_to_forward_excursion_ratio.toFixed(3)}</td><td>${v.mean_plane_normal_dot_lateral_abs.toFixed(3)}</td><td>${v.mean_plane_normal_dot_forward_abs.toFixed(3)}</td><td class="${metricClass(v.classification)}">${v.classification}</td>`;tbody.appendChild(tr)}document.getElementById("overall").textContent=`自动结论：${c.metrics.overall_classification}`;const axial=c.metrics.anatomical_axial_pose_gate,abody=document.getElementById("axial-body"),aoverall=document.getElementById("axial-overall");abody.innerHTML="";if(axial){for(const [name,v] of Object.entries(axial.metrics)){const tr=document.createElement("tr");tr.innerHTML=`<td>${name}</td><td>${v.maximum_abs_deg.toFixed(3)}°</td><td>${v.maximum_allowed_abs_deg.toFixed(3)}°</td><td class="${v.status==="passed"?"good":"bad"}">${v.status}</td>`;abody.appendChild(tr)}aoverall.textContent=`轴向结论：${axial.overall_classification}`}else{aoverall.textContent="旧 bundle：没有轴向身体门禁。"}document.getElementById("active").textContent=`当前：${active} (${c.yaw_degrees}°)`;document.querySelectorAll(".rot").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.id===active)))}
async function switchTo(id){if(switching||!candidates[id])return;switching=true;const time=videos[0].currentTime||0,paused=videos[0].paused,rate=videos[0].playbackRate;active=id;for(const [i,v] of videos.entries()){v.pause();v.src=candidates[id].media[views[i]];v.load()}await Promise.all(videos.map(v=>new Promise(resolve=>{if(v.readyState>=1)resolve();else v.addEventListener("loadedmetadata",resolve,{once:true})})));for(const v of videos){v.currentTime=Math.min(time,v.duration||time);v.playbackRate=rate}if(!paused)await Promise.all(videos.map(v=>v.play().catch(()=>{})));updateMetrics();switching=false}
document.querySelectorAll(".rot").forEach(b=>b.onclick=()=>switchTo(b.dataset.id));
function sync(){if(switching)return;const t=videos[0].currentTime;for(const v of videos.slice(1))if(Math.abs(v.currentTime-t)>1/fps)v.currentTime=t}videos[0].addEventListener("timeupdate",sync);videos[0].addEventListener("play",()=>videos.slice(1).forEach(v=>v.play().catch(()=>{})));videos[0].addEventListener("pause",()=>videos.slice(1).forEach(v=>v.pause()));
document.getElementById("play").onclick=()=>videos[0].paused?videos[0].play():videos[0].pause();document.getElementById("back").onclick=()=>videos.forEach(v=>v.currentTime=Math.max(0,v.currentTime-1/fps));document.getElementById("forward").onclick=()=>videos.forEach(v=>v.currentTime=Math.min(v.duration||999,v.currentTime+1/fps));document.getElementById("rate").onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
async function save(candidate){const status=document.getElementById("status");status.textContent="正在校验并写入…";const response=await fetch("/selection",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({csrf_token:csrf,manifest_sha256:snapshot,candidate_id:candidate,reviewer:document.getElementById("reviewer").value})});const value=await response.json();status.textContent=response.ok?`已写入：${value.path}`:`写入失败：${value.error}`}
document.getElementById("select").onclick=()=>save(active);document.getElementById("none").onclick=()=>save(null);switchTo(active);
</script></body></html>"""


def _candidate_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for candidate_id, candidate in manifest["candidates"].items():
        result[candidate_id] = {
            "yaw_degrees": candidate["yaw_degrees"],
            "metrics": candidate["metrics"],
            "media": {
                view: url_for(
                    "media",
                    candidate_id=candidate_id,
                    filename=f"walking_{view}.mp4",
                )
                for view in VIEWS
            },
        }
    return result


def create_app(bundle_dir: Path, selection_dir: Path) -> Flask:
    bundle = Path(bundle_dir).resolve()
    selection = Path(selection_dir).resolve()
    startup_manifest = validate_review_bundle(bundle)
    manifest_path = bundle / "motion_basis_review_manifest.json"
    startup_sha256 = sha256_file(manifest_path)
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secrets.token_urlsafe(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )

    @app.after_request
    def no_store(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    def unchanged_manifest() -> None:
        if sha256_file(manifest_path) != startup_sha256:
            abort(409, description="review manifest changed after server startup")

    @app.get("/")
    def index():
        unchanged_manifest()
        token = session.get("csrf_token")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        labels = {
            "front": "Front：看左右横摆与手臂位置",
            "side": "Side：看前后摆动与迈步",
            "top": "Top：看动作平面和 root 轨迹",
            "feet": "Feet：看双脚、膝盖和落地",
            "skeleton": "Skeleton：看四肢链和手腕/脚踝",
        }
        return render_template_string(
            PAGE,
            candidates=_candidate_payload(startup_manifest),
            views=list(VIEWS),
            labels=labels,
            csrf=token,
            snapshot=startup_sha256,
            reviewer=getpass.getuser(),
            bundle_dir=str(bundle),
        )

    @app.get("/manifest")
    def manifest():
        unchanged_manifest()
        return jsonify(startup_manifest)

    @app.get("/media/<candidate_id>/<filename>")
    def media(candidate_id: str, filename: str):
        unchanged_manifest()
        if candidate_id not in CANDIDATE_ANGLES:
            abort(404)
        artifacts = startup_manifest["candidates"][candidate_id]["artifacts"]
        record = artifacts.get(filename)
        if not isinstance(record, Mapping):
            abort(404)
        path = bundle / record["filename"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            abort(409, description="requested review artifact changed")
        mimetype = "video/mp4" if filename.endswith(".mp4") else "image/png"
        return send_file(path, mimetype=mimetype, conditional=True, max_age=0)

    @app.post("/selection")
    def selection_write():
        unchanged_manifest()
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            return jsonify(error="selection request must be JSON"), 400
        if value.get("csrf_token") != session.get("csrf_token"):
            return jsonify(error="CSRF token mismatch"), 403
        candidate_id = value.get("candidate_id")
        if candidate_id is not None and candidate_id not in CANDIDATE_ANGLES:
            return jsonify(error="candidate is not one of the exact reviewed bases"), 400
        try:
            output = record_selection(
                bundle_dir=bundle,
                selection_dir=selection,
                candidate_id=candidate_id,
                submitted_manifest_sha256=str(value.get("manifest_sha256", "")),
                reviewer=str(value.get("reviewer", "")),
            )
        except MotionBasisReviewError as error:
            status = 409 if "already exists" in str(error) or "stale" in str(error) else 400
            return jsonify(error=str(error)), status
        return jsonify(path=str(output), decision=json.loads(output.read_text())["decision"]), 201

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(args.bundle_dir, args.selection_dir)
    print(f"Shared limb motion-basis review: http://{args.host}:{args.port}/", flush=True)
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
