#!/usr/bin/env python3
"""Serve the mandatory pre-animation generated-animal motion-basis gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


PREVIEW_SCHEMA = "avengine_generated_animal_motion_basis_preview_v1"
DECISION_SCHEMA = "generated_animal_motion_basis_manual_decision_v1"


class ReviewError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def hash_without(value: dict[str, Any], key: str) -> str:
    return hashlib.sha256(
        canonical_json({name: item for name, item in value.items() if name != key}).encode(
            "utf-8"
        )
    ).hexdigest()


def load_previews(root: Path) -> dict[str, dict[str, Any]]:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ReviewError(f"preview root is missing: {root}")
    previews = {}
    for path in sorted(root.glob("*/preview.json")):
        if path.is_symlink() or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        asset_id = payload.get("asset_id")
        if (
            payload.get("schema") != PREVIEW_SCHEMA
            or not isinstance(asset_id, str)
            or not asset_id
            or payload.get("target_animation_generated") is not False
            or payload.get("preview_sha256") != hash_without(payload, "preview_sha256")
            or len(payload.get("candidates", [])) != 8
        ):
            raise ReviewError(f"invalid pre-animation preview: {path}")
        candidates = payload["candidates"]
        ids = {item.get("candidate_id") for item in candidates}
        if len(ids) != len(candidates) or None in ids:
            raise ReviewError(f"duplicate candidate ids: {path}")
        payload["_path"] = str(path)
        previews[asset_id] = payload
    if not previews:
        raise ReviewError(f"no previews found under {root}")
    return previews


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>绑定前动作方向人工审核</title>
<style>
:root{color-scheme:dark;--bg:#080d14;--panel:#111927;--line:#2d3b50;--text:#e8eef8;--muted:#91a0b7;--blue:#54a7ff;--green:#39d98a;--yellow:#ffd166;--red:#ff6474}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Noto Sans SC",sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:#0c131e;position:sticky;top:0;z-index:2}h1{font-size:22px;margin:0 0 5px}.warn{color:var(--yellow)}
main{max-width:1500px;margin:auto;padding:20px;display:grid;grid-template-columns:260px 1fr;gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
#assets button,.choice{width:100%;text-align:left;margin:5px 0;padding:10px;border:1px solid var(--line);border-radius:8px;background:#172235;color:var(--text);cursor:pointer}.active{border-color:var(--blue)!important;background:#183a62!important}
.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}.controls button,.actions button{padding:11px;border:1px solid var(--line);border-radius:8px;background:#172235;color:var(--text);cursor:pointer}.toggle{display:grid;grid-template-columns:1fr 1fr;gap:8px}
canvas{display:block;width:100%;height:min(62vh,720px);background:#f8fafc;border-radius:10px;margin-top:12px}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);margin:8px 0}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}
.axis-contract{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:14px 0}.axis-card{display:flex;align-items:center;gap:12px;min-height:82px;padding:10px 14px;border:2px solid;border-radius:10px;background:#f8fafc;color:#122033}.axis-card .glyph{font:900 54px/1 system-ui}.axis-card b{display:block;font-size:18px}.axis-card small{display:block;font-size:13px;color:#43536b}.axis-card.forward{border-color:#08752d}.axis-card.forward .glyph,.axis-card.forward b{color:#08752d}.axis-card.up{border-color:#2563eb}.axis-card.up .glyph,.axis-card.up b{color:#2563eb}.axis-card.walk{border-color:#d59a00}.axis-card.walk .glyph,.axis-card.walk b{color:#9b6900}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.approve{background:#116b3b!important}.reject{background:#842c38!important}textarea{width:100%;min-height:70px;background:#0c131e;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:12px}.status{margin-top:10px;color:var(--muted)}
@media(max-width:850px){main{grid-template-columns:1fr}.controls{grid-template-columns:repeat(2,1fr)}.axis-contract{grid-template-columns:1fr}.axis-card{min-height:66px}.axis-card .glyph{font-size:42px}}
</style></head><body>
<header><h1>绑定前动作方向人工审核</h1><div class="warn">这里尚未生成目标蒙皮动画。请先确认源 Walk 的方向、摆腿平面和目标原生 3D 模型一致。</div></header>
<main><aside class="panel"><b>待审核资产</b><div id="assets"></div></aside>
<section class="panel"><h2 id="title">加载中</h2>
<div><b>判断规则：</b>绿色箭头是目标模型 +X 正方向。先让灰色原生模型的躯干/头沿绿色箭头朝右、身体沿蓝色箭头直立；再选择整 90° 动作 basis，让黄色源 Walk 箭头也朝右。不要用动作旋转补偿歪头或斜身。</div>
<div class="axis-contract" aria-label="Hunyuan style direction reference">
  <div class="axis-card forward"><span class="glyph">→</span><span><b>目标正方向 +X</b><small>动物躯干与头必须朝右</small></span></div>
  <div class="axis-card up"><span class="glyph">↑</span><span><b>世界向上 +Z</b><small>四脚动物必须保持直立</small></span></div>
  <div class="axis-card walk"><span class="glyph">→</span><span><b>源 Walk 方向</b><small>黄色箭头必须与绿色箭头同向</small></span></div>
</div>
<div class="legend"><span><i class="dot" style="background:#aeb8c7"></i>原生网格/rest</span><span><i class="dot" style="background:#54a7ff"></i>候选动作骨架</span><span><i class="dot" style="background:#ff6474"></i>脚端骨骼</span></div>
<h3>源动作 basis：只允许整90°</h3><div class="controls" id="yawControls"></div>
<h3>左右腿链映射</h3><div class="toggle"><button id="matched">同侧匹配</button><button id="swapped">交换左右腿链</button></div>
<canvas id="view"></canvas><div class="status" id="candidate"></div>
<textarea id="notes" placeholder="拒绝时请写明：方向侧着、前后腿摆动面错误、脚外翻等；批准可选填"></textarea>
<div class="actions"><button class="approve" id="approve">保存动作 basis，允许后续生成动画</button><button class="reject" id="reject">拒绝，禁止生成动画</button></div><div class="status" id="result"></div>
</section></main>
<script>
const S={assets:[],preview:null,asset:null,yaw:0,side:'matched',frame:0,last:0};
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
function fitCanvas(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0)}
function candidate(){return S.preview.candidates.find(x=>x.motion_basis_yaw_deg===S.yaw&&x.side_chain_mode===S.side)}
function selectButtons(){document.querySelectorAll('[data-yaw]').forEach(b=>b.classList.toggle('active',+b.dataset.yaw===S.yaw));document.getElementById('matched').classList.toggle('active',S.side==='matched');document.getElementById('swapped').classList.toggle('active',S.side==='swapped');const c=candidate();document.getElementById('candidate').textContent=`当前候选 ${c.candidate_id}；源 Walk forward = [${c.source_motion_forward.join(', ')}]；目标动画仍未生成`;}
function project(p,kind,box){const w=canvas.clientWidth,h=canvas.clientHeight,padX=62,padTop=66,padBottom=94,panelW=(w-30)/2,usableH=h-padTop-padBottom;let a,b,amin,amax,bmin,bmax;if(kind==='side'){a=p[0];b=p[2];amin=box.min[0];amax=box.max[0];bmin=box.min[2];bmax=box.max[2]}else{a=p[0];b=p[1];amin=box.min[0];amax=box.max[0];bmin=box.min[1];bmax=box.max[1]}const scale=Math.min((panelW-2*padX)/(amax-amin||1),usableH/(bmax-bmin||1));const ox=kind==='side'?0:panelW+30;return [ox+panelW/2+(a-(amin+amax)/2)*scale,padTop+usableH/2-(b-(bmin+bmax)/2)*scale]}
function line(seg,kind,box,color,width=1){const a=project(seg.slice(0,3),kind,box),b=project(seg.slice(3),kind,box);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function screenArrow(x1,y1,x2,y2,color,label,labelDy=-10,width=6,fontSize=16,head=18){const a=Math.atan2(y2-y1,x2-x1);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-head*Math.cos(a-Math.PI/6),y2-head*Math.sin(a-Math.PI/6));ctx.lineTo(x2-head*Math.cos(a+Math.PI/6),y2-head*Math.sin(a+Math.PI/6));ctx.closePath();ctx.fill();ctx.font=`900 ${fontSize}px system-ui`;ctx.fillText(label,Math.min(x1,x2)+4,Math.min(y1,y2)+labelDy);ctx.restore()}
function sourceArrow(vec,kind,box){const center=[(box.min[0]+box.max[0])/2,(box.min[1]+box.max[1])/2,(box.min[2]+box.max[2])/2],scale=box.max[0]-box.min[0],end=[center[0]+vec[0]*scale*.38,center[1]+vec[1]*scale*.38,center[2]],a=project(center,kind,box),b=project(end,kind,box),dx=b[0]-a[0],dy=b[1]-a[1];if(Math.hypot(dx,dy)<12){ctx.save();ctx.strokeStyle='#d59a00';ctx.fillStyle='#9b6900';ctx.lineWidth=4;ctx.beginPath();ctx.arc(a[0],a[1],12,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(a[0],a[1],4,0,Math.PI*2);ctx.fill();ctx.font='900 15px system-ui';ctx.fillText('源 Walk 垂直屏幕（不是 +X）',a[0]+18,a[1]-10);ctx.restore();return}screenArrow(a[0],a[1]+12,b[0],b[1]+12,'#d59a00','源 WALK',-10,6,16,18)}
function referenceAxes(kind,w,h){const panelW=(w-30)/2,ox=kind==='side'?0:panelW+30,forwardY=h-40;screenArrow(ox+panelW*.26,forwardY,ox+panelW*.78,forwardY,'#08752d','目标正方向  FORWARD +X  →',-12,8,18,22);if(kind==='side'){screenArrow(ox+38,h*.70,ox+38,h*.25,'#2563eb','↑  世界向上  UP +Z',-10,8,18,22)}ctx.save();ctx.fillStyle='#a12424';ctx.font='900 14px system-ui';ctx.fillText('← BACK / -X',ox+24,forwardY+24);ctx.restore()}
function draw(){if(!S.preview)return;fitCanvas();const w=canvas.clientWidth,h=canvas.clientHeight,panelW=(w-30)/2;ctx.clearRect(0,0,w,h);ctx.save();ctx.strokeStyle='#b9c4d2';ctx.lineWidth=2;ctx.strokeRect(1,1,panelW-2,h-2);ctx.strokeRect(panelW+30,1,panelW-2,h-2);ctx.fillStyle='#18202b';ctx.font='900 16px system-ui';ctx.fillText('侧视图 SIDE：模型直立，头/躯干朝右',18,28);ctx.fillText('俯视图 TOP-DOWN：绿色 +X 和黄色 Walk 都朝右',panelW+48,28);ctx.restore();const t=S.preview.target,box={min:t.bbox_min,max:t.bbox_max},c=candidate(),f=c.frames[S.frame%c.frames.length];for(const kind of ['side','top']){ctx.fillStyle='rgba(88,102,120,.17)';for(const p of t.mesh_points){const q=project(p,kind,box);ctx.fillRect(q[0],q[1],1.35,1.35)}t.target_rest_segments.forEach(s=>line(s,kind,box,'rgba(92,107,128,.55)',1.2));f.segments.forEach((s,i)=>line(s,kind,box,t.foot_leaves.includes(t.bone_order[i])?'#ff3e55':'#1678d2',2.4));referenceAxes(kind,w,h);sourceArrow(c.source_motion_forward,kind,box)}}
function tick(ts){if(S.preview&&ts-S.last>70){S.frame++;S.last=ts;draw()}requestAnimationFrame(tick)}
async function loadAsset(id){S.asset=id;S.preview=await (await fetch(`/api/preview/${encodeURIComponent(id)}`)).json();S.yaw=0;S.side='matched';S.frame=0;document.getElementById('title').textContent=id;document.querySelectorAll('[data-asset]').forEach(b=>b.classList.toggle('active',b.dataset.asset===id));selectButtons();draw()}
async function decide(status){const c=candidate(),notes=document.getElementById('notes').value;const res=await fetch(`/api/decision/${encodeURIComponent(S.asset)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,candidate_id:c.candidate_id,notes})});const data=await res.json();document.getElementById('result').textContent=res.ok?`已保存：${data.status}，decision=${data.decision_sha256}`:`保存失败：${data.error}`}
async function init(){S.assets=await (await fetch('/api/assets')).json();const list=document.getElementById('assets');S.assets.forEach(x=>{const b=document.createElement('button');b.textContent=x.asset_id;b.dataset.asset=x.asset_id;b.onclick=()=>loadAsset(x.asset_id);list.appendChild(b)});const yc=document.getElementById('yawControls');[-90,0,90,180].forEach(y=>{const b=document.createElement('button');b.textContent=(y>0?'+':'')+y+'°';b.dataset.yaw=y;b.onclick=()=>{S.yaw=y;selectButtons();draw()};yc.appendChild(b)});document.getElementById('matched').onclick=()=>{S.side='matched';selectButtons();draw()};document.getElementById('swapped').onclick=()=>{S.side='swapped';selectButtons();draw()};document.getElementById('approve').onclick=()=>decide('motion_basis_approved');document.getElementById('reject').onclick=()=>decide('motion_basis_rejected');window.onresize=draw;if(S.assets.length)await loadAsset(S.assets[0].asset_id);requestAnimationFrame(tick)}init();
</script></body></html>'''


def create_app(preview_root: Path, state_root: Path) -> Flask:
    previews = load_previews(preview_root)
    state_root = state_root.absolute()
    state_root.mkdir(parents=True, exist_ok=True)
    decisions_root = state_root / "decisions"
    decisions_root.mkdir(exist_ok=True)
    app = Flask(__name__)

    @app.after_request
    def disable_review_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.errorhandler(ReviewError)
    def handle_review_error(error):
        return jsonify({"error": str(error)}), 400

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/assets")
    def assets():
        return jsonify(
            [
                {
                    "asset_id": asset_id,
                    "preview_sha256": preview["preview_sha256"],
                    "target_animation_generated": False,
                }
                for asset_id, preview in sorted(previews.items())
            ]
        )

    @app.get("/api/preview/<asset_id>")
    def preview(asset_id: str):
        if asset_id not in previews:
            raise ReviewError("unknown asset")
        payload = {name: value for name, value in previews[asset_id].items() if name != "_path"}
        return jsonify(payload)

    @app.post("/api/decision/<asset_id>")
    def decision(asset_id: str):
        if asset_id not in previews:
            raise ReviewError("unknown asset")
        payload = request.get_json(silent=True) or {}
        status = payload.get("status")
        if status not in {"motion_basis_approved", "motion_basis_rejected"}:
            raise ReviewError("invalid decision status")
        notes = str(payload.get("notes", "")).strip()
        if status == "motion_basis_rejected" and not notes:
            raise ReviewError("rejection notes are required")
        preview_payload = previews[asset_id]
        candidates = {
            item["candidate_id"]: item for item in preview_payload["candidates"]
        }
        candidate_id = payload.get("candidate_id")
        if candidate_id not in candidates:
            raise ReviewError("unknown motion-basis candidate")
        candidate = candidates[candidate_id]
        approved = status == "motion_basis_approved"
        record = {
            "schema": DECISION_SCHEMA,
            "asset_id": asset_id,
            "status": status,
            "human_approved": approved,
            "human_approved_by": getpass.getuser(),
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "preview_sha256": preview_payload["preview_sha256"],
            "preview_path": preview_payload["_path"],
            "target": {
                name: preview_payload["target"][name]
                for name in ("path", "sha256", "size_bytes", "reviewed_front_axis")
            },
            "source_motion": {
                name: preview_payload["source_motion"][name]
                for name in ("path", "sha256", "size_bytes", "action")
            },
            "candidate_id": candidate_id,
            "manual_cardinal_motion_basis_yaw_deg": candidate[
                "motion_basis_yaw_deg"
            ],
            "side_chain_mode": candidate["side_chain_mode"],
            "rotation_transfer_mode": candidate["rotation_transfer_mode"],
            "target_animation_generation_authorized": approved,
            "formal_dataset_registration_authorized": False,
        }
        record["decision_sha256"] = hash_without(record, "decision_sha256")
        try:
            write_json_exclusive(decisions_root / f"{asset_id}.json", record)
        except FileExistsError as error:
            raise ReviewError("this asset already has an immutable decision") from error
        return jsonify(record)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(args.preview_root, args.state_root)
    print(
        "GENERATED_ANIMAL_MOTION_BASIS_REVIEW_SERVER_OK "
        f"url=http://{args.host}:{args.port}/",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
