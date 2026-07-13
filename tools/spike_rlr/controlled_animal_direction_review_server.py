#!/usr/bin/env python3
"""Interactive, non-destructive controlled-animal direction review server.

This is the Hunyuan-style manual correction gate for the controlled Pixal
animals.  It renders a disposable 100k pre-bind preview after the pipeline's
existing X mirror, lets a reviewer adjust only the remaining yaw, and writes a
small transform decision.  It never rewrites a GLB, registry, historical
decision, or historical video.
"""

from __future__ import annotations

import argparse
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
    "controlled_animal_direction_revalidation_v1_20260713/review_manifest.json"
)
DEFAULT_STATE_ROOT = (
    SPEAR_ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "controlled_animal_direction_review_state_v1_20260713"
)
MANIFEST_SCHEMA = "controlled_animal_direction_revalidation_manifest_v1"
STATE_SCHEMA = "controlled_animal_direction_review_state_v1"
DECISION_SCHEMA = "controlled_animal_geometry_orientation_decision_v1"
ALLOWED_DELTAS = {-90.0, -15.0, -5.0, 5.0, 15.0, 90.0, 180.0}


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


def _combined_preview_matrix(yaw_deg: float) -> np.ndarray:
    mirror_x = np.diag([-1.0, 1.0, 1.0])
    return _yaw_matrix_y_up(yaw_deg) @ mirror_x


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


def _render_orientation_preview(
    mesh: trimesh.Trimesh,
    destination: Path,
    *,
    yaw_deg: float,
    max_points: int = 45_000,
) -> None:
    """Render a fast side/top silhouette without exporting a scratch mesh.

    The previous Hunyuan review renderer accepted a mesh filename, which made
    every button click serialize a 100k-face GLB before drawing it.  That is
    unnecessarily slow for a yaw-only gate.  Here we transform the certified
    vertices in memory and plot a deterministic point silhouette; the PBR
    contact sheet remains visible next to it for appearance/identity review.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices @ _combined_preview_matrix(yaw_deg).T
    if len(vertices) > max_points:
        indices = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
        vertices = vertices[indices]

    bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
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
        (top, 2, "TOP-DOWN: head must point right", "+Z SIDE"),
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
            "HEAD / +X",
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
            "TAIL / BACK",
            color="#a12424",
            fontweight="bold",
            ha="left",
            va="top",
        )

    side.axhline(bounds[0, 1], color="#6b7280", linestyle=":", linewidth=1.2)
    fig.suptitle(
        f"existing mirror-X + reviewer yaw {yaw_deg:+.1f} deg\n"
        "Adjust yaw until the visible head points to the green +X arrow",
        fontsize=13,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = path.resolve()
    manifest = _read_json(path)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
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
            or entry.get("species") not in {"cat", "dog"}
            or entry.get("current_evidence_status", {}).get("walking_direction")
            != "rejected_by_user_visual_review"
        ):
            raise ReviewServerError("direction review entry identity/status is invalid")
        for name in (
            "prebind_lod_glb",
            "static_contact_sheet",
            "walking_side",
            "walking_front",
            "idle_side",
            "apartment_walking_review",
        ):
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
        "prebind_lod_glb",
        "static_contact_sheet",
        "current_bound_glb",
        "walking_side",
        "walking_front",
        "idle_side",
        "apartment_walking_review",
        "apartment_walking_main",
        "apartment_walking_topdown",
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
        },
        "current_evidence_status": entry["current_evidence_status"],
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>猫狗方向人工纠正</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#0b0e14;color:#e7eaf0;height:100vh;overflow:hidden}button,input,select,textarea{font:inherit}
.app{display:grid;grid-template-columns:340px minmax(0,1fr);height:100vh}aside{background:#111620;border-right:1px solid #273043;display:flex;flex-direction:column;min-height:0}
header{padding:16px;border-bottom:1px solid #273043}h1{font-size:19px;margin:0 0 7px}.summary,.muted{font-size:12px;color:#9aa7ba}.links{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
.link,.btn{color:#e7eaf0;background:#182131;border:1px solid #344157;border-radius:8px;padding:7px 10px;text-decoration:none;cursor:pointer}.link:hover,.btn:hover{background:#24466e}
.filters{display:grid;gap:8px;padding:10px;border-bottom:1px solid #273043}input,select,textarea{width:100%;color:#e7eaf0;background:#171e2b;border:1px solid #303b50;border-radius:8px;padding:8px}
#list{overflow:auto;padding:7px}.item{width:100%;text-align:left;color:inherit;background:transparent;border:1px solid transparent;border-radius:9px;padding:9px;cursor:pointer}.item:hover{background:#171e2b}.item.active{background:#1b2940;border-color:#3c6ea8}.item-title{font-size:12px;font-weight:650;overflow-wrap:anywhere}.item-meta{font-size:11px;color:#9aa7ba;margin-top:3px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:#f59e0b}.dot.ok{background:#22c55e}.dot.bad{background:#ef4444}
main{overflow:auto;padding:20px}.stage{max-width:1450px;margin:auto}.danger{background:#441b22;border:1px solid #9f3446;border-radius:10px;padding:12px;margin-bottom:14px;color:#ffd8df}.title-row{display:flex;justify-content:space-between;gap:12px}h2{font-size:20px;margin:0;overflow-wrap:anywhere}.pills{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.pill{font-size:12px;border:1px solid #334155;background:#172131;border-radius:999px;padding:4px 8px}
.grid{display:grid;grid-template-columns:minmax(400px,1.2fr) minmax(330px,1fr);gap:14px}.card{background:#111620;border:1px solid #273043;border-radius:12px;padding:14px}h3{font-size:15px;margin:0 0 8px}.instructions{font-size:13px;color:#ccd6e5;line-height:1.55;margin-bottom:10px}.preview-wrap{background:#fff;border-radius:8px;min-height:380px;display:flex;align-items:center;justify-content:center;overflow:hidden}.preview-wrap img{width:100%;max-height:590px;object-fit:contain}.static img{width:100%;border-radius:8px;background:#05070a}
.controls{display:grid;grid-template-columns:repeat(7,minmax(44px,1fr));gap:6px;margin-top:9px}.controls button{padding:8px 3px}.yaw{font-family:ui-monospace,monospace;color:#7dd3fc;text-align:center;margin:8px 0}.decision{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.approve{background:#126137;border-color:#1f9d57}.reject{background:#7d2631;border-color:#bd4353}
.video-card{margin-top:14px}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.tabs button.active{background:#24466e;border-color:#4e83bd}video{width:100%;max-height:560px;background:#000;border-radius:9px}.path{font-size:11px;color:#b8c6dc;white-space:nowrap;overflow:auto;margin-top:7px;padding:7px;background:#0c111a;border-radius:6px}
.note{margin-top:8px}.nav{display:flex;gap:6px}.status-line{font-size:12px;color:#fbbf24;margin-top:5px}
@media(max-width:950px){body{height:auto;overflow:auto}.app{display:block;height:auto}aside{height:42vh}.grid{grid-template-columns:1fr}main{padding:12px}.controls{grid-template-columns:repeat(4,1fr)}}
</style></head><body>
<script id="assets" type="application/json">{{assets_json|safe}}</script>
<div class="app"><aside><header><h1>猫狗方向人工纠正</h1><div class="summary">{{asset_count}} 个资产 · 当前 Walking 全部已撤销方向通过状态</div><div class="links"><a class="link" href="/docs/controlled_animal_video_review.html">动物成片</a><a class="link" href="/docs/rocketbox_human_video_review.html">人类视频</a></div></header>
<div class="filters"><input id="search" type="search" placeholder="搜索 asset / breed / 属性"><select id="species"><option value="">猫 + 狗</option><option value="cat">猫</option><option value="dog">狗</option></select></div><div id="list"></div></aside>
<main><div class="stage"><div class="danger"><b>旧结果已拒绝：</b>现有 Walking 的“自动方向通过”仅来自数字骨骼，不能证明可见头尾及移动方向正确。这里的操作只保存下一次绑定要用的方向参数；不会覆盖旧 GLB 或视频。</div>
<div class="title-row"><div><h2 id="title"></h2><div id="sub" class="muted"></div><div id="status" class="status-line"></div></div><div class="nav"><button class="btn" id="prev">←</button><button class="btn" id="next">→</button></div></div><div id="pills" class="pills"></div>
<div class="grid"><section class="card"><h3>第一道门：绑定前可见方向（实时调整）</h3><div class="instructions">系统先重放现有 <code>mirror X</code>，再应用下面的 yaw。请让动物<strong style="color:#4ade80">头朝绿色 +X 箭头</strong>、背部朝上。这里只调整绕 UP 的角度，不动源 PBR 模型。</div><div class="preview-wrap"><img id="preview" alt="方向预览"></div><div id="yaw" class="yaw"></div><div class="controls"><button class="btn rot" data-d="-90">↺90</button><button class="btn rot" data-d="-15">−15</button><button class="btn rot" data-d="-5">−5</button><button class="btn" id="reset">重置</button><button class="btn rot" data-d="5">+5</button><button class="btn rot" data-d="15">+15</button><button class="btn rot" data-d="90">↻90</button></div><div class="controls" style="grid-template-columns:1fr"><button class="btn rot" data-d="180">⇄ 头尾翻转 180°</button></div><textarea id="notes" class="note" rows="2" placeholder="可选备注，例如：头仍偏左约 10°"></textarea><div class="decision"><button class="btn approve" id="approve">保存方向（仅批准静态朝向）</button><button class="btn reject" id="reject">拒绝该绑定方向</button></div></section>
<section class="card static"><h3>原始 PBR / 多视图证据</h3><div class="instructions">这张图不随按钮变化，用来确认当前审核的是同一只动物及其原始外观。右侧实时预览来自认证的 100k 绑定前 LOD。</div><img id="contact" alt="static contact sheet"><div id="rawpath" class="path"></div></section></div>
<section class="card video-card"><h3>第二道门：绑定后动作与真实移动方向</h3><div class="instructions">以下都是已被你判定方向有问题的旧证据，仅用于定位问题；静态方向保存后仍必须重新绑定并生成直线 + 转弯 canary，不能直接恢复通过状态。</div><div class="tabs" id="tabs"></div><video id="video" controls preload="metadata"></video><div id="videopath" class="path"></div></section>
</div></main></div>
<script>
const all=JSON.parse(document.getElementById('assets').textContent);let filtered=[...all],idx=0,state={};const $=x=>document.getElementById(x);
async function loadState(){state=await (await fetch('/api/state')).json();render()}
function current(){return filtered[idx]}
function apply(){const q=$('search').value.toLowerCase(),s=$('species').value;filtered=all.filter(a=>(!s||a.species===s)&&(!q||(`${a.asset_id} ${a.breed} ${JSON.stringify(a.sampled_attributes)}`).toLowerCase().includes(q)));idx=0;render()}
function itemStatus(id){return state[id]?.decision?.status||'pending'}
function renderList(){const list=$('list');list.replaceChildren();filtered.forEach((a,i)=>{const b=document.createElement('button'),st=itemStatus(a.asset_id);b.className='item'+(i===idx?' active':'');b.innerHTML=`<div class="item-title"><span class="dot ${st.includes('approved')?'ok':st.includes('rejected')?'bad':''}"></span></div><div class="item-meta"></div>`;b.querySelector('.item-title').append(a.asset_id);b.querySelector('.item-meta').textContent=`${a.species} · ${a.breed} · ${st}`;b.onclick=()=>{idx=i;render()};list.append(b)})}
let view='apartment_walking_review';const labels={walking_side:'绑定后 Walk 侧面',walking_front:'绑定后 Walk 正面',idle_side:'绑定后 Idle 侧面',apartment_walking_review:'UE + Top-down 成片',apartment_walking_main:'UE 主视图',apartment_walking_topdown:'Top-down'};
function render(){renderList();const a=current();if(!a)return;const st=state[a.asset_id]||{yaw_deg:0,revision:0};$('title').textContent=a.asset_id;$('sub').textContent=`${a.species} · ${a.breed} · ${a.profile_schema_id}`;$('status').textContent=`当前 Walking：已拒绝；静态方向：${st.decision?.status||'待审核'}`;$('pills').replaceChildren(...Object.entries(a.sampled_attributes).map(([k,v])=>{const x=document.createElement('span');x.className='pill';x.textContent=`${k}=${v}`;return x}));$('yaw').textContent=`post-mirror yaw = ${st.yaw_deg||0}°`;$('preview').src=`/preview/${encodeURIComponent(a.asset_id)}.png?r=${st.revision||0}`;$('contact').src=a.artifacts.static_contact_sheet.url;$('rawpath').textContent=a.artifacts.pixal_raw_glb.absolute_path;renderVideo();const locked=Boolean(st.decision);document.querySelectorAll('.rot,#reset,#approve,#reject').forEach(x=>x.disabled=locked)}
function renderVideo(){const a=current();const tabs=$('tabs');tabs.replaceChildren(...Object.keys(labels).map(k=>{const b=document.createElement('button');b.className='btn'+(view===k?' active':'');b.textContent=labels[k];b.onclick=()=>{view=k;renderVideo()};return b}));const m=a.artifacts[view];$('video').src=m.url;$('videopath').textContent=m.absolute_path}
async function post(url,payload={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const j=await r.json();if(!r.ok){alert(j.error||'操作失败');throw Error(j.error)}state[j.asset_id]=j;render()}
document.querySelectorAll('.rot').forEach(b=>b.onclick=()=>post(`/api/rotate/${encodeURIComponent(current().asset_id)}`,{delta_deg:Number(b.dataset.d)}));$('reset').onclick=()=>post(`/api/reset/${encodeURIComponent(current().asset_id)}`);$('approve').onclick=()=>{if(confirm('只保存绑定前朝向参数；旧 Walking 仍保持 rejected。确认？'))post(`/api/decision/${encodeURIComponent(current().asset_id)}`,{status:'geometry_orientation_approved',notes:$('notes').value})};$('reject').onclick=()=>{if(confirm('确认拒绝该绑定前方向？'))post(`/api/decision/${encodeURIComponent(current().asset_id)}`,{status:'geometry_orientation_rejected',notes:$('notes').value||'visible orientation rejected'})};
$('prev').onclick=()=>{if(filtered.length){idx=(idx-1+filtered.length)%filtered.length;render()}};$('next').onclick=()=>{if(filtered.length){idx=(idx+1)%filtered.length;render()}};$('search').oninput=apply;$('species').oninput=apply;window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};loadState();
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
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / "previews").mkdir(exist_ok=True)
    (state_root / "states").mkdir(exist_ok=True)
    (state_root / "decisions").mkdir(exist_ok=True)
    locks = {asset_id: threading.Lock() for asset_id in entries}

    app = Flask(__name__)
    app.config.update(
        DIRECTION_MANIFEST=manifest,
        DIRECTION_ENTRIES=entries,
        DIRECTION_STATE_ROOT=state_root,
        DIRECTION_HOST=host,
        DIRECTION_PORT=port,
    )

    def state_path(asset_id: str) -> Path:
        return state_root / "states" / f"{asset_id}.json"

    def decision_path(asset_id: str) -> Path:
        return state_root / "decisions" / f"{asset_id}.json"

    def read_state(asset_id: str) -> dict[str, Any]:
        path = state_path(asset_id)
        if not path.exists():
            base = {
                "schema": STATE_SCHEMA,
                "asset_id": asset_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "yaw_deg": 0.0,
                "history": [],
                "revision": 0,
            }
        else:
            base = _read_json(path)
            if (
                base.get("schema") != STATE_SCHEMA
                or base.get("asset_id") != asset_id
                or base.get("manifest_sha256") != manifest["manifest_sha256"]
                or not isinstance(base.get("history"), list)
            ):
                raise ReviewServerError(f"invalid review state: {asset_id}")
        decision_file = decision_path(asset_id)
        if decision_file.exists():
            decision = _read_json(decision_file)
            if (
                decision.get("schema") != DECISION_SCHEMA
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
        return state_root / "previews" / asset_id / f"yaw_{token}.png"

    def render_preview(asset_id: str, yaw_deg: float) -> Path:
        destination = preview_path(asset_id, yaw_deg)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        entry = entries[asset_id]
        source = Path(entry["artifacts"]["prebind_lod_glb"]["absolute_path"])
        mesh = _load_preview_mesh(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _render_orientation_preview(mesh, destination, yaw_deg=yaw_deg)
        return destination

    @app.errorhandler(ReviewServerError)
    def handle_review_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": str(error)}), 409
        return str(error), 409

    @app.get("/")
    def index():
        public = [_public_entry(entry) for entry in entries.values()]
        return render_template_string(
            HTML,
            asset_count=len(public),
            assets_json=json.dumps(public, ensure_ascii=False).replace("</", "<\\/"),
        )

    @app.get("/api/state")
    def api_state():
        return jsonify({asset_id: read_state(asset_id) for asset_id in entries})

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
        try:
            delta = float(payload.get("delta_deg"))
        except (TypeError, ValueError) as error:
            raise ReviewServerError("delta_deg must be numeric") from error
        if delta not in ALLOWED_DELTAS:
            raise ReviewServerError(f"unsupported yaw delta: {delta}")
        with locks[asset_id]:
            if decision_path(asset_id).exists():
                raise ReviewServerError("direction decision is already immutable")
            state = read_state(asset_id)
            state["yaw_deg"] = _normalize_yaw(float(state["yaw_deg"]) + delta)
            state["history"].append({"operation": "yaw_delta_deg", "value": delta})
            state["revision"] = int(state.get("revision", 0)) + 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_state(asset_id, state)
            render_preview(asset_id, float(state["yaw_deg"]))
        return jsonify(state)

    @app.post("/api/reset/<asset_id>")
    def reset(asset_id: str):
        require_asset(asset_id)
        with locks[asset_id]:
            if decision_path(asset_id).exists():
                raise ReviewServerError("direction decision is already immutable")
            previous = read_state(asset_id)
            state = {
                "schema": STATE_SCHEMA,
                "asset_id": asset_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "yaw_deg": 0.0,
                "history": [],
                "revision": int(previous.get("revision", 0)) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_state(asset_id, state)
            render_preview(asset_id, 0.0)
        return jsonify(state)

    @app.post("/api/decision/<asset_id>")
    def decide(asset_id: str):
        entry = require_asset(asset_id)
        payload = request.get_json(silent=True) or {}
        status = payload.get("status")
        if status not in {
            "geometry_orientation_approved",
            "geometry_orientation_rejected",
        }:
            raise ReviewServerError("invalid geometry orientation decision")
        notes = str(payload.get("notes", "")).strip()
        if status.endswith("rejected") and not notes:
            raise ReviewServerError("a rejection note is required")
        with locks[asset_id]:
            state = read_state(asset_id)
            yaw_deg = float(state["yaw_deg"])
            matrix = _combined_preview_matrix(yaw_deg)
            decision: dict[str, Any] = {
                "schema": DECISION_SCHEMA,
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
                "existing_binding_pretransform": "mirror_x",
                "post_mirror_yaw_about_gltf_positive_y_deg": yaw_deg,
                "combined_preview_matrix_3x3": matrix.tolist(),
                "determinant": float(np.linalg.det(matrix)),
                "downstream_candidate": {
                    "flip_x": True,
                    "target_rotate_z_deg_after_flip_x": yaw_deg,
                    "coordinate_mapping_status": "requires_straight_line_ue_canary",
                },
                "current_walking_media_status": "rejected_by_user_visual_review",
                "next_gate": (
                    "regenerate_binding_then_straight_line_and_curve_dynamic_direction_canary"
                    if status.endswith("approved")
                    else "stop_or_rework_orientation"
                ),
                "formal_dataset_registration_authorized": False,
                "source_assets_modified": False,
                "history": copy.deepcopy(state["history"]),
            }
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
