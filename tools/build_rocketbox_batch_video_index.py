"""Build stable Markdown and browser video indexes for native Rocketbox."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_MANIFEST = (
    SPEAR_ROOT
    / "tmp/rocketbox_batch_apartment_review_v1/batch_spec_manifest.json"
)
DEFAULT_DOCUMENT = (
    AVENGINE_ROOT / "docs/rocketbox_batch_apartment_video_index.md"
)
DEFAULT_HTML = AVENGINE_ROOT / "docs/rocketbox_human_video_review.html"
DEFAULT_FEATURED_MANIFEST = (
    SPEAR_ROOT
    / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/"
    "representative_spec_manifest.json"
)

VIDEO_NAMES = {
    "审核": "side_by_side_review_annotated.mp4",
    "主视图": "apartment_v1_view0.mp4",
    "Top-down": "topdown_review.mp4",
}
CATEGORY_ORDER = {"Adults": 0, "Children": 1, "Professions": 2}
GENDER_ORDER = {"female": 0, "male": 1}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o644)


def _absolute_link(target: Path) -> str:
    return str(Path(target).resolve())


def _server_url(target: Path) -> str:
    target = Path(target).resolve()
    try:
        relative = target.relative_to(AVENGINE_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"video escaped AVEngine server root: {target}") from error
    return "/" + relative.as_posix()


def _video_paths(output_dir: Path) -> dict[str, Path]:
    videos_dir = Path(output_dir).resolve() / "videos"
    return {label: videos_dir / name for label, name in VIDEO_NAMES.items()}


def _video_cell(output_dir: Path) -> tuple[str, bool]:
    paths = _video_paths(output_dir)
    complete = all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    if not complete:
        return "⏳ 待生成", False
    links = [
        f"[{label}]({_absolute_link(path)})"
        for label, path in paths.items()
    ]
    return "✅ " + " · ".join(links), True


def _browser_media(output_dir: Path) -> dict[str, dict[str, str]]:
    paths = _video_paths(output_dir)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return {}
    return {
        label: {
            "absolute_path": str(path.resolve()),
            "url": _server_url(path),
        }
        for label, path in paths.items()
    }


def _featured_entries(path: Path | None) -> list[dict]:
    if path is None:
        return []
    path = Path(path).resolve()
    manifest = _read_json(path)
    records = manifest.get("records")
    if (
        manifest.get("schema") != "rocketbox_camera_pass_table_loop_specs_v2"
        or not isinstance(records, list)
        or manifest.get("avatar_count") != len(records)
        or manifest.get("clip_count") != len(records)
    ):
        raise RuntimeError("featured Rocketbox trajectory manifest is invalid")
    result = []
    for record in records:
        walking = record.get("actions", {}).get("Walking", {})
        media = _browser_media(Path(str(walking.get("output_dir", ""))))
        if not media:
            raise RuntimeError(f"featured Rocketbox media incomplete: {record.get('tag')}")
        role = str(record.get("role_label", "unknown"))
        gender = "female" if "female" in role else "male"
        category = "Children" if "child" in role else "Professions" if "nurse" in role else "Adults"
        result.append(
            {
                "review_id": f"featured:{walking.get('clip_id')}",
                "source_set": "最新 6 角色圆桌轨迹",
                "featured": True,
                "avatar_id": str(record.get("tag")),
                "base_avatar_id": str(record.get("base_avatar_id")),
                "role_label": role,
                "category": category,
                "gender": gender,
                "height_cm": float(record.get("authored_height_cm")),
                "action": "Walking",
                "trajectory": "camera_pass_table_loop",
                "media": media,
                "speech": record.get("speech"),
            }
        )
    return result


HUMAN_REVIEW_HTML = r"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rocketbox 人类视频审核</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#0b0e14;color:#e8edf5;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:350px minmax(0,1fr);height:100vh}aside{display:flex;flex-direction:column;min-height:0;background:#111722;border-right:1px solid #283248}header,.filters{padding:14px;border-bottom:1px solid #283248}h1{font-size:19px;margin:0 0 6px}.muted{font-size:12px;color:#9eacc1}.links{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.link,.btn{color:#e8edf5;text-decoration:none;background:#192335;border:1px solid #34435d;border-radius:8px;padding:7px 10px;cursor:pointer}.link:hover,.btn:hover{background:#254b78}.filters{display:grid;grid-template-columns:1fr 1fr;gap:7px}.filters input{grid-column:1/-1}input,select{width:100%;background:#151d2b;border:1px solid #344057;border-radius:8px;padding:8px;color:#e8edf5}#list{overflow:auto;padding:7px}.item{width:100%;text-align:left;color:inherit;background:transparent;border:1px solid transparent;border-radius:9px;padding:9px;cursor:pointer}.item:hover{background:#171f2d}.item.active{background:#1b2a42;border-color:#4778af}.item.featured{border-left:3px solid #f4b942}.it{font-size:12px;font-weight:650;overflow-wrap:anywhere}.im{font-size:11px;color:#9eacc1;margin-top:3px}main{overflow:auto;padding:20px}.stage{max-width:1400px;margin:auto}.banner{padding:12px;background:#142a20;border:1px solid #286348;border-radius:10px;margin-bottom:14px}.title{display:flex;justify-content:space-between;gap:10px}h2{font-size:21px;margin:0;overflow-wrap:anywhere}.pills{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.pill{font-size:12px;background:#172234;border:1px solid #35445c;border-radius:999px;padding:4px 8px}.card{background:#111722;border:1px solid #283248;border-radius:12px;padding:14px}.tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:9px}.tabs .active{background:#27588d;border-color:#5991ca}video{display:block;width:100%;max-height:690px;background:#000;border-radius:9px}.path{font:11px ui-monospace,monospace;color:#bac7da;white-space:nowrap;overflow:auto;margin-top:9px;padding:8px;background:#0b1019;border-radius:7px}.copy{margin-top:7px}.nav{display:flex;gap:7px}.speech{font-size:12px;color:#c9d5e6;line-height:1.5;margin-top:10px}.empty{padding:20px;color:#fbbf24}@media(max-width:900px){body{height:auto;overflow:auto}.app{display:block;height:auto}aside{height:43vh}main{padding:12px}.filters{grid-template-columns:1fr 1fr}}
</style></head><body><script id="dataset" type="application/json">__DATA__</script>
<div class="app"><aside><header><h1>Rocketbox 人类视频审核</h1><div class="muted">__SUMMARY__</div><div class="links"><a class="link" href="/">猫狗方向纠正</a><a class="link" href="/docs/controlled_animal_video_review.html">动物成片</a><a class="link" href="/docs/rocketbox_video_catalog.md">人类目录文档</a></div></header><div class="filters"><input id="search" type="search" placeholder="搜索角色 / 职业 / 动作"><select id="source"><option value="">全部批次</option><option value="featured">最新 6 角色</option><option value="batch">115 角色基线</option></select><select id="action"><option value="">Walk + Idle</option><option>Walking</option><option>Standing_Idle</option></select><select id="category"><option value="">全部类别</option><option>Adults</option><option>Children</option><option>Professions</option></select><select id="gender"><option value="">男女</option><option value="female">女</option><option value="male">男</option></select></div><div id="list"></div></aside>
<main><div class="stage"><div class="banner"><b>可直接审核：</b>每个条目都只播放已存在且非空的 MP4；下方同时显示服务器绝对路径。最新 6 个代表角色排在最前，随后是 115 个 Rocketbox 的 Walking / Standing Idle。</div><div class="title"><div><h2 id="title"></h2><div id="sub" class="muted"></div></div><div class="nav"><button class="btn" id="prev">←</button><button class="btn" id="next">→</button></div></div><div id="pills" class="pills"></div><section class="card"><div id="tabs" class="tabs"></div><video id="video" controls preload="metadata"></video><div id="path" class="path"></div><button class="btn copy" id="copy">复制绝对路径</button><div id="speech" class="speech"></div></section><div id="empty" class="empty" hidden>当前筛选没有已完成视频。</div></div></main></div>
<script>const all=JSON.parse(document.getElementById('dataset').textContent);let rows=[...all],i=0,view='审核';const $=x=>document.getElementById(x);function cur(){return rows[i]}function apply(){const q=$('search').value.toLowerCase(),s=$('source').value,a=$('action').value,c=$('category').value,g=$('gender').value;rows=all.filter(x=>(!q||(`${x.avatar_id} ${x.base_avatar_id} ${x.role_label} ${x.category} ${x.action}`).toLowerCase().includes(q))&&(!s||(s==='featured')===x.featured)&&(!a||x.action===a)&&(!c||x.category===c)&&(!g||x.gender===g));i=0;render()}function list(){const el=$('list');el.replaceChildren();rows.forEach((x,n)=>{const b=document.createElement('button');b.className='item'+(x.featured?' featured':'')+(n===i?' active':'');b.innerHTML='<div class="it"></div><div class="im"></div>';b.querySelector('.it').textContent=x.avatar_id;b.querySelector('.im').textContent=`${x.action} · ${x.category} · ${x.gender} · ${x.height_cm.toFixed(1)} cm`;b.onclick=()=>{i=n;render()};el.append(b)})}function renderVideo(){const x=cur();if(!x)return;const keys=Object.keys(x.media);if(!keys.includes(view))view=keys[0];$('tabs').replaceChildren(...keys.map(k=>{const b=document.createElement('button');b.className='btn'+(view===k?' active':'');b.textContent=k;b.onclick=()=>{view=k;renderVideo()};return b}));const m=x.media[view];$('video').src=m.url;$('path').textContent=m.absolute_path}function render(){list();const x=cur(),empty=!x;$('empty').hidden=!empty;$('video').closest('.card').hidden=empty;if(empty){$('title').textContent='没有匹配结果';$('sub').textContent='';$('pills').replaceChildren();return}$('title').textContent=x.avatar_id;$('sub').textContent=`${x.source_set} · ${x.action}`;$('pills').replaceChildren(...[['类别',x.category],['性别',x.gender],['身高',x.height_cm.toFixed(2)+' cm'],['轨迹',x.trajectory]].map(([k,v])=>{const e=document.createElement('span');e.className='pill';e.textContent=`${k}: ${v}`;return e}));$('speech').textContent=x.speech?.transcript?`语音：${x.speech.transcript}`:'';renderVideo()}$('copy').onclick=async()=>{await navigator.clipboard.writeText($('path').textContent);$('copy').textContent='已复制';setTimeout(()=>$('copy').textContent='复制绝对路径',1000)};$('prev').onclick=()=>{if(rows.length){i=(i-1+rows.length)%rows.length;render()}};$('next').onclick=()=>{if(rows.length){i=(i+1)%rows.length;render()}};['search','source','action','category','gender'].forEach(k=>$(k).oninput=apply);window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};render();</script></body></html>"""


def _write_browser_index(path: Path, entries: list[dict], *, avatar_count: int) -> None:
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    featured = sum(bool(item.get("featured")) for item in entries)
    batch = len(entries) - featured
    summary = f"{featured} 个推荐轨迹 + {batch} 个完整基线动作 · {avatar_count} 个原始角色"
    text = HUMAN_REVIEW_HTML.replace("__DATA__", payload).replace("__SUMMARY__", summary)
    _atomic_text(path, text)


def build_video_index(
    manifest_path: Path,
    document_path: Path,
    *,
    html_path: Path | None = None,
    featured_manifest_path: Path | None = None,
) -> dict:
    """Write one row per avatar and canonical links for both required actions."""
    manifest_path = Path(manifest_path).resolve()
    document_path = Path(document_path).resolve()
    manifest = _read_json(manifest_path)
    records = manifest.get("records")
    if (
        manifest.get("schema") != "rocketbox_batch_apartment_specs_v1"
        or not isinstance(records, list)
        or manifest.get("avatar_count") != len(records)
        or manifest.get("clip_count") != len(records) * 2
    ):
        raise RuntimeError("Rocketbox Apartment spec manifest is invalid")

    inventory_path = Path(manifest.get("inventory", "")).resolve()
    inventory = _read_json(inventory_path)
    avatars = inventory.get("avatars")
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or not isinstance(avatars, list)
    ):
        raise RuntimeError("Rocketbox inventory is invalid")
    inventory_by_id = {item.get("base_avatar_id"): item for item in avatars}
    record_by_id = {item.get("base_avatar_id"): item for item in records}
    if (
        None in inventory_by_id
        or None in record_by_id
        or len(inventory_by_id) != len(avatars)
        or len(record_by_id) != len(records)
        or set(inventory_by_id) != set(record_by_id)
    ):
        raise RuntimeError("Rocketbox manifest/inventory identity set changed")

    completed_clip_count = 0
    complete_pair_count = 0
    grouped: dict[tuple[str, str], list[tuple[dict, dict, str, str]]] = {}
    browser_entries: list[dict] = []
    for avatar_id in sorted(record_by_id):
        avatar = inventory_by_id[avatar_id]
        record = record_by_id[avatar_id]
        tag = f"{avatar_id}_original_ue_v1"
        actions = record.get("actions", {})
        if record.get("tag") != tag or set(actions) != {"Walking", "Standing_Idle"}:
            raise RuntimeError(f"Rocketbox action pair changed: {avatar_id}")
        walking_dir = Path(actions["Walking"]["output_dir"]).resolve()
        idle_dir = Path(actions["Standing_Idle"]["output_dir"]).resolve()
        walking_cell, walking_complete = _video_cell(walking_dir)
        idle_cell, idle_complete = _video_cell(idle_dir)
        completed_clip_count += int(walking_complete) + int(idle_complete)
        complete_pair_count += int(walking_complete and idle_complete)
        category = str(avatar.get("category"))
        gender = str(avatar.get("gender"))
        if category not in CATEGORY_ORDER or gender not in GENDER_ORDER:
            raise RuntimeError(f"Rocketbox demographic changed: {avatar_id}")
        if html_path is not None:
            height = float(avatar.get("height_contract", {}).get("authored_height_cm"))
            for action, output_dir, complete in (
                ("Walking", walking_dir, walking_complete),
                ("Standing_Idle", idle_dir, idle_complete),
            ):
                if complete:
                    browser_entries.append(
                        {
                            "review_id": f"batch:{avatar_id}:{action}",
                            "source_set": "115 角色 Rocketbox 基线",
                            "featured": False,
                            "avatar_id": avatar_id,
                            "base_avatar_id": avatar_id,
                            "role_label": avatar_id,
                            "category": category,
                            "gender": gender,
                            "height_cm": height,
                            "action": action,
                            "trajectory": "original_batch_apartment",
                            "media": _browser_media(output_dir),
                            "speech": None,
                        }
                    )
        grouped.setdefault((category, gender), []).append(
            (avatar, record, walking_cell, idle_cell)
        )

    avatar_count = len(records)
    clip_count = avatar_count * 2
    lines = [
        "# Rocketbox UE Apartment 视频索引",
        "",
        "该文档由 `external/SPEAR/tools/build_rocketbox_batch_video_index.py` "
        "从固定批次 manifest 自动生成。每个 clip 收录三个长期入口：带标注的组合审核、"
        "UE 主视图、同步 Top-down/轨迹。中间编码文件不作为稳定入口。",
        "",
        f"- 更新时间：`{datetime.now(timezone.utc).isoformat()}`",
        f"- 角色总数：**{avatar_count}**",
        f"- 已完成：**{completed_clip_count} / {clip_count}**",
        f"- 完整 Walk/Idle 对：**{complete_pair_count} / {avatar_count}**",
        f"- 待生成：**{clip_count - completed_clip_count}**",
        f"- 批次 manifest：[{manifest_path.name}]({_absolute_link(manifest_path)})",
        (
            f"- 浏览器审核页：[{Path(html_path).name}]({_absolute_link(Path(html_path))})"
            if html_path is not None
            else "- 浏览器审核页：本次未生成"
        ),
        "",
        "状态说明：`✅` 表示三个稳定视频均存在且非空；`⏳` 表示该动作仍在生成或证据不完整。",
        "",
    ]
    for category, gender in sorted(
        grouped,
        key=lambda key: (CATEGORY_ORDER[key[0]], GENDER_ORDER[key[1]]),
    ):
        lines.extend(
            [
                f"## {category} · {gender.title()}",
                "",
                "| base_avatar_id | 身高 (cm) | Walking | Standing Idle |",
                "|---|---:|---|---|",
            ]
        )
        for avatar, _record, walking_cell, idle_cell in grouped[(category, gender)]:
            height = float(avatar.get("height_contract", {}).get("authored_height_cm"))
            lines.append(
                f"| `{avatar['base_avatar_id']}` | {height:.2f} | "
                f"{walking_cell} | {idle_cell} |"
            )
        lines.append("")

    _atomic_text(document_path, "\n".join(lines).rstrip() + "\n")
    if html_path is not None:
        browser_entries = _featured_entries(featured_manifest_path) + browser_entries
        _write_browser_index(
            Path(html_path).resolve(), browser_entries, avatar_count=avatar_count
        )
    return {
        "avatar_count": avatar_count,
        "clip_count": clip_count,
        "completed_clip_count": completed_clip_count,
        "complete_pair_count": complete_pair_count,
        "pending_clip_count": clip_count - completed_clip_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument(
        "--featured-manifest", type=Path, default=DEFAULT_FEATURED_MANIFEST
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_video_index(
        args.manifest,
        args.out,
        html_path=args.html,
        featured_manifest_path=args.featured_manifest,
    )
    print(
        "ROCKETBOX_VIDEO_INDEX_OK "
        f"avatars={summary['avatar_count']} "
        f"clips={summary['completed_clip_count']}/{summary['clip_count']} "
        f"pairs={summary['complete_pair_count']}/{summary['avatar_count']} "
        f"document={args.out.resolve()} html={args.html.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
