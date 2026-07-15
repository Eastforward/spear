#!/usr/bin/env python3
"""Build a self-contained browser index for an authenticated animal OFAT review."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import finalize_stable_animal_ofat_review as review_lib  # noqa: E402


class IndexError(RuntimeError):
    """Raised when the review manifest cannot safely drive a media index."""


def media_url(absolute_path: str) -> str:
    path = Path(absolute_path).resolve()
    try:
        relative = path.relative_to(AVENGINE_ROOT.resolve())
    except ValueError as error:
        raise IndexError(f"artifact is outside AVEngine: {path}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise IndexError(f"artifact is missing or empty: {path}")
    return "/" + relative.as_posix()


def artifact(record: dict[str, Any]) -> dict[str, str]:
    absolute_path = str(record.get("absolute_path", ""))
    if not absolute_path:
        raise IndexError("artifact record is missing absolute_path")
    return {"absolute_path": absolute_path, "url": media_url(absolute_path)}


def browser_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != review_lib.SCHEMA:
        raise IndexError("unsupported OFAT review schema")
    if manifest.get("manifest_sha256") != contracts.manifest_sha256(manifest):
        raise IndexError("OFAT review manifest hash mismatch")
    entries = []
    for item in manifest.get("entries", []):
        videos = item.get("videos", {})
        entries.append(
            {
                "label": item["label"],
                "instance_id": item["instance_id"],
                "sampled_attributes": item["sampled_attributes"],
                "changed_attribute_from_baseline": item.get(
                    "changed_attribute_from_baseline", "baseline"
                ),
                "static": artifact(item["static_fixed_scale_review"]),
                "glb": artifact(item["realization"]["glb"]),
                "videos": {
                    "Walking": artifact(videos["Walking"]),
                    "Idle": artifact(videos["Idle"]),
                },
                "qa": item["qa"],
            }
        )
    if not entries or len(entries) != manifest.get("entry_count"):
        raise IndexError("OFAT review entry count mismatch")
    return {
        "classification": manifest["state_classification"],
        "formal_dataset_registration_authorized": manifest[
            "formal_dataset_registration_authorized"
        ],
        "ofat": manifest["ofat"],
        "entries": entries,
        "contact_sheets": [artifact(item) for item in manifest["contact_sheets"]],
    }


def render(payload: dict[str, Any], title: str) -> str:
    safe_title = html.escape(title)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root{{--bg:#080d14;--panel:#111927;--line:#293a51;--text:#e7eef8;--muted:#98a9bd;--accent:#53d38a;--blue:#73b7ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
.app{{display:grid;grid-template-columns:330px 1fr;min-height:100vh}} aside{{border-right:1px solid var(--line);padding:20px;position:sticky;top:0;height:100vh;overflow:auto}}
main{{padding:24px;min-width:0}} h1{{font-size:21px;margin:0 0 8px}} .muted{{color:var(--muted)}} .status{{padding:10px 12px;border:1px solid #725d26;background:#251f10;border-radius:9px;margin:14px 0}}
.entry{{width:100%;text-align:left;color:var(--text);background:#101827;border:1px solid var(--line);border-radius:9px;padding:10px;margin:5px 0;cursor:pointer}}
.entry.active{{border-color:var(--blue);background:#152541}} .entry b{{display:block}} .entry small{{color:var(--muted)}}
.header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}} h2{{margin:0}} .pills{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 18px}}
.pill{{border:1px solid var(--line);border-radius:999px;padding:4px 9px;background:#131d2c}} .grid{{display:grid;grid-template-columns:minmax(380px,1fr) minmax(380px,1fr);gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}} .card h3{{margin:0 0 10px}}
video,img{{width:100%;max-height:67vh;object-fit:contain;background:#030609;border-radius:8px}} .tabs{{display:flex;gap:8px;margin-bottom:10px}} button.tab{{color:var(--text);background:#19263a;border:1px solid var(--line);padding:7px 15px;border-radius:8px;cursor:pointer}} button.tab.active{{background:#285486;border-color:#73b7ff}}
code{{display:block;color:#b8cae0;overflow-wrap:anywhere;background:#080d14;padding:9px;border-radius:7px;margin-top:9px}} .sheets{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}
@media(max-width:950px){{.app{{grid-template-columns:1fr}}aside{{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}}.grid,.sheets{{grid-template-columns:1fr}}}}
</style></head><body><div class="app"><aside><h1>{safe_title}</h1><div class="muted">9 个互斥 OFAT 实例 · 每个含 Walking + Idle</div><div class="status">当前为 research candidate；人体视觉、UE Apartment 与音频审核完成前不会注册为正式资产。</div><div id="list"></div></aside>
<main><div class="header"><div><h2 id="name"></h2><div id="id" class="muted"></div></div><div id="changed" class="pill"></div></div><div id="attrs" class="pills"></div>
<div class="grid"><section class="card"><h3>固定尺度静态对比</h3><img id="static"><code id="staticPath"></code><code id="glbPath"></code></section><section class="card"><h3>动画审核</h3><div class="tabs"><button class="tab active" data-action="Walking">Walking</button><button class="tab" data-action="Idle">Idle</button></div><video id="video" controls loop muted playsinline preload="metadata"></video><code id="videoPath"></code></section></div>
<div class="sheets" id="sheets"></div></main></div><script id="payload" type="application/json">{encoded}</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent),list=document.getElementById('list');let index=0,action='Walking';
function show(){{const e=data.entries[index];document.querySelectorAll('.entry').forEach((n,i)=>n.classList.toggle('active',i===index));document.getElementById('name').textContent=e.label;document.getElementById('id').textContent=e.instance_id;document.getElementById('changed').textContent='changed: '+e.changed_attribute_from_baseline;document.getElementById('attrs').innerHTML=Object.entries(e.sampled_attributes).map(([k,v])=>`<span class="pill">${{k}}=${{v}}</span>`).join('');document.getElementById('static').src=e.static.url;document.getElementById('staticPath').textContent=e.static.absolute_path;document.getElementById('glbPath').textContent='GLB: '+e.glb.absolute_path;setVideo();}}
function setVideo(){{const v=data.entries[index].videos[action];document.querySelectorAll('.tab').forEach(n=>n.classList.toggle('active',n.dataset.action===action));const player=document.getElementById('video');player.src=v.url;player.load();document.getElementById('videoPath').textContent=v.absolute_path;}}
data.entries.forEach((e,i)=>{{const b=document.createElement('button');b.className='entry';b.innerHTML=`<b>${{e.label}}</b><small>${{Object.entries(e.sampled_attributes).map(([k,v])=>k+'='+v).join(' · ')}}</small>`;b.onclick=()=>{{index=i;show()}};list.appendChild(b)}});document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{{action=b.dataset.action;setVideo()}});
const sheets=document.getElementById('sheets');data.contact_sheets.forEach((s,i)=>{{const c=document.createElement('section');c.className='card';c.innerHTML=`<h3>${{i===0?'九实例静态总览':'九实例 Walking 中间帧'}}</h3><img src="${{s.url}}"><code>${{s.absolute_path}}</code>`;sheets.appendChild(c)}});show();
</script></body></html>"""


def build(args: argparse.Namespace) -> Path:
    manifest_path, manifest = review_lib.load_json(args.manifest, "OFAT review manifest")
    payload = browser_payload(manifest)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise IndexError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(render(payload, args.title))
        stream.flush()
        os.fsync(stream.fileno())
    if output.stat().st_size <= 0 or "<video" not in output.read_text(encoding="utf-8"):
        raise IndexError(f"generated page readback failed: {output}")
    print(f"source_manifest={manifest_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="比格受控实例审核")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = build(args)
    except (IndexError, review_lib.ReviewError, OSError, ValueError) as error:
        print(f"STABLE_ANIMAL_OFAT_INDEX_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_ANIMAL_OFAT_INDEX_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
