#!/usr/bin/env python3
"""Authenticate and index stable-animal OFAT Apartment Walk/Idle evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from tools import finalize_stable_animal_ofat_review as ofat_lib  # noqa: E402


SPEC_SCHEMA = "stable_animal_walk_idle_apartment_specs_v1"
STATUS_SCHEMA = "stable_animal_apartment_render_status_v1"
REGISTRY_SCHEMA = "stable_animal_apartment_research_candidate_registry_v1"
ACTIONS = ("Walking", "Idle")
MEDIA_FIELDS = {
    "review": "annotated_review_video",
    "main": "apartment_video",
    "topdown": "topdown_review_video",
}


class ReviewError(RuntimeError):
    """Raised when one Apartment review authority is incomplete or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReviewError(f"{label} is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"{label} must be a JSON object: {path}")
    return path, value


def descriptor(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReviewError(f"artifact is missing or empty: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_descriptor(record: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ReviewError(f"{label} descriptor is missing")
    path = Path(str(record.get("path", ""))).resolve()
    observed = descriptor(path)
    if (
        observed["sha256"] != record.get("sha256")
        or observed["size_bytes"] != record.get("size_bytes")
    ):
        raise ReviewError(f"{label} descriptor changed: {path}")
    return observed


def server_url(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(AVENGINE_ROOT.resolve())
    except ValueError as error:
        raise ReviewError(f"media escaped AVEngine HTTP root: {resolved}") from error
    return "/" + relative.as_posix()


def authenticate(args: argparse.Namespace) -> dict[str, Any]:
    ofat_path, ofat = load_json(args.ofat_review, "OFAT review")
    if (
        ofat.get("schema") != ofat_lib.SCHEMA
        or ofat.get("manifest_sha256") != contracts.manifest_sha256(ofat)
        or ofat.get("formal_dataset_registration_authorized") is not False
    ):
        raise ReviewError("OFAT review hash or classification failed")

    spec_path, spec = load_json(args.spec_manifest, "Apartment spec manifest")
    records = spec.get("records")
    if (
        spec.get("schema") != SPEC_SCHEMA
        or not isinstance(records, list)
        or spec.get("avatar_count") != len(records)
        or spec.get("clip_count") != len(records) * len(ACTIONS)
        or spec.get("formal_registration_authorized") is not False
    ):
        raise ReviewError("Apartment spec manifest contract failed")

    status_path, status = load_json(args.batch_status, "Apartment batch status")
    if (
        status.get("schema") != STATUS_SCHEMA
        or Path(str(status.get("manifest", ""))).resolve() != spec_path
        or status.get("job_count") != spec["clip_count"]
        or status.get("passed_job_count") != spec["clip_count"]
        or status.get("failed_job_count") != 0
        or status.get("incomplete_job_count") != 0
        or status.get("incomplete_jobs") != []
    ):
        raise ReviewError("Apartment batch did not finish 100% successfully")

    ofat_entries = {item["instance_id"]: item for item in ofat.get("entries", [])}
    spec_records = {item["asset_id"]: item for item in records}
    if (
        len(ofat_entries) != ofat.get("entry_count")
        or len(spec_records) != len(records)
        or set(ofat_entries) != set(spec_records)
    ):
        raise ReviewError("OFAT and Apartment instance identity sets differ")

    entries = []
    clip_count = 0
    for instance_id in sorted(ofat_entries):
        source = ofat_entries[instance_id]
        record = spec_records[instance_id]
        actions = record.get("actions", {})
        if set(actions) != set(ACTIONS):
            raise ReviewError(f"Walk/Idle pair is incomplete: {instance_id}")
        action_roots = [Path(actions[action]["output_dir"]).resolve() for action in ACTIONS]
        registry_path = action_roots[0].parent / "registry" / f"{record['tag']}.json"
        registry_path, registry = load_json(registry_path, "per-instance UE registry")
        if (
            registry.get("schema_version") != REGISTRY_SCHEMA
            or registry.get("usage_scope") != "research_candidate"
            or registry.get("formal_registry_promotion") is not False
            or registry.get("human_visual_review") != "pending"
            or registry.get("tag") != record["tag"]
            or registry.get("asset_id") != instance_id
            or registry.get("sampled_attributes") != source["sampled_attributes"]
            or registry.get("source_sha256") != record["source_glb"]["sha256"]
            or registry.get("direction", {}).get("automatic_fine_yaw_inference") is not False
            or set(registry.get("clips", {})) != set(ACTIONS)
        ):
            raise ReviewError(f"per-instance UE registry contract failed: {instance_id}")

        action_evidence = {}
        for action in ACTIONS:
            clip = registry["clips"][action]
            planned = actions[action]
            if clip.get("clip_id") != planned.get("clip_id"):
                raise ReviewError(f"clip identity changed: {instance_id}/{action}")
            artifacts = {
                key: verify_descriptor(value, f"{instance_id}/{action}/{key}")
                for key, value in clip.items()
                if key != "clip_id"
            }
            required = {
                "spec",
                "runtime_gate",
                "actor_visual_metadata",
                "apartment_video",
                "topdown_review_video",
                "annotated_review_video",
                "binaural_audio",
                "binaural_source_schedule",
            }
            if set(artifacts) != required:
                raise ReviewError(f"clip evidence set changed: {instance_id}/{action}")
            # Finalization copies the immutable planned spec into the clip
            # directory, so authenticate content rather than demanding the
            # two evidence paths be identical.
            if (
                artifacts["spec"]["sha256"]
                != planned["spec_evidence"]["sha256"]
                or artifacts["spec"]["size_bytes"]
                != planned["spec_evidence"]["size_bytes"]
            ):
                raise ReviewError(f"final clip spec changed: {instance_id}/{action}")
            _visual_path, visual = load_json(
                Path(artifacts["actor_visual_metadata"]["path"]),
                "actor visual metadata",
            )
            rig_evidence = visual.get("rig_direction_evidence", {})
            if (
                visual.get("automatic_checks", {}).get("overall") != "passed"
                or not rig_evidence
                or {item.get("status") for item in rig_evidence.values()} != {"passed"}
            ):
                raise ReviewError(f"runtime direction/ground QA failed: {instance_id}/{action}")
            action_evidence[action] = {
                "clip_id": clip["clip_id"],
                "artifacts": artifacts,
                "media": {
                    name: {
                        **artifacts[field],
                        "url": server_url(artifacts[field]["path"]),
                    }
                    for name, field in MEDIA_FIELDS.items()
                },
                "audio": {
                    **artifacts["binaural_audio"],
                    "url": server_url(artifacts["binaural_audio"]["path"]),
                },
            }
            clip_count += 1

        entries.append(
            {
                "instance_id": instance_id,
                "label": source["label"],
                "species": record["species"],
                "breed": record["breed"],
                "sampled_attributes": source["sampled_attributes"],
                "changed_attribute_from_baseline": source.get(
                    "changed_attribute_from_baseline", "baseline"
                ),
                "source_glb": record["source_glb"],
                "registry": descriptor(registry_path),
                "actions": action_evidence,
            }
        )

    return {
        "schema": "stable_animal_ofat_apartment_review_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": (
            "research_candidate_automatic_ue_walk_idle_audio_passed_"
            "pending_human_visual_review"
        ),
        "formal_dataset_registration_authorized": False,
        "inputs": {
            "ofat_review": descriptor(ofat_path),
            "spec_manifest": descriptor(spec_path),
            "batch_status": descriptor(status_path),
        },
        "entry_count": len(entries),
        "clip_count": clip_count,
        "actions": list(ACTIONS),
        "entries": entries,
        "automatic_checks": {
            "all_instance_ids_match": True,
            "all_walk_idle_pairs_complete": True,
            "all_registry_artifacts_rehashed": True,
            "all_runtime_direction_and_ground_checks_passed": True,
            "all_audio_and_event_schedules_present": True,
            "human_visual_review": "pending",
            "overall": "passed_pending_human_visual_review",
        },
    }


def render_html(manifest: dict[str, Any], title: str) -> str:
    payload = json.dumps(
        {"classification": manifest["state_classification"], "entries": manifest["entries"]},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{safe_title}</title>
<style>:root{{--bg:#090e16;--panel:#111a28;--line:#2a3a51;--text:#edf3fb;--muted:#9dafc3;--blue:#70b7ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}.app{{display:grid;grid-template-columns:345px 1fr;min-height:100vh}}aside{{height:100vh;position:sticky;top:0;overflow:auto;padding:18px;border-right:1px solid var(--line)}}main{{padding:22px;min-width:0}}h1{{font-size:20px;margin:0 0 6px}}h2{{margin:0}}.muted{{color:var(--muted)}}.banner{{margin:14px 0;padding:11px;border:1px solid #7d6524;background:#29210e;border-radius:9px}}.item{{width:100%;color:inherit;text-align:left;background:#101827;border:1px solid var(--line);border-radius:9px;padding:10px;margin:5px 0;cursor:pointer}}.item.active{{border-color:var(--blue);background:#172a44}}.item b,.item small{{display:block}}.item small{{color:var(--muted)}}.head{{display:flex;justify-content:space-between;gap:12px}}.pills,.tabs{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}.pill,.tab{{color:inherit;border:1px solid var(--line);background:#172235;border-radius:999px;padding:5px 10px}}button.tab{{cursor:pointer}}button.tab.active{{background:#285785;border-color:var(--blue)}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}video{{width:100%;max-height:70vh;background:#000;border-radius:8px}}audio{{width:100%;margin-top:10px}}code{{display:block;white-space:nowrap;overflow:auto;background:#080d14;color:#b9cce2;padding:8px;border-radius:7px;margin-top:8px}}@media(max-width:900px){{.app{{display:block}}aside{{height:auto;position:static;border-right:0;border-bottom:1px solid var(--line)}}}}</style></head>
<body><script id="data" type="application/json">{payload}</script><div class="app"><aside><h1>{safe_title}</h1><div class="muted">9 个 OFAT 实例 · 18 个 UE Walk/Idle 音视频</div><div class="banner">自动门均通过；当前仍是 research candidate，等待人工视觉审核，不等于正式资产注册。</div><div id="list"></div></aside><main><div class="head"><div><h2 id="name"></h2><div id="id" class="muted"></div></div><div id="changed" class="pill"></div></div><div id="attrs" class="pills"></div><div class="tabs" id="actions"></div><div class="tabs" id="media"></div><section class="card"><video id="video" controls loop playsinline preload="metadata"></video><audio id="audio" controls preload="metadata"></audio><code id="path"></code><code id="audioPath"></code></section></main></div>
<script>const d=JSON.parse(document.getElementById('data').textContent),list=document.getElementById('list');let i=0,a='Walking',m='review';const $=x=>document.getElementById(x);function buttons(root,values,current,setter){{$(root).replaceChildren(...values.map(v=>{{const b=document.createElement('button');b.className='tab'+(v===current?' active':'');b.textContent=v;b.onclick=()=>setter(v);return b}}))}}function media(){{const e=d.entries[i],x=e.actions[a];buttons('actions',['Walking','Idle'],a,v=>{{a=v;media()}});buttons('media',['review','main','topdown'],m,v=>{{m=v;media()}});$('video').src=x.media[m].url;$('video').load();$('audio').src=x.audio.url;$('audio').load();$('path').textContent=x.media[m].path;$('audioPath').textContent='Audio: '+x.audio.path}}function show(){{const e=d.entries[i];document.querySelectorAll('.item').forEach((n,j)=>n.classList.toggle('active',j===i));$('name').textContent=e.label;$('id').textContent=e.instance_id;$('changed').textContent='changed: '+e.changed_attribute_from_baseline;$('attrs').replaceChildren(...Object.entries(e.sampled_attributes).map(([k,v])=>{{const s=document.createElement('span');s.className='pill';s.textContent=k+'='+v;return s}}));media()}}d.entries.forEach((e,j)=>{{const b=document.createElement('button');b.className='item';const attrs=Object.entries(e.sampled_attributes).map(([k,v])=>k+'='+v).join(' · ');b.innerHTML='<b></b><small></small>';b.querySelector('b').textContent=e.label;b.querySelector('small').textContent=attrs;b.onclick=()=>{{i=j;show()}};list.appendChild(b)}});show();</script></body></html>"""


def publish(args: argparse.Namespace) -> tuple[Path, Path]:
    manifest = authenticate(args)
    manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
    output_manifest = args.output_manifest.resolve()
    output_html = args.output_html.resolve()
    for output in (output_manifest, output_html):
        if output.exists() or output.is_symlink():
            raise ReviewError(f"refusing to replace output: {output}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    with output_html.open("x", encoding="utf-8") as stream:
        stream.write(render_html(manifest, args.title))
        stream.flush()
        os.fsync(stream.fileno())
    observed = json.loads(output_manifest.read_text(encoding="utf-8"))
    if observed["manifest_sha256"] != contracts.manifest_sha256(observed):
        raise ReviewError("output manifest hash readback failed")
    return output_manifest, output_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofat-review", type=Path, required=True)
    parser.add_argument("--spec-manifest", type=Path, required=True)
    parser.add_argument("--batch-status", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", default="稳定动物受控实例 UE 审核")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, page = publish(args)
    except (ReviewError, ofat_lib.ReviewError, OSError, ValueError) as error:
        print(f"STABLE_ANIMAL_OFAT_APARTMENT_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "STABLE_ANIMAL_OFAT_APARTMENT_REVIEW_OK "
        f"manifest={manifest} html={page}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
