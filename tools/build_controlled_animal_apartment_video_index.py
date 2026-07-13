#!/usr/bin/env python3
"""Build a stable Markdown index for controlled-animal Apartment evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_DOCUMENT = AVENGINE_ROOT / "docs/controlled_animal_video_catalog.md"
DEFAULT_HTML = AVENGINE_ROOT / "docs/controlled_animal_video_review.html"
SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
VIDEO_NAMES = {
    "审核": "side_by_side_review_annotated.mp4",
    "主视图": "apartment_v1_view0.mp4",
    "Top-down": "topdown_review.mp4",
}
VIDEO_VIEW_KEYS = {
    "审核": "review",
    "主视图": "main",
    "Top-down": "topdown",
}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o644)


def _link(target: Path, label: str) -> str:
    return f"[{label}]({target.resolve().as_posix()})"


def _server_url(target: Path, server_root: Path) -> str:
    target = target.resolve()
    server_root = server_root.resolve()
    try:
        relative = target.relative_to(server_root)
    except ValueError as error:
        raise RuntimeError(
            f"review media is outside the configured server root: {target}"
        ) from error
    return "/" + quote(relative.as_posix(), safe="/")


def _action_cell(
    *, record: dict, action: str, document: Path
) -> tuple[str, bool, dict[str, Path]]:
    action_record = record["actions"][action]
    output = Path(action_record["output_dir"]).resolve()
    tag = str(record["tag"])
    paths = {
        label: output / "videos" / filename
        for label, filename in VIDEO_NAMES.items()
    }
    paths.update(
        {
            "音频": output / "binaural.wav",
            "事件": output / "binaural_source_schedule.json",
        }
    )
    registry_path = output.parent / "registry" / f"{tag}.json"
    try:
        registry = _read_json(registry_path)
        registry_clip = registry.get("clips", {}).get(action, {})
        registry_ok = (
            registry.get("usage_scope") == "research_candidate"
            and registry.get("formal_registry_promotion") is False
            and registry_clip.get("clip_id") == action_record["clip_id"]
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        registry_ok = False
    complete = registry_ok and all(
        path.is_file() and path.stat().st_size > 0 for path in paths.values()
    )
    if not complete:
        return "⏳ 待生成/证据不完整", False, paths
    links = [_link(path, label) for label, path in paths.items()]
    links.append(_link(registry_path, "Registry"))
    paths["Registry"] = registry_path
    return "📼 " + " · ".join(links), True, paths


def _build_review_html(entries: list[dict], output_path: Path) -> None:
    encoded_entries = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>受控动物视频审核</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0b0e14; color: #e7eaf0; height: 100vh; overflow: hidden; }
    button, input, select { font: inherit; }
    .app { display: grid; grid-template-columns: 340px minmax(0, 1fr); height: 100vh; }
    aside { border-right: 1px solid #273043; background: #111620; display: flex; flex-direction: column; min-height: 0; }
    header { padding: 18px; border-bottom: 1px solid #273043; }
    h1 { font-size: 18px; margin: 0 0 6px; }
    .summary { color: #99a4b7; font-size: 13px; }
    .links { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 10px; }
    .filters { display: grid; gap: 8px; padding: 12px; border-bottom: 1px solid #273043; }
    input, select { width: 100%; color: #e7eaf0; background: #171e2b; border: 1px solid #303b50; border-radius: 8px; padding: 9px 10px; }
    .filter-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    #clip-list { overflow: auto; padding: 8px; }
    .clip-item { width: 100%; text-align: left; color: inherit; background: transparent; border: 1px solid transparent; border-radius: 9px; padding: 10px; cursor: pointer; }
    .clip-item:hover { background: #171e2b; }
    .clip-item.active { background: #1b2940; border-color: #3c6ea8; }
    .clip-title { font-size: 13px; font-weight: 650; overflow-wrap: anywhere; }
    .clip-meta { margin-top: 4px; color: #99a4b7; font-size: 12px; }
    main { min-width: 0; overflow: auto; padding: 24px; }
    .stage { max-width: 1380px; margin: 0 auto; }
    .warning { color: #ffdce2; background: #421922; border: 1px solid #a33a4e; border-radius: 10px; padding: 12px; margin-bottom: 14px; line-height: 1.5; }
    .title-row { display: flex; gap: 12px; justify-content: space-between; align-items: start; }
    h2 { margin: 0; font-size: 21px; overflow-wrap: anywhere; }
    .subhead { color: #99a4b7; margin-top: 5px; }
    .nav { display: flex; gap: 7px; }
    .button { color: #e7eaf0; background: #171e2b; border: 1px solid #303b50; border-radius: 8px; padding: 8px 12px; cursor: pointer; text-decoration: none; }
    .button:hover, .button.active { background: #24466e; border-color: #4e83bd; }
    .attributes { display: flex; flex-wrap: wrap; gap: 7px; margin: 15px 0; }
    .pill { background: #172131; border: 1px solid #2b3a51; border-radius: 999px; padding: 5px 9px; font-size: 12px; }
    .views { display: flex; gap: 8px; margin-bottom: 10px; }
    video { width: 100%; max-height: calc(100vh - 310px); background: #000; border: 1px solid #2c3547; border-radius: 10px; }
    .path-row { margin-top: 10px; display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; align-items: center; }
    code { display: block; min-width: 0; overflow: auto; white-space: nowrap; color: #b9c9df; background: #111620; border: 1px solid #273043; border-radius: 8px; padding: 9px; }
    .audio-row { margin-top: 14px; display: grid; grid-template-columns: minmax(250px, 520px) 1fr; gap: 12px; align-items: center; }
    audio { width: 100%; }
    .empty { padding: 24px; color: #99a4b7; }
    @media (max-width: 850px) {
      body { overflow: auto; height: auto; }
      .app { grid-template-columns: 1fr; height: auto; }
      aside { height: 44vh; border-right: 0; border-bottom: 1px solid #273043; }
      main { padding: 14px; }
      video { max-height: none; }
      .path-row, .audio-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <script id="clip-data" type="application/json">__CLIP_DATA__</script>
  <div class="app">
    <aside>
      <header>
        <h1>受控动物视频审核</h1>
        <div class="summary" id="summary"></div>
        <div class="links"><a class="button" href="/">先做猫狗方向纠正</a><a class="button" href="/docs/rocketbox_human_video_review.html">人类视频</a></div>
      </header>
      <div class="filters">
        <input id="search" type="search" placeholder="搜索 asset / 品种 / 属性">
        <div class="filter-row">
          <select id="species"><option value="">全部物种</option></select>
          <select id="breed"><option value="">全部品种</option></select>
        </div>
        <select id="action">
          <option value="">Walk + Idle</option>
          <option value="Walking">Walking</option>
          <option value="Idle">Idle</option>
        </select>
      </div>
      <div id="clip-list"></div>
    </aside>
    <main>
      <div class="stage" id="stage">
        <div class="warning"><b>这些是旧的诊断成片，不是方向通过结果。</b>用户已判定猫 Walking 斜跑、狗 Walking 后退/斜跑；原来的骨骼自动方向检查已失效。请先在“猫狗方向纠正”页保存可见朝向，重新绑定并通过直线 + 转弯动态 canary 后，才能恢复动作通过状态。</div>
        <div class="title-row">
          <div><h2 id="asset-title"></h2><div class="subhead" id="asset-subhead"></div></div>
          <div class="nav"><button class="button" id="previous">← 上一个</button><button class="button" id="next">下一个 →</button></div>
        </div>
        <div class="attributes" id="attributes"></div>
        <div class="views" id="views"></div>
        <video id="player" controls preload="metadata" src=""></video>
        <div class="path-row">
          <code id="absolute-path"></code>
          <button class="button" id="copy-path">复制绝对路径</button>
          <a class="button" id="open-video" target="_blank" rel="noopener">单独打开</a>
        </div>
        <div class="audio-row">
          <audio id="audio" controls preload="none"></audio>
          <div><a class="button" id="open-schedule" target="_blank" rel="noopener">查看叫声 schedule</a></div>
        </div>
      </div>
      <div class="empty" id="empty" hidden>没有符合筛选条件的视频。</div>
    </main>
  </div>
  <script>
    const allClips = JSON.parse(document.getElementById('clip-data').textContent);
    const state = { clips: [...allClips], selected: 0, view: 'review' };
    const $ = id => document.getElementById(id);
    const search = $('search'), species = $('species'), breed = $('breed'), action = $('action');
    const list = $('clip-list'), player = $('player'), audio = $('audio');

    function addOptions(select, values) {
      for (const value of [...new Set(values)].sort()) {
        const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option);
      }
    }
    addOptions(species, allClips.map(item => item.species));
    addOptions(breed, allClips.map(item => item.breed));
    $('summary').textContent = `${new Set(allClips.map(item => item.asset_id)).size} 个实例 · ${allClips.length} 个动作片段 · ${allClips.length * 3} 个视频`;

    function applyFilters() {
      const needle = search.value.trim().toLowerCase();
      state.clips = allClips.filter(item => {
        const haystack = `${item.asset_id} ${item.species} ${item.breed} ${JSON.stringify(item.attributes)}`.toLowerCase();
        return (!needle || haystack.includes(needle)) && (!species.value || item.species === species.value) && (!breed.value || item.breed === breed.value) && (!action.value || item.action === action.value);
      });
      state.selected = 0; render();
    }

    function renderList() {
      list.replaceChildren();
      state.clips.forEach((item, index) => {
        const button = document.createElement('button');
        button.className = `clip-item${index === state.selected ? ' active' : ''}`;
        button.innerHTML = `<div class="clip-title"></div><div class="clip-meta"></div>`;
        button.firstElementChild.textContent = item.asset_id;
        button.lastElementChild.textContent = `${item.species} · ${item.breed} · ${item.action}`;
        button.onclick = () => { state.selected = index; render(); };
        list.append(button);
      });
      const activeItem = list.querySelector('.active');
      if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });
    }

    function renderDetail() {
      const item = state.clips[state.selected];
      $('stage').hidden = !item; $('empty').hidden = Boolean(item);
      if (!item) { player.removeAttribute('src'); player.load(); return; }
      $('asset-title').textContent = item.asset_id;
      $('asset-subhead').textContent = `${item.species} · ${item.breed} · ${item.action} · actor scale ${item.actor_scale.toFixed(4)}`;
      $('attributes').replaceChildren(...Object.entries(item.attributes).map(([key, value]) => {
        const span = document.createElement('span'); span.className = 'pill'; span.textContent = `${key}=${value}`; return span;
      }));
      $('views').replaceChildren(...Object.entries(item.videos).map(([key, media]) => {
        const button = document.createElement('button'); button.className = `button${key === state.view ? ' active' : ''}`; button.textContent = media.label;
        button.onclick = () => { state.view = key; renderDetail(); }; return button;
      }));
      const media = item.videos[state.view] || item.videos.review;
      if (player.getAttribute('src') !== media.url) { player.src = media.url; player.load(); }
      $('absolute-path').textContent = media.absolute_path;
      $('open-video').href = media.url;
      audio.src = item.audio.url;
      $('open-schedule').href = item.schedule.url;
    }

    function render() { renderList(); renderDetail(); }
    for (const control of [search, species, breed, action]) control.addEventListener('input', applyFilters);
    $('previous').onclick = () => { if (state.clips.length) { state.selected = (state.selected - 1 + state.clips.length) % state.clips.length; render(); } };
    $('next').onclick = () => { if (state.clips.length) { state.selected = (state.selected + 1) % state.clips.length; render(); } };
    $('copy-path').onclick = async () => {
      const value = $('absolute-path').textContent;
      try { await navigator.clipboard.writeText(value); } catch (_) { const area = document.createElement('textarea'); area.value = value; document.body.append(area); area.select(); document.execCommand('copy'); area.remove(); }
    };
    window.addEventListener('keydown', event => { if (event.key === 'ArrowLeft') $('previous').click(); if (event.key === 'ArrowRight') $('next').click(); });
    render();
  </script>
</body>
</html>
"""
    _atomic_text(output_path, template.replace("__CLIP_DATA__", encoded_entries))


def build_video_index(
    manifest_paths: Sequence[Path],
    document_path: Path,
    *,
    html_path: Path | None = None,
    server_root: Path = AVENGINE_ROOT,
) -> dict:
    document_path = Path(document_path).resolve()
    records = []
    manifest_links = []
    seen = set()
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path).resolve()
        payload = _read_json(manifest_path)
        manifest_records = payload.get("records")
        if (
            payload.get("schema") != SCHEMA
            or not isinstance(manifest_records, list)
            or payload.get("avatar_count") != len(manifest_records)
            or payload.get("clip_count") != len(manifest_records) * 2
        ):
            raise RuntimeError(f"invalid controlled-animal manifest: {manifest_path}")
        manifest_links.append(_link(manifest_path, manifest_path.parent.name))
        for record in manifest_records:
            asset_id = str(record.get("base_avatar_id") or "")
            if not asset_id or asset_id in seen:
                raise RuntimeError(f"duplicate/empty controlled animal id: {asset_id!r}")
            if set(record.get("actions", {})) != {"Idle", "Walking"}:
                raise RuntimeError(f"controlled animal action pair changed: {asset_id}")
            seen.add(asset_id)
            records.append(record)

    grouped: dict[tuple[str, str], list[dict]] = {}
    complete_clips = 0
    complete_pairs = 0
    rendered_rows = []
    for record in records:
        walking_spec = _read_json(Path(record["actions"]["Walking"]["spec"]))
        sources = walking_spec.get("sources", [])
        if len(sources) != 1:
            raise RuntimeError(f"expected one source: {record['base_avatar_id']}")
        source = sources[0]
        if (
            source.get("tag") != record.get("tag")
            or source.get("asset_id") != record.get("base_avatar_id")
            or source.get("asset_class") != "animal"
        ):
            raise RuntimeError(f"controlled animal source identity changed: {record['base_avatar_id']}")
        walking, walking_ok, walking_paths = _action_cell(
            record=record, action="Walking", document=document_path
        )
        idle, idle_ok, idle_paths = _action_cell(
            record=record, action="Idle", document=document_path
        )
        complete_clips += int(walking_ok) + int(idle_ok)
        complete_pairs += int(walking_ok and idle_ok)
        row = {
            "record": record,
            "source": source,
            "walking": walking,
            "idle": idle,
            "action_paths": {
                "Walking": walking_paths,
                "Idle": idle_paths,
            },
            "action_complete": {
                "Walking": walking_ok,
                "Idle": idle_ok,
            },
        }
        grouped.setdefault(
            (str(source.get("species")), str(source.get("breed"))), []
        ).append(row)
        rendered_rows.append(row)

    lines = [
        "# 受控动物 UE Apartment 视频索引",
        "",
        "该文档由 `external/SPEAR/tools/build_controlled_animal_apartment_video_index.py` "
        "从认证 spec/registry 自动生成。稳定入口包括带标注审核、UE 主视图、同步 "
        "Top-down、双耳音频和叫声事件 schedule。",
        "",
        "> **方向状态（2026-07-13）：旧 Walking 已被用户视觉审核拒绝。** "
        "猫存在斜跑、狗存在后退/斜跑；下表的 `📼` 只表示媒体与 Registry 文件完整，"
        "不表示方向或动画重新通过。必须先通过 `http://127.0.0.1:8102/` 的可见朝向审核，"
        "再重新绑定并验证直线 + 转弯动态 canary。旧资产和视频保持不覆盖，作为失败诊断证据。",
        "",
        f"- 更新时间：`{datetime.now(timezone.utc).isoformat()}`",
        f"- 动物实例：**{len(records)}**",
        f"- 已完成 clips：**{complete_clips} / {len(records) * 2}**",
        f"- 完整 Walk/Idle 对：**{complete_pairs} / {len(records)}**",
        f"- 输入 manifests：{' · '.join(manifest_links)}",
    ]
    if html_path is not None:
        lines.append(f"- 网页审核文件：{_link(Path(html_path), '打开审核网页')}")
    lines.append("")
    for (species, breed), rows in sorted(grouped.items()):
        lines.extend(
            [
                f"## {species} · {breed}",
                "",
                "| asset_id | 绝对实例属性 | actor scale | Walking | Idle |",
                "|---|---|---:|---|---|",
            ]
        )
        for row in sorted(rows, key=lambda item: item["record"]["base_avatar_id"]):
            source = row["source"]
            attributes = ", ".join(
                f"{key}={value}"
                for key, value in sorted(source.get("sampled_attributes", {}).items())
            )
            lines.append(
                f"| `{row['record']['base_avatar_id']}` | `{attributes}` | "
                f"{float(source['actor_scale']):.4f} | {row['walking']} | {row['idle']} |"
            )
        lines.append("")
    _atomic_text(document_path, "\n".join(lines).rstrip() + "\n")
    if html_path is not None:
        entries = []
        for row in sorted(
            rendered_rows, key=lambda item: item["record"]["base_avatar_id"]
        ):
            for action in ("Walking", "Idle"):
                if not row["action_complete"][action]:
                    continue
                paths = row["action_paths"][action]
                entries.append(
                    {
                        "asset_id": row["record"]["base_avatar_id"],
                        "species": str(row["source"]["species"]),
                        "breed": str(row["source"]["breed"]),
                        "action": action,
                        "actor_scale": float(row["source"]["actor_scale"]),
                        "attributes": dict(
                            sorted(row["source"].get("sampled_attributes", {}).items())
                        ),
                        "videos": {
                            VIDEO_VIEW_KEYS[label]: {
                                "label": label,
                                "absolute_path": str(paths[label].resolve()),
                                "url": _server_url(paths[label], server_root),
                            }
                            for label in VIDEO_NAMES
                        },
                        "audio": {
                            "absolute_path": str(paths["音频"].resolve()),
                            "url": _server_url(paths["音频"], server_root),
                        },
                        "schedule": {
                            "absolute_path": str(paths["事件"].resolve()),
                            "url": _server_url(paths["事件"], server_root),
                        },
                    }
                )
        _build_review_html(entries, Path(html_path).resolve())
    return {
        "animal_count": len(records),
        "clip_count": len(records) * 2,
        "completed_clip_count": complete_clips,
        "complete_pair_count": complete_pairs,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--server-root", type=Path, default=AVENGINE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_video_index(
        args.manifest,
        args.out,
        html_path=args.html_out,
        server_root=args.server_root,
    )
    print(
        "CONTROLLED_ANIMAL_VIDEO_INDEX_OK "
        f"animals={summary['animal_count']} "
        f"clips={summary['completed_clip_count']}/{summary['clip_count']} "
        f"pairs={summary['complete_pair_count']}/{summary['animal_count']} "
        f"document={args.out.resolve()} "
        f"html={args.html_out.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
