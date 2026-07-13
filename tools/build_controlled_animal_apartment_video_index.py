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


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENT = SPEAR_ROOT.parents[1] / "docs/controlled_animal_video_catalog.md"
SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
VIDEO_NAMES = {
    "审核": "side_by_side_review_annotated.mp4",
    "主视图": "apartment_v1_view0.mp4",
    "Top-down": "topdown_review.mp4",
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


def _link(target: Path, document: Path, label: str) -> str:
    relative = Path(os.path.relpath(target.resolve(), document.parent)).as_posix()
    return f"[{label}]({relative})"


def _action_cell(
    *, record: dict, action: str, document: Path
) -> tuple[str, bool]:
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
        return "⏳ 待生成/证据不完整", False
    links = [_link(path, document, label) for label, path in paths.items()]
    links.append(_link(registry_path, document, "Registry"))
    return "✅ " + " · ".join(links), True


def build_video_index(manifest_paths: Sequence[Path], document_path: Path) -> dict:
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
        manifest_links.append(_link(manifest_path, document_path, manifest_path.parent.name))
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
        walking, walking_ok = _action_cell(
            record=record, action="Walking", document=document_path
        )
        idle, idle_ok = _action_cell(
            record=record, action="Idle", document=document_path
        )
        complete_clips += int(walking_ok) + int(idle_ok)
        complete_pairs += int(walking_ok and idle_ok)
        row = {
            "record": record,
            "source": source,
            "walking": walking,
            "idle": idle,
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
        f"- 更新时间：`{datetime.now(timezone.utc).isoformat()}`",
        f"- 动物实例：**{len(records)}**",
        f"- 已完成 clips：**{complete_clips} / {len(records) * 2}**",
        f"- 完整 Walk/Idle 对：**{complete_pairs} / {len(records)}**",
        f"- 输入 manifests：{' · '.join(manifest_links)}",
        "",
    ]
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_video_index(args.manifest, args.out)
    print(
        "CONTROLLED_ANIMAL_VIDEO_INDEX_OK "
        f"animals={summary['animal_count']} "
        f"clips={summary['completed_clip_count']}/{summary['clip_count']} "
        f"pairs={summary['complete_pair_count']}/{summary['animal_count']} "
        f"document={args.out.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
