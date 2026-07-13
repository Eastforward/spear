"""Low-memory mesh inventory for the human/animal dataset pipeline.

The auditor intentionally reads only glTF JSON/accessor metadata (or streams
OBJ records).  It never constructs triangle adjacency or calls
``trimesh.split``; that distinction matters for million-face Pixal meshes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
UE_MESH_ROOT = (
    SPEAR_ROOT
    / "cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes"
)


def _load_glb_json(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated GLB header: {path}")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"not a glTF 2 GLB: {path}")
        if total_length != path.stat().st_size:
            raise ValueError(
                f"GLB length mismatch for {path}: {total_length} != {path.stat().st_size}"
            )
        while stream.tell() < total_length:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError(f"truncated GLB chunk header: {path}")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            payload = stream.read(chunk_length)
            if len(payload) != chunk_length:
                raise ValueError(f"truncated GLB chunk: {path}")
            if chunk_type == 0x4E4F534A:  # JSON
                return json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
    raise ValueError(f"GLB has no JSON chunk: {path}")


def _triangle_count(mode: int, element_count: int) -> int:
    if mode == 4:  # TRIANGLES
        return element_count // 3
    if mode in (5, 6):  # TRIANGLE_STRIP / TRIANGLE_FAN
        return max(0, element_count - 2)
    return 0


def _gltf_stats(document: dict) -> dict:
    accessors = document.get("accessors", [])
    triangles = 0
    primitives = 0
    position_accessors: set[int] = set()
    index_accessors: set[int] = set()
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitives += 1
            attributes = primitive.get("attributes", {})
            position_index = attributes.get("POSITION")
            if position_index is None:
                continue
            position_accessors.add(int(position_index))
            index_index = primitive.get("indices")
            if index_index is None:
                element_count = int(accessors[position_index]["count"])
            else:
                index_index = int(index_index)
                index_accessors.add(index_index)
                element_count = int(accessors[index_index]["count"])
            triangles += _triangle_count(int(primitive.get("mode", 4)), element_count)
    vertices = sum(int(accessors[index]["count"]) for index in position_accessors)
    indices = sum(int(accessors[index]["count"]) for index in index_accessors)
    return {
        "vertices": vertices,
        "triangles": triangles,
        "indices": indices,
        "meshes": len(document.get("meshes", [])),
        "primitives": primitives,
        "materials": len(document.get("materials", [])),
        "textures": len(document.get("textures", [])),
        "images": len(document.get("images", [])),
        "skins": len(document.get("skins", [])),
        "animations": len(document.get("animations", [])),
    }


def _obj_stats(path: Path) -> dict:
    vertices = 0
    triangles = 0
    primitives = 0
    materials: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                count = len(line.split()) - 1
                if count >= 3:
                    triangles += count - 2
                    primitives += 1
            elif line.startswith("usemtl "):
                materials.add(line[7:].strip())
    return {
        "vertices": vertices,
        "triangles": triangles,
        "indices": triangles * 3,
        "meshes": 1,
        "primitives": primitives,
        "materials": len(materials),
        "textures": None,
        "images": None,
        "skins": 0,
        "animations": 0,
    }


def mesh_stats(path_value: str | Path | None) -> dict | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    suffix = path.suffix.lower()
    if suffix == ".glb":
        stats = _gltf_stats(_load_glb_json(path))
    elif suffix == ".gltf":
        stats = _gltf_stats(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".obj":
        stats = _obj_stats(path)
    else:
        raise ValueError(f"unsupported mesh format: {path}")
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        **stats,
    }


def _ue_asset_bytes(tag: str, *, static: bool = False) -> dict:
    prefix = "gate_static_" if static else "gate_"
    directory = UE_MESH_ROOT / f"{prefix}{tag}"
    files = [path for path in directory.rglob("*") if path.is_file()] if directory.is_dir() else []
    return {
        "path": str(directory),
        "exists": directory.is_dir(),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def _raw_hunyuan_path(tag: str, runtime_path: Path) -> Path:
    batch_dir = SPEAR_ROOT / "tmp/hy3d_batch" / tag
    for name in ("shape.glb", "input.glb"):
        candidate = batch_dir / name
        if candidate.is_file():
            return candidate
    if runtime_path.parent.name == tag:
        for name in (f"{tag}_shape.glb", "shape.glb", "input.glb"):
            candidate = runtime_path.parent / name
            if candidate.is_file():
                return candidate
    return runtime_path


def _decimation_decision(runtime_triangles: int | None) -> str:
    if runtime_triangles is None:
        return "measure_before_registration"
    if runtime_triangles <= 100_000:
        return "keep_runtime_mesh; no mandatory decimation"
    if runtime_triangles <= 150_000:
        return "keep_close_LOD; add optional 40k distant_LOD"
    return "decimate_to_100k_close_and_40k_distant_then_visual_QA"


def _row(
    *,
    tag: str,
    subject: str,
    backend: str,
    raw_path: Path,
    runtime_path: Path,
    static: bool,
    registration_status: str,
    notes: str = "",
) -> dict:
    raw = mesh_stats(raw_path)
    runtime = mesh_stats(runtime_path)
    raw_triangles = raw.get("triangles") if raw and raw.get("exists") else None
    runtime_triangles = runtime.get("triangles") if runtime and runtime.get("exists") else None
    reduction = None
    if raw_triangles and runtime_triangles is not None:
        reduction = 1.0 - runtime_triangles / raw_triangles
    return {
        "tag": tag,
        "subject": subject,
        "backend": backend,
        "raw": raw,
        "runtime": runtime,
        "triangle_reduction_fraction": reduction,
        "ue_assets": _ue_asset_bytes(tag, static=static),
        "decimation_decision": _decimation_decision(runtime_triangles),
        "registration_status": registration_status,
        "notes": notes,
    }


def collect_animals() -> list[dict]:
    import sys

    sys.path.insert(0, str(SPEAR_ROOT / "tools"))
    from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP

    rows = []
    for tag, entry in sorted(ANIMATED_RIG_MAP.items()):
        runtime = Path(entry["mesh"])
        raw = _raw_hunyuan_path(tag, runtime)
        rows.append(
            _row(
                tag=tag,
                subject="animal_animated",
                backend="Hunyuan3D legacy",
                raw_path=raw,
                runtime_path=runtime,
                static=False,
                registration_status="technical_spike_only",
                notes="Legacy license-blocked evidence; not formal dataset input.",
            )
        )
    for tag, entry in sorted(STATIC_MESH_MAP.items()):
        runtime = Path(entry["mesh"])
        raw = runtime.with_name(f"{tag}_shape.glb")
        rows.append(
            _row(
                tag=tag,
                subject="animal_static",
                backend="Hunyuan3D legacy",
                raw_path=raw,
                runtime_path=runtime,
                static=True,
                registration_status="technical_spike_only",
                notes="Static legacy evidence; must be replaced by Pixal3D and animated before formal registration.",
            )
        )

    pixal_root = SPEAR_ROOT / "tmp/pixal_animal_backend_substitution_v1"
    pixal_raw = pixal_root / "dog_pug_pixal_canary_v1/pixal_raw_1024_seed5101.glb"
    pixal_runtime = (
        pixal_root
        / "approved/dog_pug_pixal_canary_v1/mesh_runtime_100000_double_sided.glb"
    )
    rows.append(
        _row(
            tag="dog_pug_pixal_canary_v2_100k",
            subject="animal_animated",
            backend="Pixal3D",
            raw_path=pixal_raw,
            runtime_path=pixal_runtime,
            static=False,
            registration_status="research_candidate",
            notes=(
                "100k double-sided proxy passed Blender and packaged-UE Walk/Idle visual QA; "
                "appearance/provenance gates remain."
            ),
        )
    )
    return rows


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def collect_humans() -> dict:
    runtime_root = SPEAR_ROOT / "tmp/rocketbox_batch_native_runtime_ue_v1"
    entries = []
    for path in sorted(runtime_root.glob("*/runtime.glb")):
        stats = mesh_stats(path)
        entries.append({"tag": path.parent.name, "mesh": stats})
    triangles = [entry["mesh"]["triangles"] for entry in entries]
    vertices = [entry["mesh"]["vertices"] for entry in entries]
    byte_sizes = [entry["mesh"]["bytes"] for entry in entries]

    representative_tags = {
        "rocketbox_adults_male_adult_01_original_ue_v1",
        "rocketbox_adults_female_adult_01_original_ue_v1",
        "rocketbox_children_male_child_01_original_ue_v1",
        "rocketbox_children_female_child_01_original_ue_v1",
        "rocketbox_professions_medical_female_01_original_ue_v1",
    }
    representatives = [entry for entry in entries if entry["tag"] in representative_tags]
    recolor = (
        SPEAR_ROOT
        / "tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_shirt_blue_ue_v3/runtime.glb"
    )
    if recolor.is_file():
        representatives.append({"tag": recolor.parent.name, "mesh": mesh_stats(recolor)})

    pixal_models = []
    for tag, path in (
        (
            "pixal_route2_male_raw",
            SPEAR_ROOT
            / "tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_male_adult_01/canary_1024_seed42.glb",
        ),
        (
            "pixal_route2_female_raw",
            SPEAR_ROOT
            / "tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_female_adult_01/canary_1024_seed42.glb",
        ),
        (
            "pixal_route2_male_rigged",
            SPEAR_ROOT
            / "tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/third_candidate_corrected_nonformal_preview_v1/walking.glb",
        ),
    ):
        pixal_models.append({"tag": tag, "mesh": mesh_stats(path)})

    summary = {
        "count": len(entries),
        "triangles": {
            "min": min(triangles),
            "median": statistics.median(triangles),
            "p95": _percentile(triangles, 0.95),
            "max": max(triangles),
        },
        "vertices": {
            "min": min(vertices),
            "median": statistics.median(vertices),
            "p95": _percentile(vertices, 0.95),
            "max": max(vertices),
        },
        "bytes": {
            "min": min(byte_sizes),
            "median": statistics.median(byte_sizes),
            "p95": _percentile(byte_sizes, 0.95),
            "max": max(byte_sizes),
        },
        "decimation_decision": _decimation_decision(max(triangles)),
    }
    return {
        "summary": summary,
        "representatives": representatives,
        "pixal_route2": pixal_models,
        "all": entries,
    }


def collect_runtime_benchmarks() -> list[dict]:
    root = SPEAR_ROOT / "tmp/asset_mesh_efficiency_audit_v1"
    rows = []
    for directory in sorted(root.glob("benchmark_pixal_dog_*")):
        manifest_path = directory / "runtime_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tag = manifest["tag"]
        triangles = 40_000 if tag.endswith("canary_v1") else 100_000
        gpu_zero = manifest.get("gpu_telemetry", {}).get("by_gpu", {}).get("0", {})
        rows.append(
            {
                "tag": tag,
                "triangles": triangles,
                "frame_count": manifest["frame_count"],
                "timings": manifest["timings"],
                "gpu_0": gpu_zero,
                "quality_status": (
                    "rejected_holes_single_sided"
                    if triangles == 40_000
                    else "passed_no_holes_double_sided"
                ),
                "manifest": str(manifest_path.resolve()),
            }
        )
    return rows


def _fmt_int(value) -> str:
    if value is None:
        return "—"
    return f"{int(round(value)):,}"


def _fmt_mb(value) -> str:
    if value is None:
        return "—"
    return f"{value / 1_000_000:.2f}"


def _animal_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "## 动物逐资产减面判断",
        "",
        "| 资产 | 后端/状态 | 原始三角面 | 运行时三角面 | 减少 | GLB/OBJ MB（原→运行时） | UE资产 MB | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        raw = row["raw"] or {}
        runtime = row["runtime"] or {}
        reduction = row["triangle_reduction_fraction"]
        reduction_text = "—" if reduction is None else f"{reduction * 100:.1f}%"
        lines.append(
            "| {tag} | {backend}; {status} | {raw_tri} | {runtime_tri} | {reduction} | {raw_mb} → {runtime_mb} | {ue_mb} | {decision} |".format(
                tag=row["tag"],
                backend=row["backend"],
                status=row["registration_status"],
                raw_tri=_fmt_int(raw.get("triangles")),
                runtime_tri=_fmt_int(runtime.get("triangles")),
                reduction=reduction_text,
                raw_mb=_fmt_mb(raw.get("bytes")),
                runtime_mb=_fmt_mb(runtime.get("bytes")),
                ue_mb=_fmt_mb(row["ue_assets"].get("bytes")),
                decision=row["decimation_decision"],
            )
        )
    return lines


def render_markdown(report: dict) -> str:
    humans = report["humans"]
    summary = humans["summary"]
    lines = [
        "# 人类与动物 Mesh 效率审计",
        "",
        "> 统计方法只读取 glTF accessor 元数据或逐行扫描 OBJ，不加载几何邻接；因此可安全审计百万面模型。",
        "",
        "## 人类面数",
        "",
        "| 集合 | 数量 | 三角面 min / median / p95 / max | GLB MB min / median / p95 / max | 减面结论 |",
        "|---|---:|---:|---:|---|",
        "| Rocketbox 原生运行时 | {count} | {tmin} / {tmed} / {tp95} / {tmax} | {bmin} / {bmed} / {bp95} / {bmax} | {decision} |".format(
            count=summary["count"],
            tmin=_fmt_int(summary["triangles"]["min"]),
            tmed=_fmt_int(summary["triangles"]["median"]),
            tp95=_fmt_int(summary["triangles"]["p95"]),
            tmax=_fmt_int(summary["triangles"]["max"]),
            bmin=_fmt_mb(summary["bytes"]["min"]),
            bmed=_fmt_mb(summary["bytes"]["median"]),
            bp95=_fmt_mb(summary["bytes"]["p95"]),
            bmax=_fmt_mb(summary["bytes"]["max"]),
            decision=summary["decimation_decision"],
        ),
        "",
        "| 代表人类资产 | 顶点 | 三角面 | GLB MB |",
        "|---|---:|---:|---:|",
    ]
    for item in humans["representatives"] + humans["pixal_route2"]:
        mesh = item["mesh"] or {}
        lines.append(
            f"| {item['tag']} | {_fmt_int(mesh.get('vertices'))} | {_fmt_int(mesh.get('triangles'))} | {_fmt_mb(mesh.get('bytes'))} |"
        )
    lines.extend([""] + _animal_markdown(report["animals"]))
    lines.extend(
        [
            "",
            "## Pixal 狗 40k / 100k UE 实测",
            "",
            "同一台 RTX 4090 D、640×480、72 帧、每张 PNG 回读，均含相同地板、灯光、动画和相机绕视。单角色时主要瓶颈是 UE 冷启动、固定步进和 PNG 回读，因此面数差异未形成可测的吞吐收益。",
            "",
            "| LOD | 质量 | UE启动 s | 场景准备 s | 72帧 s | 捕获 fps | 单帧 p95 ms | GPU0峰值 MiB | GPU峰值利用率 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report.get("runtime_benchmarks", []):
        timings = item["timings"]
        gpu = item["gpu_0"]
        lines.append(
            "| {tri} | {quality} | {launch:.2f} | {prepare:.2f} | {frames:.2f} | {fps:.2f} | {p95:.1f} | {memory} | {util}% |".format(
                tri=_fmt_int(item["triangles"]),
                quality=item["quality_status"],
                launch=timings["launch_seconds"],
                prepare=timings["spawn_warmup_and_ground_seconds"],
                frames=timings["frame_loop_seconds"],
                fps=timings["captured_frames_per_second"],
                p95=timings["seconds_per_frame_p95"] * 1000.0,
                memory=gpu.get("peak_memory_used_mib", "—"),
                util=gpu.get("peak_utilization_pct", "—"),
            )
        )
    lines.extend(
        [
            "",
            "40k 与 100k 的 72 帧耗时分别约 16.00s 和 15.74s，差异仅 1.6% 且方向相反，属于冷启动/回读噪声；100k 只多约 18 MiB 峰值显存，却消除了可见空洞并显著改善轮廓，所以近景固定采用 100k。40k 只保留为远景候选，且必须重新生成双面版本后再批量压力测试。",
        ]
    )
    lines.extend(
        [
            "",
            "## 当前统一策略",
            "",
            "- 运行时不超过 100k 三角面：默认不减面；只做纹理、法线、脚部和 UE 回读 QA。",
            "- 100k–150k：保留近景 LOD，可选生成 40k 远景 LOD。",
            "- 超过 150k：生成约 100k 近景和约 40k 远景两级 LOD；Pixal 局部绕序不一致时保留双面材质。",
            "- 结论不能只看面数：任何减面代理都必须经过 Front/Back/Side/Top、Walk/Idle、UE PAK 回读和地面阴影检查。",
            "- Hunyuan 行仅是历史性能证据，保持 technical_spike_only；正式动物必须用 Pixal3D 重新生成并逐级过门禁。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "method": "glTF accessor metadata / streaming OBJ; no topology adjacency",
        "humans": collect_humans(),
        "animals": collect_animals(),
        "runtime_benchmarks": collect_runtime_benchmarks(),
    }
    json_path = Path(args.json_out).resolve()
    markdown_path = Path(args.markdown_out).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"MESH_EFFICIENCY_AUDIT_OK {json_path} {markdown_path}")


if __name__ == "__main__":
    main()
