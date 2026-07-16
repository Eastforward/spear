"""Aggregate measured AVEngine stage timings and write scaling tables."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_OUTPUT = SPEAR_ROOT / "tmp/pipeline_timing_audit_v1/report.json"
DEFAULT_DOC = AVENGINE_ROOT / "docs/pipeline_timing_and_scaling_audit.md"


def _atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def summarize(values):
    values = [float(value) for value in values if float(value) >= 0.0]
    if not values:
        return None
    return {
        "count": len(values),
        "min_seconds": min(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
        "values_seconds": values,
    }


def _successful_stage_durations(command_log: Path, stage: str):
    from datetime import datetime

    starts = []
    durations = []
    try:
        lines = command_log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("stage") != stage:
            continue
        if record.get("event") == "start":
            starts.append(record)
        elif record.get("event") == "finish" and record.get("status") == "passed":
            matching = next(
                (
                    item
                    for item in reversed(starts)
                    if item.get("started_at") == record.get("started_at")
                ),
                None,
            )
            if matching is None:
                continue
            durations.append(
                (
                    datetime.fromisoformat(record["finished_at"])
                    - datetime.fromisoformat(record["started_at"])
                ).total_seconds()
            )
    return durations


def collect_human_timings():
    root = SPEAR_ROOT / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/clips"
    render = []
    finalize = []
    audio = []
    for log in sorted(root.glob("*/*/command.log")):
        render.extend(_successful_stage_durations(log, "render"))
        finalize.extend(_successful_stage_durations(log, "finalize")[-1:])
        console = log.parent / "batch_console.log"
        if console.is_file():
            matches = re.findall(
                r"\[rlr\] TOTAL wall time: ([0-9.]+)s",
                console.read_text(encoding="utf-8", errors="ignore"),
            )
            if matches:
                audio.append(float(matches[-1]))
    return {"ue_render": render, "rlr_audio": audio, "finalize": finalize}


def collect_pixal_timings():
    root = SPEAR_ROOT / "tmp/pixal_animal_backend_substitution_v1/generated_batch_v1"
    timings = []
    manifests = []
    for path in sorted(root.glob("*_pixal_v1/*.manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        timing = payload.get("timings", {})
        value = timing.get("inference_seconds")
        if value is None:
            value = timing.get("inference_and_export_seconds")
        if value is not None:
            timings.append(float(value))
            manifests.append(str(path.resolve()))
    return timings, manifests


def _hours(count, service_seconds):
    return float(count) * float(service_seconds) / 3600.0


def build_report():
    human = collect_human_timings()
    pixal_values, pixal_manifests = collect_pixal_timings()
    stages = {
        "safe_mesh_inventory": {
            "summary": summarize([2.39]),
            "resource": "CPU/read-only",
            "unit": "115 humans + legacy animals per audit",
            "evidence": str(
                (SPEAR_ROOT / "tmp/asset_mesh_efficiency_audit_v1/report.json").resolve()
            ),
        },
        "pixal3d_cold_per_asset": {
            "summary": summarize(pixal_values),
            "resource": "1 GPU + heavy CPU/storage model load and GLB extraction",
            "unit": "one 1024 animal asset",
            "evidence": pixal_manifests,
        },
        "pixal_raw_to_100k_double_sided": {
            "summary": summarize([20.0]),
            "resource": "CPU/Blender",
            "unit": "one approximately 931k-triangle animal",
            "evidence": str(
                (
                    SPEAR_ROOT
                    / "tmp/pixal_animal_backend_substitution_v1/approved/"
                    "dog_pug_pixal_canary_v1/mesh_runtime_100000_double_sided.log"
                ).resolve()
            ),
        },
        "animal_weight_transfer_and_glb": {
            "summary": summarize([164.0]),
            "resource": "CPU/Blender",
            "unit": "one 100k-face/300k-vertex animal",
            "evidence": str(
                (
                    SPEAR_ROOT
                    / "tmp/pixal_animal_backend_substitution_v1/rigged/"
                    "dog_pug_pixal_canary_v1_rigged_v4_flipx_100k_double_sided.log"
                ).resolve()
            ),
        },
        "incremental_ue_cook_and_pak": {
            "summary": summarize([188.52]),
            "resource": "CPU/storage; shared batch cost",
            "unit": "one incremental package containing the Pixal pug",
            "evidence": str(
                (SPEAR_ROOT / "tmp/pixal_animal_backend_substitution_v1/ue_pak_entries.txt").resolve()
            ),
        },
        "apartment_ue_render_18s_270_frames": {
            "summary": summarize(human["ue_render"]),
            "resource": "1 GPU; fixed stepping + PNG readback",
            "unit": "one 18-second/270-frame human clip",
            "evidence": str(
                (
                    SPEAR_ROOT
                    / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2"
                ).resolve()
            ),
        },
        "rlr_binaural_18s_270_positions": {
            "summary": summarize(human["rlr_audio"]),
            "resource": "CPU + small headless graphics context",
            "unit": "one 18-second single-source clip",
            "evidence": str(
                (
                    SPEAR_ROOT
                    / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2"
                ).resolve()
            ),
        },
        "topdown_metadata_and_review_finalize": {
            "summary": summarize(human["finalize"]),
            "resource": "CPU/FFmpeg/Matplotlib",
            "unit": "one 18-second/270-frame clip",
            "evidence": str(
                (
                    SPEAR_ROOT
                    / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2"
                ).resolve()
            ),
        },
    }
    ue = stages["apartment_ue_render_18s_270_frames"]["summary"]
    audio = stages["rlr_binaural_18s_270_positions"]["summary"]
    finalize = stages["topdown_metadata_and_review_finalize"]["summary"]
    pixal = stages["pixal3d_cold_per_asset"]["summary"]
    scaling = {}
    if ue and audio and finalize:
        current_service = max(
            ue["median_seconds"] / 4.0,
            audio["median_seconds"] / 4.0,
            finalize["median_seconds"] / 6.0,
        )
        optimized_service = max(
            ue["median_seconds"] / 4.0,
            audio["median_seconds"] / 8.0,
            finalize["median_seconds"] / 16.0,
        )
        scaling["route1_existing_asset_clips"] = {
            "current_slots": {"ue_gpu": 4, "audio": 4, "finalize": 6},
            "steady_state_seconds_per_clip": current_service,
            "clips_per_hour": 3600.0 / current_service,
            "100_clips_hours": _hours(100, current_service),
            "1000_clips_hours": _hours(1000, current_service),
            "higher_cpu_parallelism_seconds_per_clip": optimized_service,
            "higher_cpu_parallelism_clips_per_hour": 3600.0 / optimized_service,
            "bottleneck": (
                "CPU review finalization at six workers"
                if current_service == finalize["median_seconds"] / 6.0
                else "UE capture"
            ),
        }
    if pixal:
        service = pixal["median_seconds"] / 4.0
        scaling["new_pixal_assets_cold_runner"] = {
            "gpu_slots": 4,
            "steady_state_seconds_per_asset": service,
            "assets_per_hour": 3600.0 / service,
            "100_assets_hours": _hours(100, service),
            "1000_assets_hours": _hours(1000, service),
            "bottleneck": "repeated Pixal model load plus inference/GLB extraction",
            "persistent_worker_target": (
                "load the pinned model once per GPU; pending complete-batch measured result"
            ),
        }
    return {
        "schema": "avengine_pipeline_timing_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_policy": (
            "successful artifact-producing attempts only; failed diagnostic retries excluded"
        ),
        "stages": stages,
        "scaling": scaling,
    }


def _fmt(value):
    return "—" if value is None else f"{float(value):.2f}"


def render_markdown(report):
    lines = [
        "# AVEngine 全流程耗时与规模化瓶颈",
        "",
        "> 所有表格只统计成功并产生可回读产物的运行；方向诊断失败、窗口化副卡启动失败等重试不计入正常吞吐。",
        "",
        "## 实测阶段耗时",
        "",
        "| 阶段 | 资源 | 计量单位 | 样本数 | median s | p95 s | 大规模判断 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    judgments = {
        "safe_mesh_inventory": "可忽略；不要换成邻接拓扑审计",
        "pixal3d_cold_per_asset": "当前最大单资产瓶颈",
        "pixal_raw_to_100k_double_sided": "较小，可与其他资产并行",
        "animal_weight_transfer_and_glb": "第二大资产准备阶段，适合 CPU 多进程",
        "incremental_ue_cook_and_pak": "批次共享成本，不应逐资产 cook",
        "apartment_ue_render_18s_270_frames": "4 卡 offscreen 可线性并行",
        "rlr_binaural_18s_270_positions": "占比很小",
        "topdown_metadata_and_review_finalize": "单任务比 UE 慢；需独立 CPU 池",
    }
    for name, stage in report["stages"].items():
        summary = stage["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    stage["resource"],
                    stage["unit"],
                    str(summary["count"] if summary else 0),
                    _fmt(summary["median_seconds"] if summary else None),
                    _fmt(summary["p95_seconds"] if summary else None),
                    judgments[name],
                ]
            )
            + " |"
        )
    lines.extend(["", "## 规模化估算", ""])
    route1 = report["scaling"].get("route1_existing_asset_clips")
    pixal = report["scaling"].get("new_pixal_assets_cold_runner")
    lines.extend(
        [
            "| 场景 | 并发假设 | 稳态吞吐 | 100 单位 | 1000 单位 | 当前瓶颈 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if route1:
        lines.append(
            f"| 已导入 Rocketbox/Pixal 资产生成审核 clip | 4 UE GPU + 4 audio + 6 finalize | "
            f"{route1['clips_per_hour']:.1f} clips/h | "
            f"{route1['100_clips_hours']:.2f} h | {route1['1000_clips_hours']:.2f} h | "
            f"{route1['bottleneck']} |"
        )
    if pixal:
        lines.append(
            f"| 新 Pixal3D 动物资产（冷启动 runner） | 4 GPU | "
            f"{pixal['assets_per_hour']:.1f} assets/h | "
            f"{pixal['100_assets_hours']:.2f} h | {pixal['1000_assets_hours']:.2f} h | "
            f"{pixal['bottleneck']} |"
        )
    lines.extend(
        [
            "",
            "## 结论与优化顺序",
            "",
            "1. Pixal3D 冷启动是第一瓶颈。每张 GPU 必须使用 persistent worker，只加载一次固定 revision；当前冷启动 runner 保留为对照证据。",
            "2. 审核视频 finalization 是 clip 生产的主要 CPU 瓶颈。它应与 UE GPU 槽完全解耦，并缓存静态 Top-down 背景；增加 CPU worker 前先监控内存和磁盘写入。",
            "3. UE 单演员 40k 与 100k 的捕获耗时差异落在噪声内，因此近景使用无空洞的 100k；减面主要降低多演员显存/绘制压力，不会解决固定步进和 PNG 回读。",
            "4. UE cook/package 必须按资产批次执行，不能逐角色重复。RLR 音频只有约数秒，不值得牺牲事件语义或空间同步来换速度。",
            "5. 规模化前再做 8/16/32 同场演员压力测试；当前 40k/100k 结论只覆盖单演员，不能外推到密集场景。",
            "",
            "机器可读报告：[report.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/pipeline_timing_audit_v1/report.json)。Mesh 证据见 [asset_mesh_efficiency_audit.md](/data/jzy/code/AVEngine/docs/asset_mesh_efficiency_audit.md)。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report()
    _atomic_json(args.output.resolve(), report)
    _atomic_text(args.doc.resolve(), render_markdown(report))
    print(f"PIPELINE_TIMING_AUDIT_OK output={args.output.resolve()} doc={args.doc.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
