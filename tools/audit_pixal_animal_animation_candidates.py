"""Freeze Walk/Idle QA and emit the non-destructive Pixal UE import batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.audit_mesh_efficiency import mesh_stats
except ModuleNotFoundError:
    from audit_mesh_efficiency import mesh_stats


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_ROOT = SPEAR_ROOT / "tmp/pixal_animal_backend_substitution_v1/generated_batch_v1"
DEFAULT_OUTPUT = DEFAULT_ROOT / "animation_qa_manifest.json"
DEFAULT_IMPORTS = DEFAULT_ROOT / "ue_import_jobs.json"
DEFAULT_DOC = AVENGINE_ROOT / "docs/pixal_animal_animation_qa.md"


DECISIONS = {
    "cat_british_shorthair_v2": {
        "status": "rejected",
        "reason": "severe foreleg folding/stretching and detached mesh islands during Walking",
    },
    "cat_siamese_v1": {
        "status": "continue_to_ue",
        "reason": "stable torso and recognizable gait; minor paw artifacts accepted for UE canary",
    },
    "cat_tabby": {
        "status": "continue_to_ue",
        "reason": "stable body, continuous limbs, and recognizable Walk/Idle",
    },
    "dog_beagle_v2": {
        "status": "continue_to_ue",
        "reason": "continuous quadruped gait; unusually long tail belongs to the Pixal reconstruction, not a rig-direction failure",
    },
    "dog_golden": {
        "status": "continue_to_ue",
        "reason": "stable body and plausible Walk/Idle with only minor fur-edge artifacts",
    },
    "donkey_ass": {
        "status": "rejected",
        "reason": "farm-axis fix exports correctly but Walking still stretches hind feet and leaves detached foot fragments",
    },
    "horse": {
        "status": "rejected",
        "reason": "uniform/nonuniform and proximity/skeleton-graph fallbacks all retain crossed forelegs or missing/misassigned hind-leg deformation",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _timing(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = float(value)
    return result


def _paths(root: Path, tag: str) -> tuple[Path, Path, Path]:
    candidate = root / f"{tag}_pixal_v1"
    if tag in {"donkey_ass", "horse"}:
        rigged = candidate / "rigged/animated_100000_double_sided_canonical.glb"
        timing = candidate / "rigged/timing_walk_idle_axisfix.txt"
        review = candidate / "animation_review_axisfix"
    else:
        rigged = candidate / "rigged/animated_100000_double_sided.glb"
        timing = candidate / "rigged/timing.txt"
        review = candidate / "animation_review"
    return rigged, timing, review


def build_report(root: Path) -> tuple[dict, dict]:
    static = json.loads((root / "static_qa_manifest.json").read_text(encoding="utf-8"))
    continued = {
        row["legacy_tag"]
        for row in static["candidates"]
        if row["static_qa_status"] == "continue_to_lod_and_rig"
    }
    if continued != set(DECISIONS):
        raise ValueError(f"animation decision coverage mismatch: {sorted(continued)}")
    rows = []
    import_jobs = []
    for tag in sorted(DECISIONS):
        decision = DECISIONS[tag]
        rigged, timing_path, review = _paths(root, tag)
        stats = mesh_stats(rigged)
        media = {
            name: _record(review / f"{name}.mp4")
            for name in ("walking_side", "walking_front", "idle_side")
        }
        media["walk_contact_sheet"] = _record(review / "walk_contact_sheet.png")
        row = {
            "legacy_tag": tag,
            "animation_qa_status": decision["status"],
            "reason": decision["reason"],
            "rigged_glb": {**stats, "sha256": _sha256(rigged)},
            "binding_timing": _timing(timing_path),
            "media": media,
            "registration_status": (
                "research_candidate" if decision["status"] == "continue_to_ue" else "rejected"
            ),
            "formal_dataset_asset": False,
        }
        if tag == "horse":
            row["fallback_evidence"] = {
                name: _record(
                    root / "horse_pixal_v1/rigged_fallbacks" / name / "walk_contact_sheet.png"
                )
                for name in ("graph_uniform", "proximity_nonuniform", "graph_nonuniform")
            }
        rows.append(row)
        if decision["status"] == "continue_to_ue":
            ue_dir = rigged.parent.parent / "ue_compatible"
            ue_glb = ue_dir / "animated_100000_double_sided_png.glb"
            transcode_manifest = ue_dir / "ue_texture_transcode_manifest.json"
            ue_stats = mesh_stats(ue_glb)
            transcode = json.loads(transcode_manifest.read_text(encoding="utf-8"))
            if (
                transcode["input"]["sha256"] != row["rigged_glb"]["sha256"]
                or transcode["output"]["sha256"] != _sha256(ue_glb)
                or transcode["geometry_skin_animation_byte_graph_changed"] is not False
            ):
                raise ValueError(f"UE texture transcode lineage mismatch for {tag}")
            row["ue_compatible_glb"] = {
                **ue_stats,
                "sha256": _sha256(ue_glb),
                "transcode_manifest": _record(transcode_manifest),
                "transcode_timing": _timing(ue_dir / "timing.txt"),
                "texture_policy": "embedded WebP losslessly transcoded to core glTF PNG",
            }
            ue_tag = f"pixal_{tag}"
            import_jobs.append(
                {
                    "tag": ue_tag,
                    "legacy_tag": tag,
                    "rigged_glb": str(ue_glb.resolve()),
                    "rigged_glb_sha256": _sha256(ue_glb),
                    "upstream_rigged_glb": str(rigged.resolve()),
                    "upstream_rigged_glb_sha256": _sha256(rigged),
                    "texture_transcode_manifest": str(transcode_manifest.resolve()),
                    "expected_actions": ["Idle", "Walking"],
                }
            )
        else:
            row["ue_compatible_glb"] = None
    report = {
        "schema": "pixal_animal_animation_qa_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "minor sliding or fur-edge artifacts may continue; folding, missing, detached, or grossly stretched limbs are rejected before UE",
        "counts": {
            "tested": len(rows),
            "continue_to_ue": len(import_jobs),
            "rejected": len(rows) - len(import_jobs),
        },
        "candidates": rows,
        "preexisting_pixal_pug": "already passed packaged-UE Walk/Idle and is not regenerated here",
    }
    imports = {
        "schema": "pixal_animal_ue_import_batch_v1",
        "generated_at": report["generated_at"],
        "non_destructive_policy": "unique gate_pixal_* content directories; do not replace legacy Hunyuan assets",
        "jobs": import_jobs,
    }
    return report, imports


def render_markdown(report: dict) -> str:
    lines = [
        "# Pixal3D 动物 Walk / Idle 绑定 QA",
        "",
        "| 动物 | 绑定 s | 动作/skin | 动画结论 | 原因 | Walk 视频 | Idle 视频 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in report["candidates"]:
        glb = row["rigged_glb"]
        lines.append(
            f"| {row['legacy_tag']} | {row['binding_timing']['wall_seconds']:.2f} | "
            f"{glb['animations']} actions / {glb['skins']} skin | {row['animation_qa_status']} | "
            f"{row['reason']} | [Walking]({row['media']['walking_side']['path']}) | "
            f"[Idle]({row['media']['idle_side']['path']}) |"
        )
    counts = report["counts"]
    lines.extend(
        [
            "",
            f"结果：{counts['tested']} 个静态通过项全部完成绑定测试；{counts['continue_to_ue']} 个进入 UE，"
            f"{counts['rejected']} 个因明显肢体错误拒绝。此前 Pixal pug 已单独通过完整 UE 门禁。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imports", type=Path, default=DEFAULT_IMPORTS)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report, imports = build_report(args.root.resolve())
    _atomic_json(args.output.resolve(), report)
    _atomic_json(args.imports.resolve(), imports)
    _atomic_text(args.doc.resolve(), render_markdown(report))
    print(
        f"PIXAL_ANIMAL_ANIMATION_QA_OK tested={report['counts']['tested']} "
        f"continue={report['counts']['continue_to_ue']} rejected={report['counts']['rejected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
