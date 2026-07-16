"""Freeze visual/static QA for the Pixal3D animal replacement batch.

Generation success is deliberately kept separate from dataset eligibility.  This
auditor records the immutable raw GLB evidence, four-view media, mesh/PBR counts,
and the first gate that each candidate is allowed to enter next.
"""
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
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from audit_mesh_efficiency import mesh_stats


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_ROOT = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/generated_batch_v1"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "static_qa_manifest.json"
DEFAULT_DOC = AVENGINE_ROOT / "docs/pixal_animal_static_qa.md"


# These are visual decisions over the four-view evidence, not model-generation
# return codes.  Obvious static failures must not silently proceed to binding.
DECISIONS = {
    "cat_british_shorthair_v2": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_cat",
        "reason": "complete quadruped silhouette, separated limbs, usable texture",
    },
    "cat_persian": {
        "status": "rejected",
        "rig_family": None,
        "reason": "planar/spiky fur geometry and fragmented silhouette",
    },
    "cat_siamese_v1": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_cat",
        "reason": "complete standing quadruped with separated limbs",
    },
    "cat_tabby": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_cat",
        "reason": "complete standing quadruped with preserved markings",
    },
    "cattle_bovinae": {
        "status": "rejected",
        "rig_family": None,
        "reason": "lying/cropped body fused with grass and background geometry",
    },
    "chipmunk": {
        "status": "rejected",
        "rig_family": None,
        "reason": "good appearance but seated/crouched rest pose is incompatible with the walking cat rig",
    },
    "dog_beagle_v2": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_dog",
        "reason": "complete standing quadruped with separated legs",
    },
    "dog_golden": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_dog",
        "reason": "complete quadruped; pale/faceted fur retained as a conditional animation candidate",
    },
    "donkey_ass": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_farm_horse",
        "reason": "complete standing ungulate; requires semantic farm-rig transfer validation",
    },
    "goat": {
        "status": "rejected",
        "rig_family": None,
        "reason": "severe polygon fragments and broken limb surfaces",
    },
    "horse": {
        "status": "continue_to_lod_and_rig",
        "rig_family": "quaternius_farm_horse",
        "reason": "complete standing ungulate; requires semantic farm-rig transfer validation",
    },
    "pig": {
        "status": "rejected",
        "rig_family": None,
        "reason": "body is fused with ground/background debris",
    },
    "sheep": {
        "status": "rejected",
        "rig_family": None,
        "reason": "incomplete bust/head reconstruction with no body or legs",
    },
    "yak": {
        "status": "rejected",
        "rig_family": None,
        "reason": "crystalline/fragmented legs and severely faceted body geometry",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _key_value_file(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = float(value)
    return values


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


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_report(root: Path) -> dict:
    batch_path = root / "batch_status.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    generated = {item["legacy_tag"]: item for item in batch["results"]}
    if set(generated) != set(DECISIONS):
        raise ValueError(
            f"decision coverage mismatch: generated={sorted(generated)} "
            f"decisions={sorted(DECISIONS)}"
        )

    rows = []
    for legacy_tag in sorted(generated):
        result = generated[legacy_tag]
        decision = DECISIONS[legacy_tag]
        raw = Path(result["output"])
        review_dir = raw.parent / "static_review_raw"
        render_manifest = review_dir / "render_manifest.json"
        contact_sheet = review_dir / "contact_sheet.png"
        required = [raw, render_manifest, contact_sheet]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{legacy_tag} missing static evidence: {missing}")
        stats = mesh_stats(raw)
        row = {
                "legacy_tag": legacy_tag,
                "candidate_tag": result["candidate_tag"],
                "generation_status": result["status"],
                "static_qa_status": decision["status"],
                "next_rig_family": decision["rig_family"],
                "reason": decision["reason"],
                "raw_mesh": {**stats, "sha256": _sha256(raw)},
                "front_axis": "negative-x",
                "render_manifest": _evidence(render_manifest),
                "contact_sheet": _evidence(contact_sheet),
                "registration_status": (
                    "rejected"
                    if decision["status"] == "rejected"
                    else "research_candidate"
                ),
                "formal_dataset_asset": False,
            }
        if decision["status"] == "continue_to_lod_and_rig":
            lod_dir = raw.parent / "runtime_lod"
            lod_mesh = lod_dir / "mesh_runtime_100000_double_sided.glb"
            lod_metadata = lod_dir / "mesh_runtime_100000_double_sided.json"
            lod_timing = lod_dir / "timing.txt"
            lod_review = raw.parent / "static_review_lod100k" / "contact_sheet.png"
            required_lod = [lod_mesh, lod_metadata, lod_timing, lod_review]
            missing_lod = [str(path) for path in required_lod if not path.is_file()]
            if missing_lod:
                raise FileNotFoundError(f"{legacy_tag} missing LOD evidence: {missing_lod}")
            lod_stats = mesh_stats(lod_mesh)
            row["runtime_lod"] = {
                **lod_stats,
                "sha256": _sha256(lod_mesh),
                "triangle_reduction_fraction": (
                    1.0 - lod_stats["triangles"] / stats["triangles"]
                ),
                "metadata": _evidence(lod_metadata),
                "timing": _key_value_file(lod_timing),
                "contact_sheet": _evidence(lod_review),
                "visual_status": "passed_no_visible_holes_or_missing_parts",
                "material_mode": "double_sided",
            }
        else:
            row["runtime_lod"] = None
        rows.append(row)
    counts = {
        "generated": len(rows),
        "continue_to_lod_and_rig": sum(
            row["static_qa_status"] == "continue_to_lod_and_rig" for row in rows
        ),
        "rejected": sum(row["static_qa_status"] == "rejected" for row in rows),
    }
    return {
        "schema": "pixal_animal_static_qa_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_manifest": _evidence(batch_path),
        "policy": (
            "generation_status does not authorize binding; rejected static candidates "
            "must not enter LOD, rig, animation, UE, or formal registration"
        ),
        "counts": counts,
        "candidates": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Pixal3D 动物替换静态 QA",
        "",
        "> Pixal3D 进程成功只代表生成了可回读 GLB；本表用四视图决定是否允许进入减面与绑定。",
        "",
        "| 动物 | 原始→运行时三角面 | 减面耗时 s | PBR | 静态结论 | 下一骨架 | 原因 | 四视图 |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in report["candidates"]:
        mesh = row["raw_mesh"]
        pbr = f"{mesh['materials']} material / {mesh['textures']} textures"
        lod = row["runtime_lod"]
        triangles = (
            f"{mesh['triangles']:,} → {lod['triangles']:,}"
            if lod
            else f"{mesh['triangles']:,} → —"
        )
        lod_seconds = f"{lod['timing']['wall_seconds']:.2f}" if lod else "—"
        review = (
            lod["contact_sheet"]["path"] if lod else row["contact_sheet"]["path"]
        )
        lines.append(
            f"| {row['legacy_tag']} | {triangles} | {lod_seconds} | {pbr} | "
            f"{row['static_qa_status']} | {row['next_rig_family'] or '—'} | "
            f"{row['reason']} | [contact]({review}) |"
        )
    counts = report["counts"]
    lines.extend(
        [
            "",
            f"结果：14/14 已生成并完成静态检查；{counts['continue_to_lod_and_rig']} 个进入减面/绑定，"
            f"{counts['rejected']} 个在动画前拒绝。所有继续项仍是 `research_candidate`，并未自动成为正式数据资产。",
            "",
            f"机器可读 manifest：[{DEFAULT_OUTPUT.name}]({DEFAULT_OUTPUT.resolve()})。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_report(args.root.resolve())
    _atomic_json(args.output.resolve(), report)
    _atomic_text(args.doc.resolve(), render_markdown(report))
    print(
        f"PIXAL_ANIMAL_STATIC_QA_OK output={args.output.resolve()} "
        f"continue={report['counts']['continue_to_lod_and_rig']} "
        f"rejected={report['counts']['rejected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
