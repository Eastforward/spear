#!/usr/bin/env python3
"""Strict v2-manifest adapter for the complete TokenRig human static audit.

This wrapper authenticates a lower-limb sanitation v2 publication, then delegates
unchanged geometry/PBR/render/static checks and owner-bundle publication to the
existing ``run_static_audit`` core.  It does not weaken or extend the v1 manifest
authenticator used by historical assets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_sanitize_tokenrig_lower_limb_weights_v2 as sanitizer
from tools import blender_tokenrig_human_static_audit as static_audit


class LowerLimbStaticAdapterError(RuntimeError):
    """The v2 sanitation publication cannot own a full static audit."""


def validate_static_canonical_lower_limb(
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate the world-space canonical positions emitted by the static core."""

    try:
        return sanitizer.validate_lower_limb_bilateral_contamination(
            canonical_world_positions=positions,
            vertex_weights=vertex_weights,
            chains=chains,
        )
    except sanitizer.LowerLimbSanitationError as error:
        raise static_audit.StaticAuditError(str(error)) from error


def _record(path: Path, description: str) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise LowerLimbStaticAdapterError(f"{description} must be a direct file")
    return {
        "path": str(path),
        "sha256": static_audit.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def authenticate_v2_static_inputs(
    *,
    asset_id: str,
    source_glb: Path,
    tokenrig_glb: Path,
    tokenrig_manifest: Path,
) -> dict[str, Any]:
    try:
        manifest = sanitizer.validate_published_manifest(tokenrig_manifest)
    except sanitizer.LowerLimbSanitationError as error:
        raise LowerLimbStaticAdapterError(
            f"lower-limb v2 manifest authentication failed: {error}"
        ) from error
    source = _record(source_glb, "source Pixal GLB")
    tokenrig = _record(tokenrig_glb, "lower-limb sanitized TokenRig GLB")
    manifest_record = _record(tokenrig_manifest, "lower-limb v2 manifest")
    if manifest.get("asset_id") != asset_id:
        raise LowerLimbStaticAdapterError("v2 manifest asset_id mismatch")
    manifest_source = manifest.get("input", {}).get("source_glb")
    manifest_output = manifest.get("output")
    if not isinstance(manifest_source, Mapping) or any(
        manifest_source.get(field) != source[field]
        for field in ("path", "sha256", "size_bytes")
    ):
        raise LowerLimbStaticAdapterError("source GLB differs from v2 source lineage")
    if not isinstance(manifest_output, Mapping) or any(
        manifest_output.get(field) != tokenrig[field]
        for field in ("path", "sha256", "size_bytes")
    ):
        raise LowerLimbStaticAdapterError("TokenRig GLB differs from v2 output lineage")
    if Path(tokenrig["path"]).parent != Path(manifest_record["path"]).parent:
        raise LowerLimbStaticAdapterError("v2 GLB and manifest must share one output root")
    changes = manifest.get("artifacts", {}).get("weight_changes")
    if not isinstance(changes, Mapping):
        raise LowerLimbStaticAdapterError("v2 change-log lineage is missing")
    adapter = _record(Path(__file__).resolve(), "v2 static adapter code")
    return {
        "asset_id": asset_id,
        "attempt": "lower_limb_weight_sanitation_v2",
        "manifest_schema": sanitizer.SCHEMA,
        "input_mode": manifest["input_mode"],
        "task3_gate_status": "failed",
        "recovered_candidate": False,
        "fitted_candidate": False,
        "sanitized_candidate": True,
        "lower_limb_sanitized_candidate": True,
        "source_glb_sha256": source["sha256"],
        "conditioning_glb_sha256": manifest["input"]["glb"]["sha256"],
        "conditioning_manifest_sha256": manifest["input"]["manifest"]["sha256"],
        "prior_static_failure_sha256": [
            record["sha256"] for record in manifest["input"]["prior_failures"]
        ],
        "tokenrig_glb_sha256": tokenrig["sha256"],
        "tokenrig_manifest_sha256": manifest_record["sha256"],
        "lower_limb_sanitizer_sha256": manifest["code"]["sanitizer_v2"]["sha256"],
        "static_audit_runner_sha256": static_audit.sha256_file(
            Path(static_audit.__file__).resolve()
        ),
        "static_audit_adapter_sha256": adapter["sha256"],
        "weight_changes_sha256": changes["sha256"],
        "source_front": static_audit.SOURCE_FRONT,
        "canonical_front": static_audit.CANONICAL_FRONT,
        "fresh_full_static_audit_required": True,
    }


def run_v2_static_audit(
    *,
    asset_id: str,
    source_glb: Path,
    tokenrig_glb: Path,
    tokenrig_manifest: Path,
    output_dir: Path,
) -> Path:
    authenticated = authenticate_v2_static_inputs(
        asset_id=asset_id,
        source_glb=source_glb,
        tokenrig_glb=tokenrig_glb,
        tokenrig_manifest=tokenrig_manifest,
    )
    original_authenticator = static_audit.authenticate_task3_inputs
    original_bilateral_validator = static_audit.validate_bilateral_contamination

    def adapter_authenticator(**kwargs: Any) -> dict[str, Any]:
        current = authenticate_v2_static_inputs(
            asset_id=kwargs["asset_id"],
            source_glb=kwargs["source_glb"],
            tokenrig_glb=kwargs["tokenrig_glb"],
            tokenrig_manifest=kwargs["tokenrig_manifest"],
        )
        if current != authenticated:
            raise LowerLimbStaticAdapterError(
                "v2 inputs changed between adapter and static-audit authentication"
            )
        return current

    static_audit.authenticate_task3_inputs = adapter_authenticator

    static_audit.validate_bilateral_contamination = (
        validate_static_canonical_lower_limb
    )
    try:
        return static_audit.run_static_audit(
            asset_id=asset_id,
            source_glb=source_glb,
            tokenrig_glb=tokenrig_glb,
            tokenrig_manifest=tokenrig_manifest,
            output_dir=output_dir,
        )
    finally:
        static_audit.authenticate_task3_inputs = original_authenticator
        static_audit.validate_bilateral_contamination = original_bilateral_validator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--tokenrig-glb", type=Path, required=True)
    parser.add_argument("--tokenrig-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_v2_static_audit(
        asset_id=args.asset_id,
        source_glb=args.source_glb,
        tokenrig_glb=args.tokenrig_glb,
        tokenrig_manifest=args.tokenrig_manifest,
        output_dir=args.output_dir,
    )
    print(f"TOKENRIG_LOWER_LIMB_STATIC_AUDIT_V2_OK {output}")
    return 0


if __name__ == "__main__":
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(blender_args))
