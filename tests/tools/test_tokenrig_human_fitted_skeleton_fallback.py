"""CPU/source contracts for the required fitted-skeleton TokenRig fallback."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


fallback = importlib.import_module("tools.tokenrig_human_fitted_skeleton_fallback")


def test_pins_direct_derivative_recovery_failures_and_new_output():
    contract = fallback.PINNED_FALLBACK

    assert contract.asset_id == "rocketbox_male_adult_01"
    assert contract.input_glb.name == "tokenrig_transfer.glb"
    assert contract.input_glb_sha256 == (
        "8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c"
    )
    assert contract.input_glb_size == 50_843_552
    assert contract.recovery_manifest_sha256 == (
        "cae4aac8f6472b893ce695173ad9a1766ef8f2ecf86cbbc7a80440b2ad949e96"
    )
    assert contract.recovery_manifest_size == 5_071
    assert contract.output_dir.name == "fitted_skeleton_v1"
    assert contract.output_dir.parent.name == contract.asset_id
    assert [(item.sha256, item.size_bytes) for item in contract.static_failures] == [
        (
            "0eab61c2dfcb5a7fe0a05ee8f5109c60a9c756a6ae048a0545df6d9e64c590ce",
            1_127,
        ),
        (
            "c78d5f1e4d7c127781f8b24b5a9bed906f96b5034d3b5bda9509c23c8f492e26",
            1_154,
        ),
    ]


def test_static_failure_payloads_must_prove_both_direct_rejections():
    records = fallback.validate_static_failure_payloads(
        (
            {
                "decision": "rejected",
                "readiness_bundle_published": False,
                "failure": {"message": "raw GLB triangle count changed: source=976970 output=976951"},
            },
            {
                "decision": "rejected",
                "readiness_bundle_published": False,
                "failure": {
                    "message": "opposite-limb contamination on distal vertices: count=55593 maximum=0.0778309777379036"
                },
            },
        )
    )

    assert records == {
        "strict_topology_attempt": "rejected",
        "bounded_skin_attempt": "rejected",
        "animation_authorized": False,
    }
    with pytest.raises(fallback.FallbackError, match="opposite-limb"):
        fallback.validate_static_failure_payloads(
            (
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": "raw GLB triangle count changed"},
                },
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": "different failure"},
                },
            )
        )


def test_manifest_augmentation_requires_skeleton_conditioning_and_records_failed_direct_gate():
    base_manifest = {
        "schema": "pixal_tokenrig_canary_v1",
        "asset_id": "rocketbox_male_adult_01",
        "attempt": "fitted_skeleton_transfer",
        "inference_parameters": {"use_skeleton": True, "use_transfer": True},
        "input": {
            "glb": {"path": "/direct.glb", "sha256": "a" * 64},
            "fallback_provenance": {"static_failures": ["a", "b"]},
        },
    }

    result = fallback.augment_fitted_manifest(base_manifest)

    assert result["schema"] == "pixal_tokenrig_fitted_skeleton_v1"
    assert result["base_runner_schema"] == "pixal_tokenrig_canary_v1"
    assert result["task3_direct_gate_status"] == "failed"
    assert result["animation_authorized"] is False
    assert result["fitted_skeleton"]["use_skeleton_input"] is True
    with pytest.raises(fallback.FallbackError, match="use_skeleton"):
        fallback.augment_fitted_manifest(
            {
                **base_manifest,
                "inference_parameters": {"use_skeleton": False, "use_transfer": True},
            }
        )


def test_base_call_is_forced_to_use_skeleton_and_dedicated_contract():
    kwargs = fallback.build_base_call()

    assert kwargs["use_skeleton_input"] is True
    assert kwargs["seed"] == 42
    assert kwargs["input_glb"] == fallback.PINNED_FALLBACK.input_glb
    assert kwargs["input_manifest"] == fallback.PINNED_FALLBACK.recovery_manifest
    assert kwargs["output_dir"] == fallback.PINNED_FALLBACK.output_dir
    assert kwargs["contract"].output_dir == fallback.PINNED_FALLBACK.output_dir
    assert kwargs["contract"].input_glb == fallback.PINNED_FALLBACK.input_glb


def test_deferred_hygiene_does_not_import_src_during_site_startup():
    source = fallback.DEFERRED_HYGIENE_SOURCE

    ast.parse(source)
    assert "builtins.__import__" in source
    assert "_install_parser_hook" in source
    assert "from src." not in source
    assert "parser.clean_bpy()" in source
    assert "before_clean" in source and "after_import" in source


def _module_source() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "tokenrig_human_fitted_skeleton_fallback.py"
    ).read_text(encoding="utf-8")


def test_wrapper_restores_base_injections_and_has_no_animation_path():
    source = _module_source()
    tree = ast.parse(source)
    assert "finally:" in source
    assert "base._read_input_contract = original_reader" in source
    assert "base.SERVER_HYGIENE_SOURCE = original_hygiene" in source
    assert "base._rename_noreplace = original_rename" in source
    assert '"use_skeleton_input": True' in source
    assert "animation" not in " ".join(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def test_cli_has_no_way_to_disable_skeleton_conditioning():
    args = fallback.parse_args([])

    assert args.seed == 42
    assert not hasattr(args, "use_skeleton_input")
    with pytest.raises(SystemExit):
        fallback.parse_args(["--seed", "7"])


def test_direct_script_cli_bootstraps_tools_package_from_any_cwd(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "tokenrig_human_fitted_skeleton_fallback.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--seed" not in result.stdout
