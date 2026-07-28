"""Contracts for the direct-native authored quadruped review v2 runner."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import encode_quadruped_review_media as media_encoder
from tools import run_direct_native_authored_quadruped_animation_review as review
from tools import run_target_native_generated_quadruped_review as strict_review


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/run_direct_native_authored_quadruped_animation_review.py"
)


def timing_signature():
    actions = {
        action: [
            {
                "interpolation": "LINEAR",
                "sample_count": 3,
                "times_seconds": [0.033333, 0.066667, 0.1],
            }
        ]
        for action in ("Idle", "Walking")
    }
    return {
        "actions": actions,
        "fractional_24fps_time_count": 3,
        "preserves_non_integer_24fps_times": True,
        "sha256": review.canonical_sha256(actions),
    }


def dense_plan():
    return SimpleNamespace(
        maximum_sampled_skinned_world_delta=0.001,
        maximum_animation_duration_delta_frames=0.001,
        roundtrip_canonical_precision_decimals=3,
        motion_root_bone="Bone",
    )


def dense_invariants():
    phases = [index / 80 for index in range(81)]
    semantic_actions = {}
    comparison_actions = {}
    for action in ("Idle", "Walking"):
        semantic_actions[action] = {
            "action_name": f"{action}_Armature",
            "duration_frames": 40.0,
            "root_bone": "Bone",
            "samples": [
                {
                    "phase": phase,
                    "joint_world_matrices_sha256": "a" * 64,
                    "skinned_world_surface_sha256": "b" * 64,
                }
                for phase in phases
            ],
        }
        comparison_actions[action] = [
            {
                "phase": phase,
                "source_evaluated_vertex_count": 100,
                "output_evaluated_vertex_count": 110,
                "maximum_skinned_world_vertex_delta": index * 1.0e-7,
                "within_gate": True,
                "maximum_root_trajectory_delta": index * 2.0e-8,
                "root_trajectory_within_gate": True,
            }
            for index, phase in enumerate(phases)
        ]
    semantic = {
        "precision_decimals": 3,
        "sample_phases": phases,
        "actions": semantic_actions,
        "sha256": "c" * 64,
    }
    return {
        "animation_semantic_comparison": {
            "maximum_animation_duration_delta_frames": 0.0,
            "maximum_animation_duration_delta_frames_gate": 0.001,
            "maximum_root_trajectory_delta": 80 * 2.0e-8,
            "maximum_root_trajectory_delta_gate": 0.001,
            "maximum_skinned_world_vertex_delta": 80 * 1.0e-7,
            "maximum_skinned_world_vertex_delta_gate": 0.001,
            "per_action": comparison_actions,
            "joint_world_matrix_hashes_recorded_but_not_used_as_blender_rig_gate": (
                True
            ),
            "overall": "passed",
        },
        "animation_semantics_before_export": copy.deepcopy(semantic),
        "animation_semantics_after_reimport": copy.deepcopy(semantic),
    }


def minimal_receipt():
    descriptor = {"path": "artifact", "sha256": "a" * 64, "size_bytes": 1}
    executables = {}
    for name in ("blender", "ffmpeg", "ffprobe", "python"):
        path = f"/usr/bin/{name}"
        version_argument = "-version" if name in {"ffmpeg", "ffprobe"} else "--version"
        executables[name] = {
            "identity": {
                "path": path,
                "sha256": "b" * 64,
                "size_bytes": 1,
                "device": 1,
                "inode": 1,
            },
            "version_argv": [path, version_argument],
            "version_stdout": f"{name} version 1",
        }
    tools = {
        name: {
            "external": {
                "path": str(review.REVIEW_TOOL_FILES[name]),
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
            "immutable_snapshot": {
                "path": f"evidence/tool_{name}.py",
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
        }
        for name in review.REVIEW_TOOL_FILES
    }
    commands = []
    for stage in review.REVIEW_STAGES:
        if stage == "deformation_audit":
            tool_name = "deformation_auditor"
            argv = [
                executables["blender"]["identity"]["path"],
                "-b",
                "--python-exit-code",
                "2",
                "--python",
                tools[tool_name]["external"]["path"],
                "--",
            ]
        elif stage.startswith("render_"):
            tool_name = "animation_renderer"
            argv = [
                executables["blender"]["identity"]["path"],
                "-b",
                "--python-exit-code",
                "2",
                "--python",
                tools[tool_name]["external"]["path"],
                "--",
            ]
        else:
            tool_name = "media_encoder"
            argv = [
                executables["python"]["identity"]["path"],
                tools[tool_name]["external"]["path"],
            ]
        commands.append(
            {
                "stage": stage,
                "observed_argv": argv,
                "published_equivalent_argv": list(argv),
            }
        )
    subprocess_contract = {
        "scope": "authenticated_encoder_process_only",
        **{
            executable_name: [
                {
                    "label": label,
                    "resolved_executable_identity": (
                        f"toolchain.executables.{executable_name}"
                    ),
                    "observed_argv": [executable_name, label],
                }
                for label in review.MEDIA_LABELS
            ]
            for executable_name in ("ffmpeg", "ffprobe")
        },
    }
    media_outputs = {
        label: {
            "video": dict(descriptor),
            "render_manifest": dict(descriptor),
            "encode_manifest": dict(descriptor),
            "frames": [dict(descriptor) for _ in range(24)],
        }
        for label in review.MEDIA_LABELS
    }
    value = {
        "schema": review.SCHEMA,
        "created": "2026-07-28T00:00:00+00:00",
        "status": review.STATUS,
        "state": {
            "classification": "research_candidate",
            "owner_animation_decision": "pending",
            "owner_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "native_merge_authorized": False,
        },
        "formal": {
            "formal_dataset_registration_authorized": False,
            "ue_import_authorized": False,
            "native_merge_authorized": False,
        },
        "route": dict(review.ROUTE),
        "lineage": {
            "realization_manifest": {
                "external_path": "/source/realization_manifest.json",
                "external_raw_sha256_authority": "d" * 64,
                "immutable_snapshot": dict(descriptor),
            },
            "reviewed_glb": dict(descriptor),
            "reviewed_coat": dict(descriptor),
            "bounded_realization_exact_root_authenticated": True,
            "bounded_realization_exact_evidence_authenticated": True,
        },
        "toolchain": {
            "executables": executables,
            "tools": tools,
            "commands": commands,
            "media_subprocess_contract": subprocess_contract,
        },
        "automatic_admission_gates": {
            "realization_external_raw_sha256_authenticated": True,
            "realization_exact_immutable_closure_authenticated": True,
            "raw_idle_walking_timing_signature_exactly_equal": True,
            "raw_timing_signature": timing_signature(),
            "dense_animation_semantic_gate": {
                "phases_per_action": 81,
                "phase_interval": "index/80",
                "maximum_skinned_world_vertex_delta": 0.0,
                "maximum_skinned_world_vertex_delta_gate": 0.001,
                "maximum_root_trajectory_delta": 0.0,
                "maximum_root_trajectory_delta_gate": 0.001,
                "maximum_animation_duration_delta_frames": 0.0,
                "maximum_animation_duration_delta_frames_gate": 0.001,
                "status": "passed",
            },
            "structural_skin_skeleton_bind_gate": {
                "topology_unchanged_in_memory": True,
                "weights_unchanged_in_memory": True,
                "skeleton_unchanged_in_memory": True,
                "actions_unchanged_in_memory": True,
                "roundtrip_geometry_and_weight_clusters_passed": True,
                "joint_count": 34,
                "maximum_bind_matrix_semantic_delta": 0.0,
                "maximum_bind_matrix_semantic_delta_gate": 0.001,
                "status": "passed",
            },
            "srgb_and_glb_readback_gate": {
                "mesh_count": 1,
                "skin_count": 1,
                "animation_names": ["Idle", "Walking"],
                "material_count": 1,
                "texture_count": 1,
                "image_count": 1,
                "primitive_count": 1,
                "required_attributes": [],
                "sRGB_transfer_applied_exactly_once": True,
                "base_rgba8": [1, 2, 3, 255],
                "white_rgba8": [4, 5, 6, 255],
                "status": "passed",
            },
            "deformation_audit": {
                "samples_per_action": 24,
                "thresholds": dict(review.DEFORMATION_THRESHOLDS),
                "overall": "passed",
                "status": "passed",
            },
            "six_view_media": {
                "labels": list(review.MEDIA_LABELS),
                "frames_per_view": 24,
                "resolution": [512, 384],
                "fps": 8,
                "render_encode_video_lineage_authenticated": True,
                "status": "passed",
            },
            "all_automatic_gates_passed": True,
        },
        "outputs": {
            "deformation_audit": dict(descriptor),
            "media": media_outputs,
        },
        "artifact_closure": {},
        "authority_boundary": {
            "automatic_technical_review_completed": True,
            "owner_animation_decision": "pending",
            "owner_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "formal_dataset_registration_authorized": False,
            "native_merge_authorized": False,
            "same_uid_continuous_write_adversary_out_of_scope": True,
            "pathnames_are_not_authority": True,
            "caller_supplied_raw_review_sha256_required": True,
        },
        "next_gate": "exact_artifact_bound_human_animation_and_identity_review",
    }
    value["review_sha256"] = review.canonical_sha256(value)
    return value


def publication_with_payload(tmp_path):
    output = tmp_path / "review"
    publication = review.SecureReviewPublication(output)
    publication.write_exclusive("payload.bin", b"original")
    records = publication.seal({"payload.bin"})
    return publication, records


def fixed_protocol_fixture():
    receipt = minimal_receipt()
    records = {
        relative: review.relative_record(relative, relative.encode("utf-8"))
        for relative in review._expected_review_files()
    }
    receipt["outputs"] = review._output_descriptor(records)
    receipt["artifact_closure"] = review._records_to_closure(records)
    receipt["lineage"]["realization_manifest"]["immutable_snapshot"] = records[
        "evidence/source_realization_manifest.json"
    ]
    receipt["lineage"]["reviewed_glb"] = records["evidence/source_bounded_identity.glb"]
    receipt["lineage"]["reviewed_coat"] = records["evidence/source_identity_coat.png"]
    for label, tool in receipt["toolchain"]["tools"].items():
        snapshot = records[f"evidence/tool_{label}.py"]
        tool["immutable_snapshot"] = snapshot
        tool["external"]["sha256"] = snapshot["sha256"]
        tool["external"]["size_bytes"] = snapshot["size_bytes"]
    directories = {
        "",
        "evidence",
        "media",
        *(f"media/{label}_frames" for label in review.MEDIA_LABELS),
    }
    return receipt, records, directories


def test_static_route_is_explicitly_direct_native_and_never_fakes_generated_lineage():
    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        'SCHEMA = "avengine_direct_native_authored_quadruped_animation_review_v2"'
        in text
    )
    assert '"pixal3d_used": False' in text
    assert '"tokenrig_used": False' in text
    assert '"native_animated_source_used": True' in text
    assert '"candidate_ranking_used": False' in text
    assert '"retry_selection_used": False' in text
    assert review.AUTHORIZED_RESEARCH_OUTPUT_SCOPES == frozenset(
        {"new_research_candidate_sealed_at_publication_only"}
    )
    assert "new_immutable_research_candidate_only" not in text
    assert "blender_retarget_quaternius_to_generated_quadruped.py" not in text
    assert "blender_repair_animated_quadruped_weight_stretch.py" not in text


def test_realization_consumer_requires_current_publication_evidence_schema():
    assert "toolchain_publication_snapshots" in review.REALIZATION_TOP_LEVEL_FIELDS
    assert "publication_security_boundary" in review.REALIZATION_TOP_LEVEL_FIELDS
    assert "toolchain_immutable_snapshots" not in review.REALIZATION_TOP_LEVEL_FIELDS

    review._validate_realization_publication_security_boundary(
        copy.deepcopy(review.EXPECTED_REALIZATION_PUBLICATION_SECURITY_BOUNDARY)
    )
    weakened = copy.deepcopy(review.EXPECTED_REALIZATION_PUBLICATION_SECURITY_BOUNDARY)
    weakened["consumer_requirement"][
        "rehash_entire_declared_file_closure_before_use"
    ] = False
    with pytest.raises(review.DirectNativeReviewError, match="changed or weakened"):
        review._validate_realization_publication_security_boundary(weakened)

    payload = b"sealed producer evidence"
    record = {
        "path": "/producer/source.bin",
        "sha256": review.sha256_bytes(payload),
        "size_bytes": len(payload),
        "publication_snapshot": review.relative_record(
            "evidence/source_snapshot.glb", payload
        ),
    }
    review._require_external_and_publication_snapshot_binding(
        record,
        snapshot_relative="evidence/source_snapshot.glb",
        snapshot_payload=payload,
        label="producer source",
    )
    obsolete = dict(record)
    obsolete["immutable_snapshot"] = obsolete.pop("publication_snapshot")
    with pytest.raises(review.DirectNativeReviewError, match="fields changed"):
        review._require_external_and_publication_snapshot_binding(
            obsolete,
            snapshot_relative="evidence/source_snapshot.glb",
            snapshot_payload=payload,
            label="obsolete producer source",
        )


def test_review_tool_closure_contains_every_local_runtime_import():
    snapshotted = {path.resolve() for path in review.REVIEW_TOOL_FILES.values()}
    missing = {}
    for label, path in review.REVIEW_TOOL_FILES.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "tools":
                modules.update(
                    f"tools.{alias.name}" for alias in node.names if alias.name != "*"
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("tools.")
            ):
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("tools.")
                )
        local_imports = {
            SCRIPT.parents[1] / f"{module.replace('.', '/')}.py" for module in modules
        }
        unresolved = sorted(
            str(imported.relative_to(SCRIPT.parents[1]))
            for imported in local_imports
            if imported.resolve() not in snapshotted
        )
        if unresolved:
            missing[label] = unresolved

    assert missing == {}


def test_published_tree_binding_uses_fixed_185_file_protocol_not_receipt_self_report():
    receipt, records, directories = fixed_protocol_fixture()

    assert len(review._expected_review_files()) == 185
    review._validate_fixed_protocol_tree_bindings(receipt, records, directories)

    with_extra = dict(records)
    with_extra["media/unreviewed_redirect.mp4"] = review.relative_record(
        "media/unreviewed_redirect.mp4", b"unreviewed"
    )
    with pytest.raises(review.DirectNativeReviewError, match="fixed protocol"):
        review._validate_fixed_protocol_tree_bindings(
            receipt,
            with_extra,
            directories,
        )

    redirected = copy.deepcopy(receipt)
    redirected["outputs"]["media"]["walking_side"]["video"] = records[
        "media/walking_side_render_manifest.json"
    ]
    with pytest.raises(review.DirectNativeReviewError, match="fixed protocol paths"):
        review._validate_fixed_protocol_tree_bindings(
            redirected,
            records,
            directories,
        )

    tool_redirect = copy.deepcopy(receipt)
    tool_redirect["toolchain"]["tools"]["review_runner"]["immutable_snapshot"] = (
        records["evidence/source_realization_manifest.json"]
    )
    with pytest.raises(review.DirectNativeReviewError, match="tool snapshot"):
        review._validate_fixed_protocol_tree_bindings(
            tool_redirect,
            records,
            directories,
        )


def test_execution_lineage_requires_exact_commands_labels_and_one_held_root(tmp_path):
    receipt = minimal_receipt()
    published_root = tmp_path / "published-review"
    observed_root = Path("/proc/4242/fd/73")
    commands = review.build_commands(
        root=observed_root,
        blender=Path(
            receipt["toolchain"]["executables"]["blender"]["identity"]["path"]
        ),
        python=Path(receipt["toolchain"]["executables"]["python"]["identity"]["path"]),
        input_glb=observed_root / "evidence/source_bounded_identity.glb",
    )
    receipt["toolchain"]["commands"] = review._command_records(
        commands,
        old_prefix=str(observed_root),
        new_prefix=str(published_root),
    )
    receipt["toolchain"]["media_subprocess_contract"] = (
        review._media_subprocess_contract_for_root(observed_root)
    )

    review._validate_execution_lineage_against_root(receipt, published_root)

    suffix = copy.deepcopy(receipt)
    suffix["toolchain"]["commands"][0]["observed_argv"].append("--unrecorded")
    suffix["toolchain"]["commands"][0]["published_equivalent_argv"].append(
        "--unrecorded"
    )
    with pytest.raises(review.DirectNativeReviewError, match="command plan changed"):
        review._validate_execution_lineage_against_root(suffix, published_root)

    duplicate_label = copy.deepcopy(receipt)
    duplicate_label["toolchain"]["media_subprocess_contract"]["ffprobe"][1]["label"] = (
        review.MEDIA_LABELS[0]
    )
    with pytest.raises(review.DirectNativeReviewError, match="identity/label"):
        review._validate_execution_lineage_against_root(
            duplicate_label,
            published_root,
        )

    split_root = copy.deepcopy(receipt)
    split_root["toolchain"]["media_subprocess_contract"]["ffprobe"][0]["observed_argv"][
        -1
    ] = split_root["toolchain"]["media_subprocess_contract"]["ffprobe"][0][
        "observed_argv"
    ][-1].replace("/proc/4242/fd/73", "/proc/4242/fd/74")
    with pytest.raises(review.DirectNativeReviewError, match="publication roots"):
        review._validate_execution_lineage_against_root(
            split_root,
            published_root,
        )

    mixed_command = copy.deepcopy(receipt)
    command = mixed_command["toolchain"]["commands"][0]
    root_index = next(
        index
        for index, argument in enumerate(command["observed_argv"])
        if argument.startswith(str(observed_root) + "/")
    )
    command["observed_argv"][root_index] = command["published_equivalent_argv"][
        root_index
    ]
    with pytest.raises(review.DirectNativeReviewError, match="bypasses the held"):
        review._validate_execution_lineage_against_root(
            mixed_command,
            published_root,
        )

    mixed_media = copy.deepcopy(receipt)
    media_argv = mixed_media["toolchain"]["media_subprocess_contract"]["ffprobe"][0][
        "observed_argv"
    ]
    media_argv[-1] = media_argv[-1].replace(
        str(observed_root),
        str(published_root),
    )
    with pytest.raises(review.DirectNativeReviewError, match="bypasses the held"):
        review._validate_execution_lineage_against_root(
            mixed_media,
            published_root,
        )

    for invalid_root in (
        "/proc/0/fd/73",
        "/proc/04242/fd/73",
        "/proc/4242/fd/0",
        "/proc/4242/fd/073",
        "/proc/self/fd/73",
    ):
        with pytest.raises(
            review.DirectNativeReviewError,
            match="invalid held-fd path",
        ):
            review._normalize_historical_proc_argv(
                [f"{invalid_root}/artifact"],
                expected_argv=[str(published_root / "artifact")],
                published_root=published_root,
                expected_proc_prefix=None,
                label="invalid held root",
            )


def test_embedded_gate_parity_rejects_self_reported_pass_values():
    expected = {
        "dense_animation_semantic_gate": {
            "maximum_skinned_world_vertex_delta": 0.0001,
            "status": "passed",
        }
    }
    review._validate_embedded_gate_parity(copy.deepcopy(expected), expected)

    claimed = copy.deepcopy(expected)
    claimed["dense_animation_semantic_gate"]["maximum_skinned_world_vertex_delta"] = 999
    with pytest.raises(review.DirectNativeReviewError, match="contradicts embedded"):
        review._validate_embedded_gate_parity(claimed, expected)


def test_validation_ffprobe_binds_receipt_identity_and_resolved_basename(
    tmp_path,
    monkeypatch,
):
    ffprobe_path = tmp_path / "ffprobe"
    ffprobe_path.write_bytes(b"probe")
    receipt = minimal_receipt()
    recorded = receipt["toolchain"]["executables"]["ffprobe"]
    executable = SimpleNamespace(
        path=ffprobe_path,
        record=dict(recorded["identity"]),
        version_argv=list(recorded["version_argv"]),
        version_stdout=recorded["version_stdout"],
        verify=lambda: None,
    )
    monkeypatch.setattr(review.shutil, "which", lambda _name, path: str(ffprobe_path))
    review._require_validation_ffprobe(receipt, executable)

    wrong_identity = copy.deepcopy(receipt)
    wrong_identity["toolchain"]["executables"]["ffprobe"]["identity"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(review.DirectNativeReviewError, match="does not match"):
        review._require_validation_ffprobe(wrong_identity, executable)

    different = tmp_path / "different"
    different.write_bytes(b"different")
    monkeypatch.setattr(review.shutil, "which", lambda _name, path: str(different))
    with pytest.raises(review.DirectNativeReviewError, match="not the executable"):
        review._require_validation_ffprobe(receipt, executable)


def test_review_tool_execution_file_is_held_rehashed_and_snapshot_bound(tmp_path):
    path = tmp_path / "tool.py"
    path.write_bytes(b"print('bounded')\n")
    tool = review.authenticate_tool_file(path, name="fixture")
    try:
        tool.verify()
        assert tool.payload == b"print('bounded')\n"
        assert tool.record == {
            "path": str(path),
            "sha256": review.sha256_bytes(tool.payload),
            "size_bytes": len(tool.payload),
        }
        path.write_bytes(b"print('changed')\n")
        with pytest.raises(review.DirectNativeReviewError, match="identity/bytes"):
            tool.verify()
    finally:
        tool.close()


def test_resource_close_attempts_every_handle_before_reporting_success():
    events = []

    def resource(name, *, fail=False):
        def close():
            events.append(name)
            if fail:
                raise OSError(f"{name} close failed")

        return SimpleNamespace(close=close)

    with pytest.raises(OSError, match="publication close failed"):
        review._close_producer_resources(
            resource("publication", fail=True),
            {"blender": resource("executable")},
            {"runner": resource("tool")},
            resource("realization"),
        )
    assert events == ["publication", "executable", "tool", "realization"]

    text = SCRIPT.read_text(encoding="utf-8")
    success_print = text.index("print(success)")
    preview_print = text.index("print(preview)")
    assert text.rfind("_close_producer_resources(", 0, success_print) > 0
    assert text.rfind("_close_producer_resources(", 0, preview_print) > 0


def test_static_commands_and_thresholds_cannot_silently_shrink_review_scope():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "REVIEW_FRAMES = 24" in text
    assert "DEFORMATION_SAMPLES = 24" in text
    assert "REVIEW_MEDIA_SPECS" in text
    assert '"blender_audit_skinned_deformation.py"' in text
    assert '"blender_render_glb_animation.py"' in text
    assert '"encode_quadruped_review_media.py"' in text
    assert "require_deformation_audit(" in text
    assert "require_review_media_set(" in text
    assert 'executables["python"] = authenticate_executable(' in text
    assert "review_tools[name] = authenticate_tool_file(path, name=name)" in text
    stage_call = text.index("subprocess.run(\n                command,")
    before_stage = text[:stage_call]
    after_stage = text[stage_call:]
    verification = "for tool in review_tools.values():\n                tool.verify()"
    assert before_stage.rfind(verification) > before_stage.rfind("for stage, command")
    assert after_stage.find(verification) < after_stage.find(
        'if stage == "deformation_audit"'
    )
    for label in (
        "walking_side",
        "walking_front",
        "walking_rear",
        "idle_side",
        "idle_front",
        "idle_rear",
    ):
        assert label in str(review.MEDIA_LABELS)


def test_static_publication_is_no_replace_post_verified_and_quarantine_only():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "renameat2" in text
    assert "self.verify(expected)" in text
    assert "self.published = True" in text
    assert text.index("self.published = True") < text.rindex("self.verify(expected)")
    assert "def quarantine(" in text
    assert "os.unlink(" not in text
    assert "os.rmdir(" not in text
    assert "shutil.rmtree" not in text


def test_strict_json_and_receipt_reject_duplicate_nan_and_extra_fields():
    with pytest.raises(review.DirectNativeReviewError, match="duplicate"):
        review.strict_json_bytes(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(review.DirectNativeReviewError, match="non-finite"):
        review.strict_json_bytes(b'{"a":NaN}', "fixture")

    receipt = minimal_receipt()
    review._validate_receipt_shape(receipt)
    changed = copy.deepcopy(receipt)
    changed["unexpected"] = True
    with pytest.raises(review.DirectNativeReviewError, match="fields changed"):
        review._validate_receipt_shape(changed)

    changed = copy.deepcopy(receipt)
    changed["state"]["unexpected"] = True
    changed["review_sha256"] = review.canonical_sha256(
        {key: value for key, value in changed.items() if key != "review_sha256"}
    )
    with pytest.raises(review.DirectNativeReviewError, match="review state fields"):
        review._validate_receipt_shape(changed)

    changed = copy.deepcopy(receipt)
    changed["automatic_admission_gates"]["dense_animation_semantic_gate"][
        "unexpected"
    ] = True
    changed["review_sha256"] = review.canonical_sha256(
        {key: value for key, value in changed.items() if key != "review_sha256"}
    )
    with pytest.raises(review.DirectNativeReviewError, match="dense animation"):
        review._validate_receipt_shape(changed)

    changed = copy.deepcopy(receipt)
    del changed["toolchain"]["executables"]["python"]
    with pytest.raises(review.DirectNativeReviewError, match="executables fields"):
        review._validate_receipt_shape(changed)

    changed = copy.deepcopy(receipt)
    changed["toolchain"]["commands"][1]["stage"] = "wrong_stage"
    with pytest.raises(review.DirectNativeReviewError, match="13 stages"):
        review._validate_receipt_shape(changed)

    changed = copy.deepcopy(receipt)
    encode = next(
        command
        for command in changed["toolchain"]["commands"]
        if command["stage"].startswith("encode_")
    )
    encode["observed_argv"][0] = "/unbound/python"
    with pytest.raises(review.DirectNativeReviewError, match="executable/tool binding"):
        review._validate_receipt_shape(changed)


def test_raw_timing_requires_exact_idle_walking_equality_and_recomputed_hash():
    timing = timing_signature()
    invariants = {
        "raw_source_animation_time_signature": copy.deepcopy(timing),
        "raw_output_animation_time_signature": copy.deepcopy(timing),
    }
    result = review._validate_raw_animation_timing(invariants)
    assert result["fractional_24fps_time_count"] == 3

    changed = copy.deepcopy(invariants)
    changed["raw_output_animation_time_signature"]["actions"]["Walking"][0][
        "times_seconds"
    ][1] = 0.07
    with pytest.raises(review.DirectNativeReviewError, match="not exactly equal"):
        review._validate_raw_animation_timing(changed)

    changed = copy.deepcopy(invariants)
    changed["raw_source_animation_time_signature"]["sha256"] = "0" * 64
    changed["raw_output_animation_time_signature"]["sha256"] = "0" * 64
    with pytest.raises(review.DirectNativeReviewError, match="summary contradicts"):
        review._validate_raw_animation_timing(changed)


def test_dense_gate_requires_all_81_phases_and_recomputes_aggregates():
    invariants = dense_invariants()
    result = review._validate_dense_animation_gate(invariants, dense_plan())

    assert result["phases_per_action"] == 81
    assert result["maximum_skinned_world_vertex_delta"] == pytest.approx(8.0e-6)

    changed = copy.deepcopy(invariants)
    changed["animation_semantic_comparison"]["per_action"]["Walking"].pop(41)
    with pytest.raises(review.DirectNativeReviewError, match="exactly 81"):
        review._validate_dense_animation_gate(changed, dense_plan())

    changed = copy.deepcopy(invariants)
    changed["animation_semantic_comparison"]["maximum_skinned_world_vertex_delta"] = 0.0
    with pytest.raises(review.DirectNativeReviewError, match="aggregate contradicts"):
        review._validate_dense_animation_gate(changed, dense_plan())

    changed = copy.deepcopy(invariants)
    changed["animation_semantic_comparison"]["per_action"]["Idle"][20][
        "maximum_skinned_world_vertex_delta"
    ] = 0.002
    with pytest.raises(review.DirectNativeReviewError, match="boolean contradicts"):
        review._validate_dense_animation_gate(changed, dense_plan())


def test_ancestor_symlink_is_rejected_before_private_staging(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        review.SecureReviewPublication(link / "review")
    assert not any(real.iterdir())


def test_proc_root_is_resolvable_by_nested_media_subprocesses(tmp_path):
    output_root = tmp_path / "review"
    publication = review.SecureReviewPublication(output_root)
    source = b"held publication payload"
    output_relative = "media/walking_side_frames/nested_probe.json"
    publication.write_exclusive("evidence/probe.bin", source)
    nested_script = (
        "import json\n"
        "from pathlib import Path\n"
        "import sys\n"
        "payload = Path(sys.argv[1]).read_bytes()\n"
        "output = Path(sys.argv[2])\n"
        "output.write_text(json.dumps({\n"
        "    'source': str(Path(sys.argv[1]).resolve()),\n"
        "    'manifest': str(output.resolve()),\n"
        "    'size_bytes': len(payload),\n"
        "}), encoding='utf-8')\n"
    )
    child_script = (
        "import subprocess\n"
        "import sys\n"
        f"nested_script = {nested_script!r}\n"
        "subprocess.run(\n"
        "    [sys.executable, '-c', nested_script, sys.argv[1], sys.argv[2]],\n"
        "    check=True,\n"
        "    close_fds=True,\n"
        ")\n"
    )
    try:
        assert str(publication.proc_root).startswith(f"/proc/{os.getpid()}/fd/")
        subprocess.run(
            [
                sys.executable,
                "-c",
                child_script,
                str(publication.proc_path("evidence/probe.bin")),
                str(publication.proc_path(output_relative)),
            ],
            check=True,
            close_fds=True,
        )
        generated = review.strict_json_bytes(
            publication.read_file(output_relative),
            "nested generated lineage",
        )
        resolved_prefix = str(publication.resolved_staging_root)
        assert generated["source"].startswith(resolved_prefix + "/")
        assert generated["manifest"].startswith(resolved_prefix + "/")
        normalized = review._normalize_publication_lineage_value(
            generated,
            publication,
        )
        publication.rewrite_owned(
            output_relative,
            review.pretty_json_bytes(normalized),
        )
        records = publication.seal({"evidence/probe.bin", output_relative})
        publication.publish(records)
    finally:
        publication.close()
    published = json.loads((output_root / output_relative).read_text(encoding="utf-8"))
    assert published == {
        "source": str(output_root / "evidence/probe.bin"),
        "manifest": str(output_root / output_relative),
        "size_bytes": len(source),
    }


def test_direct_encoder_media_subprocess_argv_matches_receipt_contract(
    tmp_path,
    monkeypatch,
):
    publication = review.SecureReviewPublication(tmp_path / "review")
    publication.write_exclusive("evidence/input.glb", b"glb")
    publication.write_exclusive("media/walking_side_render_manifest.json", b"{}")
    captured = {}

    def fake_render_manifest(*_args, **_kwargs):
        return {"frames": []}

    def fake_ffmpeg(argv, **_kwargs):
        captured["ffmpeg"] = list(argv)
        Path(argv[-1]).write_bytes(b"video")
        return SimpleNamespace()

    def fake_verify_video(path, _expected_frames, **kwargs):
        captured["probe_path"] = Path(kwargs["probe_path"])
        return strict_review.file_record(path)

    monkeypatch.setattr(
        media_encoder,
        "require_render_manifest",
        fake_render_manifest,
    )
    monkeypatch.setattr(media_encoder.subprocess, "run", fake_ffmpeg)
    monkeypatch.setattr(media_encoder, "verify_video", fake_verify_video)
    try:
        paths = review.media_paths(publication.proc_root, "walking_side")
        assert (
            media_encoder.main(
                [
                    "--input-glb",
                    str(publication.proc_path("evidence/input.glb")),
                    "--render-manifest",
                    str(paths["render_manifest"]),
                    "--frame-dir",
                    str(paths["frame_dir"]),
                    "--label",
                    "walking_side",
                    "--action",
                    "Walking",
                    "--view",
                    "side",
                    "--asset-yaw-deg",
                    "0.0",
                    "--n-frames",
                    "24",
                    "--width",
                    "512",
                    "--height",
                    "384",
                    "--fps",
                    "8",
                    "--output",
                    str(paths["video"]),
                    "--manifest",
                    str(paths["encode_manifest"]),
                    "--preserve-lexical-media-subprocess-paths",
                ]
            )
            == 0
        )
        contract = review._media_subprocess_contract_for_root(publication.proc_root)
        assert captured["ffmpeg"] == contract["ffmpeg"][0]["observed_argv"]
        assert captured["probe_path"] == paths["video"]
        manifest = review.strict_json_bytes(
            publication.read_file("media/walking_side_encode_manifest.json"),
            "encoder manifest",
        )
        assert manifest["ffmpeg"]["input_pattern"].startswith(
            str(publication.resolved_staging_root) + "/"
        )
    finally:
        publication.close()


def test_verify_video_uses_authenticated_lexical_probe_alias(
    tmp_path,
    monkeypatch,
):
    publication = review.SecureReviewPublication(tmp_path / "review")
    publication.write_exclusive("media/walking_side.mp4", b"video")
    captured = {}

    def fake_probe(argv, **_kwargs):
        captured["ffprobe"] = list(argv)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 512,
                            "height": 384,
                            "nb_frames": "24",
                            "r_frame_rate": "8/1",
                            "avg_frame_rate": "8/1",
                        }
                    ],
                    "format": {"duration": "3.0"},
                }
            )
        )

    monkeypatch.setattr(strict_review.subprocess, "run", fake_probe)
    try:
        lexical_video = publication.proc_path("media/walking_side.mp4")
        video_record = strict_review.verify_video(
            lexical_video,
            24,
            probe_path=lexical_video,
        )
        contract = review._media_subprocess_contract_for_root(publication.proc_root)
        assert contract["scope"] == "authenticated_encoder_process_only"
        assert captured["ffprobe"] == contract["ffprobe"][0]["observed_argv"]
        publication.write_exclusive(
            "media/walking_side_recorded_probe.json",
            review.pretty_json_bytes({"video": video_record}),
        )
        recorded = strict_review.require_recorded_video_readback(
            publication.proc_path("media/walking_side_recorded_probe.json"),
            lexical_video,
            24,
        )
        assert recorded == video_record
        changed = dict(video_record)
        changed["codec"] = "forged"
        publication.rewrite_owned(
            "media/walking_side_recorded_probe.json",
            review.pretty_json_bytes({"video": changed}),
        )
        with pytest.raises(RuntimeError, match="contradicts the sealed video"):
            strict_review.require_recorded_video_readback(
                publication.proc_path("media/walking_side_recorded_probe.json"),
                lexical_video,
                24,
            )
    finally:
        publication.close()


def test_parent_swap_is_rejected_and_hidden_root_is_quarantined(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    publication = review.SecureReviewPublication(parent / "review")
    publication.write_exclusive("payload.bin", b"original")
    records = publication.seal({"payload.bin"})
    moved_parent = tmp_path / "moved-parent"
    parent.rename(moved_parent)
    parent.mkdir()
    try:
        with pytest.raises(review.DirectNativeReviewError, match="parent path"):
            publication.publish(records)
        assert (moved_parent / publication.staging_name).is_dir()
        assert not (parent / "review").exists()
    finally:
        publication.close()


def test_concurrent_output_created_after_preverify_is_never_replaced(
    tmp_path,
    monkeypatch,
):
    publication, records = publication_with_payload(tmp_path)
    original_verify = publication.verify
    calls = 0

    def create_concurrent_after_preverify(expected):
        nonlocal calls
        calls += 1
        original_verify(expected)
        if calls == 1:
            publication.output_root.mkdir()
            (publication.output_root / "owner").write_text(
                "concurrent",
                encoding="utf-8",
            )

    monkeypatch.setattr(publication, "verify", create_concurrent_after_preverify)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="concurrent"):
            publication.publish(records)
        marker = publication.output_root / "owner"
        assert marker.read_text(encoding="utf-8") == "concurrent"
        assert (tmp_path / publication.staging_name).is_dir()
    finally:
        publication.close()


def test_same_inode_byte_change_after_seal_is_detected(tmp_path):
    publication, records = publication_with_payload(tmp_path)
    payload = publication.proc_path("payload.bin")
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")
    payload.chmod(0o444)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="bytes"):
            publication.verify(records)
    finally:
        publication.close()


def test_verify_to_rename_same_inode_tamper_fails_post_verify_and_keeps_final(
    tmp_path, monkeypatch
):
    publication, records = publication_with_payload(tmp_path)
    original_verify = publication.verify
    calls = 0

    def tampering_verify(expected):
        nonlocal calls
        calls += 1
        original_verify(expected)
        if calls == 1:
            payload = publication.proc_path("payload.bin")
            payload.chmod(0o644)
            payload.write_bytes(b"tampered")
            payload.chmod(0o444)

    monkeypatch.setattr(publication, "verify", tampering_verify)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="bytes"):
            publication.publish(records)
        assert publication.output_root.is_dir()
        assert (publication.output_root / "payload.bin").read_bytes() == b"tampered"
    finally:
        publication.close()


def test_post_publish_verifier_failure_preserves_final(tmp_path, monkeypatch):
    publication, records = publication_with_payload(tmp_path)
    original_verify = publication.verify
    calls = 0

    def failing_second_verify(expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise review.DirectNativeReviewError("injected post-publish failure")
        original_verify(expected)

    monkeypatch.setattr(publication, "verify", failing_second_verify)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="post-publish"):
            publication.publish(records)
        assert publication.output_root.is_dir()
        assert (publication.output_root / "payload.bin").read_bytes() == b"original"
    finally:
        publication.close()


def test_post_rename_final_name_replacement_is_detected_and_retained(
    tmp_path,
    monkeypatch,
):
    publication, records = publication_with_payload(tmp_path)
    original_verify = publication.verify
    detached = tmp_path / "detached-published-review"
    victim = tmp_path / "victim-review"
    victim.mkdir()
    (victim / "sentinel").write_bytes(b"external victim")
    calls = 0

    def replace_final_after_second_verify(expected):
        nonlocal calls
        calls += 1
        original_verify(expected)
        if calls == 2:
            publication.output_root.rename(detached)
            victim.rename(publication.output_root)

    monkeypatch.setattr(publication, "verify", replace_final_after_second_verify)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="published name"):
            publication.publish(records)
        assert publication.published is True
        assert (detached / "payload.bin").read_bytes() == b"original"
        assert (publication.output_root / "sentinel").read_bytes() == (
            b"external victim"
        )
        publication.quarantine()
        assert (detached / "payload.bin").read_bytes() == b"original"
    finally:
        publication.close()


def test_post_rename_parent_alias_swap_is_detected_without_cleanup(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    publication = review.SecureReviewPublication(parent / "review")
    publication.write_exclusive("payload.bin", b"original")
    records = publication.seal({"payload.bin"})
    original_verify = publication.verify
    moved_parent = tmp_path / "moved-parent"
    calls = 0

    def swap_parent_after_second_verify(expected):
        nonlocal calls
        calls += 1
        original_verify(expected)
        if calls == 2:
            parent.rename(moved_parent)
            parent.mkdir()

    monkeypatch.setattr(publication, "verify", swap_parent_after_second_verify)
    try:
        with pytest.raises(review.DirectNativeReviewError, match="parent path"):
            publication.publish(records)
        assert publication.published is True
        assert (moved_parent / "review/payload.bin").read_bytes() == b"original"
        assert not (parent / "review").exists()
        publication.quarantine()
        assert (moved_parent / "review/payload.bin").read_bytes() == b"original"
    finally:
        publication.close()


def test_quarantine_never_unlinks_unknown_or_substituted_external_victim(tmp_path):
    publication = review.SecureReviewPublication(tmp_path / "review")
    publication.write_exclusive("unknown.bin", b"unknown")
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_file = victim / "must-survive"
    victim_file.write_text("external", encoding="utf-8")
    moved = f"{publication.staging_name}.moved"
    os.rename(
        publication.staging_name,
        moved,
        src_dir_fd=publication.parent_fd,
        dst_dir_fd=publication.parent_fd,
    )
    os.symlink(
        victim,
        publication.staging_name,
        target_is_directory=True,
        dir_fd=publication.parent_fd,
    )
    try:
        publication.quarantine()
        assert victim_file.read_text(encoding="utf-8") == "external"
        assert (tmp_path / moved / "unknown.bin").read_bytes() == b"unknown"
        assert (tmp_path / publication.staging_name).is_symlink()
    finally:
        publication.close()


def test_sealed_tree_rejects_unknown_symlink(tmp_path):
    root = tmp_path / "sealed"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"payload")
    payload.chmod(0o444)
    (root / "link").symlink_to(payload)
    root.chmod(0o555)
    root_fd = os.open(root, review._directory_flags())
    try:
        with pytest.raises(review.DirectNativeReviewError, match="link/special"):
            review.scan_sealed_tree(root_fd)
    finally:
        os.close(root_fd)


def test_build_commands_contains_one_audit_and_all_six_fixed_media_pairs(tmp_path):
    commands = review.build_commands(
        root=tmp_path / "private",
        blender=Path("/opt/blender"),
        python=Path("/opt/python"),
        input_glb=tmp_path / "private/evidence/input.glb",
    )

    assert [stage for stage, _ in commands] == [
        "deformation_audit",
        *[
            stage
            for label in review.MEDIA_LABELS
            for stage in (f"render_{label}", f"encode_{label}")
        ],
    ]
    flattened = json.dumps(commands)
    assert '"--samples", "24"' in flattened
    assert flattened.count('"--n-frames", "24"') == 12
    assert flattened.count('"--width", "512"') == 12
    assert flattened.count('"--height", "384"') == 12
    assert flattened.count('"--fps", "8"') == 12
    assert flattened.count('"--preserve-lexical-media-subprocess-paths"') == len(
        review.MEDIA_LABELS
    )
