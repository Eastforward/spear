"""Artifact gate tests for stable-template humanoids in apartment_0000."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, payload=b"artifact"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(payload)
    return {"filename": path.name, "sha256": _sha(path)}


def _ready_spike(tmp_path, *, tag="hy3d_rocketbox_male_adult_01_spike"):
    from human_apartment_gate import ALLOWED_HUMAN_SPIKES

    asset_id = ALLOWED_HUMAN_SPIKES[tag]
    stable_root = tmp_path / "hy3d_rocketbox_template_fit_v1"
    asset_dir = stable_root / asset_id
    asset_dir.mkdir(parents=True)

    bind_manifest = {
        "schema_version": "hy3d_rocketbox_bind_v1",
        "asset_id": asset_id,
        "binding_mode": "stable_rocketbox_template_fit_v1",
        "usage_scope": "technical_spike_only",
    }
    bind_manifest_path = asset_dir / "bind_manifest.json"
    _write(bind_manifest_path, bind_manifest)
    pixel_qa = _write(asset_dir / "pixel_qa.json", {"decision": "ready"})
    review_manifest = _write(asset_dir / "review_manifest.json", {"automatic_checks": {"overall": "passed"}})
    bind_metrics = _write(asset_dir / "bind_metrics.json")
    bound_blend = _write(asset_dir / "bound.blend")
    contact_sheet = _write(asset_dir / "bind_contact_sheet.png")
    walk_glb = _write(asset_dir / "bound_walk.glb")
    idle_glb = _write(asset_dir / "bound_idle.glb")
    videos = {
        action: {
            view: _write(asset_dir / f"{action}_{view}.mp4")
            for view in ("front", "side", "feet")
        }
        for action in ("walk", "idle")
    }
    ready = {
        "schema_version": "hy3d_rocketbox_direct_attempt_ready_v1",
        "status": "ready",
        "asset_id": asset_id,
        "bind_manifest_sha256": _sha(bind_manifest_path),
        "pixel_qa": pixel_qa,
        "review_manifest_sha256": review_manifest["sha256"],
        "bind_metrics": bind_metrics,
        "bound_blend": bound_blend,
        "contact_sheet": contact_sheet,
        "glbs": {"walk": walk_glb, "idle": idle_glb},
        "videos": videos,
    }
    _write(asset_dir / "direct_attempt_ready.json", ready)

    runtime_glb = asset_dir / "ue_runtime.glb"
    _write(runtime_glb, b"runtime-glb")
    import_manifest = {
        "schema": "hy3d_rocketbox_ue_import_v1",
        "tag": tag,
        "asset_id": asset_id,
        "usage_scope": "technical_spike_only",
        "source_glb": str(runtime_glb.resolve()),
        "source_glb_sha256": _sha(runtime_glb),
        "reload_verification": {"status": "passed"},
        "runtime_contract": {"bone_count": 80},
        "content": {
            "animations": {"Walking": "/Game/Walking", "Standing_Idle": "/Game/Standing_Idle"},
            "blueprint": f"/Game/BP_gate_{tag}",
            "material_slots": ["body", "head", "opacity"],
        },
    }
    _write(asset_dir / "ue_import_manifest.json", import_manifest)
    registry_root = tmp_path / "source_assets_v1"
    registry_root.mkdir()
    return tag, stable_root, registry_root, asset_dir


def _ready_native_rocketbox(
    tmp_path, *, tag="rocketbox_male_adult_01_original_ue_v3"
):
    runtime_root = tmp_path / "rocketbox_native_runtime_ue_v3"
    runtime_dir = runtime_root / tag
    runtime_dir.mkdir(parents=True)
    runtime_glb = runtime_dir / "runtime.glb"
    _write(runtime_glb, b"native-v3-runtime")
    source_manifest = {
        "schema": "rocketbox_native_ue_runtime_v3",
        "tag": tag,
        "asset_id": "rocketbox_male_adult_01",
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "runtime_glb": {
            "filename": "runtime.glb",
            "size_bytes": runtime_glb.stat().st_size,
            "sha256": _sha(runtime_glb),
        },
        "normalization": {
            "schema": (
                "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
            ),
            "normalized_joint_count": 80,
            "in_place_actions": ["Walking"],
            "root_motion": {
                "Walking": {
                    "maximum_horizontal_deviation_after_m": 1e-8,
                    "maximum_vertical_world_error_m": 1e-8,
                }
            },
        },
        "runtime_motion_contract": {
            "walking_embedded_horizontal_root_motion": "removed",
            "walking_vertical_motion": "preserved",
            "dynamic_ground_snap_to_floor_required": True,
        },
        "expected_ue_qa": {
            "actor_scale": 1.0,
            "height_range_cm": [165.0, 200.0],
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": 15.0,
        },
        "automatic_checks": {"overall": "passed"},
    }
    source_manifest_path = runtime_dir / "normalization_manifest.json"
    _write(source_manifest_path, source_manifest)

    import_root = tmp_path / "rocketbox_native_ue_import_v3"
    import_dir = import_root / tag
    import_dir.mkdir(parents=True)
    import_manifest = {
        "schema": "rocketbox_native_ue_import_v3",
        "tag": tag,
        "asset_id": "rocketbox_male_adult_01",
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "source_glb": str(runtime_glb.resolve()),
        "source_glb_sha256": _sha(runtime_glb),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha(source_manifest_path),
        "reload_verification": {"status": "passed"},
        "runtime_contract": {
            "actor_scale": 1.0,
            "bone_count": 80,
            "bounds": {
                "height_cm": 183.1,
                "height_range_cm": [165.0, 200.0],
                "height_passed": True,
                "ground_passed": True,
            },
        },
        "content": {
            "animations": {
                "Walking": "/Game/Walking",
                "Standing_Idle": "/Game/Standing_Idle",
            },
            "blueprint": f"/Game/BP_gate_{tag}",
        },
    }
    import_manifest_path = import_dir / "ue_import_manifest.json"
    _write(import_manifest_path, import_manifest)
    registry_root = tmp_path / "source_assets_v1"
    registry_root.mkdir()
    return tag, runtime_root, import_root, registry_root


def test_gate_accepts_only_current_stable_template_and_reloaded_ue_assets(tmp_path):
    from human_apartment_gate import assert_human_apartment_ready

    tag, stable_root, registry_root, asset_dir = _ready_spike(tmp_path)
    evidence = assert_human_apartment_ready(
        tag,
        stable_root=stable_root,
        formal_registry_root=registry_root,
        skip_review_gate=False,
    )

    assert evidence["tag"] == tag
    assert evidence["asset_dir"] == str(asset_dir.resolve())
    assert evidence["usage_scope"] == "technical_spike_only"
    assert evidence["bone_count"] == 80
    assert set(evidence["animations"]) == {"Walking", "Standing_Idle"}


def test_gate_rejects_skip_review_override_for_human_evidence(tmp_path):
    from human_apartment_gate import HumanApartmentGateError, assert_human_apartment_ready

    tag, stable_root, registry_root, _asset_dir = _ready_spike(tmp_path)
    with pytest.raises(HumanApartmentGateError, match="SPEAR_SKIP_REVIEW_GATE"):
        assert_human_apartment_ready(
            tag,
            stable_root=stable_root,
            formal_registry_root=registry_root,
            skip_review_gate=True,
        )


def test_gate_rejects_old_direct_generated_topology_root(tmp_path):
    from human_apartment_gate import HumanApartmentGateError, assert_human_apartment_ready

    tag, stable_root, registry_root, _asset_dir = _ready_spike(tmp_path)
    old_root = tmp_path / "hy3d_rocketbox_spike_v1"
    stable_root.rename(old_root)
    with pytest.raises(HumanApartmentGateError, match="stable-template root"):
        assert_human_apartment_ready(
            tag,
            stable_root=old_root,
            formal_registry_root=registry_root,
        )


def test_gate_rejects_stale_runtime_glb_and_missing_import_manifest(tmp_path):
    from human_apartment_gate import HumanApartmentGateError, assert_human_apartment_ready

    tag, stable_root, registry_root, asset_dir = _ready_spike(tmp_path)
    (asset_dir / "ue_runtime.glb").write_bytes(b"changed-after-import")
    with pytest.raises(HumanApartmentGateError, match="runtime GLB hash"):
        assert_human_apartment_ready(tag, stable_root=stable_root, formal_registry_root=registry_root)

    tag, stable_root, registry_root, asset_dir = _ready_spike(tmp_path / "missing")
    (asset_dir / "ue_import_manifest.json").unlink()
    with pytest.raises(HumanApartmentGateError, match="UE import manifest"):
        assert_human_apartment_ready(tag, stable_root=stable_root, formal_registry_root=registry_root)


def test_gate_rejects_any_formal_registry_promotion(tmp_path):
    from human_apartment_gate import HumanApartmentGateError, assert_human_apartment_ready

    tag, stable_root, registry_root, _asset_dir = _ready_spike(tmp_path)
    _write(
        registry_root / "bad.json",
        {"asset_id": "forbidden_human", "legacy_tag": tag},
    )
    with pytest.raises(HumanApartmentGateError, match="formal source registry"):
        assert_human_apartment_ready(tag, stable_root=stable_root, formal_registry_root=registry_root)


def test_native_v3_gate_accepts_in_place_real_height_reloaded_runtime(tmp_path):
    from human_apartment_gate import assert_native_rocketbox_apartment_ready

    tag, runtime_root, import_root, registry_root = _ready_native_rocketbox(
        tmp_path
    )
    evidence = assert_native_rocketbox_apartment_ready(
        tag,
        runtime_root=runtime_root,
        ue_import_root=import_root,
        formal_registry_root=registry_root,
    )

    assert evidence["tag"] == tag
    assert evidence["usage_scope"] == "research_candidate"
    assert evidence["bone_count"] == 80
    assert evidence["actor_scale"] == 1.0
    assert evidence["height_cm"] == pytest.approx(183.1)
    assert evidence["walking_in_place"] is True
    assert evidence["dynamic_ground_snap_to_floor_required"] is True
    assert set(evidence["animations"]) == {"Walking", "Standing_Idle"}


def test_native_v3_gate_rejects_stale_runtime_and_wrong_height(tmp_path):
    from human_apartment_gate import (
        HumanApartmentGateError,
        assert_native_rocketbox_apartment_ready,
    )

    tag, runtime_root, import_root, registry_root = _ready_native_rocketbox(
        tmp_path
    )
    (runtime_root / tag / "runtime.glb").write_bytes(b"stale")
    with pytest.raises(HumanApartmentGateError, match="runtime GLB hash"):
        assert_native_rocketbox_apartment_ready(
            tag,
            runtime_root=runtime_root,
            ue_import_root=import_root,
            formal_registry_root=registry_root,
        )

    tag, runtime_root, import_root, registry_root = _ready_native_rocketbox(
        tmp_path / "height"
    )
    manifest_path = import_root / tag / "ue_import_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_contract"]["bounds"]["height_cm"] = 93.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HumanApartmentGateError, match="adult height"):
        assert_native_rocketbox_apartment_ready(
            tag,
            runtime_root=runtime_root,
            ue_import_root=import_root,
            formal_registry_root=registry_root,
        )


def test_apartment_runner_routes_human_tags_through_artifact_gate():
    runner = (REPO / "tools/spike_rlr/run_render_pass_apartment.py").read_text()

    assert "def _assert_source_review_gates" in runner
    assert "ALLOWED_HUMAN_SPIKES" in runner
    assert "assert_human_apartment_ready" in runner
    assert "NATIVE_ROCKETBOX_HUMAN_CANDIDATES" in runner
    assert "assert_native_rocketbox_apartment_ready" in runner
    assert "skip_review_gate=skip_review_gate" in runner
    assert "human_gate_evidence = _assert_source_review_gates(spec)" in runner
    assert '"human_gate_evidence": human_gate_evidence' in runner


def test_human_smoke_launcher_uses_one_stable_command_and_sets_render_environment():
    launcher = REPO / "tools/spike_rlr/run_human_apartment_smoke.py"
    source = launcher.read_text()

    assert 'os.environ.setdefault("DISPLAY", ":99")' in source
    assert 'os.environ.setdefault("VK_ICD_FILENAMES"' in source
    assert source.index('os.environ.setdefault("DISPLAY"') < source.index(
        "from run_render_pass_apartment import render_apartment"
    )
    assert "render_function(" in source


def test_human_smoke_launcher_can_publish_complete_evidence_and_command_log():
    launcher = (
        REPO / "tools" / "spike_rlr" / "run_human_apartment_smoke.py"
    ).read_text()

    assert '"--finalize-evidence"' in launcher
    assert "from human_apartment_evidence import finalize_human_apartment_clip" in launcher
    assert "finalize_function(" in launcher
    assert 'out_dir / "command.log"' in launcher
    assert 'os.environ.setdefault("MPLCONFIGDIR"' in launcher


def test_human_smoke_explicit_finalize_stage_skips_ue_render(tmp_path):
    import run_human_apartment_smoke as launcher

    calls = []
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    out_dir = tmp_path / "clip"

    launcher.execute_stage(
        stage="finalize",
        spec_path=spec,
        out_dir=out_dir,
        clip_id="clip_001",
        render_function=lambda *args: calls.append("render"),
        finalize_function=lambda **kwargs: calls.append("finalize"),
    )

    assert calls == ["finalize"]


def test_human_smoke_explicit_render_stage_skips_cpu_finalize(tmp_path):
    import run_human_apartment_smoke as launcher

    calls = []
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    out_dir = tmp_path / "clip"

    launcher.execute_stage(
        stage="render",
        spec_path=spec,
        out_dir=out_dir,
        clip_id="clip_001",
        render_function=lambda *args: calls.append("render"),
        finalize_function=lambda **kwargs: calls.append("finalize"),
    )

    assert calls == ["render"]


def test_apartment_runner_allows_short_canary_warmup_without_changing_defaults():
    runner = (REPO / "tools/spike_rlr/run_render_pass_apartment.py").read_text()

    assert 'render_config.get("streaming_warmup_frames", 120)' in runner
    assert 'render_config.get("camera_warmup_frames", 40)' in runner
    assert "instance.step(num_frames=streaming_warmup_frames)" in runner
    assert "instance.step(num_frames=camera_warmup_frames)" in runner


def test_apartment_capture_fps_controls_spear_fixed_delta_time():
    apartment_api = (REPO / "examples/render_in_apartment.py").read_text()
    runner = (REPO / "tools/spike_rlr/run_render_pass_apartment.py").read_text()

    assert "def configure_instance(rpc_port, fixed_delta_time=None):" in apartment_api
    assert "FIXED_DELTA_TIME = float(fixed_delta_time)" in apartment_api
    assert "fixed_delta_time=1.0 / fps" in runner


def test_apartment_render_supports_isolated_parallel_rpc_and_gpu_selection():
    apartment_api = (REPO / "examples/render_in_apartment.py").read_text()
    runner = (REPO / "tools/spike_rlr/run_render_pass_apartment.py").read_text()

    assert 'os.environ.get("SPEAR_APARTMENT_RPC_PORT", "39004")' in runner
    assert 'os.environ.get("SPEAR_GRAPHICS_ADAPTER")' in apartment_api
    assert 'os.environ.get("SPEAR_RENDER_OFFSCREEN", "0")' in apartment_api
    assert "parallel_instance_settings(" in apartment_api
    assert "SHARED_MEMORY_INITIAL_UNIQUE_ID" in apartment_api
    assert "COMMAND_LINE_ARGS.graphicsadapter" in apartment_api
    assert "COMMAND_LINE_ARGS.renderoffscreen" in apartment_api
