import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.spike_rlr.acoustic_scene_contract import (
    approved_independent_rlr_subprocess_contract,
    approved_rlr_renderer_contract,
)
from tools.spike_rlr import sample_active_frame_rir_rlr as worker
from tools.spike_rlr import run_audio_pass_rlr as runner


def _descriptor(name, payload):
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_snapshot(root, name, payload):
    path = root / name
    path.write_bytes(payload)
    path.chmod(0o400)
    return path


def test_audio_only_stub_exposes_fail_closed_imported_scipy_surfaces(
    monkeypatch,
):
    for name in list(sys.modules):
        if name == "scipy" or name.startswith("scipy."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    runner._install_habitat_audio_only_scipy_stub()

    from scipy.interpolate import CubicSpline
    from scipy.stats import truncnorm

    with pytest.raises(RuntimeError, match="random-scene splines"):
        CubicSpline([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(RuntimeError, match="noisy locomotion"):
        truncnorm.rvs()


@pytest.mark.parametrize(
    "coordinate_frame",
    [
        None,
        "right-handed Z-up meters",
        {"system": "right-handed Y-up meters"},
    ],
)
def test_worker_rejects_explicit_noncanonical_coordinate_frame(
    coordinate_frame,
):
    assert worker._uses_canonical_coordinate_frame_if_declared({}) is True
    assert worker._uses_canonical_coordinate_frame_if_declared(
        {"coordinate_frame": {"system": "right-handed Z-up meters"}}
    ) is True
    assert (
        worker._uses_canonical_coordinate_frame_if_declared(
            {"coordinate_frame": coordinate_frame}
        )
        is False
    )


def test_worker_consumes_rename_stable_readonly_snapshot(
    tmp_path,
    monkeypatch,
):
    tag = "dog_snapshot_canary"
    # Production apartment_v1 specs leave this descriptive field absent; the
    # authenticated request is the authority for source-position coordinates.
    spec = {
        "spec_version": "apartment_v1",
        "audio_config": {"sample_rate_hz": 16000},
        "render_config": {"n_frames": 3, "fps": 15},
        "mic": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
        "sources": [{"tag": tag}],
    }
    spec_payload = json.dumps(spec).encode()
    mesh_payload = b"approved mesh snapshot"
    materials_payload = b'{"materials":[]}'
    derived_payload = b"approved derived snapshot"
    subprocess_contract = approved_independent_rlr_subprocess_contract()
    renderer_contract = approved_rlr_renderer_contract(
        sample_rate_hz=16000,
        quality_mode="high",
    )
    trajectory = np.asarray(
        [[1.0, 0.0, 0.4], [1.1, 0.0, 0.4], [1.2, 0.0, 0.4]],
        dtype=np.dtype("<f8"),
    )
    trajectory_sha256 = hashlib.sha256(
        trajectory.tobytes(order="C")
    ).hexdigest()
    request = {
        "schema": worker.REQUEST_SCHEMA,
        "source_tag": tag,
        "frame_index": 2,
        "selection_seed_sha256": "a" * 64,
        "source_position_coordinate_frame": "right-handed Z-up meters",
        "trajectory_prefix_shape": [3, 3],
        "trajectory_prefix_sha256": trajectory_sha256,
        "warmup_source_positions_scene_m": trajectory.tolist(),
        "mic_position_scene_m": [0.0, 0.0, 1.2],
        "mic_yaw_deg": 0.0,
        "quality_mode": "high",
        "renderer_contract": renderer_contract,
        "subprocess_contract": subprocess_contract,
        "spec": _descriptor("spec.json", spec_payload),
        "acoustic_mesh": _descriptor("acoustic_mesh.glb", mesh_payload),
        "acoustic_materials": _descriptor(
            "acoustic_materials.json",
            materials_payload,
        ),
        "derived_rlr_materials": _descriptor(
            "rlr_materials.json",
            derived_payload,
        ),
    }
    request_payload = (
        json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    original = tmp_path / "snapshot"
    original.mkdir()
    _write_snapshot(original, "spec.json", spec_payload)
    _write_snapshot(original, "acoustic_mesh.glb", mesh_payload)
    _write_snapshot(
        original,
        "acoustic_materials.json",
        materials_payload,
    )
    _write_snapshot(original, "rlr_materials.json", derived_payload)
    _write_snapshot(original, "request.json", request_payload)
    original.chmod(0o500)
    snapshot_fd = os.open(
        original,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )

    # Replace the pathname after pinning the directory inode.  The worker must
    # still feed the original authenticated bytes to every path-only loader.
    moved = tmp_path / "pinned_original"
    original.rename(moved)
    original.mkdir()
    for name in (
        "spec.json",
        "acoustic_mesh.glb",
        "acoustic_materials.json",
        "rlr_materials.json",
        "request.json",
    ):
        _write_snapshot(original, name, b"forged replacement")
    original.chmod(0o500)

    monkeypatch.setattr(
        worker,
        "_validate_runtime_identity",
        lambda _fd: (
            subprocess_contract,
            subprocess_contract["python"]["sha256"],
            subprocess_contract["worker"]["sha256"],
        ),
    )
    monkeypatch.setattr(worker, "np", np)
    monkeypatch.setattr(worker, "_activate_runtime", lambda _contract: None)
    monkeypatch.setattr(
        worker,
        "validate_approved_acoustic_scene_inputs",
        lambda value, *, mesh_payload, materials_payload: (
            value,
            mesh_payload,
            materials_payload,
        ),
    )
    monkeypatch.setattr(
        worker,
        "build_rlr_materials_payload",
        lambda _materials: derived_payload,
    )
    observed_inputs = {}

    class FakeSensor:
        def setAudioSourceTransform(self, value):
            observed_inputs.setdefault("positions", []).append(value)

    class FakeSim:
        def get_sensor_observations(self):
            return {
                "audio_sensor": np.asarray(
                    [[1.0, 0.1], [0.65, 0.05]],
                    dtype=np.float32,
                )
            }

        def close(self):
            observed_inputs["closed"] = True

    def fake_build(mesh_path, materials_path, **_kwargs):
        observed_inputs["mesh"] = Path(mesh_path).read_bytes()
        observed_inputs["derived"] = Path(materials_path).read_bytes()
        return FakeSim(), FakeSensor()

    monkeypatch.setattr(worker, "build_rlr_sim", fake_build)
    monkeypatch.setattr(worker, "_set_agent_pose", lambda *_args: None)
    monkeypatch.setattr(worker, "_habitat_from_scene", lambda value: value)

    worker_source = tmp_path / "worker.py"
    worker_source.write_bytes(b"pinned worker")
    worker_fd = os.open(worker_source, os.O_RDONLY)
    result_path = tmp_path / "anonymous_result"
    output_fd = os.open(
        result_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    result_path.unlink()
    try:
        worker.run(
            snapshot_fd=snapshot_fd,
            request_name="request.json",
            output_fd=output_fd,
            worker_fd=worker_fd,
        )
        os.lseek(output_fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(output_fd), "rb") as stream:
            with np.load(stream, allow_pickle=False) as archive:
                rir = np.asarray(archive["rir"])
    finally:
        os.close(output_fd)
        os.close(worker_fd)
        os.close(snapshot_fd)

    assert observed_inputs["mesh"] == mesh_payload
    assert observed_inputs["derived"] == derived_payload
    assert len(observed_inputs["positions"]) == 3
    assert observed_inputs["closed"] is True
    assert np.array_equal(
        rir,
        np.asarray([[1.0, 0.1], [0.65, 0.05]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "variant",
    [
        "missing_frame",
        "extra_frame",
        "integer_coordinate",
        "nonfinite_coordinate",
        "wrong_declared_shape",
        "wrong_prefix_hash",
    ],
)
def test_worker_rejects_malformed_continuous_trajectory_prefix(
    monkeypatch,
    variant,
):
    monkeypatch.setattr(worker, "np", np)
    prefix = [
        [1.0, 0.0, 0.4],
        [1.1, 0.0, 0.4],
        [1.2, 0.0, 0.4],
    ]
    payload = np.asarray(prefix, dtype=np.dtype("<f8")).tobytes(order="C")
    request = {
        "trajectory_prefix_shape": [3, 3],
        "trajectory_prefix_sha256": hashlib.sha256(payload).hexdigest(),
        "warmup_source_positions_scene_m": prefix,
    }
    if variant == "missing_frame":
        request["warmup_source_positions_scene_m"] = prefix[:-1]
    elif variant == "extra_frame":
        request["warmup_source_positions_scene_m"] = prefix + [prefix[-1]]
    elif variant == "integer_coordinate":
        request["warmup_source_positions_scene_m"][0][0] = 1
    elif variant == "nonfinite_coordinate":
        request["warmup_source_positions_scene_m"][0][0] = float("nan")
    elif variant == "wrong_declared_shape":
        request["trajectory_prefix_shape"] = [2, 3]
    elif variant == "wrong_prefix_hash":
        request["trajectory_prefix_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="trajectory prefix"):
        worker._validated_trajectory_prefix(request, frame_index=2)


def test_worker_process_rejects_wrong_environment_before_render(tmp_path):
    contract = approved_independent_rlr_subprocess_contract()
    worker_path = Path(__file__).resolve().parents[3] / contract["worker"]["path"]
    worker_fd = os.open(
        worker_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.chmod(0o500)
    snapshot_fd = os.open(
        snapshot,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    result_path = tmp_path / "result"
    output_fd = os.open(
        result_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    result_path.unlink()
    environment = dict(contract["environment"])
    environment.pop("PYTHONNOUSERSITE")
    try:
        completed = subprocess.run(
            [
                contract["python"]["resolved_path"],
                *contract["python"]["isolated_flags"],
                f"/proc/self/fd/{worker_fd}",
                "--snapshot-fd",
                str(snapshot_fd),
                "--request-name",
                "request.json",
                "--output-fd",
                str(output_fd),
                "--worker-fd",
                str(worker_fd),
            ],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(worker_fd, snapshot_fd, output_fd),
            timeout=15,
            check=False,
        )
    finally:
        os.close(output_fd)
        os.close(snapshot_fd)
        os.close(worker_fd)

    assert completed.returncode != 0
    assert b"subprocess environment changed" in completed.stderr
