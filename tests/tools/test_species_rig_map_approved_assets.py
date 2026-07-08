import hashlib
import json
import os
import subprocess
import sys

import pytest


def test_animated_map_prefers_approved_oriented_mesh(tmp_path):
    approved = tmp_path / "approved"
    tag_dir = approved / "dog_golden"
    tag_dir.mkdir(parents=True)
    mesh = tag_dir / "mesh_oriented.glb"
    mesh.write_bytes(b"approved mesh")
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "auto_orient_v1",
        "human_approved": True,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": False,
        "mesh_sha256": hashlib.sha256(b"approved mesh").hexdigest(),
        "detection": {
            "head_direction_original_mesh_frame": [1, 0, 0],
            "rotation_applied_to_align_to_plus_x": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }))

    code = """
import json
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
import species_rig_map
entry = species_rig_map.ANIMATED_RIG_MAP["dog_golden"]
print(json.dumps({"mesh": entry["mesh"], "mesh_sha256": entry.get("mesh_sha256")}))
"""
    env = {**os.environ, "HY3D_APPROVED_DIR": str(approved)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    entry = json.loads(proc.stdout)

    assert entry["mesh"] == str(mesh)
    assert entry["mesh_sha256"] == hashlib.sha256(b"approved mesh").hexdigest()


def test_animated_map_prefers_current_runtime_proxy_for_downstream(tmp_path):
    approved = tmp_path / "approved"
    tag_dir = approved / "dog_golden"
    tag_dir.mkdir(parents=True)
    canonical = tag_dir / "mesh_oriented.glb"
    runtime = tag_dir / "mesh_runtime.glb"
    canonical.write_bytes(b"approved mesh")
    runtime.write_bytes(b"runtime proxy")
    canonical_sha = hashlib.sha256(b"approved mesh").hexdigest()
    runtime_sha = hashlib.sha256(b"runtime proxy").hexdigest()
    (tag_dir / "mesh_runtime.json").write_text(json.dumps({
        "algorithm": "blender_decimate_v1",
        "source_mesh_sha256": canonical_sha,
        "target_faces": 80000,
        "actual_faces": 79999,
        "runtime_mesh_sha256": runtime_sha,
    }))
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "auto_orient_v1",
        "human_approved": True,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": False,
        "mesh_sha256": canonical_sha,
        "detection": {
            "head_direction_original_mesh_frame": [1, 0, 0],
            "rotation_applied_to_align_to_plus_x": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }))

    code = """
import json
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
import species_rig_map
entry = species_rig_map.ANIMATED_RIG_MAP["dog_golden"]
print(json.dumps({
    "mesh": entry["mesh"],
    "approved_mesh": entry.get("approved_mesh"),
    "runtime_mesh": entry.get("runtime_mesh"),
    "mesh_sha256": entry.get("mesh_sha256"),
}))
"""
    env = {**os.environ, "HY3D_APPROVED_DIR": str(approved)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    entry = json.loads(proc.stdout)

    assert entry["mesh"] == str(runtime)
    assert entry["approved_mesh"] == str(canonical)
    assert entry["runtime_mesh"] == str(runtime)
    assert entry["mesh_sha256"] == canonical_sha


def test_approved_animated_map_uses_diffuse_from_approved_dir(tmp_path):
    approved = tmp_path / "approved"
    batch = tmp_path / "batch"
    tag_dir = approved / "dog_golden"
    tag_dir.mkdir(parents=True)
    batch.mkdir()
    mesh = tag_dir / "mesh_oriented.glb"
    diffuse = tag_dir / "hy3d_diffuse.jpg"
    mesh.write_bytes(b"approved mesh")
    diffuse.write_bytes(b"diffuse")
    mesh_sha = hashlib.sha256(b"approved mesh").hexdigest()
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "auto_orient_v1",
        "human_approved": True,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": False,
        "mesh_sha256": mesh_sha,
        "detection": {
            "head_direction_original_mesh_frame": [1, 0, 0],
            "rotation_applied_to_align_to_plus_x": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }))

    code = """
import json
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
import species_rig_map
entry = species_rig_map.ANIMATED_RIG_MAP["dog_golden"]
print(json.dumps({"diffuse": entry.get("diffuse")}))
"""
    env = {
        **os.environ,
        "HY3D_APPROVED_DIR": str(approved),
        "HY3D_BATCH_DIR": str(batch),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    entry = json.loads(proc.stdout)

    assert entry["diffuse"] == str(diffuse)


def test_approved_animated_inputs_reject_missing_diffuse_for_untextured_mesh(tmp_path):
    approved = tmp_path / "approved"
    batch = tmp_path / "batch"
    tag_dir = approved / "dog_golden"
    tag_dir.mkdir(parents=True)
    batch.mkdir()
    mesh = tag_dir / "mesh_oriented.glb"
    mesh.write_bytes(b"approved mesh without embedded texture")
    mesh_sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "auto_orient_v1",
        "human_approved": True,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": False,
        "mesh_sha256": mesh_sha,
        "detection": {
            "head_direction_original_mesh_frame": [1, 0, 0],
            "rotation_applied_to_align_to_plus_x": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }))

    code = """
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
import species_rig_map
species_rig_map.assert_inputs_exist("dog_golden")
"""
    env = {
        **os.environ,
        "HY3D_APPROVED_DIR": str(approved),
        "HY3D_BATCH_DIR": str(batch),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode != 0
    assert "diffuse" in (proc.stderr + proc.stdout)


def test_gate_check_script_uses_repo_relative_spear_dir():
    script = "/data/jzy/code/AVEngine/external/SPEAR/tools/gate_check_animal.sh"
    text = open(script).read()
    assert "/data/jzy/code/SPEAR" not in text
    assert "dirname" in text and "BASH_SOURCE" in text
    assert 'if [ -n "$DIFF" ]' in text
    assert 'BLENDER_ARGS+=(--new-diffuse "$DIFF")' in text
    assert 'PYTHONPATH="$SPEAR_DIR/examples:$SPEAR_DIR/tools' in text
    assert 'DISPLAY="${DISPLAY:-:99}"' in text
    assert 'VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"' in text
    assert "GATE_CHECK_FAIL cook failed" in text
    assert "ensure_runtime_proxy_mesh.py" in text
    assert "GATE_RUNTIME_TARGET_FACES" in text
