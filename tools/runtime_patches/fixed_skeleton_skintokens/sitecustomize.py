"""Deterministic, fail-closed SkinTokens child-process hygiene.

This module is loaded only when its containing directory is explicitly added
to ``PYTHONPATH`` by ``run_fixed_skeleton_skintokens.py``.
"""

import builtins
import hashlib
import json
import os
import random
import sys
from pathlib import Path


seed = int(os.environ["TOKENRIG_CANARY_SEED"])
random.seed(seed)

import numpy as np

np.random.seed(seed)

import torch

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


_load_sequence = 0
_parser_hook_installed = False
_load_audit_path = Path(os.environ["TOKENRIG_LOAD_AUDIT_PATH"])
_load_audit_path.parent.mkdir(parents=True, exist_ok=True)
_bpy_port = int(os.environ["TOKENRIG_BPY_PORT"])
if not 1024 <= _bpy_port <= 65535:
    raise RuntimeError("TOKENRIG_BPY_PORT must be in [1024, 65535]")
_bpy_server = f"http://localhost:{_bpy_port}"


def _inventory(parser):
    objects = sorted(
        ({"name": obj.name, "type": obj.type} for obj in parser.bpy.data.objects),
        key=lambda item: (item["name"], item["type"]),
    )
    return {
        "objects": objects,
        "mesh_count": len(parser.bpy.data.meshes),
        "material_count": len(parser.bpy.data.materials),
        "image_count": len(parser.bpy.data.images),
    }


def _record_load_event(parser, sequence, phase, filepath):
    event = {
        "sequence": sequence,
        "phase": phase,
        "filepath": str(Path(filepath).resolve()),
        "inventory": _inventory(parser),
    }
    payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    with _load_audit_path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _install_parser_hook():
    global _parser_hook_installed
    if _parser_hook_installed:
        return
    parser = sys.modules.get("src.rig_package.parser.bpy")
    if parser is None or not hasattr(parser, "BpyParser"):
        return
    original_load = parser.BpyParser.load.__func__

    def _clean_load(cls, filepath, **kwargs):
        global _load_sequence
        _load_sequence += 1
        sequence = _load_sequence
        _record_load_event(parser, sequence, "before_clean", filepath)
        parser.clean_bpy()
        _record_load_event(parser, sequence, "after_clean", filepath)
        result = original_load(cls, filepath, **kwargs)
        _record_load_event(parser, sequence, "after_import", filepath)
        return result

    parser.BpyParser.load = classmethod(_clean_load)
    _parser_hook_installed = True


def _install_bpy_port():
    spec = sys.modules.get("src.server.spec")
    if spec is not None:
        spec.BPY_PORT = _bpy_port
        spec.BPY_SERVER = _bpy_server
    server = sys.modules.get("src.server.bpy_server")
    if server is not None:
        server.BPY_PORT = _bpy_port
    datapath = sys.modules.get("src.data.datapath")
    if datapath is not None:
        datapath.BPY_SERVER = _bpy_server


_original_import = builtins.__import__


def _deferred_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    _install_bpy_port()
    _install_parser_hook()
    return result


builtins.__import__ = _deferred_import
_install_bpy_port()
_install_parser_hook()


marker_dir = Path(os.environ["TOKENRIG_HYGIENE_MARKER_DIR"])
marker_dir.mkdir(parents=True, exist_ok=True)
marker = {
    "argv": list(sys.argv),
    "patch_sha256": os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"],
    "pid": os.getpid(),
    "seed": seed,
    "bpy_port": _bpy_port,
}
marker_bytes = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
marker_path = marker_dir / f"{os.getpid()}.json"
with marker_path.open("xb") as stream:
    stream.write(marker_bytes)
    stream.flush()
    os.fsync(stream.fileno())
