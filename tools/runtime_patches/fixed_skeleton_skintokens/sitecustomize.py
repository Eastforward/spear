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
_service_generation = os.environ.get("TOKENRIG_SERVICE_GENERATION")
if _service_generation is None:
    _service_generation = hashlib.sha256(
        f"standalone:{os.getpid()}:{os.getppid()}".encode("ascii")
    ).hexdigest()[:32]
if (
    len(_service_generation) != 32
    or any(character not in "0123456789abcdef" for character in _service_generation)
):
    raise RuntimeError("TOKENRIG_SERVICE_GENERATION must be 32 lowercase hex digits")
_service_patch_sha256 = os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"]
if (
    len(_service_patch_sha256) != 64
    or any(
        character not in "0123456789abcdef"
        for character in _service_patch_sha256
    )
):
    raise RuntimeError("TOKENRIG_SERVER_HYGIENE_SHA256 must be a SHA-256 hex digest")
_bpy_port = int(os.environ["TOKENRIG_BPY_PORT"])
if not 1024 <= _bpy_port <= 65535:
    raise RuntimeError("TOKENRIG_BPY_PORT must be in [1024, 65535]")
_bpy_bind_host = "127.0.0.1"
if os.environ.get("TOKENRIG_BPY_BIND_HOST", _bpy_bind_host) != _bpy_bind_host:
    raise RuntimeError("TOKENRIG_BPY_BIND_HOST must be 127.0.0.1")
_bpy_server = f"http://{_bpy_bind_host}:{_bpy_port}"
_health_schema = "avengine_tokenrig_warm_health_v1"
_health_payload = json.dumps(
    {
        "bind_host": _bpy_bind_host,
        "generation": _service_generation,
        "patch_sha256": _service_patch_sha256,
        "pid": os.getpid(),
        "port": _bpy_port,
        "schema": _health_schema,
    },
    sort_keys=True,
    separators=(",", ":"),
)


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
        "generation": _service_generation,
        "pid": os.getpid(),
        "patch_sha256": _service_patch_sha256,
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


def _install_bottle_loopback():
    """Override the upstream public bind for its unauthenticated pickle API."""

    module = sys.modules.get("bottle")
    if module is None or not hasattr(module, "run"):
        return
    current_run = module.run
    if getattr(current_run, "_avengine_loopback_only", False):
        return

    def _run_loopback(*args, **kwargs):
        positional = list(args)
        application = positional[0] if positional else kwargs.get("app")
        if (
            application is not None
            and hasattr(application, "route")
            and not getattr(application, "_avengine_health_installed", False)
        ):
            application.route(
                "/avengine-health",
                method="GET",
                callback=lambda: _health_payload,
            )
            application._avengine_health_installed = True
        if len(positional) >= 3:
            positional[2] = _bpy_bind_host
            kwargs.pop("host", None)
        else:
            kwargs["host"] = _bpy_bind_host
        return current_run(*positional, **kwargs)

    _run_loopback._avengine_loopback_only = True
    module.run = _run_loopback


_original_import = builtins.__import__


def _deferred_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    _install_bpy_port()
    _install_bottle_loopback()
    _install_parser_hook()
    return result


builtins.__import__ = _deferred_import
_install_bpy_port()
_install_bottle_loopback()
_install_parser_hook()


marker_dir = Path(os.environ["TOKENRIG_HYGIENE_MARKER_DIR"])
marker_dir.mkdir(parents=True, exist_ok=True)
marker = {
    "argv": list(sys.argv),
    "patch_sha256": _service_patch_sha256,
    "pid": os.getpid(),
    "seed": seed,
    "generation": _service_generation,
    "bpy_port": _bpy_port,
    "bpy_bind_host": _bpy_bind_host,
}
marker_bytes = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
marker_path = marker_dir / f"{os.getpid()}.json"
with marker_path.open("xb") as stream:
    stream.write(marker_bytes)
    stream.flush()
    os.fsync(stream.fileno())
