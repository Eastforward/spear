from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_geometry_tokenrig_preflight_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_geometry_tokenrig_preflight_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_all_eight_pixal_contract_assets_are_available_for_preflight():
    assets = runner.allowed_assets()
    assert len(assets) == 8
    assert len(set(assets)) == 8
    assert all(asset.startswith("route2_v3_") for asset in assets)


def test_first_four_static_agent_gates_are_authenticated():
    for asset in (
        "route2_v3_male_long_sleeve",
        "route2_v3_female_long_sleeve",
        "route2_v3_male_shorts",
        "route2_v3_female_shorts",
    ):
        gate = runner.authenticate_static_gate(asset)
        assert gate["decision"]["size_bytes"] > 0
        assert gate["contact_sheet"]["size_bytes"] > 0


def test_controlled_inputs_are_packed_pbr_and_exactly_pinned():
    asset = "route2_v3_male_long_sleeve"
    contract = runner._contract(asset)
    snapshot = runner.read_controlled_input(
        contract.input_glb, contract.input_manifest, contract
    )
    assert snapshot["glb"]["readback"]["mesh_count"] == 1
    assert snapshot["glb"]["readback"]["skin_count"] == 0
    assert snapshot["pbr_glb_readback"]["packed_pbr"] is True
    assert snapshot["pbr_glb_readback"]["image_count"] >= 2


def test_authenticated_port_override_updates_demo_client_and_bpy_server_spec(monkeypatch):
    src = types.ModuleType("src")
    src.__path__ = []
    server = types.ModuleType("src.server")
    server.__path__ = []
    spec = types.ModuleType("src.server.spec")
    spec.BPY_PORT = 59876
    spec.BPY_SERVER = "http://localhost:59876"
    server.spec = spec
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.server", server)
    monkeypatch.setitem(sys.modules, "src.server.spec", spec)
    monkeypatch.setenv("TOKENRIG_BPY_PORT", "59879")

    exec(runner.PORT_OVERRIDE_SOURCE, {})

    assert spec.BPY_PORT == 59879
    assert spec.BPY_SERVER == "http://localhost:59879"
    skin_root = MODULE_PATH.parents[2] / "SkinTokens"
    demo = (skin_root / "demo.py").read_text(encoding="utf-8")
    bpy_server = (skin_root / "src/server/bpy_server.py").read_text(encoding="utf-8")
    assert "BPY_SERVER" in demo and "from src.server.spec import" in demo
    assert "from .spec import bytes_to_object, object_to_bytes, BPY_PORT" in bpy_server


def test_runner_forces_direct_transfer_and_records_parallel_port_assignment():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "use_skeleton_input=False" in source
    assert '"use_transfer": True, "use_skeleton": False' in source
    assert "fcntl.LOCK_EX" in source
    assert "ALLOWED_PORTS = (59876, 59877, 59878, 59879)" in source
    assert '"TOKENRIG_BPY_PORT": str(port)' in source
    assert '"bpy_port": port' in source
    assert '"static_binding_audit": "pending"' in source
    assert '"formal_dataset_registration_authorized": False' in source
