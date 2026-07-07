"""Offline tests for the apartment shell dump.

Validates the on-disk JSON contents. Live SPEAR verification is manual
(see the CLI regression check in dump_apartment_shell.py's docstring).
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_shell_map_json_exists_and_has_actors():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped — run dump_apartment_shell.py")
    d = json.loads(p.read_text())
    assert "shell_actors" in d
    assert len(d["shell_actors"]) > 0
    assert "meta" in d


def test_shell_map_no_furniture_labels():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped")
    d = json.loads(p.read_text())
    for a in d["shell_actors"]:
        assert a["shell_label"] != "furniture", f"furniture actor leaked: {a['actor_name']}"


def test_shell_map_has_walls_floor_ceiling():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped")
    d = json.loads(p.read_text())
    labels = {a["shell_label"] for a in d["shell_actors"]}
    assert "shell_wall" in labels, f"no walls in shell dump; got {labels}"
    assert "shell_floor" in labels, f"no floor in shell dump; got {labels}"
    assert "shell_ceiling" in labels, f"no ceiling in shell dump; got {labels}"
