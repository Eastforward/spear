"""Validate apartment_v1_spec.json and apartment_furniture_categories.json."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CATEGORIES = REPO / "tools" / "spike_rlr" / "apartment_furniture_categories.json"
SPEC = REPO / "data" / "apartment_v1_spec.json"
FURNITURE_MAP = REPO / "data" / "apartment_furniture_map.json"


def test_categories_covers_all_furniture_actors():
    cats = json.loads(CATEGORIES.read_text())
    fmap = json.loads(FURNITURE_MAP.read_text())
    all_actors = {f["actor_name"] for f in fmap["furniture"]}
    classified = set(cats["core"]) | set(cats["decoration"]) | set(cats["misc"])
    missing = all_actors - classified
    extra = classified - all_actors
    assert not missing, f"unclassified actors: {missing}"
    assert not extra, f"unknown actors classified: {extra}"


def test_categories_are_disjoint():
    cats = json.loads(CATEGORIES.read_text())
    a = set(cats["core"]); b = set(cats["decoration"]); c = set(cats["misc"])
    assert not (a & b), f"core & decoration overlap: {a & b}"
    assert not (a & c), f"core & misc overlap: {a & c}"
    assert not (b & c), f"decoration & misc overlap: {b & c}"


def test_apartment_v1_spec_schema():
    s = json.loads(SPEC.read_text())
    assert s["spec_version"] == "apartment_v1"
    assert s["room_backend"] == "apartment_shell"
    assert "mic" in s and "pos_m" in s["mic"] and "yaw_deg" in s["mic"]
    assert "camera_configs" in s and len(s["camera_configs"]) == 1
    assert s["camera_configs"][0]["fov_deg"] == 90.0
    assert "furniture_mode" in s and s["furniture_mode"] in ("shell", "subset", "full")
    assert "sources" in s and len(s["sources"]) == 2  # golden + husky, hand-tuned
    for src in s["sources"]:
        assert "tag" in src and "audio_lookup" in src
        assert "start_pos_m" in src and "end_pos_m" in src


def test_apartment_v1_spec_mic_and_camera_glued():
    s = json.loads(SPEC.read_text())
    mic_pos = s["mic"]["pos_m"]
    mic_yaw = s["mic"]["yaw_deg"]
    cam = s["camera_configs"][0]
    assert cam["pos_m"] == mic_pos, "camera pos must equal mic pos (C-glued)"
    assert cam["yaw_deg"] == mic_yaw, "camera yaw must equal mic yaw (C-glued)"


def test_apartment_v1_spec_subset_categories_are_valid():
    s = json.loads(SPEC.read_text())
    cats = json.loads(CATEGORIES.read_text())
    valid = {"core", "decoration", "misc"}
    for c in s["furniture_include_categories"]:
        assert c in valid, f"unknown category in furniture_include_categories: {c!r}"
    # Verify referenced categories exist in the categories JSON
    for c in s["furniture_include_categories"]:
        assert c in cats, f"spec references category {c!r} not present in categories JSON"
