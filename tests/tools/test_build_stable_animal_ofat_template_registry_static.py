from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_stable_animal_ofat_template_registry.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_registry_authenticates_ofat_ue_and_two_license_sources():
    text = source()
    assert "contracts.manifest_sha256(review)" in text
    assert "input_manifest_sha256" in text
    assert '"rocketbox_geometry_material_skeleton"' in text
    assert '"quaternius_motion"' in text


def test_registry_preserves_instance_attributes_and_pending_review_state():
    text = source()
    assert '"sampled_attributes": source["sampled_attributes"]' in text
    assert '"target_physical_profile": source["target_physical_profile"]' in text
    assert 'PENDING_REVIEW_STATUS = "local_ofat_visual_review_pending"' in text
    assert '"ue_apartment_media": "pending"' in text
    assert '"formal_dataset_registration_authorized": False' in text


def test_registry_uses_generic_schema_and_refuses_overwrite():
    text = source()
    assert 'SCHEMA = "avengine_stable_animal_template_registry_v2"' in text
    assert 'payload["manifest_sha256"] = contracts.manifest_sha256(payload)' in text
    assert 'output.open("x"' in text
    assert "refusing to replace output" in text
