from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prepare_stable_animal_ofat_ue_imports.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_preparation_authenticates_review_glbs_deformation_and_all_domains():
    text = source()
    assert "contracts.manifest_sha256(review)" in text
    assert "instance GLB hash mismatch" in text
    assert "deformation hash mismatch" in text
    assert '"all_domain_values_covered": True' in text


def test_preparation_keeps_direction_cardinal_and_never_infers_fine_yaw():
    text = source()
    assert "CARDINAL_YAWS" in text
    assert '"automatic_fine_yaw_inference": False' in text
    assert '"authority": "explicit_cli_cardinal_value"' in text


def test_preparation_keeps_per_instance_attributes_and_never_overwrites():
    text = source()
    assert '"sampled_attributes": entry["sampled_attributes"]' in text
    assert '"target_physical_profile": entry["target_physical_profile"]' in text
    assert "refusing to replace output" in text
    assert "os.rename(staging, output_root)" in text
