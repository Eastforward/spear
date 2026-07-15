from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "finalize_stable_animal_ofat_review.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_finalizer_requires_full_domain_and_true_ofat_coverage():
    text = source()
    assert "all_domain_values_realized" in text
    assert "every_nonbaseline_entry_changes_exactly_one_attribute" in text
    assert "selection is not one-factor-at-a-time" in text
    assert "one --selection must be named baseline" in text


def test_finalizer_authenticates_invariants_actions_and_media():
    text = source()
    assert "topology_uv_skin_unchanged" in text
    assert "actions_unchanged" in text
    assert 'REQUIRED_ACTIONS = {"Walking", "Idle"}' in text
    assert "passed_automatic_deformation_measurements" in text
    assert "ffprobe" in text


def test_finalizer_never_promotes_without_human_ue_and_audio_review():
    text = source()
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"human_visual_review": "pending"' in text
    assert '"ue_apartment": "pending"' in text
    assert '"audio": "pending"' in text
    assert 'output.open("x"' in text
