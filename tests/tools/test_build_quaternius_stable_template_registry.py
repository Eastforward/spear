from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "build_quaternius_stable_template_registry.py"
)


def test_registry_requires_complete_evidence_and_keeps_review_pending():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_quaternius_stable_template_registry_v1" in text
    assert 'deformation.get("overall") != "passed"' in text
    assert 'set(decisions) != {"Walking", "Idle"}' in text
    assert '"agent_selected_pending_human_review"' in text
    assert '"automatic_fine_yaw_inference": False' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert "expected 12 stable templates" in text
    assert '"sha256": sha256_file(path)' in text
