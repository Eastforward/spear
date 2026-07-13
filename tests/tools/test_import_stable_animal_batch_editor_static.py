from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "import_stable_animal_batch_editor.py"
)


def test_stable_import_reuses_gate_import_without_pixal_namespace():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "import_gate_animal_editor.py" in text
    assert 'tag.startswith("stable_")' in text
    assert "stable_animal_ue_import_batch_v1" in text
    assert "stable_animal_ue_import_result_v1" in text
    assert 'set(job.get("expected_actions", [])) != {"Idle", "Walking"}' in text
    assert 'os.environ["GATE_RIGGED_GLB"]' in text
    assert 'job["human_review_status"]' in text
    assert '"formal_dataset_registration_authorized": False' in text
