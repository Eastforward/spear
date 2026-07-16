import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/audit_controlled_animal_one_shot_batch.py"
)


def test_one_shot_auditor_is_parseable_and_has_no_seed_override_cli():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)

    assert "--flux-batch" in source
    assert "--pixal-batch" in source
    assert "--output" in source
    assert "--seed" not in source
    assert "best_of" not in source
