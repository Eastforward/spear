from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_color_fastlane_decision_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_color_fastlane_decision_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fastlane_can_override_only_texture_correlation_advisories():
    base = {
        "automatic_2d_gate": "rejected",
        "metrics": {
            "checks": {
                **{name: True for name in runner.CRITICAL_CHECKS},
                "texture_edges_retained": False,
                "texture_luminance_retained": False,
            }
        },
    }
    assert runner.advisory_only_automatic_rejection(base) is True
    base["metrics"]["checks"]["source_alpha_byte_identical"] = False
    assert runner.advisory_only_automatic_rejection(base) is False


def test_current_eight_candidates_have_exact_critical_pixel_and_alpha_checks():
    for case_id in runner.color.CASE_BY_ID:
        _, manifest = runner._candidate(case_id)
        checks = manifest["metrics"]["checks"]
        assert all(checks[name] is True for name in runner.CRITICAL_CHECKS)
        assert runner.advisory_only_automatic_rejection(manifest) is True


def test_decisions_never_erase_original_rejection_or_claim_formal_status():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"automatic_2d_gate": manifest["automatic_2d_gate"]' in source
    assert "preserved_texture_correlation_advisory" in source
    assert '"user_acceptance": "not_claimed"' in source
    assert '"formal_dataset_registration_authorized": False' in source
