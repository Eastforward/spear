from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_geometry_pixal_static_decision_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_geometry_pixal_static_decision_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
decision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decision)


def test_decision_requires_all_five_views_and_contact_sheet():
    assert decision.REQUIRED_ARTIFACTS == {
        "front.png",
        "back.png",
        "side.png",
        "top.png",
        "quarter.png",
        "contact_sheet.png",
    }


def test_cli_accepts_only_explicit_pass_or_rejection():
    args = decision.parse_args(
        [
            "--asset-id",
            "route2_v3_male_shorts",
            "--status",
            "agent_static_visual_passed",
            "--notes",
            "five-view evidence passed",
        ]
    )
    assert args.asset_id == "route2_v3_male_shorts"
    with pytest.raises(SystemExit):
        decision.parse_args(
            [
                "--asset-id",
                "route2_v3_male_shorts",
                "--status",
                "approved",
                "--notes",
                "invalid formal claim",
            ]
        )


def test_publication_is_immutable_no_replace_and_never_formal():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "os.O_EXCL" in source
    assert "os.O_NOFOLLOW" in source
    assert '"formal_dataset_registration_authorized": False' in source
    assert '"user_acceptance": "not_claimed"' in source
    assert '"tokenrig_preflight_authorized": passed' in source
