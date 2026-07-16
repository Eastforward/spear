from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "build_fixed_skeleton_agent_motion_decision.py"
)


def test_agent_decision_is_research_only_and_never_claims_human_approval():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_fixed_skeleton_motion_basis_agent_decision_v1" in text
    assert '"human_approved": False' in text
    assert '"agent_approved_for_research": True' in text
    assert '"user_delegated_autonomous_review": True' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert "explicit user delegation is required" in text


def test_agent_decision_authenticates_full_fixed_skeleton_lineage():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "conditioning_input_is_carrier" in text
    assert "conditioning_output_is_skintokens_input" in text
    assert "skintokens_output_is_target" in text
    assert "static rig audit" in text
    assert '"manual_cardinal_motion_basis_yaw_deg": 0' in text
    assert '"side_chain_mode": "matched"' in text
    assert "refusing to replace decision" in text
