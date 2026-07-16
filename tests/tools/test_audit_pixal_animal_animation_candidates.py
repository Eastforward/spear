from tools.audit_pixal_animal_animation_candidates import DECISIONS


def test_animation_decisions_are_complete_and_conservative():
    assert len(DECISIONS) == 7
    assert sum(item["status"] == "continue_to_ue" for item in DECISIONS.values()) == 4
    assert DECISIONS["horse"]["status"] == "rejected"
    assert DECISIONS["donkey_ass"]["status"] == "rejected"
