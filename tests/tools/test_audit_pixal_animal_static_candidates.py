from tools.audit_pixal_animal_static_candidates import DECISIONS


def test_static_decisions_cover_the_fourteen_generated_replacements():
    assert len(DECISIONS) == 14
    assert {item["status"] for item in DECISIONS.values()} == {
        "continue_to_lod_and_rig",
        "rejected",
    }
    assert sum(
        item["status"] == "continue_to_lod_and_rig"
        for item in DECISIONS.values()
    ) == 7
    assert all(
        item["rig_family"] is None
        for item in DECISIONS.values()
        if item["status"] == "rejected"
    )
