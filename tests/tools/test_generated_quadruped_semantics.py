import pytest

from tools.generated_quadruped_semantics import (
    SemanticRigError,
    infer_quadruped_semantics,
    quadruped_semantic_labels,
)


def bone(name, parent, x, y, z, children=()):
    return {
        "name": name,
        "parent": parent,
        "children": list(children),
        "head_world": [x, y, z],
        "tail_world": [x, y, z - 0.02],
    }


def synthetic_quadruped():
    return [
        bone("root", None, 0.2, 0.0, 0.5, ("spine", "hl", "hr", "tail")),
        bone("spine", "root", 0.0, 0.0, 0.5, ("neck", "fl", "fr")),
        bone("neck", "spine", -0.2, 0.0, 0.6, ("head",)),
        bone("head", "neck", -0.4, 0.0, 0.65),
        bone("tail", "root", 0.45, 0.0, 0.55, ("tail_tip",)),
        bone("tail_tip", "tail", 0.65, 0.0, 0.6),
        bone("fl", "spine", -0.15, -0.2, 0.35, ("fl_foot",)),
        bone("fl_foot", "fl", -0.2, -0.2, 0.02),
        bone("fr", "spine", -0.15, 0.2, 0.35, ("fr_foot",)),
        bone("fr_foot", "fr", -0.2, 0.2, 0.02),
        bone("hl", "root", 0.2, -0.2, 0.35, ("hl_foot",)),
        bone("hl_foot", "hl", 0.25, -0.2, 0.02),
        bone("hr", "root", 0.2, 0.2, 0.35, ("hr_foot",)),
        bone("hr_foot", "hr", 0.25, 0.2, 0.02),
    ]


def test_infers_quadruped_chains_without_bone_names():
    result = infer_quadruped_semantics(
        synthetic_quadruped(),
        bbox_min=(-0.5, -0.3, 0.0),
        bbox_extent=(1.2, 0.6, 0.8),
        front_axis="negative-x",
    )

    assert result.root == "root"
    assert result.axial == ("root", "spine")
    assert result.head_chain == ("neck", "head")
    assert result.tail_chain == ("tail", "tail_tip")
    assert result.front_side_negative == ("fl", "fl_foot")
    assert result.front_side_positive == ("fr", "fr_foot")
    assert result.hind_side_negative == ("hl", "hl_foot")
    assert result.hind_side_positive == ("hr", "hr_foot")
    assert set(result.all_bones()) == {item["name"] for item in synthetic_quadruped()}


def test_rejects_non_quadruped_low_endpoint_count():
    records = synthetic_quadruped()
    records[-1]["head_world"][2] = 0.4

    with pytest.raises(SemanticRigError, match="exactly four"):
        infer_quadruped_semantics(
            records,
            bbox_min=(-0.5, -0.3, 0.0),
            bbox_extent=(1.2, 0.6, 0.8),
            front_axis="negative-x",
        )


def test_infers_same_chains_for_cardinal_negative_y_front_axis():
    records = []
    for record in synthetic_quadruped():
        converted = dict(record)
        converted["children"] = list(record["children"])
        # The synthetic animal faces -X. Rotate its coordinates +90 degrees so
        # that the same animal faces -Y while preserving vertical Z.
        x, y, z = record["head_world"]
        converted["head_world"] = [-y, x, z]
        x, y, z = record["tail_world"]
        converted["tail_world"] = [-y, x, z]
        records.append(converted)

    result = infer_quadruped_semantics(
        records,
        bbox_min=(-0.3, -0.5, 0.0),
        bbox_extent=(0.6, 1.2, 0.8),
        front_axis="negative-y",
    )

    assert result.root == "root"
    assert result.axial == ("root", "spine")
    assert result.head_chain == ("neck", "head")
    assert result.tail_chain == ("tail", "tail_tip")
    assert set(result.foot_leaves) == {
        "fl_foot",
        "fr_foot",
        "hl_foot",
        "hr_foot",
    }


def test_preserves_high_auxiliary_head_branches_without_calling_them_limbs():
    records = synthetic_quadruped()
    neck = next(record for record in records if record["name"] == "neck")
    neck["children"] = ["head", "ear_l", "ear_r"]
    records.extend(
        [
            bone("ear_l", "neck", -0.28, -0.08, 0.72, ("ear_l_tip",)),
            bone("ear_l_tip", "ear_l", -0.30, -0.10, 0.78),
            bone("ear_r", "neck", -0.28, 0.08, 0.72, ("ear_r_tip",)),
            bone("ear_r_tip", "ear_r", -0.30, 0.10, 0.78),
        ]
    )

    result = infer_quadruped_semantics(
        records,
        bbox_min=(-0.5, -0.3, 0.0),
        bbox_extent=(1.2, 0.6, 0.8),
        front_axis="negative-x",
    )

    assert result.head_chain == ("neck", "head")
    assert result.auxiliary_branches == (
        ("ear_l", "ear_l_tip"),
        ("ear_r", "ear_r_tip"),
    )
    assert set(result.all_bones()) == {item["name"] for item in records}


def test_clusters_disconnected_low_hoof_controls_into_four_anatomical_limbs():
    records = synthetic_quadruped()
    root = next(record for record in records if record["name"] == "root")
    controls = []
    for prefix, x, y in (
        ("fl", -0.28, -0.2),
        ("fr", -0.28, 0.2),
        ("hl", 0.33, -0.2),
        ("hr", 0.33, 0.2),
    ):
        control = f"{prefix}_hoof_control"
        endpoint = f"{control}_end"
        root["children"].append(control)
        controls.extend(
            [
                bone(control, "root", x, y, 0.02, (endpoint,)),
                bone(endpoint, control, x - 0.04, y, 0.02),
            ]
        )
    records.extend(controls)

    result = infer_quadruped_semantics(
        records,
        bbox_min=(-0.5, -0.3, 0.0),
        bbox_extent=(1.2, 0.6, 0.8),
        front_axis="negative-x",
    )

    assert set(result.foot_leaves) == {
        "fl_foot",
        "fr_foot",
        "hl_foot",
        "hr_foot",
    }
    assert result.front_side_negative == ("fl", "fl_foot")
    assert result.front_side_positive == ("fr", "fr_foot")
    assert result.hind_side_negative == ("hl", "hl_foot")
    assert result.hind_side_positive == ("hr", "hr_foot")
    assert len(result.auxiliary_branches) == 4
    assert set(result.all_bones()) == {item["name"] for item in records}

    labels = quadruped_semantic_labels(
        result,
        records,
        bbox_min=(-0.5, -0.3, 0.0),
        bbox_extent=(1.2, 0.6, 0.8),
        front_axis="negative-x",
    )
    assert labels["fl_hoof_control"] == "front_side_negative"
    assert labels["fr_hoof_control_end"] == "front_side_positive"
    assert labels["hl_hoof_control"] == "hind_side_negative"
    assert labels["hr_hoof_control_end"] == "hind_side_positive"
