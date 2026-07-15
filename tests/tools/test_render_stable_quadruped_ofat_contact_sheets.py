from pathlib import Path

from tools import render_stable_quadruped_ofat_contact_sheets as review


def test_visible_mesh_filter_keeps_accessories_and_drops_material_free_helpers():
    inventory = {
        "meshes": [
            {
                "name": "Body",
                "materials": ["Coat"],
                "vertices_with_weights": 100,
                "armature_modifiers": ["Rig"],
            },
            {
                "name": "Horns",
                "materials": ["Horn"],
                "vertices_with_weights": 0,
                "armature_modifiers": [],
            },
            {
                "name": "Icosphere",
                "materials": [],
                "vertices_with_weights": 0,
                "armature_modifiers": [],
            },
        ]
    }

    assert [item["name"] for item in review.visible_meshes(inventory, skinned_only=False)] == [
        "Body",
        "Horns",
    ]
    assert [item["name"] for item in review.visible_meshes(inventory, skinned_only=True)] == [
        "Body"
    ]


def evidence(label, changed, attributes, diagonal, width, luminance, head, gray, desat):
    return {
        "entry": {
            "label": label,
            "changed_attribute_from_baseline": changed,
            "instance_id": f"instance_{label}",
            "sampled_attributes": attributes,
        },
        "skinned_diagonal": diagonal,
        "skinned_extent": [2.0, width, 1.0],
        "torso_lateral_rms": width,
        "head_radius_rms": head,
        "coat_luminance": luminance,
        "manifest": {
            "realization": {
                "shape": {"head_scale": head},
                "materials": {
                    "muzzle_gray_mix": gray,
                    "senior_coat_desaturation": desat,
                },
            }
        },
    }


def test_ofat_order_and_automatic_attribute_ordering_are_absolute():
    base = {
        "size": "medium",
        "body_build": "standard",
        "coat_tone": "natural",
        "life_stage": "adult",
    }
    values = [
        evidence("baseline", None, base, 10.0, 2.0, 0.5, 1.0, 0.0, 0.0),
        evidence("size_small", "size", {**base, "size": "small"}, 9.0, 1.8, 0.5, 1.0, 0.0, 0.0),
        evidence("size_large", "size", {**base, "size": "large"}, 11.0, 2.2, 0.5, 1.0, 0.0, 0.0),
        evidence("build_slim", "body_build", {**base, "body_build": "slim"}, 10.0, 1.8, 0.5, 1.0, 0.0, 0.0),
        evidence("build_stocky", "body_build", {**base, "body_build": "stocky"}, 10.0, 2.2, 0.5, 1.0, 0.0, 0.0),
        evidence("coat_light", "coat_tone", {**base, "coat_tone": "light"}, 10.0, 2.0, 0.7, 1.0, 0.0, 0.0),
        evidence("coat_dark", "coat_tone", {**base, "coat_tone": "dark"}, 10.0, 2.0, 0.3, 1.0, 0.0, 0.0),
        evidence("age_young", "life_stage", {**base, "life_stage": "young"}, 10.0, 2.0, 0.5, 1.06, 0.0, 0.0),
        evidence("age_senior", "life_stage", {**base, "life_stage": "senior"}, 10.0, 2.0, 0.5, 0.98, 0.5, 0.18),
    ]

    ordered = review.order_entries(list(reversed(values)))
    result = review.automatic_checks(ordered)

    assert [item["entry"]["label"] for item in ordered] == [
        "baseline",
        "size_small",
        "size_large",
        "build_slim",
        "build_stocky",
        "coat_light",
        "coat_dark",
        "age_young",
        "age_senior",
    ]
    assert result["size_order"] is True
    assert result["body_build_width_order"] is True
    assert result["coat_luminance_strict_order"] is True


def test_publication_record_rewrites_only_the_staging_prefix(tmp_path: Path):
    staging = tmp_path / ".review.staging"
    output = tmp_path / "review"
    artifact = staging / "profile" / "frame.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")

    result = review.publication_record(artifact, staging, output)

    assert result["absolute_path"] == str(output / "profile" / "frame.png")
    assert result["size_bytes"] == 3
