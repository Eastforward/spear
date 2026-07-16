from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tools import route2_deterministic_material_canary as canary
from tools import blender_render_route2_material_canary as renderer


SPEAR_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def authenticated_real_source():
    return canary.load_authenticated_source()


@pytest.fixture(scope="module")
def real_semantic_masks(authenticated_real_source):
    return canary.build_semantic_masks(authenticated_real_source)


def _synthetic_semantic_mesh():
    # Six independent triangles: valid top, skin-colored top, bottom, shoes,
    # hair, and face-colored head.  The classifier must fail closed on skin.
    centers = np.asarray(
        [
            [0.0, 0.65, 0.02],
            [0.0, 0.65, 0.02],
            [0.0, 0.30, 0.02],
            [0.0, 0.04, 0.02],
            [0.0, 0.87, 0.05],
            [0.0, 0.83, 0.11],
        ],
        dtype=np.float32,
    )
    positions = np.repeat(centers, 3, axis=0)
    positions[0::3, 0] -= 0.01
    positions[1::3, 0] += 0.01
    positions[2::3, 2] += 0.01
    triangles = np.arange(18, dtype=np.int32).reshape(-1, 3)
    joint_names = ["spine_anchor", "leg_anchor", "foot_anchor", "head_anchor"]
    dominant_slots = np.repeat([0, 0, 1, 2, 3, 3], 3)
    joints = np.zeros((18, 4), dtype=np.uint8)
    joints[:, 0] = dominant_slots
    weights = np.zeros((18, 4), dtype=np.float32)
    weights[:, 0] = 1.0
    triangle_rgb = np.asarray(
        [
            [27, 39, 21],
            [165, 121, 113],
            [56, 56, 59],
            [151, 148, 144],
            [76, 56, 47],
            [166, 122, 114],
        ],
        dtype=np.uint8,
    )
    semantic_bones = {
        "pelvis": "spine_anchor",
        "spine": ["spine_anchor"],
        "neck": "head_anchor",
        "head": "head_anchor",
        "left_clavicle": "spine_anchor",
        "left_upper_arm": "spine_anchor",
        "right_clavicle": "spine_anchor",
        "right_upper_arm": "spine_anchor",
        "left_thigh": "leg_anchor",
        "left_calf": "leg_anchor",
        "right_thigh": "leg_anchor",
        "right_calf": "leg_anchor",
        "left_foot": "foot_anchor",
        "left_toe": "foot_anchor",
        "right_foot": "foot_anchor",
        "right_toe": "foot_anchor",
    }
    return positions, triangles, joints, weights, joint_names, triangle_rgb, semantic_bones


def test_production_contract_pins_static_passed_pixal_surface_and_new_output_root():
    assert canary.SOURCE_GLB == (
        SPEAR_ROOT
        / "tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/"
        "fitted_skeleton_v1/sanitized_weights_v1/static_audit_v1/bind_pose.glb"
    )
    assert canary.SOURCE_GLB_SHA256 == (
        "1a85f2d22e6bdac230379bb57f389db7fc4c73a8f7c50f786e353374f89d6785"
    )
    assert canary.STATIC_QA_SHA256 == (
        "31cd5bf745526913d2226efd180ca10b6623db1b34111f02ae4feef6feae8990"
    )
    assert canary.OUTPUT_ROOT == SPEAR_ROOT / "tmp/route2_deterministic_material_canary_v1"


def test_semantic_triangle_classifier_accepts_only_color_and_skin_supported_cores():
    args = _synthetic_semantic_mesh()

    labels, metrics = canary.classify_semantic_triangles(
        *args[:-1], semantic_bones=args[-1]
    )

    assert labels.tolist() == [
        canary.LABELS["top"],
        canary.LABELS["unknown"],
        canary.LABELS["bottom"],
        canary.LABELS["shoes"],
        canary.LABELS["hair"],
        canary.LABELS["unknown"],
    ]
    assert metrics["triangle_counts"] == {
        "unknown": 2,
        "top": 1,
        "bottom": 1,
        "shoes": 1,
        "hair": 1,
    }
    assert metrics["classification_version"] == "male_pixal_semantic_core_v1"


def test_semantic_classifier_rejects_dark_foot_texel_instead_of_dyeing_trouser_cuff():
    positions, triangles, joints, weights, names, rgb, semantic_bones = (
        _synthetic_semantic_mesh()
    )
    rgb[3] = [40, 40, 42]

    labels, _ = canary.classify_semantic_triangles(
        positions,
        triangles,
        joints,
        weights,
        names,
        rgb,
        semantic_bones=semantic_bones,
    )

    assert labels[3] == canary.LABELS["unknown"]


def test_random_joint_renaming_with_same_semantic_roles_preserves_labels_and_uv_masks():
    positions, triangles, joints, weights, names, rgb, semantic_bones = (
        _synthetic_semantic_mesh()
    )
    original, _ = canary.classify_semantic_triangles(
        positions,
        triangles,
        joints,
        weights,
        names,
        rgb,
        semantic_bones=semantic_bones,
    )
    rename = {name: f"randomized_joint_{index}_a9f" for index, name in enumerate(names)}
    renamed_mapping = {
        role: [rename[name] for name in value]
        if isinstance(value, list)
        else rename[value]
        for role, value in semantic_bones.items()
    }
    renamed, _ = canary.classify_semantic_triangles(
        positions,
        triangles,
        joints,
        weights,
        [rename[name] for name in names],
        rgb,
        semantic_bones=renamed_mapping,
    )
    uvs = np.tile(
        np.asarray([[0.10, 0.10], [0.20, 0.10], [0.15, 0.20]], dtype=np.float32),
        (len(triangles), 1),
    )
    for triangle_index in range(len(triangles)):
        uvs[triangle_index * 3 : triangle_index * 3 + 3, 0] += triangle_index * 0.12
    for region in canary.REGIONS:
        original_mask = canary.rasterize_uv_triangles(
            uvs,
            triangles,
            np.flatnonzero(original == canary.LABELS[region]),
            width=128,
            height=128,
        )
        renamed_mask = canary.rasterize_uv_triangles(
            uvs,
            triangles,
            np.flatnonzero(renamed == canary.LABELS[region]),
            width=128,
            height=128,
        )
        assert np.array_equal(original_mask, renamed_mask)


def test_uv_triangle_rasterization_is_bounded_and_nonempty():
    uvs = np.asarray([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]], dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)

    mask = canary.rasterize_uv_triangles(uvs, triangles, [0], width=32, height=32)

    assert mask.dtype == np.bool_
    assert 80 < int(mask.sum()) < 200
    assert not mask[0].any()
    assert not mask[-1].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()


def test_overlap_resolution_clears_every_conflicting_texel_from_all_regions():
    masks = {name: np.zeros((8, 8), dtype=bool) for name in canary.REGIONS}
    masks["top"][2:5, 2:5] = True
    masks["bottom"][4:7, 4:7] = True

    resolved, conflict = canary.resolve_mask_conflicts(masks)

    assert int(conflict.sum()) == 1
    assert not resolved["top"][4, 4]
    assert not resolved["bottom"][4, 4]
    total = sum(mask.astype(np.uint8) for mask in resolved.values())
    assert int(total.max()) <= 1


def test_registered_color_transform_changes_only_masked_rgb_and_preserves_alpha():
    rgba = np.full((12, 12, 4), [72, 75, 70, 211], dtype=np.uint8)
    rgba[4:8, 4:8, :3] = [31, 42, 24]
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    first, proof = canary.apply_registered_color(
        rgba, mask, region="top", palette_name="cobalt_blue"
    )
    second, second_proof = canary.apply_registered_color(
        rgba, mask, region="top", palette_name="cobalt_blue"
    )

    assert np.array_equal(first, second)
    assert proof == second_proof
    assert proof["outside_mask_changed_texels"] == 0
    assert proof["alpha_changed_texels"] == 0
    assert proof["inside_mask_changed_texels"] == 16
    assert np.array_equal(first[~mask], rgba[~mask])
    assert np.array_equal(first[:, :, 3], rgba[:, :, 3])


def test_color_transform_rejects_unregistered_region_palette_pair():
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(canary.MaterialCanaryError, match="registered palette"):
        canary.apply_registered_color(
            rgba, mask, region="top", palette_name="invented_magenta"
        )
    with pytest.raises(canary.MaterialCanaryError, match="registered palette"):
        canary.apply_registered_color(
            rgba, mask, region="hair", palette_name="cobalt_blue"
        )


def test_embedded_image_replacement_preserves_every_non_target_buffer_view_payload():
    document = {
        "buffers": [{"byteLength": 28}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 4},
            {"buffer": 0, "byteOffset": 4, "byteLength": 8},
            {"buffer": 0, "byteOffset": 12, "byteLength": 8},
            {"buffer": 0, "byteOffset": 20, "byteLength": 8},
        ],
    }
    binary = b"GEOM" + b"BASEBASE" + b"METALMET" + b"BINDMATR"
    before = copy.deepcopy(document)

    updated, new_binary, proof = canary.replace_buffer_view_payload(
        document, binary, view_index=1, payload=b"NEW-COLOR-WEBP"
    )

    assert document == before
    assert canary.buffer_view_payload(updated, new_binary, 0) == b"GEOM"
    assert canary.buffer_view_payload(updated, new_binary, 1) == b"NEW-COLOR-WEBP"
    assert canary.buffer_view_payload(updated, new_binary, 2) == b"METALMET"
    assert canary.buffer_view_payload(updated, new_binary, 3) == b"BINDMATR"
    assert proof["non_target_buffer_views_changed"] == 0
    assert updated["bufferViews"][2]["byteOffset"] % 4 == 0
    assert updated["buffers"][0]["byteLength"] == len(new_binary)


def test_glb_encoder_roundtrips_document_and_binary_exactly():
    document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}
    binary = b"ABCD"

    raw = canary.encode_glb(document, binary)
    decoded_document, decoded_binary = canary.decode_glb_bytes(raw)

    assert decoded_document == document
    assert decoded_binary == binary


def test_renderer_uses_canonical_negative_y_front_and_three_required_views():
    assert renderer.review_view_directions() == {
        "front": (0.0, -1.0, 0.0),
        "back": (0.0, 1.0, 0.0),
        "side": (1.0, 0.0, 0.0),
    }


def test_real_source_preflight_reads_exact_skin_uv_and_pbr_contract(authenticated_real_source):
    source = authenticated_real_source

    assert source["vertex_count"] == 712037
    assert source["triangle_count"] == 976951
    assert source["joint_count"] == 52
    assert source["texture_size"] == [4096, 4096]
    assert source["base_color_payload_sha256"] == (
        "91af1ad07f799f38532f30be856b535c86b9d22b76e073114c37e4854746f096"
    )
    assert source["metallic_roughness_payload_sha256"] == (
        "215567ed643a730cbbbf03fd59e5597dfeb484b53e8338fd8664376845226416"
    )
    assert source["base_color_image"].shape == (4096, 4096, 4)


def test_real_semantic_masks_are_large_nonoverlapping_conservative_cores(
    real_semantic_masks,
):
    bundle = real_semantic_masks

    total = sum(bundle["masks"][region].astype(np.uint8) for region in canary.REGIONS)
    assert int(total.max()) <= 1
    assert bundle["conflict_pixel_count"] >= 0
    assert bundle["post_conflict_overlap_pixel_count"] == 0
    for region in canary.REGIONS:
        assert bundle["pixel_counts"][region] >= 50000
        assert bundle["triangle_counts"][region] >= 5000
    assert bundle["pixel_color_guard_failures"] == {
        "top": 0,
        "bottom": 0,
        "shoes": 0,
        "hair": 0,
    }


def test_real_top_variant_roundtrip_changes_only_registered_basecolor_texels(
    authenticated_real_source, real_semantic_masks
):
    variant = canary.build_variant_bytes(
        authenticated_real_source,
        real_semantic_masks["masks"]["top"],
        region="top",
        palette_name="cobalt_blue",
    )

    assert variant["qa"]["outside_mask_changed_texels"] == 0
    assert variant["qa"]["alpha_changed_texels"] == 0
    assert variant["qa"]["non_target_buffer_views_changed"] == 0
    assert variant["qa"]["metallic_roughness_payload_unchanged"] is True
    assert variant["qa"]["mesh_skin_uv_document_unchanged"] is True
    assert variant["qa"]["decoded_output_matches_intended_rgba"] is True
    assert variant["qa"]["inside_mask_changed_texels"] >= 50000
    assert variant["glb_sha256"] != canary.SOURCE_GLB_SHA256
    document, _ = canary.decode_glb_bytes(variant["glb_bytes"])
    assert document["images"][0]["mimeType"] == "image/webp"
