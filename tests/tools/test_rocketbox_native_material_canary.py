from __future__ import annotations

import importlib
import importlib.util
import ast
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest
from PIL import Image


SPEAR_ROOT = Path(__file__).resolve().parents[2]


def _canary():
    spec = importlib.util.find_spec("tools.rocketbox_native_material_canary")
    assert spec is not None, "native Rocketbox material canary module is missing"
    return importlib.import_module("tools.rocketbox_native_material_canary")


def test_contract_pins_official_m002_inputs_and_new_output_root():
    canary = _canary()

    assert canary.ROCKETBOX_COMMIT == "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
    assert canary.ASSET_ID == "rocketbox_male_adult_01"
    assert canary.SOURCE_BODY_COLOR_SHA256 == (
        "6a048a6b2140a1f5293798ca286be64fd0de6d79572e1273ff5765bd578f463f"
    )
    assert canary.SOURCE_BODY_COLOR_GIT_BLOB_SHA1 == (
        "818ec72b6c69655c5853ab0eb1efdd6ab40b2bf0"
    )
    assert canary.SOURCE_BODY_COLOR_SIZE == 12_582_956
    assert canary.OUTPUT_ROOT == (
        SPEAR_ROOT
        / "tmp/rocketbox_native_material_canary_v1/"
        "rocketbox_male_adult_01/shirt_blue_v1"
    )


def _authenticated_source():
    loader = getattr(_canary(), "load_authenticated_source", None)
    assert callable(loader), "authenticated native Rocketbox loader is missing"
    return loader()


def _shirt_uv_bundle(source):
    extractor = getattr(_canary(), "extract_authenticated_shirt_uv", None)
    assert callable(extractor), "fixed native Rocketbox shirt UV extractor is missing"
    return extractor(source)


def _shirt_masks(source, uv_bundle):
    builder = getattr(_canary(), "build_shirt_masks", None)
    assert callable(builder), "native Rocketbox shirt mask builder is missing"
    return builder(source, uv_bundle)


def _blue_variant(source, masks):
    builder = getattr(_canary(), "build_blue_tga_variant", None)
    assert callable(builder), "native Rocketbox blue TGA variant builder is missing"
    return builder(source, masks)


def test_source_authentication_checks_commit_fbx_tga_hash_blob_size_and_layout():
    source = _authenticated_source()

    assert source["repository_commit"] == (
        "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
    )
    assert source["fbx"] == {
        "path": str(
            _canary().ROCKETBOX_ROOT
            / "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx"
        ),
        "size_bytes": 521_472,
        "sha256": "8d1edb51b4dc3427ae2456f4407fc105532c145dd019e53cd42bab31cc948a29",
        "git_blob_sha1": "fdaf4aa7d15054f1601740ea1a09cf111938d210",
    }
    assert source["body_color"]["size_bytes"] == 12_582_956
    assert source["body_color"]["sha256"] == _canary().SOURCE_BODY_COLOR_SHA256
    assert source["body_color"]["git_blob_sha1"] == (
        _canary().SOURCE_BODY_COLOR_GIT_BLOB_SHA1
    )
    assert source["tga"]["image_type"] == 2
    assert source["tga"]["width"] == 2048
    assert source["tga"]["height"] == 2048
    assert source["tga"]["pixel_depth"] == 24
    assert source["tga"]["descriptor"] == 0
    assert source["tga"]["origin"] == "bottom_left"
    assert len(source["tga"]["header_bytes"]) == 18
    assert len(source["tga"]["pixel_payload"]) == 2048 * 2048 * 3
    assert len(source["tga"]["footer_bytes"]) == 26
    assert source["tga"]["footer_bytes"].endswith(b"TRUEVISION-XFILE.\x00")
    assert source["tga"]["rgb"].shape == (2048, 2048, 3)
    assert source["tga"]["rgb"].dtype == np.uint8


def test_protected_pbr_head_and_opacity_textures_are_authenticated_by_reference():
    source = _authenticated_source()
    records = source["protected_textures"]

    assert set(records) == {
        "body_normal",
        "body_specular",
        "head_color",
        "head_normal",
        "head_specular",
        "opacity_color",
    }
    assert records["body_normal"]["sha256"] == (
        "c9892e80c56890f6f5365627835286c2d4d2cc34cc473f6788fe3d560a52fb69"
    )
    assert records["body_specular"]["sha256"] == (
        "c071a46bf86f2cc0062ccddae770a9bc21ebae325634439ac3cc1ed4d92fd684"
    )
    assert records["head_color"]["sha256"] == (
        "f5b64e4894930d438f7c419c3c9d0bba1f95fa4de9af078b762ea55bfca1ab85"
    )
    assert records["head_normal"]["sha256"] == (
        "5ef5ae404c9276f17641e392d0d4bdbe5fb23e94f28238b23a381e6a357c42ea"
    )
    assert records["head_specular"]["sha256"] == (
        "c051336b12d443cbd0e04dc3f1ce4109d420f426198aa02a1a1451b87e9a7831"
    )
    assert records["opacity_color"]["sha256"] == (
        "53818d3f45519451edd2bc60d6e59cd4057ecd381827463c5503da0439d0a5cf"
    )
    for record in records.values():
        assert Path(record["path"]).is_file()
        assert Path(record["path"]).parent == Path(
            source["body_color"]["path"]
        ).parent


def test_tga_parser_fails_closed_on_compression_or_noncanonical_footer():
    canary = _canary()
    raw = _authenticated_source()["tga"]["raw_bytes"]
    compressed = bytearray(raw)
    compressed[2] = 10
    bad_footer = bytearray(raw)
    bad_footer[-18:] = b"NOT-A-TGA-FOOTER!\x00"

    with pytest.raises(canary.NativeMaterialCanaryError, match="uncompressed"):
        canary.parse_tga_bytes(bytes(compressed))
    with pytest.raises(canary.NativeMaterialCanaryError, match="footer"):
        canary.parse_tga_bytes(bytes(bad_footer))


def test_fixed_shirt_uv_contract_is_exactly_three_islands_and_643_faces():
    source = _authenticated_source()
    bundle = _shirt_uv_bundle(source)
    indices = np.asarray(bundle["polygon_indices"], dtype="<u4")

    assert bundle["mesh_name"] == "m002_hipoly_81_bones_opacity"
    assert bundle["material_names"] == ["m002_body", "m002_head", "m002_opacity"]
    assert bundle["material_polygon_counts"] == [3939, 3007, 494]
    assert bundle["body_uv_island_count"] == 18
    assert bundle["shirt_island_signatures"] == [
        [80, 0.009, 0.3953, 0.2767, 0.5582, 123.73, 148.34],
        [483, 0.3565, 0.1508, 0.6428, 0.9439, 89.01, 157.6],
        [80, 0.7228, 0.3948, 0.9905, 0.5568, 123.3, 148.54],
    ]
    assert len(indices) == 643
    assert len(np.unique(indices)) == 643
    assert len(bundle["uv_polygons"]) == 643
    assert all(len(polygon) >= 3 for polygon in bundle["uv_polygons"])
    assert hashlib.sha256(indices.tobytes()).hexdigest() == (
        "b6e68ad480a10f9756129ca6360596e5c5a91e641a19ba0fbf9d96d91267f317"
    )


def test_shirt_masks_partition_fixed_surface_and_protect_stripes_and_buttons():
    source = _authenticated_source()
    bundle = _shirt_masks(source, _shirt_uv_bundle(source))
    surface = bundle["shirt_surface"]
    protected = bundle["stripe_detail_protect"]
    main = bundle["shirt_main_color"]

    assert surface.shape == (2048, 2048)
    assert protected.shape == surface.shape
    assert main.shape == surface.shape
    assert surface.dtype == protected.dtype == main.dtype == np.bool_
    assert int(surface.sum()) == 1_129_206
    assert np.array_equal(surface, protected | main)
    assert not np.any(protected & main)
    assert not np.any((protected | main) & ~surface)
    assert 150_000 <= int(protected.sum()) <= 400_000
    assert int(main.sum()) >= 700_000

    rgb = source["tga"]["rgb"].astype(np.uint32)
    luma8 = (54 * rgb[:, :, 0] + 183 * rgb[:, :, 1] + 19 * rgb[:, :, 2] + 128) >> 8
    dark_stripe_core = surface & (luma8 < _canary().STRIPE_LUMA_THRESHOLD)
    assert dark_stripe_core.any()
    assert np.all(protected[dark_stripe_core])
    assert not np.any(main[dark_stripe_core])
    for x, y, _radius in _canary().BUTTON_GUARDS:
        assert protected[y, x]
        assert not main[y, x]

    assert bundle["mask_sha256"] == {
        name: hashlib.sha256(bundle[name].tobytes()).hexdigest()
        for name in ("shirt_surface", "stripe_detail_protect", "shirt_main_color")
    }


def test_uv_rasterization_uses_version_independent_pixel_center_barycentrics():
    canary = _canary()
    rasterizer = getattr(canary, "rasterize_uv_polygons", None)
    assert callable(rasterizer), "version-independent UV rasterizer is missing"

    mask = rasterizer(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], width=4, height=4
    )

    assert np.array_equal(
        mask,
        np.asarray(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ),
    )
    module_path = SPEAR_ROOT / "tools/rocketbox_native_material_canary.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "cv2" not in imported_roots


def test_blue_variant_patches_only_main_mask_raw_bgr_and_preserves_tga_container():
    canary = _canary()
    source = _authenticated_source()
    masks = _shirt_masks(source, _shirt_uv_bundle(source))
    variant = _blue_variant(source, masks)
    decoded = canary.parse_tga_bytes(variant["tga_bytes"])
    original = source["tga"]
    changed = np.any(decoded["rgb"] != original["rgb"], axis=2)
    main = masks["shirt_main_color"]
    protected = masks["stripe_detail_protect"]

    assert len(variant["tga_bytes"]) == len(original["raw_bytes"])
    assert decoded["header_bytes"] == original["header_bytes"]
    assert decoded["footer_bytes"] == original["footer_bytes"]
    assert variant["tga_bytes"][:18] == original["raw_bytes"][:18]
    assert variant["tga_bytes"][-26:] == original["raw_bytes"][-26:]
    assert np.array_equal(changed, main)
    assert np.array_equal(decoded["rgb"][~main], original["rgb"][~main])
    assert np.array_equal(decoded["rgb"][protected], original["rgb"][protected])
    assert variant["qa"]["outside_mask_changed_pixels"] == 0
    assert variant["qa"]["protected_changed_pixels"] == 0
    assert variant["qa"]["header_changed_bytes"] == 0
    assert variant["qa"]["footer_changed_bytes"] == 0
    assert variant["qa"]["size_unchanged"] is True
    assert variant["qa"]["inside_mask_changed_pixels"] == int(main.sum())
    assert variant["qa"]["linear_luminance_correlation"] >= 0.98
    assert variant["qa"]["transform"] == (
        "registered_srgb_linear_luminance_scale_v1"
    )
    assert variant["qa"]["palette_srgb"] == [36, 88, 207]
    median = np.median(decoded["rgb"][main], axis=0)
    assert median[2] > median[1] > median[0]
    assert variant["source_sha256"] == canary.SOURCE_BODY_COLOR_SHA256
    assert variant["output_sha256"] != variant["source_sha256"]


def test_masks_and_blue_tga_are_frozen_to_exact_hashes_and_counts():
    canary = _canary()
    source = _authenticated_source()
    masks = _shirt_masks(source, _shirt_uv_bundle(source))
    variant = _blue_variant(source, masks)

    assert canary.FROZEN_MASK_REGISTRY == {
        "shirt_surface": {
            "pixel_count": 1_129_206,
            "raw_bool_sha256": "8ac1a005a391da508748b25f93d7e4385bcba50980ae55850bc20d7c447f382f",
        },
        "stripe_detail_protect": {
            "pixel_count": 325_654,
            "raw_bool_sha256": "8ed02cb4f9b5bdf7520787d19d338e15204978b0645685e80966f45979dca294",
        },
        "shirt_main_color": {
            "pixel_count": 803_552,
            "raw_bool_sha256": "0a5459ae32f2b74824be07ce43fb9d6673e851e78505354eafb741f30e7f6ca9",
        },
    }
    assert masks["pixel_counts"] == {
        name: record["pixel_count"]
        for name, record in canary.FROZEN_MASK_REGISTRY.items()
    }
    assert masks["mask_sha256"] == {
        name: record["raw_bool_sha256"]
        for name, record in canary.FROZEN_MASK_REGISTRY.items()
    }
    assert canary.FROZEN_BLUE_TGA_SHA256 == (
        "1098a88989e10574ec5d3c058f51cb17164b5671a47649071ad74ed514ec04e9"
    )
    assert variant["output_sha256"] == canary.FROZEN_BLUE_TGA_SHA256


def test_writer_freezes_masks_overlay_diff_variant_and_reference_only_manifest(tmp_path):
    canary = _canary()
    writer = getattr(canary, "write_canary_output", None)
    assert callable(writer), "native Rocketbox canary artifact writer is missing"
    source = _authenticated_source()
    uv_bundle = _shirt_uv_bundle(source)
    masks = _shirt_masks(source, uv_bundle)
    variant = _blue_variant(source, masks)
    output = tmp_path / "shirt_blue_v1"

    manifest_path = writer(output, source, uv_bundle, masks, variant)

    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "rocketbox_native_material_canary_v1"
    assert manifest["asset_id"] == "rocketbox_male_adult_01"
    assert manifest["source"]["repository_commit"] == canary.ROCKETBOX_COMMIT
    assert manifest["source"]["body_color"]["sha256"] == (
        canary.SOURCE_BODY_COLOR_SHA256
    )
    assert set(manifest["source"]["protected_texture_references"]) == {
        "body_normal",
        "body_specular",
        "head_color",
        "head_normal",
        "head_specular",
        "opacity_color",
    }
    assert manifest["mask_registry"]["face_count"] == 643
    assert manifest["mask_registry"]["face_indices_u32le_sha256"] == (
        canary.SHIRT_FACE_INDICES_U32LE_SHA256
    )
    assert manifest["variant"]["sha256"] == canary.FROZEN_BLUE_TGA_SHA256
    assert manifest["variant"]["qa"]["outside_mask_changed_pixels"] == 0
    assert manifest["variant"]["qa"]["protected_changed_pixels"] == 0
    assert manifest["automatic_qa"]["all_checks_passed"] is True

    required = {
        "manifest.json",
        "mask_registry.json",
        "masks/shirt_surface.png",
        "masks/stripe_detail_protect.png",
        "masks/shirt_main_color.png",
        "masks/shirt_polygon_indices.npy",
        "diagnostics/mask_overlay.png",
        "diagnostics/texture_diff.png",
        "variant/m002_body_color.tga",
    }
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == required
    assert [path.relative_to(output).as_posix() for path in output.rglob("*.tga")] == [
        "variant/m002_body_color.tga"
    ]
    for role, record in manifest["source"]["protected_texture_references"].items():
        assert Path(record["path"]).is_file(), role
        assert not str(record["path"]).startswith(str(output))
        assert not (output / Path(record["path"]).name).exists()

    for name in ("shirt_surface", "stripe_detail_protect", "shirt_main_color"):
        image = np.asarray(Image.open(output / f"masks/{name}.png").convert("L"))
        assert np.array_equal(image == 255, masks[name])
    assert (output / "variant/m002_body_color.tga").read_bytes() == variant["tga_bytes"]
    for path in output.rglob("*"):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o444 if path.is_file() else 0o555)
