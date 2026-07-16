#!/usr/bin/env python3

"""Build the sealed Female_Adult_01 native Walk + Idle runtime.

The underlying builder is the already verified immutable native Rocketbox
builder.  This wrapper supplies the separately sealed female baseline, female
neutral idle, and the seven authenticated ``f001`` PBR payloads.  No baseline
file is opened directly for writing and no source object is rescaled.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import blender_build_native_rocketbox_runtime as native


ASSET_ID = "rocketbox_female_adult_01"
TEXTURE_PREFIX = "f001"
BASELINE_BLEND_SHA256 = (
    "dd2b174301b31468511c4c49c1ea53daf3bb53148220a9e026dc5511fac4d3be"
)
BASELINE_GLB_SHA256 = (
    "b678cc011c27a42d3a7833c0529af401533a0c68cc2c63448ab0367d3359048e"
)
IDLE_SHA256 = "fd68b33ea9e290dc734ca8c3a71ef5842bb2dfe719853ff84f6336d06d39fdcb"
IDLE_GIT_BLOB_SHA1 = "aecf1d0089ccfc0c381d5395294bb1c8fe0e63ae"
IDLE_SIZE_BYTES = 2_959_360
EXPECTED_WALK_RANGE = (1, 38)
EXPECTED_IDLE_RANGE = (1, 467)

OFFICIAL_TEXTURES = {
    "f001_body_color": {
        "filename": "f001_body_color.tga",
        "size_bytes": 12_582_956,
        "sha256": "c038b01c5d05831c3f4e8c6daa18394e8a4851372acd3fcbdf30e7ee53c56cae",
        "git_blob_sha1": "25852db0db1f44a0de0643ce1e09c88aef587e6f",
    },
    "f001_body_normal": {
        "filename": "f001_body_normal.tga",
        "size_bytes": 12_582_956,
        "sha256": "86b9b5a5aa73bd97244b35f97f3799c31b3db70426d7d607cbdc2dd271ace346",
        "git_blob_sha1": "345ba24adc6d429a1e0967c0010588b88892be46",
    },
    "f001_body_specular": {
        "filename": "f001_body_specular.tga",
        "size_bytes": 12_582_956,
        "sha256": "9b813e68b243074f74c4bbfb26d41cce2a70361efa09fcabe1962b4267bedb90",
        "git_blob_sha1": "4cb0a66433aac02a3899c2ceeceb4e11880e6395",
    },
    "f001_head_color": {
        "filename": "f001_head_color.tga",
        "size_bytes": 12_582_956,
        "sha256": "0d3153c59c54908d5f5194ffe3a7d882351f0dd74f9ef44ab3a2f164a3bf9d8e",
        "git_blob_sha1": "0cdc62ab47da3dca01aff0bb179159865a559c63",
    },
    "f001_head_normal": {
        "filename": "f001_head_normal.tga",
        "size_bytes": 12_582_956,
        "sha256": "522f3372283ea4161ed56d85471be15e02628df73a966447e4456bc169d310b2",
        "git_blob_sha1": "6de1fb458c746154713e8c34039add24f4ba9f6d",
    },
    "f001_head_specular": {
        "filename": "f001_head_specular.tga",
        "size_bytes": 12_582_956,
        "sha256": "302c56927d1d032084fb4faca784591dd8ab42e8bafb1029db157f73608eb251",
        "git_blob_sha1": "47a70e2ff5ed71ff3152019bc46845ef7ff3fbf8",
    },
    "f001_opacity_color": {
        "filename": "f001_opacity_color.tga",
        "size_bytes": 16_777_260,
        "sha256": "12c9c366814e665687fe64056421213686b50a972a5ad47295f247d6a8ff7369",
        "git_blob_sha1": "de21a9e69c88e539da6a5050eaceadd026e99b62",
    },
}


def configure_native_module() -> None:
    native.ASSET_ID = ASSET_ID
    native.TEXTURE_PREFIX = TEXTURE_PREFIX
    native.BASELINE_BLEND_SHA256 = BASELINE_BLEND_SHA256
    native.BASELINE_GLB_SHA256 = BASELINE_GLB_SHA256
    native.BASELINE_ASSET_DIR = native.BASELINE_ROOT / ASSET_ID
    native.BASELINE_BLEND = native.BASELINE_ASSET_DIR / "retarget.blend"
    native.BASELINE_GLB = native.BASELINE_ASSET_DIR / "retarget.glb"
    native.IDLE_RELATIVE_PATH = Path(
        "Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_01.max.fbx"
    )
    native.IDLE_PATH = native.ROCKETBOX_ROOT / native.IDLE_RELATIVE_PATH
    native.IDLE_SHA256 = IDLE_SHA256
    native.IDLE_GIT_BLOB_SHA1 = IDLE_GIT_BLOB_SHA1
    native.IDLE_SIZE_BYTES = IDLE_SIZE_BYTES
    native.TEXTURE_DIR = (
        native.ROCKETBOX_ROOT / "Assets/Avatars/Adults/Female_Adult_01/Textures"
    )
    native.ORIGINAL_TAG = "rocketbox_female_adult_01_original_v1"
    native.BLUE_SHIRT_TAG = "rocketbox_female_adult_01_shirt_blue_v1"
    native.ORIGINAL_OUTPUT_DIR = (
        native.RUNTIME_ROOT / "rocketbox_female_adult_01_original_v1"
    )
    native.OFFICIAL_TEXTURES = OFFICIAL_TEXTURES
    native.EXPECTED_MATERIALS = (
        "f001_body",
        "f001_head",
        "f001_opacity",
    )
    native.EXPECTED_IMAGES = tuple(OFFICIAL_TEXTURES)
    native.EXPECTED_WALK_RANGE = (1, 38)
    native.EXPECTED_IDLE_RANGE = (1, 467)


def main(argv=None) -> int:
    configure_native_module()
    return native.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
