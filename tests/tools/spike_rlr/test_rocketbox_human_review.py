import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rocketbox_human_review import (  # noqa: E402
    OfficialFile,
    OfficialFileError,
    RocketboxReviewAsset,
    ensure_official_files,
    git_blob_sha1,
    git_blob_sha1_bytes,
    load_review_assets,
    verify_official_file,
)


def test_load_review_assets_selects_two_sampled_adults(tmp_path):
    tree = {
        "tree": [
            {
                "path": "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx",
                "size": 3,
                "sha": "fbx-m",
            },
            {
                "path": "Assets/Avatars/Adults/Male_Adult_01/Textures/m002_body_color.tga",
                "size": 4,
                "sha": "tex-m",
            },
            {
                "path": "Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx",
                "size": 3,
                "sha": "fbx-f",
            },
            {
                "path": "Assets/Avatars/Adults/Female_Adult_01/Textures/f001_body_color.tga",
                "size": 4,
                "sha": "tex-f",
            },
        ]
    }
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")

    assets = load_review_assets(tree_path, tmp_path / "sample")

    assert sorted(assets) == ["rocketbox_female_adult_01", "rocketbox_male_adult_01"]
    assert assets["rocketbox_male_adult_01"].forward_axis == "-Y"
    assert assets["rocketbox_female_adult_01"].up_axis == "+Z"


def test_load_review_assets_records_missing_required_textures(tmp_path):
    tree = {
        "tree": [
            {
                "path": "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx",
                "size": 3,
                "sha": "fbx-m",
            },
            {
                "path": "Assets/Avatars/Adults/Male_Adult_01/Textures/m002_body_color.tga",
                "size": 4,
                "sha": "tex-m",
            },
        ]
    }
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")

    asset = load_review_assets(tree_path, tmp_path / "sample")[
        "rocketbox_male_adult_01"
    ]

    texture_root = "Assets/Avatars/Adults/Male_Adult_01/Textures"
    assert asset.missing_required_textures == (
        f"{texture_root}/m002_body_normal.tga",
        f"{texture_root}/m002_body_specular.tga",
        f"{texture_root}/m002_head_color.tga",
        f"{texture_root}/m002_head_normal.tga",
        f"{texture_root}/m002_head_specular.tga",
        f"{texture_root}/m002_opacity_color.tga",
    )


def test_git_blob_sha1_matches_git_object_format(tmp_path):
    path = tmp_path / "hello.bin"
    path.write_bytes(b"hello\n")

    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_verify_official_file_reports_path_size_and_git_sha(tmp_path):
    path = tmp_path / "invalid.bin"
    path.write_bytes(b"x")

    with pytest.raises(OfficialFileError) as error:
        verify_official_file(path, expected_size=2, expected_git_sha="expected-sha")

    message = str(error.value)
    assert str(path) in message
    assert "size actual=1 expected=2" in message
    assert "Git blob SHA actual=" in message
    assert "expected=expected-sha" in message


def _official_file(tmp_path, payload=b"fbx", name="avatar.fbx"):
    local_path = tmp_path / name
    local_path.write_bytes(payload)
    return OfficialFile(
        rel_path=f"Assets/Test/{name}",
        size=len(payload),
        git_sha=git_blob_sha1_bytes(payload),
        local_path=local_path,
    )


def _asset_with_files(tmp_path, textures):
    return RocketboxReviewAsset(
        asset_id="rocketbox_test",
        gender="test",
        avatar_dir="Test",
        texture_prefix="test",
        fbx=_official_file(tmp_path),
        textures=textures,
    )


def test_ensure_official_files_downloads_atomically_and_verifies(tmp_path):
    payload = b"texture"
    expected = OfficialFile(
        rel_path="Assets/Test/texture.tga",
        size=len(payload),
        git_sha=git_blob_sha1_bytes(payload),
        local_path=tmp_path / "texture.tga",
    )
    asset = _asset_with_files(tmp_path, textures=(expected,))

    paths = ensure_official_files(
        asset, opener=lambda request, timeout: io.BytesIO(payload)
    )

    assert paths == [expected.local_path]
    assert expected.local_path.read_bytes() == payload
    assert not expected.local_path.with_suffix(".tga.part").exists()


def test_ensure_official_files_rejects_missing_required_textures_before_network(
    tmp_path,
):
    missing_paths = (
        "Assets/Test/m002_head_color.tga",
        "Assets/Test/m002_opacity_color.tga",
    )
    asset = replace(
        _asset_with_files(tmp_path, textures=()),
        missing_required_textures=missing_paths,
    )
    requests = []

    with pytest.raises(OfficialFileError) as error:
        ensure_official_files(
            asset,
            opener=lambda request, timeout: requests.append(request)
            or io.BytesIO(b"unused"),
        )

    assert requests == []
    assert all(path in str(error.value) for path in missing_paths)


def test_ensure_official_files_rejects_corrupt_download(tmp_path):
    expected = _official_file(tmp_path, payload=b"correct", name="texture.tga")
    expected.local_path.unlink()
    asset = _asset_with_files(tmp_path, textures=(expected,))

    with pytest.raises(OfficialFileError, match="Git blob SHA"):
        ensure_official_files(
            asset, opener=lambda request, timeout: io.BytesIO(b"wrong!!")
        )

    assert not expected.local_path.exists()


def test_ensure_official_files_preserves_invalid_existing_file(tmp_path):
    payload = b"correct"
    expected = _official_file(tmp_path, payload=payload, name="texture.tga")
    expected.local_path.write_bytes(b"corrupt")
    asset = _asset_with_files(tmp_path, textures=(expected,))

    paths = ensure_official_files(
        asset, opener=lambda request, timeout: io.BytesIO(payload)
    )

    assert paths == [expected.local_path]
    assert expected.local_path.read_bytes() == payload
    assert expected.local_path.with_name("texture.tga.invalid").read_bytes() == b"corrupt"


def test_ensure_official_files_quotes_paths_and_identifies_request(tmp_path):
    payload = b"texture"
    expected = OfficialFile(
        rel_path="Assets/Test/texture with space.tga",
        size=len(payload),
        git_sha=git_blob_sha1_bytes(payload),
        local_path=tmp_path / "texture with space.tga",
    )
    asset = _asset_with_files(tmp_path, textures=(expected,))
    requests = []

    ensure_official_files(
        asset,
        opener=lambda request, timeout: requests.append(request) or io.BytesIO(payload),
    )

    assert requests[0].full_url.endswith("Assets/Test/texture%20with%20space.tga")
    assert requests[0].get_header("User-agent")
