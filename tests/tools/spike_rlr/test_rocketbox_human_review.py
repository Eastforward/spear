import io
import json
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rocketbox_human_review import (  # noqa: E402
    OfficialFile,
    OfficialFileError,
    RocketboxReviewAsset,
    SourceReviewNotApproved,
    approve_source_review,
    assert_source_review_approved,
    build_source_inspection,
    ensure_official_files,
    git_blob_sha1,
    git_blob_sha1_bytes,
    load_review_assets,
    main,
    parse_args,
    sha256_file,
    verify_official_file,
    write_pending_source_review,
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


def _complete_local_asset(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    textures = (
        _official_file(tmp_path, payload=b"body", name="body_color.tga"),
        _official_file(tmp_path, payload=b"head", name="head_color.tga"),
    )
    return _asset_with_files(tmp_path, textures)


def test_pending_source_review_records_axes_hashes_and_pending_status(tmp_path):
    asset = _complete_local_asset(tmp_path)
    inspection = build_source_inspection(asset)

    review_path = write_pending_source_review(asset, tmp_path / "out", inspection)
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert review["schema_version"] == "rocketbox_human_source_review_v1"
    assert review["up_axis"] == "+Z"
    assert review["forward_axis"] == "-Y"
    assert review["geometry_status"] == "pending"
    assert review["appearance_status"] == "pending"
    assert review["direction_status"] == "pending"
    assert review["source_sha256"] == sha256_file(asset.fbx.local_path)
    assert review["official_files"] == inspection["official_files"]


def test_source_inspection_records_verified_absolute_file_provenance(tmp_path):
    asset = _complete_local_asset(tmp_path)

    inspection = build_source_inspection(asset)

    assert inspection["schema_version"] == "rocketbox_source_inspection_v1"
    assert inspection["missing_required_textures"] == []
    assert inspection["official_files"] == [
        {
            "role": "fbx",
            "official_rel_path": asset.fbx.rel_path,
            "local_path": str(asset.fbx.local_path.resolve()),
            "size": asset.fbx.size,
            "git_blob_sha1": asset.fbx.git_sha,
            "official_git_blob_sha1": asset.fbx.git_sha,
            "sha256": sha256_file(asset.fbx.local_path),
        },
        *[
            {
                "role": "texture",
                "official_rel_path": texture.rel_path,
                "local_path": str(texture.local_path.resolve()),
                "size": texture.size,
                "git_blob_sha1": texture.git_sha,
                "official_git_blob_sha1": texture.git_sha,
                "sha256": sha256_file(texture.local_path),
            }
            for texture in asset.textures
        ],
    ]


def test_source_inspection_rejects_incomplete_or_corrupt_task_one_asset(tmp_path):
    incomplete = replace(
        _complete_local_asset(tmp_path),
        missing_required_textures=("Assets/Test/missing.tga",),
    )

    with pytest.raises(OfficialFileError, match="missing required official textures"):
        build_source_inspection(incomplete)

    corrupt = _complete_local_asset(tmp_path / "corrupt")
    corrupt.fbx.local_path.write_bytes(b"corrupt")

    with pytest.raises(OfficialFileError, match="Official file verification failed"):
        build_source_inspection(corrupt)


def test_source_review_gate_requires_all_three_human_approvals(tmp_path):
    asset = _complete_local_asset(tmp_path)
    inspection = build_source_inspection(asset)
    review_path = write_pending_source_review(asset, tmp_path / "out", inspection)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "geometry_status": "approved",
            "appearance_status": "pending",
            "direction_status": "approved",
            "approved_by": "jzy",
            "approved_at": "2026-07-10T12:00:00+08:00",
        }
    )
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(SourceReviewNotApproved, match="appearance_status"):
        assert_source_review_approved(review_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("approved_by", "", "approved_by"),
        ("approved_at", "2026-07-10T12:00:00", "approved_at"),
    ),
)
def test_source_review_gate_requires_reviewer_and_aware_approval_time(
    tmp_path, field, value, message
):
    asset = _complete_local_asset(tmp_path)
    review_path = write_pending_source_review(
        asset, tmp_path / "out", build_source_inspection(asset)
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "geometry_status": "approved",
            "appearance_status": "approved",
            "direction_status": "approved",
            "approved_by": "jzy",
            "approved_at": "2026-07-10T12:00:00+08:00",
        }
    )
    review[field] = value
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(SourceReviewNotApproved, match=message):
        assert_source_review_approved(review_path)


def test_source_review_gate_rechecks_official_files_after_human_approval(tmp_path):
    asset = _complete_local_asset(tmp_path)
    review_path = write_pending_source_review(
        asset, tmp_path / "out", build_source_inspection(asset)
    )
    approve_source_review(
        review_path,
        reviewer="jzy",
        geometry_status="approved",
        appearance_status="approved",
        direction_status="approved",
        notes="reviewed in generated contact sheet and turntable",
    )
    asset.fbx.local_path.write_bytes(b"corrupt after approval")

    with pytest.raises(SourceReviewNotApproved, match="official_files verification failed"):
        assert_source_review_approved(review_path)


def test_source_review_gate_rejects_malformed_official_file_record(tmp_path):
    asset = _complete_local_asset(tmp_path)
    review_path = write_pending_source_review(
        asset, tmp_path / "out", build_source_inspection(asset)
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "geometry_status": "approved",
            "appearance_status": "approved",
            "direction_status": "approved",
            "approved_by": "jzy",
            "approved_at": "2026-07-10T12:00:00+08:00",
            "official_files": ["not a file record"],
        }
    )
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(SourceReviewNotApproved, match="official_files verification failed"):
        assert_source_review_approved(review_path)


def _write_complete_cli_asset_tree(tmp_path, gender="male"):
    sample_root = tmp_path / "sample"
    avatar_dir, texture_prefix = {
        "male": ("Male_Adult_01", "m002"),
        "female": ("Female_Adult_01", "f001"),
    }[gender]
    avatar_root = f"Assets/Avatars/Adults/{avatar_dir}"
    paths = [f"{avatar_root}/Export/{avatar_dir}.fbx"]
    paths.extend(
        f"{avatar_root}/Textures/{texture_prefix}_{suffix}.tga"
        for suffix in (
            "body_color",
            "body_normal",
            "body_specular",
            "head_color",
            "head_normal",
            "head_specular",
            "opacity_color",
        )
    )
    if gender == "female":
        paths.append(f"{avatar_root}/Textures/{texture_prefix}_head_normal_wrinkle.tga")
    tree = {"tree": []}
    for index, rel_path in enumerate(paths):
        payload = f"file-{index}".encode("ascii")
        local_path = sample_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payload)
        tree["tree"].append(
            {
                "path": rel_path,
                "size": len(payload),
                "sha": git_blob_sha1_bytes(payload),
            }
        )
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    return tree_path, sample_root


@pytest.mark.parametrize(
    ("gender", "asset_id", "official_file_count"),
    (
        ("male", "rocketbox_male_adult_01", 8),
        ("female", "rocketbox_female_adult_01", 9),
    ),
)
def test_download_cli_reports_total_verified_official_file_count_when_cached(
    tmp_path, capsys, gender, asset_id, official_file_count
):
    tree_json, sample_root = _write_complete_cli_asset_tree(tmp_path, gender=gender)

    exit_code = main(
        [
            "download",
            "--tree-json",
            str(tree_json),
            "--sample-root",
            str(sample_root),
            "--asset-id",
            asset_id,
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"ROCKETBOX_OFFICIAL_FILES_OK asset_id={asset_id} "
        f"files={official_file_count}\n"
    )


def test_parse_download_arguments_without_network_access(tmp_path):
    tree_json = tmp_path / "tree.json"
    sample_root = tmp_path / "sample"

    args = parse_args(
        [
            "download",
            "--tree-json",
            str(tree_json),
            "--sample-root",
            str(sample_root),
            "--asset-id",
            "rocketbox_male_adult_01",
        ]
    )

    assert args.command == "download"
    assert args.tree_json == tree_json
    assert args.sample_root == sample_root
    assert args.asset_id == "rocketbox_male_adult_01"


def test_parse_inspect_arguments_with_output_directory(tmp_path):
    tree_json = tmp_path / "tree.json"
    sample_root = tmp_path / "sample"
    output_dir = tmp_path / "out"

    args = parse_args(
        [
            "inspect",
            "--tree-json",
            str(tree_json),
            "--sample-root",
            str(sample_root),
            "--asset-id",
            "rocketbox_male_adult_01",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert args.command == "inspect"
    assert args.tree_json == tree_json
    assert args.sample_root == sample_root
    assert args.output_dir == output_dir


def test_parse_approve_requires_all_explicit_approval_statuses(tmp_path):
    review_json = tmp_path / "source_review.json"

    args = parse_args(
        [
            "approve",
            "--review-json",
            str(review_json),
            "--reviewer",
            "jzy",
            "--geometry",
            "approved",
            "--appearance",
            "approved",
            "--direction",
            "approved",
            "--notes",
            "reviewed in generated contact sheet and turntable",
        ]
    )

    assert args.command == "approve"
    assert args.review_json == review_json
    assert args.geometry == "approved"
    assert args.appearance == "approved"
    assert args.direction == "approved"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "approve",
                "--review-json",
                str(review_json),
                "--reviewer",
                "jzy",
                "--geometry",
                "approved",
                "--appearance",
                "approved",
                "--notes",
                "missing direction is never auto-approved",
            ]
        )


def test_inspect_cli_writes_fixed_sentinels_and_pending_review(tmp_path, capsys):
    tree_json, sample_root = _write_complete_cli_asset_tree(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "inspect",
            "--tree-json",
            str(tree_json),
            "--sample-root",
            str(sample_root),
            "--asset-id",
            "rocketbox_male_adult_01",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"ROCKETBOX_SOURCE_INSPECTION_WRITTEN path={output_dir / 'source_inspection.json'}",
        f"ROCKETBOX_SOURCE_REVIEW_PENDING path={output_dir / 'source_review.json'}",
    ]
    assert (output_dir / "source_inspection.json").exists()
    assert json.loads((output_dir / "source_review.json").read_text())["geometry_status"] == (
        "pending"
    )


def test_approve_cli_records_explicit_human_approval_and_aware_time(tmp_path, capsys):
    asset = _complete_local_asset(tmp_path)
    review_path = write_pending_source_review(
        asset, tmp_path / "out", build_source_inspection(asset)
    )

    exit_code = main(
        [
            "approve",
            "--review-json",
            str(review_path),
            "--reviewer",
            "jzy",
            "--geometry",
            "approved",
            "--appearance",
            "approved",
            "--direction",
            "approved",
            "--notes",
            "reviewed in generated contact sheet and turntable",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"ROCKETBOX_SOURCE_REVIEW_APPROVED path={review_path}\n"
    )
    review = assert_source_review_approved(review_path)
    assert review["approved_by"] == "jzy"
    assert review["notes"] == "reviewed in generated contact sheet and turntable"
    assert datetime.fromisoformat(review["approved_at"]).tzinfo is not None


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


def test_ensure_official_files_numbers_colliding_invalid_archives(tmp_path):
    payload = b"correct"
    expected = _official_file(tmp_path, payload=payload, name="texture.tga")
    expected.local_path.write_bytes(b"newer-corrupt")
    first_archive = expected.local_path.with_name("texture.tga.invalid")
    first_archive.write_bytes(b"older-corrupt")
    asset = _asset_with_files(tmp_path, textures=(expected,))

    paths = ensure_official_files(
        asset, opener=lambda request, timeout: io.BytesIO(payload)
    )

    assert paths == [expected.local_path]
    assert expected.local_path.read_bytes() == payload
    assert first_archive.read_bytes() == b"older-corrupt"
    assert expected.local_path.with_name("texture.tga.invalid.1").read_bytes() == (
        b"newer-corrupt"
    )


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
