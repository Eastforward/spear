"""Tests for the FLUX human reference review Flask application."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from human_reference_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    record_review,
    sha256_file,
    write_candidate_manifest,
)


MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SOURCE_APPROVAL_SHA256 = "a" * 64
PROMPTS = {
    "rocketbox_male_adult_01": "Preserve this male identity in a forest-green T-shirt.",
    "rocketbox_female_adult_01": "Preserve this female identity in a deep burgundy T-shirt.",
}


def _write_candidate_fixture(review_root: Path, asset_id: str) -> Path:
    candidate_dir = review_root / asset_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "source.png").write_bytes(f"{asset_id}:source-image".encode("ascii"))
    (candidate_dir / "candidate.png").write_bytes(
        f"{asset_id}:candidate-image".encode("ascii")
    )
    write_candidate_manifest(
        candidate_dir,
        asset_id=asset_id,
        model_revision=MODEL_REVISION,
        prompt=PROMPTS[asset_id],
        seed=42,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )
    return candidate_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    review_root = tmp_path / "reviews"
    for asset_id in EXPECTED_ASSET_IDS:
        _write_candidate_fixture(review_root, asset_id)
    return review_root


def _client(workspace: Path):
    from human_reference_review_server import create_app

    return create_app(workspace).test_client()


def _csrf_token(client, asset_id: str = "rocketbox_male_adult_01") -> str:
    response = client.get(f"/asset/{asset_id}")
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match is not None
    return match.group(1).decode("ascii")


def _snapshot_fields(workspace: Path, asset_id: str) -> dict[str, str]:
    candidate_dir = workspace / asset_id
    return {
        "candidate_manifest_sha256": sha256_file(
            candidate_dir / "candidate_manifest.json"
        ),
        "source_sha256": sha256_file(candidate_dir / "source.png"),
        "candidate_sha256": sha256_file(candidate_dir / "candidate.png"),
    }


def _current_form_data(
    client, workspace: Path, asset_id: str, decision: str
) -> dict[str, str]:
    return {
        "decision": decision,
        "reviewer": "reviewer",
        "notes": "review notes",
        "csrf_token": _csrf_token(client, asset_id),
        **_snapshot_fields(workspace, asset_id),
    }


def _media_url(workspace: Path, asset_id: str, kind: str) -> str:
    expected_hash = _snapshot_fields(workspace, asset_id)[f"{kind}_sha256"]
    return f"/media/{asset_id}/{kind}?expected_sha256={expected_hash}"


def _review_snapshot(workspace: Path) -> dict[str, bytes | None]:
    return {
        asset_id: (
            path.read_bytes() if path.exists() else None
        )
        for asset_id in EXPECTED_ASSET_IDS
        for path in [workspace / asset_id / "reference_review.json"]
    }


@pytest.mark.parametrize(
    "route",
    (
        "/asset/rocketbox_male_adult_01",
        "/gate",
    ),
)
def test_get_routes_are_read_only(workspace, route):
    before = _review_snapshot(workspace)

    response = _client(workspace).get(route)

    assert response.status_code == 200
    assert _review_snapshot(workspace) == before


def test_asset_page_shows_exact_prompt_and_source_candidate_image_labels(workspace):
    response = _client(workspace).get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    assert b"Source image" in response.data
    assert b"Candidate image" in response.data
    assert PROMPTS["rocketbox_male_adult_01"].encode("utf-8") in response.data
    assert b"/media/rocketbox_male_adult_01/source" in response.data
    assert b"/media/rocketbox_male_adult_01/candidate" in response.data
    assert b"Approve" in response.data
    assert b"Reject" in response.data
    assert b"candidate_manifest.json" not in response.data


def test_asset_page_uses_an_inline_favicon_without_an_extra_request(workspace):
    response = _client(workspace).get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    assert b'<link rel="icon" href="data:,">' in response.data


def test_asset_page_form_carries_the_displayed_snapshot_hashes(workspace):
    asset_id = "rocketbox_male_adult_01"

    response = _client(workspace).get(f"/asset/{asset_id}")

    assert response.status_code == 200
    snapshot = _snapshot_fields(workspace, asset_id)
    for field, expected in snapshot.items():
        assert f'name="{field}" value="{expected}"'.encode("ascii") in response.data
    for kind in ("source", "candidate"):
        assert _media_url(workspace, asset_id, kind).encode("ascii") in response.data


def test_asset_page_does_not_mix_an_old_prompt_with_new_hidden_hashes(
    workspace, monkeypatch
):
    import human_reference_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    candidate_dir = workspace / asset_id
    snapshot_a = _snapshot_fields(workspace, asset_id)
    original_sha256_file = sha256_file
    regenerated = False

    def regenerate_before_later_hash(path):
        nonlocal regenerated
        if Path(path).name == "candidate_manifest.json" and not regenerated:
            regenerated = True
            (candidate_dir / "source.png").write_bytes(b"source B")
            (candidate_dir / "candidate.png").write_bytes(b"candidate B")
            write_candidate_manifest(
                candidate_dir,
                asset_id=asset_id,
                model_revision=MODEL_REVISION,
                prompt="Candidate B prompt.",
                seed=43,
                width=1024,
                height=1536,
                steps=28,
                guidance_scale=4.0,
                source_approval_sha256=SOURCE_APPROVAL_SHA256,
            )
        return original_sha256_file(path)

    monkeypatch.setattr(
        review_server, "sha256_file", regenerate_before_later_hash, raising=False
    )

    response = _client(workspace).get(f"/asset/{asset_id}")

    assert response.status_code == 200
    assert PROMPTS[asset_id].encode("utf-8") in response.data
    assert b"Candidate B prompt." not in response.data
    for field, expected in snapshot_a.items():
        assert f'name="{field}" value="{expected}"'.encode("ascii") in response.data
    for kind in ("source", "candidate"):
        expected = snapshot_a[f"{kind}_sha256"]
        assert (
            f"/media/{asset_id}/{kind}?expected_sha256={expected}".encode("ascii")
            in response.data
        )


def test_media_route_allowlists_source_and_candidate_and_supports_ranges(workspace):
    client = _client(workspace)
    before = _review_snapshot(workspace)

    source = client.get(
        _media_url(workspace, "rocketbox_male_adult_01", "source")
    )
    ranged = client.get(
        _media_url(workspace, "rocketbox_male_adult_01", "candidate"),
        headers={"Range": "bytes=0-3"},
    )

    assert source.status_code == 200
    assert source.headers["Cache-Control"] == "no-store, max-age=0"
    assert ranged.status_code == 206
    assert ranged.data == b"rock"
    assert "bytes" in ranged.headers["Accept-Ranges"]
    assert client.get("/media/rocketbox_male_adult_01/manifest").status_code == 404
    assert client.get("/media/../candidate").status_code in {404, 405}
    assert _review_snapshot(workspace) == before


def test_media_route_rejects_an_old_page_hash_after_regeneration(workspace):
    asset_id = "rocketbox_male_adult_01"
    candidate_dir = workspace / asset_id
    client = _client(workspace)
    old_candidate_url = _media_url(workspace, asset_id, "candidate")
    (candidate_dir / "source.png").write_bytes(b"source B")
    (candidate_dir / "candidate.png").write_bytes(b"candidate B")
    write_candidate_manifest(
        candidate_dir,
        asset_id=asset_id,
        model_revision=MODEL_REVISION,
        prompt="Candidate B prompt.",
        seed=43,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    response = client.get(old_candidate_url)

    assert response.status_code == 409


@pytest.mark.parametrize("expected_hash", ("", "a" * 63, "A" * 64, "g" * 64))
def test_media_route_requires_a_canonical_expected_hash(workspace, expected_hash):
    response = _client(workspace).get(
        "/media/rocketbox_male_adult_01/source",
        query_string={"expected_sha256": expected_hash},
    )

    assert response.status_code == 400


def test_posted_decisions_are_independent_and_redirect_locally(workspace):
    client = _client(workspace)

    first = client.post(
        "/review/rocketbox_male_adult_01?next=https://example.test/",
        data={
            **_current_form_data(
                client, workspace, "rocketbox_male_adult_01", "approved"
            ),
            "reviewer": "reviewer-a",
            "notes": "male ready",
        },
        follow_redirects=False,
    )

    assert first.status_code == 302
    assert first.headers["Location"].endswith("/asset/rocketbox_male_adult_01")
    male_review = json.loads(
        (workspace / "rocketbox_male_adult_01" / "reference_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert male_review["decision"] == "approved"
    assert not (workspace / "rocketbox_female_adult_01" / "reference_review.json").exists()

    second = client.post(
        "/review/rocketbox_female_adult_01",
        data={
            **_current_form_data(
                client, workspace, "rocketbox_female_adult_01", "rejected"
            ),
            "reviewer": "reviewer-b",
            "notes": "female needs another pass",
        },
    )

    assert second.status_code == 302
    female_review = json.loads(
        (workspace / "rocketbox_female_adult_01" / "reference_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert female_review["decision"] == "rejected"


@pytest.mark.parametrize("decision", ("approved", "rejected"))
def test_stale_page_post_returns_409_without_writing_a_review(workspace, decision):
    asset_id = "rocketbox_male_adult_01"
    candidate_dir = workspace / asset_id
    client = _client(workspace)
    old_form = _current_form_data(client, workspace, asset_id, decision)
    (candidate_dir / "source.png").write_bytes(b"source B")
    (candidate_dir / "candidate.png").write_bytes(b"candidate B")
    write_candidate_manifest(
        candidate_dir,
        asset_id=asset_id,
        model_revision=MODEL_REVISION,
        prompt=PROMPTS[asset_id],
        seed=43,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    response = client.post(f"/review/{asset_id}", data=old_form)

    assert response.status_code == 409
    assert not (candidate_dir / "reference_review.json").exists()


def test_stale_page_post_does_not_overwrite_an_existing_review(workspace):
    asset_id = "rocketbox_male_adult_01"
    candidate_dir = workspace / asset_id
    client = _client(workspace)
    old_form = _current_form_data(client, workspace, asset_id, "approved")
    record_review(
        candidate_dir,
        "rejected",
        "first-reviewer",
        "candidate A",
        expected_snapshot=_snapshot_fields(workspace, asset_id),
    )
    review_path = candidate_dir / "reference_review.json"
    before = review_path.read_bytes()
    (candidate_dir / "source.png").write_bytes(b"source B")
    (candidate_dir / "candidate.png").write_bytes(b"candidate B")
    write_candidate_manifest(
        candidate_dir,
        asset_id=asset_id,
        model_revision=MODEL_REVISION,
        prompt=PROMPTS[asset_id],
        seed=43,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    response = client.post(f"/review/{asset_id}", data=old_form)

    assert response.status_code == 409
    assert review_path.read_bytes() == before


def test_change_after_outer_snapshot_check_before_record_review_returns_409(
    workspace, monkeypatch
):
    import human_reference_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    candidate_dir = workspace / asset_id
    client = _client(workspace)
    old_form = _current_form_data(client, workspace, asset_id, "approved")
    real_record_review = review_server.record_review

    def regenerate_then_record(*args, **kwargs):
        (candidate_dir / "source.png").write_bytes(b"source B")
        (candidate_dir / "candidate.png").write_bytes(b"candidate B")
        write_candidate_manifest(
            candidate_dir,
            asset_id=asset_id,
            model_revision=MODEL_REVISION,
            prompt="Candidate B prompt.",
            seed=43,
            width=1024,
            height=1536,
            steps=28,
            guidance_scale=4.0,
            source_approval_sha256=SOURCE_APPROVAL_SHA256,
        )
        return real_record_review(*args, **kwargs)

    monkeypatch.setattr(review_server, "record_review", regenerate_then_record)

    response = client.post(f"/review/{asset_id}", data=old_form)

    assert response.status_code == 409
    assert not (candidate_dir / "reference_review.json").exists()


@pytest.mark.parametrize(
    "field",
    ("candidate_manifest_sha256", "source_sha256", "candidate_sha256"),
)
@pytest.mark.parametrize("invalid_hash", ("a" * 63, "A" * 64, "g" * 64))
def test_review_post_rejects_noncanonical_snapshot_hashes(
    workspace, field, invalid_hash
):
    asset_id = "rocketbox_male_adult_01"
    client = _client(workspace)
    form = _current_form_data(client, workspace, asset_id, "approved")
    form[field] = invalid_hash

    response = client.post(f"/review/{asset_id}", data=form)

    assert response.status_code == 400
    assert not (
        workspace / asset_id / "reference_review.json"
    ).exists()


def test_stale_review_is_shown_as_pending_without_rewriting_it(workspace):
    candidate_dir = workspace / "rocketbox_male_adult_01"
    record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "before regeneration",
        expected_snapshot=_snapshot_fields(
            workspace, "rocketbox_male_adult_01"
        ),
    )
    review_path = candidate_dir / "reference_review.json"
    before = review_path.read_bytes()
    (candidate_dir / "candidate.png").write_bytes(b"regenerated candidate")
    write_candidate_manifest(
        candidate_dir,
        asset_id="rocketbox_male_adult_01",
        model_revision=MODEL_REVISION,
        prompt=PROMPTS["rocketbox_male_adult_01"],
        seed=43,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    response = _client(workspace).get("/asset/rocketbox_male_adult_01")
    gate = _client(workspace).get("/gate")

    assert response.status_code == 200
    assert b"pending" in response.data
    assert review_path.read_bytes() == before
    assert gate.get_json()["state"] == "locked"


def test_pair_gate_only_approves_after_both_current_reviews(workspace):
    client = _client(workspace)

    assert client.get("/gate").get_json()["state"] == "locked"
    for asset_id in EXPECTED_ASSET_IDS:
        response = client.post(
            f"/review/{asset_id}",
            data={
                **_current_form_data(client, workspace, asset_id, "approved"),
                "notes": "approved",
            },
        )
        assert response.status_code == 302

    assert client.get("/gate").get_json() == {"state": "approved", "reason": ""}


def test_unknown_asset_and_invalid_csrf_are_rejected_without_writing(workspace):
    client = _client(workspace)

    assert client.get("/asset/not-a-human-reference").status_code == 404
    assert client.post(
        "/review/rocketbox_male_adult_01",
        data={"decision": "approved", "reviewer": "cross-site"},
    ).status_code == 400
    assert not (workspace / "rocketbox_male_adult_01" / "reference_review.json").exists()
