from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "spike_rlr"
    / "tokenrig_human_review.py"
)
SPEC = importlib.util.spec_from_file_location("tokenrig_human_review_contract", MODULE_PATH)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, filename: bool = False) -> dict:
    result = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    result["filename" if filename else "path"] = path.name if filename else str(path.resolve())
    return result


@pytest.fixture()
def dynamic_review(tmp_path: Path) -> Path:
    asset_root = tmp_path / "person_01"
    static_dir = asset_root / "static_audit_v1"
    media_dir = asset_root / "dynamic_review_v1"
    static_dir.mkdir(parents=True)
    media_dir.mkdir()

    static_qa = static_dir / "static_qa.json"
    static_qa.write_text("{}", encoding="utf-8")
    bind = static_dir / "bind_pose.glb"
    bind.write_bytes(b"bind")
    static_evidence = {}
    for filename in review.STATIC_EVIDENCE:
        path = static_dir / filename
        path.write_bytes((filename + " static").encode())
        static_evidence[filename] = _record(path)

    retarget_dir = asset_root / "retarget_v1"
    retarget_dir.mkdir()
    retarget_manifest = retarget_dir / "retarget_manifest.json"
    retarget_manifest.write_text("{}", encoding="utf-8")
    retarget_metrics = retarget_dir / "retarget_metrics.json"
    retarget_metrics.write_text("{}", encoding="utf-8")
    glbs = {}
    for motion, filename in (("walking", "walking.glb"), ("standing_idle", "standing_idle.glb")):
        path = retarget_dir / filename
        path.write_bytes((motion + " glb").encode())
        glbs[motion] = _record(path)

    execution_dir = asset_root / "execution"
    execution_dir.mkdir()
    renderer_path = execution_dir / "blender_render_tokenrig_human_review.py"
    ffmpeg_path = execution_dir / "ffmpeg"
    ffprobe_path = execution_dir / "ffprobe"
    renderer_path.write_bytes(b"renderer code")
    ffmpeg_path.write_bytes(b"ffmpeg binary")
    ffprobe_path.write_bytes(b"ffprobe binary")

    actions = {}
    for motion, action_name in review.MOTIONS.items():
        views = {}
        for view in review.VIEWS:
            views[view] = {}
            for kind in ("png", "mp4"):
                filename = f"{motion}_{view}.{kind}"
                path = media_dir / filename
                path.write_bytes((filename + " bytes").encode())
                views[view][kind] = _record(path, filename=True)
        actions[motion] = {
            "action_name": action_name,
            "frame_start": 1,
            "frame_end": 31,
            "frame_count": 31,
            "fps": 30,
            "duration_s": 31 / 30,
            "views": views,
        }
    media_qa = media_dir / "media_qa.json"
    media_qa.write_text(
        json.dumps(
            {
                "schema": "tokenrig_human_media_qa_v1",
                "asset_id": "person_01",
                "automatic_checks": "passed",
                "actions": {
                    motion: {view: {"png": {"width": 1280}, "mp4": {"frame_count": 31}} for view in review.VIEWS}
                    for motion in review.MOTIONS
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "tokenrig_human_dynamic_review_v1",
        "asset_id": "person_01",
        "display_label": "Male canary",
        "instance_kind": "male_canary",
        "state_classification": "research_candidate",
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "fixed_floor_z_m": 0.0,
        "upstream": {
            "asset_id": "person_01",
            "static_qa": _record(static_qa),
            "bind_pose": _record(bind),
            "static_evidence": static_evidence,
            "retarget_manifest": _record(retarget_manifest),
            "retarget_metrics": _record(retarget_metrics),
            "glbs": glbs,
        },
        "actions": actions,
        "media_qa": _record(media_qa, filename=True),
        "execution": {
            "renderer": _record(renderer_path),
            "ffmpeg": _record(ffmpeg_path) | {"version": "ffmpeg version test"},
            "ffprobe": _record(ffprobe_path) | {"version": "ffprobe version test"},
        },
        "automatic_checks": "passed",
        "agent_visual_qa": "pending_agent_visual_qa",
        "user_acceptance": "pending_user_review",
        "environment": {"blender_version": "4.2.1", "fps": 30, "resolution": [1280, 720]},
    }
    (media_dir / "review_manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return media_dir


def _passing_checks() -> dict[str, bool]:
    return {name: True for name in review.AGENT_VISUAL_CHECKS}


def test_validated_snapshot_authenticates_all_upstreams_and_twenty_media_files(dynamic_review):
    snapshot = review.validated_review_snapshot(dynamic_review)
    assert snapshot["asset_id"] == "person_01"
    assert snapshot["review_manifest_sha256"] == _sha(dynamic_review / "review_manifest.json")
    assert len(snapshot["dynamic_media_sha256"]) == 20
    assert set(snapshot["static_evidence_sha256"]) == set(review.STATIC_EVIDENCE)
    assert snapshot["agent_visual_qa"] == "pending_agent_visual_qa"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "review manifest schema"),
        ("automatic_checks", "failed", "automatic checks"),
        ("canonical_front", "positive-y", "FRONT -Y"),
        ("canonical_up", "negative-z", r"UP \+Z"),
        ("user_acceptance", "user_approved", "user approval"),
    ],
)
def test_snapshot_rejects_stale_or_forged_manifest(dynamic_review, field, value, message):
    path = dynamic_review / "review_manifest.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match=message):
        review.validated_review_snapshot(dynamic_review)


def test_snapshot_rejects_media_and_upstream_tamper(dynamic_review):
    (dynamic_review / "walking_front.mp4").write_bytes(b"tampered")
    with pytest.raises(review.ReviewContractError, match="walking_front.mp4 SHA-256"):
        review.validated_review_snapshot(dynamic_review)

    dynamic_review = dynamic_review


def test_snapshot_rejects_symlink_media(dynamic_review):
    path = dynamic_review / "standing_idle_side.png"
    real = path.with_suffix(".real.png")
    path.rename(real)
    path.symlink_to(real.name)
    with pytest.raises(review.ReviewContractError, match="direct regular file"):
        review.validated_review_snapshot(dynamic_review)


def test_record_agent_pass_is_exclusive_hash_locked_and_never_user_approved(dynamic_review):
    decision_path = review.record_agent_visual_qa(
        dynamic_review,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex-route2-visual-qa",
        notes="Inspected all Walk/Idle views and representative frames.",
        checks=_passing_checks(),
    )
    payload = json.loads(decision_path.read_text())
    assert decision_path == dynamic_review.with_name("dynamic_review_v1.agent_visual_qa.json")
    assert payload["reviewer_kind"] == "agent"
    assert payload["status"] == "agent_qa_passed_pending_user_acceptance"
    assert payload["user_acceptance"] == "pending_user_review"
    assert "user_approved" not in decision_path.read_text()
    assert decision_path.stat().st_mode & 0o777 == 0o444
    assert review.assert_agent_qa_passed(dynamic_review)["status"] == payload["status"]
    with pytest.raises(review.ReviewContractError, match="already exists"):
        review.record_agent_visual_qa(
            dynamic_review,
            status="rejected",
            reviewer="another",
            notes="cannot replace",
            checks=_passing_checks(),
        )


def test_agent_pass_requires_every_visual_check_true(dynamic_review):
    checks = _passing_checks()
    checks["feet_contact_reasonable"] = False
    with pytest.raises(review.ReviewContractError, match="all visual checks"):
        review.record_agent_visual_qa(
            dynamic_review,
            status="agent_qa_passed_pending_user_acceptance",
            reviewer="codex",
            notes="one failure",
            checks=checks,
        )


def test_agent_decision_rejects_unknown_fields_status_or_empty_evidence(dynamic_review):
    with pytest.raises(review.ReviewContractError, match="status"):
        review.record_agent_visual_qa(dynamic_review, status="approved", reviewer="codex", notes="bad", checks=_passing_checks())
    with pytest.raises(review.ReviewContractError, match="reviewer"):
        review.record_agent_visual_qa(dynamic_review, status="rejected", reviewer="", notes="bad", checks=_passing_checks())
    with pytest.raises(review.ReviewContractError, match="notes"):
        review.record_agent_visual_qa(dynamic_review, status="rejected", reviewer="codex", notes="", checks=_passing_checks())
    checks = _passing_checks() | {"invented": True}
    with pytest.raises(review.ReviewContractError, match="checklist"):
        review.record_agent_visual_qa(dynamic_review, status="rejected", reviewer="codex", notes="bad", checks=checks)


def test_recorded_decision_invalidates_after_any_media_change(dynamic_review):
    review.record_agent_visual_qa(
        dynamic_review,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex",
        notes="all views inspected",
        checks=_passing_checks(),
    )
    (dynamic_review / "standing_idle_skeleton.mp4").write_bytes(b"changed")
    with pytest.raises(review.ReviewContractError, match="review snapshot changed"):
        review.read_agent_visual_qa(dynamic_review)


def test_pending_or_rejected_never_satisfies_downstream_gate(dynamic_review):
    with pytest.raises(review.ReviewNotAccepted, match="pending"):
        review.assert_agent_qa_passed(dynamic_review)
    review.record_agent_visual_qa(
        dynamic_review,
        status="rejected",
        reviewer="codex",
        notes="visible trouser tearing at frame 12",
        checks=_passing_checks() | {"trousers_intact": False},
    )
    with pytest.raises(review.ReviewNotAccepted, match="rejected"):
        review.assert_agent_qa_passed(dynamic_review)


def test_build_consolidated_bundle_is_appendable_no_replace_and_readable(dynamic_review, tmp_path):
    output = tmp_path / "route2_review_site_v1"
    catalog_path = review.build_consolidated_bundle([dynamic_review], output)
    assert catalog_path == output / "review_catalog.json"
    catalog = review.validate_review_catalog(output)
    assert [entry["asset_id"] for entry in catalog["entries"]] == ["person_01"]
    entry = catalog["entries"][0]
    assert set(entry["media"]["walking"]["front"]["mp4"]) == {
        "path",
        "sha256",
        "size_bytes",
    }
    assert set(entry["static_evidence"]["bind_front.png"]) == {
        "path",
        "sha256",
        "size_bytes",
    }
    assert set(entry["review_manifest"]) == {"path", "sha256", "size_bytes"}
    html = (output / "review.html").read_text(encoding="utf-8")
    for label in ("Male canary", "Walk", "Idle", "Front", "Side", "Top", "Feet", "Skeleton", "Static bind"):
        assert label in html
    assert "user_approved" not in html
    assert ".fbx" not in html.lower()
    assert "review_manifest.json" not in html
    assert "Approve" not in html
    with pytest.raises(review.ReviewContractError, match="already exists"):
        review.build_consolidated_bundle([dynamic_review], output)


def test_consolidated_bundle_accepts_multiple_unique_instances_and_rejects_duplicates(dynamic_review, tmp_path):
    source_root = dynamic_review.parent
    second_root = tmp_path / "person_02"
    shutil.copytree(source_root, second_root)
    second = second_root / "dynamic_review_v1"
    manifest_path = second / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["asset_id"] = "person_02"
    manifest["display_label"] = "Glasses attribute"
    manifest["instance_kind"] = "attribute_glasses"
    manifest["upstream"]["asset_id"] = "person_02"

    def relocated(record):
        old = Path(record["path"])
        new = second_root / old.relative_to(source_root)
        result = _record(new)
        if "version" in record:
            result["version"] = record["version"]
        return result

    for key in ("static_qa", "bind_pose", "retarget_manifest", "retarget_metrics"):
        manifest["upstream"][key] = relocated(manifest["upstream"][key])
    for motion in review.MOTIONS:
        manifest["upstream"]["glbs"][motion] = relocated(
            manifest["upstream"]["glbs"][motion]
        )
    for filename in review.STATIC_EVIDENCE:
        manifest["upstream"]["static_evidence"][filename] = relocated(
            manifest["upstream"]["static_evidence"][filename]
        )
    for name in ("renderer", "ffmpeg", "ffprobe"):
        manifest["execution"][name] = relocated(manifest["execution"][name])
    media_qa_path = second / "media_qa.json"
    media_qa = json.loads(media_qa_path.read_text())
    media_qa["asset_id"] = "person_02"
    media_qa_path.write_text(json.dumps(media_qa), encoding="utf-8")
    manifest["media_qa"] = _record(media_qa_path, filename=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    catalog = review.build_consolidated_bundle([dynamic_review, second], tmp_path / "site_two")
    assert len(review.validate_review_catalog(catalog.parent)["entries"]) == 2
    with pytest.raises(review.ReviewContractError, match="duplicate asset_id"):
        review.build_consolidated_bundle([dynamic_review, dynamic_review], tmp_path / "site_duplicate")


def test_catalog_fails_closed_if_source_media_changes_after_publication(dynamic_review, tmp_path):
    output = tmp_path / "site"
    review.build_consolidated_bundle([dynamic_review], output)
    (dynamic_review / "walking_top.png").write_bytes(b"changed")
    with pytest.raises(review.ReviewContractError, match="review source snapshot changed"):
        review.validate_review_catalog(output)


def test_catalog_locks_exact_agent_decision_bytes_not_only_status(dynamic_review, tmp_path):
    decision_path = review.record_agent_visual_qa(
        dynamic_review,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex",
        notes="original inspection notes",
        checks=_passing_checks(),
    )
    output = tmp_path / "site_with_decision"
    review.build_consolidated_bundle([dynamic_review], output)
    decision = json.loads(decision_path.read_text())
    decision["notes"] = "changed after catalog publication"
    decision_path.chmod(0o644)
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="decision changed"):
        review.validate_review_catalog(output)


def test_consolidated_page_displays_agent_reviewer_and_notes(dynamic_review, tmp_path):
    review.record_agent_visual_qa(
        dynamic_review,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex-route2-visual-qa",
        notes="All ten videos inspected at representative and contact frames.",
        checks=_passing_checks(),
    )
    output = tmp_path / "site_with_notes"
    review.build_consolidated_bundle([dynamic_review], output)
    page = (output / "review.html").read_text(encoding="utf-8")
    assert "codex-route2-visual-qa" in page
    assert "All ten videos inspected" in page


def test_external_artifact_descriptor_must_start_with_an_absolute_path():
    with pytest.raises(review.ReviewContractError, match="path is not absolute"):
        review._external_path(
            {"path": "relative/static_qa.json", "sha256": "0" * 64, "size_bytes": 1},
            "static QA",
        )


def test_snapshot_requires_renderer_and_ffmpeg_ffprobe_execution_contract(dynamic_review):
    path = dynamic_review / "review_manifest.json"
    payload = json.loads(path.read_text())
    del payload["execution"]["renderer"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="execution contract"):
        review.validated_review_snapshot(dynamic_review)


def test_snapshot_rejects_static_or_retarget_path_outside_asset_bundle_even_with_same_hash(dynamic_review, tmp_path):
    manifest_path = dynamic_review / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = Path(manifest["upstream"]["static_evidence"]["bind_front.png"]["path"])
    substitute = tmp_path / "outside_bind_front.png"
    substitute.write_bytes(original.read_bytes())
    manifest["upstream"]["static_evidence"]["bind_front.png"] = _record(substitute)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="static evidence.*contained"):
        review.validated_review_snapshot(dynamic_review)


@pytest.mark.parametrize("field", ["display_label", "agent_review", "media", "static_evidence"])
def test_catalog_is_rebuilt_from_snapshot_and_rejects_entry_field_substitution(dynamic_review, tmp_path, field):
    output = tmp_path / f"site_tamper_{field}"
    review.build_consolidated_bundle([dynamic_review], output)
    catalog_path = output / "review_catalog.json"
    catalog_path.chmod(0o644)
    catalog = json.loads(catalog_path.read_text())
    if field == "display_label":
        catalog["entries"][0][field] = "Substituted label"
    elif field == "agent_review":
        catalog["entries"][0][field] = {"reviewer": "forged", "notes": "forged"}
    elif field == "media":
        original = Path(catalog["entries"][0][field]["walking"]["front"]["mp4"]["path"])
        substitute = tmp_path / "substitute.mp4"
        substitute.write_bytes(original.read_bytes())
        catalog["entries"][0][field]["walking"]["front"]["mp4"]["path"] = str(substitute)
    else:
        original = Path(catalog["entries"][0][field]["bind_front.png"]["path"])
        substitute = tmp_path / "substitute.png"
        substitute.write_bytes(original.read_bytes())
        catalog["entries"][0][field]["bind_front.png"]["path"] = str(substitute)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="entry does not match"):
        review.validate_review_catalog(output)


def test_rejected_instance_uses_explicit_non_green_review_style(dynamic_review, tmp_path):
    checks = _passing_checks()
    checks["trousers_intact"] = False
    review.record_agent_visual_qa(
        dynamic_review,
        status="rejected",
        reviewer="codex",
        notes="trouser tearing",
        checks=checks,
    )
    output = tmp_path / "rejected_site"
    review.build_consolidated_bundle([dynamic_review], output)
    page = (output / "review.html").read_text(encoding="utf-8")
    assert 'data-status="rejected"' in page
    assert '.status[data-status="rejected"]' in page
    assert "#9a3b34" in page or "#7a2b25" in page


def test_catalog_rejects_forged_html_even_if_attacker_updates_html_descriptor(dynamic_review, tmp_path):
    output = tmp_path / "forged_html_site"
    review.build_consolidated_bundle([dynamic_review], output)
    html_path = output / "review.html"
    catalog_path = output / "review_catalog.json"
    html_path.chmod(0o644)
    catalog_path.chmod(0o644)
    html_path.write_text("<!doctype html><title>forged review</title>", encoding="utf-8")
    catalog = json.loads(catalog_path.read_text())
    catalog["review_html"] = {
        "filename": "review.html",
        "sha256": _sha(html_path),
        "size_bytes": html_path.stat().st_size,
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="review.html does not match rebuilt"):
        review.validate_review_catalog(output)


def test_catalog_recomputes_overall_agent_state_from_entries(dynamic_review, tmp_path):
    output = tmp_path / "forged_overall_site"
    review.build_consolidated_bundle([dynamic_review], output)
    catalog_path = output / "review_catalog.json"
    catalog_path.chmod(0o644)
    catalog = json.loads(catalog_path.read_text())
    catalog["overall_agent_qa"] = "agent_qa_passed_pending_user_acceptance"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(review.ReviewContractError, match="overall agent state"):
        review.validate_review_catalog(output)


def test_catalog_cli_supports_repeated_review_dirs_and_explicit_new_output():
    args = review.parse_args(
        [
            "build-site",
            "--review-dir", "/tmp/male/dynamic_review_v1",
            "--review-dir", "/tmp/female/dynamic_review_v1",
            "--output-dir", "/tmp/route2_review_site_v1",
        ]
    )
    assert len(args.review_dir) == 2
    assert args.output_dir == Path("/tmp/route2_review_site_v1")
