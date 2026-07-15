from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_stable_animal_ofat_apartment_review.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_review_requires_complete_walk_idle_batch_and_per_instance_registry():
    text = source()
    assert 'STATUS_SCHEMA = "stable_animal_apartment_render_status_v1"' in text
    assert 'status.get("passed_job_count") != spec["clip_count"]' in text
    assert 'status.get("failed_job_count") != 0' in text
    assert 'set(registry.get("clips", {})) != set(ACTIONS)' in text
    assert 'planned["spec_evidence"]["sha256"]' in text


def test_review_rehashes_all_ue_video_audio_and_runtime_artifacts():
    text = source()
    for field in (
        "actor_visual_metadata",
        "apartment_video",
        "topdown_review_video",
        "annotated_review_video",
        "binaural_audio",
        "binaural_source_schedule",
    ):
        assert f'"{field}"' in text
    assert "verify_descriptor" in text
    assert "sha256_file(path)" in text


def test_review_preserves_absolute_paths_and_browser_urls_without_promotion():
    text = source()
    assert "resolved.relative_to(AVENGINE_ROOT.resolve())" in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"human_visual_review": "pending"' in text
    assert 'output_manifest.open("x"' in text
    assert 'output_html.open("x"' in text
