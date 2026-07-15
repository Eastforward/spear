from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_stable_animal_ofat_video_index.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_index_authenticates_manifest_and_stays_inside_avengine():
    text = source()
    assert "contracts.manifest_sha256(manifest)" in text
    assert "path.relative_to(AVENGINE_ROOT.resolve())" in text
    assert "artifact is outside AVEngine" in text


def test_index_exposes_every_instance_static_glb_walk_and_idle():
    text = source()
    assert '"static": artifact(item["static_fixed_scale_review"])' in text
    assert '"glb": artifact(item["realization"]["glb"])' in text
    assert '"Walking": artifact(videos["Walking"])' in text
    assert '"Idle": artifact(videos["Idle"])' in text
    assert "sampled_attributes" in text


def test_index_does_not_overwrite_existing_review_page():
    text = source()
    assert 'output.open("x"' in text
    assert "refusing to replace output" in text
