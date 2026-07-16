import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_review_work_directory_is_unique_for_same_action_name():
    from build_review_videos import review_work_key

    female = Path("/tmp/review/female/walking")
    male = Path("/tmp/review/male/walking")

    assert review_work_key(female) != review_work_key(male)
    assert review_work_key(female) == review_work_key(female)


def test_topdown_renderer_cli_accepts_per_clip_work_directory():
    source = (
        REPO / "tools" / "spike_rlr" / "render_topdown_2d.py"
    ).read_text(encoding="utf-8")

    assert '"--tmp-frames-dir"' in source
    assert "tmp_frames_dir=args.tmp_frames_dir" in source
