"""Static contract for exclusive quadruped review video encoding."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/encode_quadruped_review_media.py"
)


def test_encoder_binds_render_frames_fixed_ffmpeg_and_video_readback():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "require_render_manifest(" in text
    assert "render_frame_set(render_payload)" in text
    assert "expected_review_ffmpeg_config" in text
    assert '"-nostdin"' in text
    assert '"-n"' in text
    assert '"-frames:v"' in text
    assert '"-c:v"' in text
    assert '"-pix_fmt"' in text
    assert '"-movflags"' in text
    assert "verify_video(" in text
    assert 'with path.open("x", encoding="utf-8")' in text
    assert "refusing to replace review video" in text
    assert "refusing to replace encode manifest" in text
