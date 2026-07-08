"""Verify CLI args exist on the audio pass + topdown render scripts.

These are cheap smoke tests — they only run `--help` and check that the
new/existing CLI flags are advertised. Full end-to-end audio rendering
is verified in Task 9's live runs.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_audio_pass_help_shows_spec_and_mesh_args():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/ss2/bin/python",
         str(REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"help failed:\n{r.stderr}"
    assert "--spec" in r.stdout
    assert "--mesh" in r.stdout
    assert "--materials" in r.stdout


def test_topdown_help_shows_spec_arg():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools" / "spike_rlr" / "render_topdown_2d.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"help failed:\n{r.stderr}"
    assert "--spec" in r.stdout


def test_audio_pass_load_scene_dispatch_shoebox():
    """The dispatcher (imported directly) should return shoebox composer for v2."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_scene_and_scene_two_dogs
    fn = _load_scene_and_scene_two_dogs(REPO / "data" / "shoebox_v2_spec.json")
    assert fn.__name__ == "compose_two_dog_scene_v2"


def test_audio_pass_load_scene_dispatch_apartment():
    """The dispatcher should return apartment composer for apartment_v1."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_scene_and_scene_two_dogs
    fn = _load_scene_and_scene_two_dogs(REPO / "data" / "apartment_v1_spec.json")
    assert fn.__name__ == "compose_two_dog_scene_apartment"


def test_audio_pass_has_explicit_sources_for_review_animals():
    text = (REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py").read_text()

    assert '"dog_beagle_v2":' in text
    assert '"cat_british_shorthair_v2":' in text
    assert "Barking Aldi Dog_358.wav" in text
    assert "cat purring" in text


def test_audio_agent_yaw_tracks_scene_mic_yaw():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _habitat_agent_yaw_deg_for_scene_yaw_deg

    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(90.0) == 180.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(0.0) == 270.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(180.0) == 90.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(270.0) == 0.0


def test_audio_pass_reads_mic_yaw_from_spec_with_camera_fallback():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _mic_yaw_deg_from_spec

    assert _mic_yaw_deg_from_spec({
        "mic": {"yaw_deg": 34.5},
        "camera_configs": [{"yaw_deg": 90.0}],
    }) == 34.5
    assert _mic_yaw_deg_from_spec({
        "mic": {},
        "camera_configs": [{"yaw_deg": 90.0}],
    }) == 90.0
    assert _mic_yaw_deg_from_spec({"mic": {}}) == 90.0


def test_topdown_load_scene_dispatch_apartment():
    """render_topdown_2d._load_scene should return an apartment SceneSpec."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from render_topdown_2d import _load_scene
    sc = _load_scene(REPO / "data" / "apartment_v1_spec.json")
    tags = {a.tag for a in sc.animals}
    assert tags == {"dog_golden", "dog_husky"}
