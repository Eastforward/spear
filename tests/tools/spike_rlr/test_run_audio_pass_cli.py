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


def test_topdown_load_scene_dispatch_apartment():
    """render_topdown_2d._load_scene should return an apartment SceneSpec."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from render_topdown_2d import _load_scene
    sc = _load_scene(REPO / "data" / "apartment_v1_spec.json")
    tags = {a.tag for a in sc.animals}
    assert tags == {"dog_golden", "dog_husky"}
