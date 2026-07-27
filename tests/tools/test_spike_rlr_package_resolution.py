from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def test_runtime_spike_rlr_package_is_not_shadowed_by_test_helpers() -> None:
    script = f"""
import pathlib
import sys

repo = pathlib.Path({str(REPO)!r})
sys.path.insert(0, str(repo / "tests"))
sys.path.insert(0, str(repo))

import tools.generated_animal_forward_contract
import tools.spike_rlr.animal_audio as animal_audio

resolved = pathlib.Path(animal_audio.__file__).resolve()
expected = (repo / "tools/spike_rlr/animal_audio.py").resolve()
if resolved != expected:
    raise SystemExit(f"shadowed animal_audio: {{resolved}} != {{expected}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
