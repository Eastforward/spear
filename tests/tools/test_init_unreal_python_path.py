from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[2]
INIT_UNREAL = REPO / "cpp" / "unreal_projects" / "SpearSim" / "Content" / "Python" / "init_unreal.py"


def test_init_unreal_adds_repo_python_package_dir():
    old_path = list(sys.path)
    try:
        runpy.run_path(str(INIT_UNREAL))

        assert str(REPO / "python") in sys.path
        assert str(REPO / "editor") in sys.path
    finally:
        sys.path[:] = old_path
