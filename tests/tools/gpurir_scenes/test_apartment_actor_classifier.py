import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "gpurir_scenes"))

from apartment_actor_classifier import (  # noqa: E402
    classify_actor, SHELL_LABELS,
)


def test_wall_actor_classified_as_shell_wall():
    label = classify_actor(
        actor_name="Meshes/24_wall/Wall_North:SM_wall_1",
        bbox_min_z=0.0, bbox_max_z=280.0,
        x_extent_cm=1500.0, y_extent_cm=10.0,
    )
    assert label == "shell_wall"
    assert label in SHELL_LABELS


def test_ceiling_by_zmin_classified_as_shell_ceiling():
    label = classify_actor(
        actor_name="Meshes/22_ceiling/Ceiling",
        bbox_min_z=310.0, bbox_max_z=350.0,
        x_extent_cm=1500.0, y_extent_cm=1200.0,
    )
    assert label == "shell_ceiling"


def test_floor_by_zmax_classified_as_shell_floor():
    label = classify_actor(
        actor_name="Meshes/21_floor/Floor",
        bbox_min_z=0.0, bbox_max_z=3.0,
        x_extent_cm=1500.0, y_extent_cm=1200.0,
    )
    assert label == "shell_floor"


def test_door_actor_classified_as_shell_door():
    label = classify_actor(
        actor_name="Meshes/08_door/Door_Front:SM_door_1",
        bbox_min_z=0.0, bbox_max_z=210.0,
        x_extent_cm=100.0, y_extent_cm=10.0,
    )
    assert label == "shell_door"


def test_window_actor_classified_as_shell_window():
    label = classify_actor(
        actor_name="Meshes/09_window/Window_1:SM_window_5",
        bbox_min_z=100.0, bbox_max_z=250.0,
        x_extent_cm=150.0, y_extent_cm=8.0,
    )
    assert label == "shell_window"


def test_curtain_actor_classified_as_shell_curtain():
    label = classify_actor(
        actor_name="Meshes/16_curtain/Curtain",
        bbox_min_z=50.0, bbox_max_z=280.0,
        x_extent_cm=200.0, y_extent_cm=5.0,
    )
    assert label == "shell_curtain"


def test_picture_actor_classified_as_shell_picture():
    label = classify_actor(
        actor_name="Meshes/11_picture/Picture_2",
        bbox_min_z=140.0, bbox_max_z=200.0,
        x_extent_cm=60.0, y_extent_cm=4.0,
    )
    assert label == "shell_picture"


def test_mirror_actor_classified_as_shell_mirror():
    label = classify_actor(
        actor_name="Meshes/19_mirror/Mirror:SM_Mirror_5",
        bbox_min_z=212.0, bbox_max_z=280.0,
        x_extent_cm=80.0, y_extent_cm=6.0,
    )
    assert label == "shell_mirror"


def test_huge_actor_classified_as_structural():
    label = classify_actor(
        actor_name="Meshes/38_otherstructure/BigStructure",
        bbox_min_z=0.0, bbox_max_z=280.0,
        x_extent_cm=1500.0, y_extent_cm=1300.0,   # 19.5 m2, above 20 m2 threshold no; try 15x15
    )
    # bbox area = 1500 * 1300 = 1.95e6 cm2 > 2e5 -> structural
    assert label == "structural"


def test_chair_actor_classified_as_furniture():
    label = classify_actor(
        actor_name="Meshes/05_chair/LivingRoom_Chair_01:SM_chair_living_2",
        bbox_min_z=28.0, bbox_max_z=163.0,
        x_extent_cm=128.0, y_extent_cm=128.0,
    )
    assert label == "furniture"
    assert label not in SHELL_LABELS


def test_sofa_actor_classified_as_furniture():
    label = classify_actor(
        actor_name="Meshes/06_sofa/Sofa",
        bbox_min_z=25.0, bbox_max_z=90.0,
        x_extent_cm=200.0, y_extent_cm=90.0,
    )
    assert label == "furniture"


def test_shell_labels_disjoint_from_furniture():
    assert "furniture" not in SHELL_LABELS
