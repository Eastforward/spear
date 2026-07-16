import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


class _Component:
    def __init__(self, bone_count):
        self.bone_count = int(bone_count)

    def GetNumBones(self):
        return self.bone_count


def test_select_skeletal_mesh_component_prefers_the_component_with_most_bones():
    from rig_direction_check import select_skeletal_mesh_component

    empty = _Component(0)
    rigged = _Component(80)
    actor = object()

    class _UnrealService:
        def get_components_by_class(self, *, actor, uclass):
            assert actor is actor_instance
            assert uclass == "USkeletalMeshComponent"
            return [empty, rigged]

    actor_instance = actor
    diagnostics = []
    selected = select_skeletal_mesh_component(
        unreal_service=_UnrealService(),
        actor=actor,
        diagnostics=diagnostics,
    )

    assert selected is rigged
    assert diagnostics == []
