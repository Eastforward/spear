"""Bone-name-independent semantic decomposition for generated quadruped rigs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class SemanticRigError(ValueError):
    pass


@dataclass(frozen=True)
class QuadrupedSemantics:
    root: str
    axial: tuple[str, ...]
    head_chain: tuple[str, ...]
    tail_chain: tuple[str, ...]
    front_side_negative: tuple[str, ...]
    front_side_positive: tuple[str, ...]
    hind_side_negative: tuple[str, ...]
    hind_side_positive: tuple[str, ...]
    foot_leaves: tuple[str, ...]

    def chains(self) -> dict[str, tuple[str, ...]]:
        return {
            "axial": self.axial,
            "head": self.head_chain,
            "tail": self.tail_chain,
            "front_side_negative": self.front_side_negative,
            "front_side_positive": self.front_side_positive,
            "hind_side_negative": self.hind_side_negative,
            "hind_side_positive": self.hind_side_positive,
        }

    def all_bones(self) -> tuple[str, ...]:
        ordered = []
        for chain in self.chains().values():
            for name in chain:
                if name not in ordered:
                    ordered.append(name)
        return tuple(ordered)


def _point(record: Mapping, key: str) -> tuple[float, float, float]:
    value = record.get(key)
    if not isinstance(value, Sequence) or len(value) != 3:
        raise SemanticRigError(f"invalid {key} for bone {record.get('name')}")
    return tuple(float(component) for component in value)


def _path_to_root(name: str, by_name: Mapping[str, Mapping]) -> list[str]:
    path = []
    seen = set()
    current = name
    while current is not None:
        if current in seen or current not in by_name:
            raise SemanticRigError("bone hierarchy is cyclic or references a missing parent")
        seen.add(current)
        path.append(current)
        current = by_name[current].get("parent")
    path.reverse()
    return path


def _common_prefix(first: Sequence[str], second: Sequence[str]) -> list[str]:
    result = []
    for left, right in zip(first, second):
        if left != right:
            break
        result.append(left)
    return result


def infer_quadruped_semantics(
    records: Iterable[Mapping],
    *,
    bbox_min: Sequence[float],
    bbox_extent: Sequence[float],
    front_axis: str,
    low_leaf_height_fraction: float = 0.22,
) -> QuadrupedSemantics:
    """Decompose a one-root quadruped hierarchy using geometry and topology.

    Leaf *heads* are used for foot detection because generated rig exporters
    often extrapolate the visual tail of a leaf bone below the actual paw.
    """
    axis_components = {
        "positive-x": (0, 1.0, 1),
        "negative-x": (0, -1.0, 1),
        "positive-y": (1, 1.0, 0),
        "negative-y": (1, -1.0, 0),
    }
    if front_axis not in axis_components:
        raise SemanticRigError(f"unsupported front axis: {front_axis}")
    records = [dict(record) for record in records]
    by_name = {record.get("name"): record for record in records}
    if None in by_name or len(by_name) != len(records):
        raise SemanticRigError("bone names must be present and unique")
    roots = [name for name, record in by_name.items() if record.get("parent") is None]
    if len(roots) != 1:
        raise SemanticRigError(f"quadruped rig needs one root, got {roots}")
    root = roots[0]
    floor = float(bbox_min[2])
    height = float(bbox_extent[2])
    if height <= 0.0:
        raise SemanticRigError("mesh height must be positive")
    low_limit = floor + float(low_leaf_height_fraction) * height
    leaves = [
        name for name, record in by_name.items() if not record.get("children", [])
    ]
    foot_leaves = [
        name for name in leaves if _point(by_name[name], "head_world")[2] <= low_limit
    ]
    if len(foot_leaves) != 4:
        raise SemanticRigError(
            f"expected exactly four low limb leaves, got {foot_leaves}"
        )
    non_foot_leaves = [name for name in leaves if name not in foot_leaves]
    if len(non_foot_leaves) < 2:
        raise SemanticRigError("quadruped rig needs distinct head and tail leaves")

    forward_index, sign, lateral_index = axis_components[front_axis]
    forward = (
        lambda name: sign
        * _point(by_name[name], "head_world")[forward_index]
    )
    head_leaf = max(non_foot_leaves, key=forward)
    tail_leaf = min(non_foot_leaves, key=forward)
    if head_leaf == tail_leaf:
        raise SemanticRigError("head and tail leaf inference collapsed")
    head_path = _path_to_root(head_leaf, by_name)
    tail_path = _path_to_root(tail_leaf, by_name)
    head_tail_common = _common_prefix(head_path, tail_path)
    if not head_tail_common:
        raise SemanticRigError("head and tail chains do not share the root")
    tail_chain = tuple(tail_path[len(head_tail_common) :])
    if not tail_chain:
        raise SemanticRigError("tail chain is empty")

    limb_entries = []
    axial_attachment_indices = []
    for leaf in foot_leaves:
        path = _path_to_root(leaf, by_name)
        common = _common_prefix(path, head_path)
        if not common:
            raise SemanticRigError(f"foot path {leaf} does not share the axial root")
        chain = tuple(path[len(common) :])
        if not chain:
            raise SemanticRigError(f"foot chain {leaf} is empty")
        attachment = common[-1]
        axial_attachment_indices.append(head_path.index(attachment))
        lateral = sum(
            _point(by_name[name], "head_world")[lateral_index] for name in chain
        ) / len(chain)
        limb_entries.append(
            {
                "leaf": leaf,
                "chain": chain,
                "attachment": attachment,
                "forward": forward(leaf),
                "lateral": lateral,
            }
        )

    # The two most forward paws are forelimbs; the remaining two are hindlimbs.
    ordered = sorted(limb_entries, key=lambda item: item["forward"], reverse=True)
    front = ordered[:2]
    hind = ordered[2:]
    for label, pair in (("front", front), ("hind", hind)):
        if len(pair) != 2:
            raise SemanticRigError(f"{label} limb pair is incomplete")
        pair.sort(key=lambda item: item["lateral"])
        if not pair[0]["lateral"] < pair[1]["lateral"]:
            raise SemanticRigError(f"{label} limb lateral ordering is degenerate")

    last_limb_attachment = max(axial_attachment_indices)
    axial = tuple(head_path[: last_limb_attachment + 1])
    head_chain = tuple(head_path[last_limb_attachment + 1 :])
    if not axial or not head_chain:
        raise SemanticRigError("axial or head chain is empty")
    semantics = QuadrupedSemantics(
        root=root,
        axial=axial,
        head_chain=head_chain,
        tail_chain=tail_chain,
        front_side_negative=front[0]["chain"],
        front_side_positive=front[1]["chain"],
        hind_side_negative=hind[0]["chain"],
        hind_side_positive=hind[1]["chain"],
        foot_leaves=tuple(sorted(foot_leaves)),
    )
    if set(semantics.all_bones()) != set(by_name):
        missing = sorted(set(by_name) - set(semantics.all_bones()))
        extra = sorted(set(semantics.all_bones()) - set(by_name))
        raise SemanticRigError(
            f"semantic decomposition must cover every bone; missing={missing} extra={extra}"
        )
    return semantics
