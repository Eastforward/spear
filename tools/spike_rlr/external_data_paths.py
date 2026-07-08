"""External dataset path helpers for AVEngine/SPEAR experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    default_path: Path
    env_var: str
    acquisition_hint: str


DATASETS: dict[str, DatasetSpec] = {
    "replicacad": DatasetSpec(
        name="replicacad",
        label="ReplicaCAD",
        default_path=Path("/data/datasets/replica_cad"),
        env_var="AVENGINE_REPLICACAD_ROOT",
        acquisition_hint=(
            "ReplicaCAD: run python -m habitat_sim.utils.datasets_download --uids "
            "replica_cad_dataset --data-path /data/datasets --no-replace. "
            "Set AVENGINE_REPLICACAD_ROOT when using a non-default location."
        ),
    ),
    "mixamo": DatasetSpec(
        name="mixamo",
        label="Mixamo",
        default_path=Path("/data/datasets/mixamo"),
        env_var="AVENGINE_MIXAMO_ROOT",
        acquisition_hint=(
            "Place user-downloaded Mixamo FBX files under this directory or set "
            "AVENGINE_MIXAMO_ROOT to the FBX dataset directory."
        ),
    ),
}


class DatasetMissingError(FileNotFoundError):
    """Raised when an expected external dataset root is missing."""


def dataset_spec(name: str) -> DatasetSpec:
    return DATASETS[name]


def dataset_root(name: str) -> Path:
    spec = dataset_spec(name)
    return Path(os.environ.get(spec.env_var, spec.default_path)).expanduser()


def require_dataset_root(name: str) -> Path:
    spec = dataset_spec(name)
    root = dataset_root(name)
    if root.exists():
        return root
    raise DatasetMissingError(
        f"{spec.label} dataset root does not exist: {root}. "
        f"Set {spec.env_var} or create the default path. {spec.acquisition_hint}"
    )
