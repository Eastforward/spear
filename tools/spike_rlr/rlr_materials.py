"""Deterministic conversion of SPEAR acoustic materials for RLR."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping


RLR_MATERIAL_BANDS_HZ = (125.0, 500.0, 2000.0, 8000.0)


def _finite_vector(value: Any, *, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{label} contains a non-finite value")
        result.append(float(item))
    return result


def build_rlr_materials_payload(materials_sidecar: Mapping[str, Any]) -> bytes:
    """Return canonical SoundSpaces/RLR material JSON bytes.

    Both the renderer and readback validator call this function.  The derived
    file is therefore evidence of a deterministic transform of the
    authenticated source sidecar, not an independently self-asserted input.
    """
    if not isinstance(materials_sidecar, Mapping):
        raise ValueError("acoustic materials sidecar must be an object")
    materials = materials_sidecar.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("acoustic materials sidecar has no materials")

    def interleave(values: list[float]) -> list[float]:
        return [
            value
            for frequency, coefficient in zip(RLR_MATERIAL_BANDS_HZ, values)
            for value in (frequency, coefficient)
        ]

    converted = []
    seen_names = set()
    for index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise ValueError(f"acoustic material {index} must be an object")
        name = material.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("acoustic material names are missing or duplicated")
        seen_names.add(name)
        absorption = _finite_vector(
            material.get("alpha"),
            label=f"acoustic material {name!r} alpha",
            length=4,
        )
        transmission = _finite_vector(
            material.get("transmission"),
            label=f"acoustic material {name!r} transmission",
            length=4,
        )
        scattering = material.get("scattering")
        if (
            isinstance(scattering, bool)
            or not isinstance(scattering, (int, float))
            or not math.isfinite(float(scattering))
        ):
            raise ValueError(
                f"acoustic material {name!r} scattering is non-finite"
            )
        converted.append(
            {
                "name": name,
                "absorption": interleave(absorption),
                "scattering": interleave([float(scattering)] * 4),
                "transmission": interleave(transmission),
                "labels": [name],
            }
        )
    payload = {"materials": converted}
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = ["RLR_MATERIAL_BANDS_HZ", "build_rlr_materials_payload"]
