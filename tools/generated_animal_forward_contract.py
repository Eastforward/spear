"""Single-point forward declaration contract for generated animal assets.

The anatomical front of a generated animal is declared exactly once, in a
self-hashed forward declaration produced before heading normalization.  Every
later stage derives from that declaration instead of re-declaring the same
fact:

- heading normalization consumes ``reviewed_source_front_yaw_deg`` from the
  declaration and rotates the rig into the one canonical target axis;
- motion retarget then runs with the *constant* motion basis of the declared
  motion donor.  A per-asset motion-basis yaw or side-chain flip is no longer
  a reviewable choice: needing one means the forward declaration (or the
  canonicalization stage) is wrong and must be fixed at the source.

New motion donor families must be declared in ``MOTION_DONOR_BASIS`` before
use; an unknown donor tag fails immediately instead of silently borrowing
another donor's convention.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

from tools.controlled_source_asset_schema import StrictJSONError, strict_json_loads


FORWARD_DECLARATION_SCHEMA = "avengine_generated_animal_forward_declaration_v1"
CANONICAL_TARGET_FRONT_AXIS = "positive-x"
DOG_MOTION_DONOR_ID = "quaternius_universal_quadruped_v1"
CAT_MOTION_DONOR_ID = "quaternius_cat_v1"

# One motion-basis convention per donor family, never per asset.  Both pinned
# Quaternius Animal Pack donors drive a +X-forward canonical target with an
# identity basis and matched side chains.  The cat donor is a skeleton/action
# source only; its rendered body, materials, textures, and skin weights are not
# authorities for a generated target.
MOTION_DONOR_BASIS = {
    DOG_MOTION_DONOR_ID: {
        "motion_basis_yaw_deg": 0,
        "side_chain_mode": "matched",
    },
    CAT_MOTION_DONOR_ID: {
        "motion_basis_yaw_deg": 0,
        "side_chain_mode": "matched",
    },
}

MOTION_DONOR_BY_SPECIES = {
    "dog": DOG_MOTION_DONOR_ID,
    "cat": CAT_MOTION_DONOR_ID,
}

# This authority contract is intentionally independent from the artifact
# identity below.  A donor may supply rest-skeleton semantics and the named
# actions, while the generated target remains the sole mesh/PBR/skeleton/skin
# authority.  The post-TokenRig runner also rechecks the matching fields in the
# retarget manifest before it admits any render.
MOTION_DONOR_CONTRACTS = {
    DOG_MOTION_DONOR_ID: {
        "target_species": "dog",
        "donor_role": "skeleton_semantics_and_animation_only",
        "required_actions": ["Walking", "Idle"],
        "animation_channels_used": ["translation", "rotation", "scale"],
        "geometry_used": False,
        "weights_used": False,
        "target_mesh_authority": "generated_target_glb",
    },
    CAT_MOTION_DONOR_ID: {
        "target_species": "cat",
        "donor_role": "skeleton_semantics_and_animation_only",
        "required_actions": ["Walking", "Idle"],
        "animation_channels_used": ["translation", "rotation", "scale"],
        "geometry_used": False,
        "weights_used": False,
        "target_mesh_authority": "generated_target_glb",
    },
}

# Donor tags are identities, not aliases for arbitrary GLBs that happen to
# expose similarly named actions.  Keep the approved source bytes independent
# from the per-asset declaration so historical declarations remain readable
# while every new execution authenticates the actual donor supplied to the
# runner.
MOTION_DONOR_ARTIFACTS = {
    DOG_MOTION_DONOR_ID: {
        "sha256": (
            "bf9d2fdaf74a36be453edf4516a0b13b042cfce2d2614e0bf3ee24d40d553032"
        ),
        "size_bytes": 143692,
    },
    CAT_MOTION_DONOR_ID: {
        "sha256": (
            "af2afb5e92c6d9daae98a918f8bd2bcb13ea4d7cfb880020d0d263e4d2f1277e"
        ),
        "size_bytes": 163516,
    },
}

HEAD_END_DECISION_SOURCES = (
    "human_review",
    "human_confirming_estimator",
)


class ForwardContractError(ValueError):
    pass


def _canonical_hash(payload: dict, excluded_key: str) -> str:
    reduced = {key: value for key, value in payload.items() if key != excluded_key}
    encoded = json.dumps(reduced, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motion_donor_for_species(target_species: str) -> str:
    try:
        return MOTION_DONOR_BY_SPECIES[target_species]
    except (KeyError, TypeError) as error:
        raise ForwardContractError(
            f"unsupported generated-animal target species {target_species!r}; "
            f"expected one of {tuple(MOTION_DONOR_BY_SPECIES)}"
        ) from error


def expected_motion_donor_contract(motion_donor_tag: str) -> dict:
    contract = MOTION_DONOR_CONTRACTS.get(motion_donor_tag)
    if contract is None:
        raise ForwardContractError(
            f"unknown motion donor tag {motion_donor_tag!r}; declare its "
            "authority contract before use"
        )
    return copy.deepcopy(contract)


def resolve_motion_donor(
    target_species: str | None,
    motion_donor_tag: str | None,
) -> tuple[str, str]:
    if target_species is None:
        if motion_donor_tag not in MOTION_DONOR_CONTRACTS:
            if motion_donor_tag is not None:
                raise ForwardContractError(
                    f"unknown motion donor tag {motion_donor_tag!r}"
                )
            raise ForwardContractError(
                "a registered target species or motion donor tag is required"
            )
        target_species = MOTION_DONOR_CONTRACTS[motion_donor_tag][
            "target_species"
        ]
    expected_tag = motion_donor_for_species(target_species)
    if motion_donor_tag is not None and motion_donor_tag != expected_tag:
        raise ForwardContractError(
            "motion donor does not match the generated target species "
            f"(species={target_species!r}, expected={expected_tag!r}, "
            f"got={motion_donor_tag!r})"
        )
    return target_species, expected_tag


def _load_glb_json(path: Path) -> dict:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ForwardContractError(f"motion donor is not a complete GLB: {path}")
    magic, version, declared_size = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_size != len(payload):
        raise ForwardContractError(f"motion donor has an invalid GLB header: {path}")
    offset = 12
    document = None
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ForwardContractError(
                f"motion donor has a truncated GLB chunk header: {path}"
            )
        chunk_size, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_size
        if chunk_end > len(payload):
            raise ForwardContractError(
                f"motion donor has a truncated GLB chunk: {path}"
            )
        if chunk_type == 0x4E4F534A:
            if document is not None:
                raise ForwardContractError(
                    f"motion donor has multiple GLB JSON chunks: {path}"
                )
            try:
                document = strict_json_loads(
                    payload[offset:chunk_end].rstrip(b" \t\r\n\x00")
                )
            except StrictJSONError as error:
                raise ForwardContractError(
                    f"motion donor has an invalid GLB JSON chunk: {path}"
                ) from error
        offset = chunk_end
    if offset != len(payload) or not isinstance(document, dict):
        raise ForwardContractError(f"motion donor GLB JSON is missing: {path}")
    return document


def _assert_motion_donor_action_semantics(
    motion_donor_tag: str,
    source_motion_glb: Path,
) -> list[str]:
    document = _load_glb_json(source_motion_glb)
    animations = document.get("animations")
    if not isinstance(animations, list):
        raise ForwardContractError(
            f"motion donor lacks animations: {source_motion_glb}"
        )
    names = [
        item.get("name")
        for item in animations
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    required = expected_motion_donor_contract(motion_donor_tag)[
        "required_actions"
    ]
    if len(names) != len(animations) or sorted(names) != sorted(required):
        raise ForwardContractError(
            "motion donor action semantics do not match the declared donor "
            f"(expected={required}, got={names}): {source_motion_glb}"
        )
    return names


def build_forward_declaration(
    *,
    asset_workspace: str,
    input_glb: Path,
    reviewed_source_front_yaw_deg: float,
    head_end_decision_source: str,
    head_end_evidence: Path,
    motion_donor_tag: str | None = None,
    target_species: str | None = None,
    estimate: dict | None = None,
) -> dict:
    if not asset_workspace or not asset_workspace.strip():
        raise ForwardContractError("asset_workspace must be a non-empty identifier")
    if not math.isfinite(reviewed_source_front_yaw_deg):
        raise ForwardContractError("reviewed_source_front_yaw_deg must be finite")
    if head_end_decision_source not in HEAD_END_DECISION_SOURCES:
        raise ForwardContractError(
            f"head_end_decision_source must be one of {HEAD_END_DECISION_SOURCES}"
        )
    target_species, motion_donor_tag = resolve_motion_donor(
        target_species, motion_donor_tag
    )
    input_glb = Path(input_glb).resolve()
    if input_glb.is_symlink() or not input_glb.is_file() or input_glb.stat().st_size <= 0:
        raise ForwardContractError(f"missing or unsafe input GLB: {input_glb}")
    head_end_evidence = Path(head_end_evidence).resolve()
    if (
        head_end_evidence.is_symlink()
        or not head_end_evidence.is_file()
        or head_end_evidence.stat().st_size <= 0
    ):
        raise ForwardContractError(
            f"missing or unsafe head-end evidence: {head_end_evidence}"
        )
    payload = {
        "schema": FORWARD_DECLARATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_workspace": asset_workspace,
        "input_glb": {
            "path": str(input_glb),
            "sha256": sha256_file(input_glb),
            "size_bytes": input_glb.stat().st_size,
        },
        "reviewed_source_front_yaw_deg": float(reviewed_source_front_yaw_deg),
        "target_front_axis": CANONICAL_TARGET_FRONT_AXIS,
        "target_species": target_species,
        "head_end_decision": {
            "source": head_end_decision_source,
            "evidence_path": str(head_end_evidence),
            "evidence_sha256": sha256_file(head_end_evidence),
        },
        "motion_donor_tag": motion_donor_tag,
        "expected_motion_basis": dict(MOTION_DONOR_BASIS[motion_donor_tag]),
        "expected_motion_donor_contract": expected_motion_donor_contract(
            motion_donor_tag
        ),
        "forward_estimate": estimate,
        "policy": "declare_front_once_derive_everywhere",
    }
    payload["declaration_sha256"] = _canonical_hash(payload, "declaration_sha256")
    return payload


def load_forward_declaration(path: Path) -> dict:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ForwardContractError(f"missing or unsafe forward declaration: {path}")
    try:
        declaration = strict_json_loads(path.read_bytes())
    except (OSError, StrictJSONError) as error:
        raise ForwardContractError(f"invalid forward declaration: {error}") from error
    if not isinstance(declaration, dict):
        raise ForwardContractError("forward declaration must be a JSON object")
    if declaration.get("schema") != FORWARD_DECLARATION_SCHEMA:
        raise ForwardContractError(
            f"forward declaration schema mismatch: {declaration.get('schema')!r}"
        )
    if declaration.get("declaration_sha256") != _canonical_hash(
        declaration, "declaration_sha256"
    ):
        raise ForwardContractError("forward declaration failed self-hash authentication")
    yaw = declaration.get("reviewed_source_front_yaw_deg")
    if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ForwardContractError("reviewed_source_front_yaw_deg must be finite")
    if declaration.get("target_front_axis") != CANONICAL_TARGET_FRONT_AXIS:
        raise ForwardContractError(
            "forward declaration target axis must be the canonical "
            f"{CANONICAL_TARGET_FRONT_AXIS!r}; per-asset target axes are forbidden"
        )
    head_end = declaration.get("head_end_decision")
    if (
        not isinstance(head_end, dict)
        or head_end.get("source") not in HEAD_END_DECISION_SOURCES
    ):
        raise ForwardContractError(
            "forward declaration head-end decision must record a human source"
        )
    evidence_path_value = head_end.get("evidence_path")
    if not isinstance(evidence_path_value, str) or not evidence_path_value:
        raise ForwardContractError(
            "forward declaration head-end evidence path is missing"
        )
    evidence_path = Path(evidence_path_value).resolve()
    if (
        evidence_path.is_symlink()
        or not evidence_path.is_file()
        or evidence_path.stat().st_size <= 0
        or str(evidence_path) != evidence_path_value
        or head_end.get("evidence_sha256") != sha256_file(evidence_path)
    ):
        raise ForwardContractError(
            "forward declaration head-end evidence is missing or changed"
        )
    donor_tag = declaration.get("motion_donor_tag")
    if donor_tag not in MOTION_DONOR_BASIS:
        raise ForwardContractError(
            f"forward declaration names unknown motion donor {donor_tag!r}"
        )
    if declaration.get("expected_motion_basis") != MOTION_DONOR_BASIS[donor_tag]:
        raise ForwardContractError(
            "forward declaration motion basis does not match the donor table; "
            "regenerate the declaration instead of editing it"
        )
    target_species = declaration.get("target_species")
    donor_contract = declaration.get("expected_motion_donor_contract")
    if (target_species is None) != (donor_contract is None):
        raise ForwardContractError(
            "forward declaration has an incomplete species/donor authority contract"
        )
    if target_species is not None:
        if motion_donor_for_species(target_species) != donor_tag:
            raise ForwardContractError(
                "forward declaration motion donor does not match target species"
            )
        if donor_contract != expected_motion_donor_contract(donor_tag):
            raise ForwardContractError(
                "forward declaration donor authority contract changed; "
                "regenerate the declaration instead of editing it"
            )
    return declaration


def expected_motion_basis(motion_donor_tag: str) -> dict:
    if motion_donor_tag not in MOTION_DONOR_BASIS:
        raise ForwardContractError(
            f"unknown motion donor tag {motion_donor_tag!r}; declare it in "
            "MOTION_DONOR_BASIS before use"
        )
    return dict(MOTION_DONOR_BASIS[motion_donor_tag])


def assert_declared_motion_donor_artifact(
    motion_donor_tag: str,
    source_motion_glb: Path,
) -> dict:
    expected = MOTION_DONOR_ARTIFACTS.get(motion_donor_tag)
    if expected is None:
        raise ForwardContractError(
            f"motion donor {motion_donor_tag!r} has no authenticated artifact"
        )
    source_motion_glb = Path(source_motion_glb).resolve()
    if (
        source_motion_glb.is_symlink()
        or not source_motion_glb.is_file()
        or source_motion_glb.stat().st_size <= 0
    ):
        raise ForwardContractError(
            f"missing or unsafe motion donor artifact: {source_motion_glb}"
        )
    if (
        source_motion_glb.stat().st_size != expected["size_bytes"]
        or sha256_file(source_motion_glb) != expected["sha256"]
    ):
        raise ForwardContractError(
            "motion donor artifact does not match the declared donor tag "
            f"{motion_donor_tag!r}: {source_motion_glb}"
        )
    animation_names = _assert_motion_donor_action_semantics(
        motion_donor_tag, source_motion_glb
    )
    return {
        "path": str(source_motion_glb),
        "sha256": expected["sha256"],
        "size_bytes": expected["size_bytes"],
        "animation_names": animation_names,
        "authority_contract": expected_motion_donor_contract(motion_donor_tag),
    }


def assert_declared_motion_basis(
    motion_donor_tag: str,
    motion_basis_yaw_deg: int,
    side_chain_mode: str,
) -> None:
    expected = expected_motion_basis(motion_donor_tag)
    if (
        motion_basis_yaw_deg != expected["motion_basis_yaw_deg"]
        or side_chain_mode != expected["side_chain_mode"]
    ):
        raise ForwardContractError(
            "motion basis deviates from the declared donor constant "
            f"(expected yaw {expected['motion_basis_yaw_deg']} / "
            f"{expected['side_chain_mode']}, got yaw {motion_basis_yaw_deg} / "
            f"{side_chain_mode}).  A deviating basis means the forward "
            "declaration or heading canonicalization is wrong; fix the "
            "declaration, do not compensate at retarget."
        )
