#!/usr/bin/env python3
"""Run the hash-locked reviewed female Pixal PBR GLB through TokenRig."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools import route2_human_contract_common as route2_common
from tools import route2_human_qualified_candidate as qualified_candidate
from tools import tokenrig_human_canary as base


RUNNER_PATH = Path(__file__).resolve()
ASSET_ID = "rocketbox_female_adult_01"
_SPEAR_ROOT = RUNNER_PATH.parents[1]
_INPUT_ROOT = _SPEAR_ROOT / "tmp/i23d_human_bakeoff_v1/pixal3d" / ASSET_ID
_OUTPUT_DIR = _SPEAR_ROOT / "tmp/pixal_tokenrig_route2_v1" / ASSET_ID
CANONICAL_MALE_QUALIFIED_CANDIDATE = (
    _SPEAR_ROOT
    / "tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01"
    / qualified_candidate.FILENAME
)
BASE_RUNNER_SHA256 = "ab9b56019acc8491777112ee3979d6d4b4a581f3cb339b0bfba0866c87b8f9b9"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FemaleGateError(RuntimeError):
    """Raised when the exact male Route-2 gate has not authorized female work."""

PINNED_FEMALE_CONTRACT = replace(
    base.PINNED_CONTRACT,
    asset_id=ASSET_ID,
    input_glb=_INPUT_ROOT / "canary_1024_seed42.glb",
    input_manifest=_INPUT_ROOT / "canary_1024_seed42.manifest.json",
    output_dir=_OUTPUT_DIR,
    input_glb_sha256=(
        "894e7f88d96d59510837bd4550a136a53fdd32e421910281351fdb20aedbb746"
    ),
    input_manifest_sha256=(
        "bbcffc16a63ee2a2cb0f7bf063a620e40aff4dcf98103da4f19bc7eea82a954b"
    ),
)


def authenticate_male_gate(pointer: Path) -> dict[str, Any]:
    pointer = Path(pointer).absolute()
    if pointer != CANONICAL_MALE_QUALIFIED_CANDIDATE:
        raise FemaleGateError("male qualified candidate path is not canonical")
    try:
        record = route2_common.file_record(
            pointer,
            root=pointer.parent,
            description="male qualified candidate",
            error_type=FemaleGateError,
            require_mode=0o444,
        )
        qualified = qualified_candidate.validate_qualified_candidate(pointer)
    except (FemaleGateError, qualified_candidate.QualificationError) as error:
        raise FemaleGateError(f"male final branch is not qualified: {error}") from error
    final_branch = qualified.get("final_branch")
    dynamic = qualified.get("dynamic")
    if (
        qualified.get("asset_id") != "rocketbox_male_adult_01"
        or qualified.get("base_avatar_id") != "rocketbox_male_adult_01"
        or qualified.get("status")
        != "agent_qa_passed_pending_user_acceptance"
        or not isinstance(final_branch, Mapping)
        or set(final_branch) != {"branch_id", "path", "relative_root"}
        or not isinstance(dynamic, Mapping)
        or not isinstance(dynamic.get("review_dir"), str)
        or not isinstance(qualified.get("inventory_sha256"), str)
        or _SHA256.fullmatch(qualified["inventory_sha256"]) is None
    ):
        raise FemaleGateError("male qualified candidate lineage is invalid")
    return {
        "schema": "route2_male_qualified_gate_snapshot_v2",
        "asset_id": "rocketbox_male_adult_01",
        "status": qualified["status"],
        "qualified_candidate": {
            "path": str(pointer),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        },
        "final_branch": dict(final_branch),
        "review_dir": dynamic["review_dir"],
        "inventory_sha256": qualified["inventory_sha256"],
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise FemaleGateError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def publish_female_gate_record(*, gate: Mapping[str, Any]) -> Path:
    parent = _OUTPUT_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve() != parent:
        raise FemaleGateError("female route output parent must be a direct real directory")
    base_path = base.RUNNER_PATH.absolute()
    if base.sha256_file(base_path) != BASE_RUNNER_SHA256:
        raise FemaleGateError("base TokenRig runner hash changed")
    payload = {
        "schema": "tokenrig_female_preflight_gate_v1",
        "asset_id": ASSET_ID,
        "male_gate": dict(gate),
        "female_wrapper": {
            "path": str(RUNNER_PATH),
            "sha256": base.sha256_file(RUNNER_PATH),
            "size_bytes": RUNNER_PATH.stat().st_size,
        },
        "base_runner": {
            "path": str(base_path),
            "sha256": BASE_RUNNER_SHA256,
            "size_bytes": base_path.stat().st_size,
        },
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise FemaleGateError("female gate may not claim user approval")
    qualified_hash = gate.get("qualified_candidate", {}).get("sha256")
    if not isinstance(qualified_hash, str) or _SHA256.fullmatch(qualified_hash) is None:
        raise FemaleGateError("female gate has no qualified candidate hash")
    destination = parent / f"{ASSET_ID}.female_gate.{qualified_hash[:16]}.json"
    return route2_common.write_json_immutable_noreplace(
        destination,
        payload,
        FemaleGateError,
        "female qualified preflight gate",
    )


def publish_female_authorization_manifest(
    *,
    gate: Mapping[str, Any],
    gate_record: Path,
    tokenrig_manifest: Path,
) -> Path:
    output_dir = route2_common.require_real_directory(
        _OUTPUT_DIR, "female TokenRig output", FemaleGateError
    )
    gate_record = Path(gate_record).absolute()
    tokenrig_manifest = Path(tokenrig_manifest).absolute()
    gate_descriptor = route2_common.file_record(
        gate_record,
        root=gate_record.parent,
        description="female gate record",
        error_type=FemaleGateError,
        require_mode=0o444,
    )
    tokenrig_descriptor = route2_common.file_record(
        tokenrig_manifest,
        root=output_dir,
        description="female TokenRig manifest",
        error_type=FemaleGateError,
        require_mode=0o444,
    )
    if tokenrig_manifest != output_dir / "tokenrig_manifest.json":
        raise FemaleGateError("female TokenRig manifest path is not canonical")
    wrapper = route2_common.file_record(
        RUNNER_PATH,
        root=RUNNER_PATH.parent,
        description="female wrapper",
        error_type=FemaleGateError,
    )
    base_runner = route2_common.file_record(
        base.RUNNER_PATH,
        root=base.RUNNER_PATH.parent,
        description="base TokenRig runner",
        error_type=FemaleGateError,
    )
    if base_runner["sha256"] != BASE_RUNNER_SHA256:
        raise FemaleGateError("base TokenRig runner hash changed")
    payload = {
        "schema": "tokenrig_human_female_authorization_v2",
        "asset_id": ASSET_ID,
        "state_classification": "research_candidate",
        "male_gate": dict(gate),
        "female_gate_record": {
            "path": gate_descriptor["path"],
            "sha256": gate_descriptor["sha256"],
            "size_bytes": gate_descriptor["size_bytes"],
        },
        "tokenrig_manifest": {
            "path": tokenrig_descriptor["path"],
            "sha256": tokenrig_descriptor["sha256"],
            "size_bytes": tokenrig_descriptor["size_bytes"],
        },
        "female_wrapper": {
            "path": wrapper["path"],
            "sha256": wrapper["sha256"],
            "size_bytes": wrapper["size_bytes"],
        },
        "base_runner": {
            "path": base_runner["path"],
            "sha256": base_runner["sha256"],
            "size_bytes": base_runner["size_bytes"],
        },
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise FemaleGateError("female authorization may not claim user approval")
    return route2_common.write_json_immutable_noreplace(
        output_dir / "tokenrig_female_authorization_v2.json",
        payload,
        FemaleGateError,
        "female TokenRig authorization manifest",
    )


def _validated_record(
    value: Any,
    *,
    expected_path: Path,
    root: Path,
    description: str,
    require_mode: int | None = None,
) -> dict[str, Any]:
    record = route2_common.file_record(
        expected_path,
        root=root,
        description=description,
        error_type=FemaleGateError,
        require_mode=require_mode,
    )
    expected = {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FemaleGateError(f"{description} descriptor changed")
    return record


def validate_female_authorization_manifest(
    path: Path,
    *,
    expected_tokenrig_manifest: Path,
    expected_asset_id: str = ASSET_ID,
) -> dict[str, Any]:
    """Owner-validate the recursively bound female gate and TokenRig producer."""
    path = Path(path).absolute()
    output_dir = route2_common.require_real_directory(
        path.parent, "female authorization root", FemaleGateError
    )
    if (
        output_dir != Path(_OUTPUT_DIR).absolute()
        or path.name != "tokenrig_female_authorization_v2.json"
        or expected_asset_id != ASSET_ID
    ):
        raise FemaleGateError("female authorization path or asset is not canonical")
    payload, authorization_record = route2_common.load_json_mapping_record(
        path,
        root=output_dir,
        description="female authorization manifest",
        error_type=FemaleGateError,
        require_mode=0o444,
    )
    expected_fields = {
        "schema",
        "asset_id",
        "state_classification",
        "male_gate",
        "female_gate_record",
        "tokenrig_manifest",
        "female_wrapper",
        "base_runner",
        "user_acceptance",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != "tokenrig_human_female_authorization_v2"
        or payload.get("asset_id") != expected_asset_id
        or payload.get("state_classification") != "research_candidate"
        or payload.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(payload)
    ):
        raise FemaleGateError("female authorization schema or state is invalid")
    tokenrig_manifest = Path(expected_tokenrig_manifest).absolute()
    if tokenrig_manifest != output_dir / "tokenrig_manifest.json":
        raise FemaleGateError("expected female TokenRig manifest path is not canonical")
    tokenrig_payload, tokenrig_record = route2_common.load_json_mapping_record(
        tokenrig_manifest,
        root=output_dir,
        description="female TokenRig manifest",
        error_type=FemaleGateError,
        require_mode=0o444,
    )
    expected_tokenrig_descriptor = {
        "path": tokenrig_record["path"],
        "sha256": tokenrig_record["sha256"],
        "size_bytes": tokenrig_record["size_bytes"],
    }
    if payload.get("tokenrig_manifest") != expected_tokenrig_descriptor:
        raise FemaleGateError("female TokenRig manifest descriptor changed")
    if (
        not isinstance(tokenrig_payload, Mapping)
        or tokenrig_payload.get("asset_id") != expected_asset_id
        or "user_approved" in json.dumps(tokenrig_payload)
    ):
        raise FemaleGateError("female TokenRig manifest asset identity is invalid")

    gate = payload.get("male_gate")
    if not isinstance(gate, Mapping):
        raise FemaleGateError("female authorization male gate is missing")
    pointer_value = gate.get("qualified_candidate")
    if not isinstance(pointer_value, Mapping) or not isinstance(
        pointer_value.get("path"), str
    ):
        raise FemaleGateError("female authorization qualified pointer is missing")
    pointer = Path(pointer_value["path"]).absolute()
    if authenticate_male_gate(pointer) != dict(gate):
        raise FemaleGateError("female authorization male final branch changed")
    pointer_record = _validated_record(
        pointer_value,
        expected_path=pointer,
        root=pointer.parent,
        description="male qualified candidate",
        require_mode=0o444,
    )

    gate_hash = pointer_record["sha256"]
    gate_path = output_dir.parent / f"{ASSET_ID}.female_gate.{gate_hash[:16]}.json"
    gate_payload, gate_record = route2_common.load_json_mapping_record(
        gate_path,
        root=gate_path.parent,
        description="female gate record",
        error_type=FemaleGateError,
        require_mode=0o444,
    )
    expected_gate_descriptor = {
        "path": gate_record["path"],
        "sha256": gate_record["sha256"],
        "size_bytes": gate_record["size_bytes"],
    }
    if payload.get("female_gate_record") != expected_gate_descriptor:
        raise FemaleGateError("female gate record descriptor changed")
    if (
        gate_payload.get("schema") != "tokenrig_female_preflight_gate_v1"
        or gate_payload.get("asset_id") != expected_asset_id
        or gate_payload.get("male_gate") != dict(gate)
        or gate_payload.get("user_acceptance") != "pending_user_review"
    ):
        raise FemaleGateError("female gate record owner snapshot changed")

    wrapper_record = _validated_record(
        payload.get("female_wrapper"),
        expected_path=RUNNER_PATH,
        root=RUNNER_PATH.parent,
        description="female wrapper",
    )
    base_record = _validated_record(
        payload.get("base_runner"),
        expected_path=base.RUNNER_PATH,
        root=base.RUNNER_PATH.parent,
        description="base TokenRig runner",
    )
    if base_record["sha256"] != BASE_RUNNER_SHA256:
        raise FemaleGateError("base TokenRig runner hash changed")
    if (
        gate_payload.get("female_wrapper") != payload.get("female_wrapper")
        or gate_payload.get("base_runner") != payload.get("base_runner")
    ):
        raise FemaleGateError("female gate producer binding changed")
    return {
        "payload": payload,
        "records": {
            "authorization": authorization_record,
            "female_gate_record": gate_record,
            "male_qualified_candidate": pointer_record,
            "tokenrig_manifest": tokenrig_record,
            "female_wrapper": wrapper_record,
            "base_runner": base_record,
        },
    }


def seal_female_success_artifacts(tokenrig_manifest: Path) -> dict[str, dict[str, Any]]:
    """Durably seal the female producer files while leaving branch dirs writable."""
    output_dir = route2_common.require_real_directory(
        _OUTPUT_DIR, "female TokenRig output", FemaleGateError
    )
    tokenrig_manifest = Path(tokenrig_manifest).absolute()
    canonical_manifest = output_dir / "tokenrig_manifest.json"
    if tokenrig_manifest != canonical_manifest:
        raise FemaleGateError("female TokenRig manifest path is not canonical")

    required = {
        canonical_manifest,
        output_dir / "tokenrig_transfer.glb",
        output_dir.with_name(f"{ASSET_ID}.tokenrig_attempt.json"),
    }
    discovered: set[Path] = set()
    directories: set[Path] = {output_dir, output_dir.parent}
    try:
        for root, directory_names, file_names in os.walk(
            output_dir, topdown=True, followlinks=False
        ):
            root_path = Path(root)
            directories.add(root_path)
            for directory_name in directory_names:
                directory = root_path / directory_name
                if directory.is_symlink():
                    raise FemaleGateError(
                        f"female producer contains a symlink directory: {directory}"
                    )
                route2_common.require_real_directory(
                    directory, "female producer directory", FemaleGateError
                )
                directories.add(directory)
            for file_name in file_names:
                discovered.add(root_path / file_name)
    except OSError as error:
        raise FemaleGateError(f"female producer inventory is unreadable: {error}") from error
    discovered.add(output_dir.with_name(f"{ASSET_ID}.tokenrig_attempt.json"))
    missing = sorted(str(path) for path in required - discovered if not path.exists())
    if missing:
        raise FemaleGateError(
            "female producer is missing required success artifacts: " + ", ".join(missing)
        )

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(discovered, key=lambda item: str(item)):
        root = output_dir if path.is_relative_to(output_dir) else output_dir.parent
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as error:
            raise FemaleGateError(
                f"female producer file could not be opened directly: {path}: {error}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise FemaleGateError(
                    f"female producer artifact must be a non-empty regular file: {path}"
                )
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or stat.S_IMODE(after.st_mode) != 0o444
            ):
                raise FemaleGateError(
                    f"female producer artifact changed while sealing: {path}"
                )
        finally:
            os.close(descriptor)
        records[str(path)] = route2_common.file_record(
            path,
            root=root,
            description="sealed female producer artifact",
            error_type=FemaleGateError,
            require_mode=0o444,
        )

    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        route2_common.fsync_directory(directory)
    return records


def run_female_canary(*, male_qualified_candidate: Path) -> Path:
    """Execute direct female TokenRig only after the exact male final branch."""
    gate = authenticate_male_gate(male_qualified_candidate)
    gate_record = publish_female_gate_record(gate=gate)
    contract = PINNED_FEMALE_CONTRACT
    tokenrig_manifest = base.run_canary(
        input_glb=contract.input_glb,
        input_manifest=contract.input_manifest,
        output_dir=contract.output_dir,
        skintokens_root=contract.skintokens_root,
        model_revision=contract.model_revision,
        seed=42,
        use_skeleton_input=False,
        contract=contract,
        orchestrator_path=RUNNER_PATH,
    )
    seal_female_success_artifacts(tokenrig_manifest)
    if authenticate_male_gate(male_qualified_candidate) != gate:
        raise FemaleGateError("male qualified candidate changed during female inference")
    authorization = publish_female_authorization_manifest(
        gate=gate,
        gate_record=gate_record,
        tokenrig_manifest=tokenrig_manifest,
    )
    validate_female_authorization_manifest(
        authorization,
        expected_tokenrig_manifest=tokenrig_manifest,
    )
    if authenticate_male_gate(male_qualified_candidate) != gate:
        raise FemaleGateError(
            "male qualified candidate changed after female authorization publication"
        )
    return authorization


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male-qualified-candidate", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_female_canary(
        male_qualified_candidate=args.male_qualified_candidate,
    )
    print(f"TOKENRIG_FEMALE_CANARY_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
