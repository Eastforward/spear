"""Artifact-locked gate for stable-template humanoids used in UE smoke tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STABLE_ROOT = REPO_ROOT / "tmp" / "hy3d_rocketbox_template_fit_v1"
DEFAULT_FORMAL_REGISTRY_ROOT = REPO_ROOT / "data" / "source_assets_v1"
DEFAULT_NATIVE_ROCKETBOX_RUNTIME_ROOT = (
    REPO_ROOT / "tmp" / "rocketbox_native_runtime_ue_v3"
)
DEFAULT_NATIVE_ROCKETBOX_IMPORT_ROOT = (
    REPO_ROOT / "tmp" / "rocketbox_native_ue_import_v3"
)
DEFAULT_BATCH_NATIVE_ROCKETBOX_RUNTIME_ROOT = (
    REPO_ROOT / "tmp" / "rocketbox_batch_native_runtime_ue_v1"
)
DEFAULT_BATCH_NATIVE_ROCKETBOX_IMPORT_ROOT = (
    REPO_ROOT / "tmp" / "rocketbox_batch_native_ue_import_v1"
)
DEFAULT_ROCKETBOX_INVENTORY = (
    REPO_ROOT / "tmp" / "rocketbox_route1_inventory_v1" / "inventory.json"
)
ALLOWED_HUMAN_SPIKES = {
    "hy3d_rocketbox_male_adult_01_spike": "rocketbox_male_adult_01",
    "hy3d_rocketbox_female_adult_01_spike": "rocketbox_female_adult_01",
}
NATIVE_ROCKETBOX_HUMAN_CANDIDATES = {
    "rocketbox_male_adult_01_original_ue_v3": "rocketbox_male_adult_01",
    "rocketbox_male_adult_01_shirt_blue_ue_v3": "rocketbox_male_adult_01",
}
_BATCH_NATIVE_ROCKETBOX_TAG = re.compile(
    r"rocketbox_(?:adults|children|professions)_[a-z0-9_]+_original_ue_v1"
)


class HumanApartmentGateError(RuntimeError):
    """Raised when a humanoid is not eligible for apartment smoke testing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise HumanApartmentGateError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanApartmentGateError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HumanApartmentGateError(f"{label} must be a JSON object: {path}")
    return payload


def _verify_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise HumanApartmentGateError(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise HumanApartmentGateError(
            f"{label} hash mismatch: expected {expected}, got {actual}: {path}"
        )
    return actual


def _verify_record(asset_dir: Path, record: dict, label: str) -> Path:
    if not isinstance(record, dict):
        raise HumanApartmentGateError(f"malformed {label} artifact record")
    filename = record.get("filename")
    expected = record.get("sha256")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise HumanApartmentGateError(f"unsafe filename in {label} artifact record")
    if not isinstance(expected, str) or len(expected) != 64:
        raise HumanApartmentGateError(f"missing sha256 in {label} artifact record")
    path = asset_dir / filename
    _verify_file_hash(path, expected, label)
    return path


def _walk_records(records, prefix="artifact"):
    if not isinstance(records, dict):
        raise HumanApartmentGateError(f"malformed {prefix} artifact tree")
    for key, value in records.items():
        label = f"{prefix}.{key}"
        if isinstance(value, dict) and "filename" in value:
            yield label, value
        elif isinstance(value, dict):
            yield from _walk_records(value, label)
        else:
            raise HumanApartmentGateError(f"malformed {label} artifact record")


def _registry_contains_tag(registry_root: Path, tag: str) -> bool:
    if not registry_root.exists():
        return False
    for path in registry_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if item.get("legacy_tag") == tag or item.get("tag") == tag:
                    return True
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return False


def is_batch_native_rocketbox_human_candidate(tag: object) -> bool:
    """Recognize batch tag syntax; the full gate still locks it to inventory."""
    return isinstance(tag, str) and _BATCH_NATIVE_ROCKETBOX_TAG.fullmatch(tag) is not None


def _single_inventory_avatar(inventory_path: Path, avatar_id: str) -> tuple[dict, dict]:
    inventory_path = Path(inventory_path).resolve()
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise HumanApartmentGateError(
            f"missing direct Rocketbox inventory: {inventory_path}"
        )
    inventory = _load_json(inventory_path, "Rocketbox inventory")
    avatars = inventory.get("avatars")
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(avatars, list)
        or inventory.get("population", {}).get("total") != len(avatars)
    ):
        raise HumanApartmentGateError("Rocketbox inventory is not Apartment-ready")
    matches = [item for item in avatars if item.get("base_avatar_id") == avatar_id]
    if len(matches) != 1 or matches[0].get("inventory_status") != "passed":
        raise HumanApartmentGateError(
            f"batch Rocketbox tag is not uniquely approved by inventory: {avatar_id}"
        )
    return inventory, matches[0]


def _same_float(left: object, right: object, tolerance: float = 1.0e-4) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_float)
        and math.isfinite(right_float)
        and abs(left_float - right_float) <= tolerance
    )


def assert_batch_native_rocketbox_apartment_ready(
    tag: str,
    *,
    runtime_root: Path | None = None,
    ue_import_root: Path | None = None,
    inventory_path: Path | None = None,
    formal_registry_root: Path | None = None,
) -> dict:
    """Authenticate one of the 115 inventory-locked native Rocketbox humans."""
    if not is_batch_native_rocketbox_human_candidate(tag):
        raise HumanApartmentGateError(
            f"unsupported batch native Rocketbox human tag: {tag!r}"
        )
    avatar_id = tag.removesuffix("_original_ue_v1")
    runtime_root = Path(
        runtime_root or DEFAULT_BATCH_NATIVE_ROCKETBOX_RUNTIME_ROOT
    ).resolve()
    ue_import_root = Path(
        ue_import_root or DEFAULT_BATCH_NATIVE_ROCKETBOX_IMPORT_ROOT
    ).resolve()
    inventory_path = Path(inventory_path or DEFAULT_ROCKETBOX_INVENTORY).resolve()
    formal_registry_root = Path(
        formal_registry_root or DEFAULT_FORMAL_REGISTRY_ROOT
    ).resolve()
    if runtime_root.name != "rocketbox_batch_native_runtime_ue_v1":
        raise HumanApartmentGateError(
            f"batch runtime escaped its versioned root: {runtime_root}"
        )
    if ue_import_root.name != "rocketbox_batch_native_ue_import_v1":
        raise HumanApartmentGateError(
            f"batch UE import escaped its versioned root: {ue_import_root}"
        )
    if _registry_contains_tag(formal_registry_root, tag):
        raise HumanApartmentGateError(
            f"research-candidate batch human appears in formal registry: {tag}"
        )

    inventory, avatar = _single_inventory_avatar(inventory_path, avatar_id)
    asset_id = avatar.get("legacy_asset_id")
    category = avatar.get("category")
    demographic = avatar.get("demographic")
    gender = avatar.get("gender")
    height_contract = avatar.get("height_contract", {})
    height_policy = inventory.get("apartment_height_policy", {})
    expected_category_prefix = f"rocketbox_{str(category).lower()}_"
    if (
        not isinstance(asset_id, str)
        or not avatar_id.startswith(expected_category_prefix)
        or demographic not in {"adult", "child"}
        or gender not in {"male", "female"}
        or height_contract.get("status") != "passed"
        or height_contract.get("actor_scale") != 1.0
        or height_policy.get("actor_scale") != 1.0
        or height_policy.get("authored_height_preserved") is not True
    ):
        raise HumanApartmentGateError(
            f"batch Rocketbox inventory identity/height contract failed: {avatar_id}"
        )

    runtime_dir = (runtime_root / tag).resolve()
    import_dir = (ue_import_root / tag).resolve()
    if runtime_dir.parent != runtime_root or import_dir.parent != ue_import_root:
        raise HumanApartmentGateError("batch Rocketbox path escaped its root")
    runtime_glb = runtime_dir / "runtime.glb"
    source_manifest_path = runtime_dir / "normalization_manifest.json"
    source_manifest = _load_json(
        source_manifest_path, "batch native Rocketbox source manifest"
    )
    if (
        source_manifest.get("schema") != "rocketbox_batch_native_ue_runtime_v1"
        or source_manifest.get("tag") != tag
        or source_manifest.get("base_avatar_id") != avatar_id
        or source_manifest.get("asset_id") != asset_id
        or source_manifest.get("usage_scope") != "research_candidate"
        or source_manifest.get("formal_registration_authorized") is not False
        or source_manifest.get("demographic") != demographic
        or source_manifest.get("gender") != gender
        or source_manifest.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise HumanApartmentGateError(
            f"batch native Rocketbox source identity/checks changed: {avatar_id}"
        )
    runtime_record = source_manifest.get("runtime_glb", {})
    if (
        runtime_glb.is_symlink()
        or not runtime_glb.is_file()
        or runtime_record.get("filename") != "runtime.glb"
        or runtime_record.get("size_bytes") != runtime_glb.stat().st_size
    ):
        raise HumanApartmentGateError(
            f"batch native Rocketbox runtime file record changed: {avatar_id}"
        )
    runtime_sha256 = _verify_file_hash(
        runtime_glb,
        runtime_record.get("sha256", ""),
        "batch native runtime GLB",
    )

    normalization = source_manifest.get("normalization", {})
    walking = normalization.get("root_motion", {}).get("Walking", {})
    runtime_motion = source_manifest.get("runtime_motion_contract", {})
    expected_qa = source_manifest.get("expected_ue_qa", {})
    allowed_height = height_contract.get("allowed_height_cm")
    authored_height_cm = height_contract.get("authored_height_cm")
    ceiling_cm = height_policy.get("ceiling_cm")
    minimum_headroom_cm = height_policy.get("minimum_headroom_cm")
    mouth_audio_height_cm = height_contract.get("mouth_audio_height_cm")
    if (
        normalization.get("schema")
        != "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        or normalization.get("normalized_joint_count") != 80
        or normalization.get("in_place_actions") != ["Walking"]
        or float(walking.get("maximum_horizontal_deviation_after_m", 1.0))
        >= 1.0e-6
        or float(walking.get("maximum_vertical_world_error_m", 1.0)) >= 1.0e-6
        or runtime_motion.get("walking_embedded_horizontal_root_motion") != "removed"
        or runtime_motion.get("walking_vertical_motion") != "preserved"
        or runtime_motion.get("dynamic_ground_snap_to_floor_required") is not True
        or expected_qa.get("actor_scale") != 1.0
        or expected_qa.get("authored_height_preserved") is not True
        or expected_qa.get("ground_snap_to_floor") is not True
        or expected_qa.get("ground_snap_max_abs_correction_cm") != 15.0
        or expected_qa.get("height_range_cm") != allowed_height
        or expected_qa.get("demographic") != demographic
        or not _same_float(expected_qa.get("authored_height_cm"), authored_height_cm)
        or not _same_float(expected_qa.get("apartment_ceiling_cm"), ceiling_cm)
        or not _same_float(expected_qa.get("mouth_audio_height_cm"), mouth_audio_height_cm)
    ):
        raise HumanApartmentGateError(
            f"batch native Rocketbox motion/height normalization changed: {avatar_id}"
        )

    ue_manifest_path = import_dir / "ue_import_manifest.json"
    ue_manifest = _load_json(
        ue_manifest_path, "batch native Rocketbox UE import manifest"
    )
    if (
        ue_manifest.get("schema") != "rocketbox_batch_native_ue_import_v1"
        or ue_manifest.get("tag") != tag
        or ue_manifest.get("base_avatar_id") != avatar_id
        or ue_manifest.get("asset_id") != asset_id
        or ue_manifest.get("usage_scope") != "research_candidate"
        or ue_manifest.get("formal_registration_authorized") is not False
        or ue_manifest.get("reload_verification", {}).get("status") != "passed"
        or Path(ue_manifest.get("source_glb", "")).resolve() != runtime_glb.resolve()
        or ue_manifest.get("source_glb_sha256") != runtime_sha256
        or Path(ue_manifest.get("source_manifest", "")).resolve()
        != source_manifest_path.resolve()
        or ue_manifest.get("source_manifest_sha256") != _sha256(source_manifest_path)
    ):
        raise HumanApartmentGateError(
            f"batch native Rocketbox UE identity/hash/reload changed: {avatar_id}"
        )
    runtime_contract = ue_manifest.get("runtime_contract", {})
    bounds = runtime_contract.get("bounds", {})
    height_cm = float(bounds.get("height_cm", 0.0))
    authored_height_delta_cm = abs(height_cm - float(authored_height_cm))
    authored_height_tolerance_cm = float(
        bounds.get("authored_height_tolerance_cm", 0.0)
    )
    if (
        runtime_contract.get("bone_count") != 80
        or runtime_contract.get("actor_scale") != 1.0
        or not isinstance(allowed_height, list)
        or len(allowed_height) != 2
        or not float(allowed_height[0]) <= height_cm <= float(allowed_height[1])
        or bounds.get("height_passed") is not True
        or bounds.get("authored_height_preserved") is not True
        or bounds.get("ground_passed") is not True
        or not _same_float(bounds.get("authored_height_cm"), authored_height_cm)
        or not _same_float(
            bounds.get("authored_height_delta_cm"), authored_height_delta_cm
        )
        or authored_height_delta_cm > authored_height_tolerance_cm
    ):
        raise HumanApartmentGateError(
            f"batch native Rocketbox authored height/runtime failed: {avatar_id}"
        )

    ceiling_cm = float(ceiling_cm)
    minimum_headroom_cm = float(minimum_headroom_cm)
    ceiling_headroom_cm = ceiling_cm - height_cm
    if (
        not math.isfinite(ceiling_headroom_cm)
        or ceiling_headroom_cm < minimum_headroom_cm
    ):
        raise HumanApartmentGateError(
            f"Apartment headroom failed for {avatar_id}: "
            f"{ceiling_headroom_cm:.3f} cm < {minimum_headroom_cm:.3f} cm"
        )
    animations = ue_manifest.get("content", {}).get("animations", {})
    if set(animations) != {"Walking", "Standing_Idle"}:
        raise HumanApartmentGateError(
            f"batch native Rocketbox lacks exact Walk/Idle actions: {avatar_id}"
        )
    blueprint = ue_manifest.get("content", {}).get("blueprint")
    if not isinstance(blueprint, str) or not blueprint:
        raise HumanApartmentGateError(
            f"batch native Rocketbox Blueprint is missing: {avatar_id}"
        )

    return {
        "tag": tag,
        "base_avatar_id": avatar_id,
        "asset_id": asset_id,
        "category": category,
        "demographic": demographic,
        "gender": gender,
        "usage_scope": "research_candidate",
        "runtime_glb_sha256": runtime_sha256,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "ue_import_manifest_sha256": _sha256(ue_manifest_path),
        "ue_import_manifest_path": str(ue_manifest_path.resolve()),
        "inventory_path": str(inventory_path),
        "bone_count": 80,
        "actor_scale": 1.0,
        "authored_height_cm": float(authored_height_cm),
        "height_cm": height_cm,
        "height_range_cm": list(allowed_height),
        "apartment_ceiling_cm": ceiling_cm,
        "minimum_headroom_cm": minimum_headroom_cm,
        "ceiling_headroom_cm": ceiling_headroom_cm,
        "audio_source_height_m": float(mouth_audio_height_cm) / 100.0,
        "walking_in_place": True,
        "dynamic_ground_snap_to_floor_required": True,
        "animations": animations,
        "blueprint": blueprint,
    }


def assert_native_rocketbox_apartment_ready(
    tag: str,
    *,
    runtime_root: Path | None = None,
    ue_import_root: Path | None = None,
    formal_registry_root: Path | None = None,
) -> dict:
    """Authenticate one in-place native Rocketbox UE runtime for rendering."""
    if tag not in NATIVE_ROCKETBOX_HUMAN_CANDIDATES:
        raise HumanApartmentGateError(
            f"unsupported native Rocketbox human tag: {tag!r}"
        )
    runtime_root = Path(
        runtime_root or DEFAULT_NATIVE_ROCKETBOX_RUNTIME_ROOT
    ).resolve()
    ue_import_root = Path(
        ue_import_root or DEFAULT_NATIVE_ROCKETBOX_IMPORT_ROOT
    ).resolve()
    formal_registry_root = Path(
        formal_registry_root or DEFAULT_FORMAL_REGISTRY_ROOT
    ).resolve()
    if runtime_root.name != "rocketbox_native_runtime_ue_v3":
        raise HumanApartmentGateError(
            f"native Rocketbox runtime escaped the v3 root: {runtime_root}"
        )
    if ue_import_root.name != "rocketbox_native_ue_import_v3":
        raise HumanApartmentGateError(
            f"native Rocketbox import escaped the v3 root: {ue_import_root}"
        )
    if _registry_contains_tag(formal_registry_root, tag):
        raise HumanApartmentGateError(
            f"research-candidate native human appears in formal registry: {tag}"
        )

    asset_id = NATIVE_ROCKETBOX_HUMAN_CANDIDATES[tag]
    runtime_dir = (runtime_root / tag).resolve()
    import_dir = (ue_import_root / tag).resolve()
    if runtime_dir.parent != runtime_root or import_dir.parent != ue_import_root:
        raise HumanApartmentGateError("native Rocketbox path escaped its root")
    runtime_glb = runtime_dir / "runtime.glb"
    source_manifest_path = runtime_dir / "normalization_manifest.json"
    source_manifest = _load_json(
        source_manifest_path, "native Rocketbox v3 source manifest"
    )
    if (
        source_manifest.get("schema") != "rocketbox_native_ue_runtime_v3"
        or source_manifest.get("tag") != tag
        or source_manifest.get("asset_id") != asset_id
        or source_manifest.get("usage_scope") != "research_candidate"
        or source_manifest.get("formal_registration_authorized") is not False
        or source_manifest.get("automatic_checks", {}).get("overall")
        != "passed"
    ):
        raise HumanApartmentGateError(
            "native Rocketbox v3 source identity/scope/checks changed"
        )
    runtime_record = source_manifest.get("runtime_glb", {})
    if (
        runtime_record.get("filename") != "runtime.glb"
        or runtime_record.get("size_bytes") != runtime_glb.stat().st_size
    ):
        raise HumanApartmentGateError(
            "native Rocketbox runtime GLB hash/size record changed"
        )
    runtime_sha256 = _verify_file_hash(
        runtime_glb,
        runtime_record.get("sha256", ""),
        "native runtime GLB",
    )

    normalization = source_manifest.get("normalization", {})
    walking = normalization.get("root_motion", {}).get("Walking", {})
    runtime_motion = source_manifest.get("runtime_motion_contract", {})
    expected_qa = source_manifest.get("expected_ue_qa", {})
    height_range = expected_qa.get("height_range_cm")
    if (
        normalization.get("schema")
        != "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        or normalization.get("normalized_joint_count") != 80
        or normalization.get("in_place_actions") != ["Walking"]
        or float(walking.get("maximum_horizontal_deviation_after_m", 1.0))
        >= 1.0e-6
        or float(walking.get("maximum_vertical_world_error_m", 1.0))
        >= 1.0e-6
        or runtime_motion.get("walking_embedded_horizontal_root_motion")
        != "removed"
        or runtime_motion.get("walking_vertical_motion") != "preserved"
        or runtime_motion.get("dynamic_ground_snap_to_floor_required") is not True
        or expected_qa.get("actor_scale") != 1.0
        or expected_qa.get("ground_snap_to_floor") is not True
        or expected_qa.get("ground_snap_max_abs_correction_cm") != 15.0
        or height_range != [165.0, 200.0]
    ):
        raise HumanApartmentGateError(
            "native Rocketbox in-place/height/grounding contract changed"
        )

    ue_manifest_path = import_dir / "ue_import_manifest.json"
    ue_manifest = _load_json(
        ue_manifest_path, "native Rocketbox v3 UE import manifest"
    )
    if (
        ue_manifest.get("schema") != "rocketbox_native_ue_import_v3"
        or ue_manifest.get("tag") != tag
        or ue_manifest.get("asset_id") != asset_id
        or ue_manifest.get("usage_scope") != "research_candidate"
        or ue_manifest.get("formal_registration_authorized") is not False
        or ue_manifest.get("reload_verification", {}).get("status") != "passed"
    ):
        raise HumanApartmentGateError(
            "native Rocketbox UE import identity/scope/reload changed"
        )
    if Path(ue_manifest.get("source_glb", "")).resolve() != runtime_glb.resolve():
        raise HumanApartmentGateError("native Rocketbox UE source GLB path changed")
    if ue_manifest.get("source_glb_sha256") != runtime_sha256:
        raise HumanApartmentGateError("native Rocketbox UE runtime GLB hash changed")
    if (
        Path(ue_manifest.get("source_manifest", "")).resolve()
        != source_manifest_path.resolve()
        or ue_manifest.get("source_manifest_sha256") != _sha256(source_manifest_path)
    ):
        raise HumanApartmentGateError(
            "native Rocketbox UE source manifest hash/path changed"
        )
    runtime_contract = ue_manifest.get("runtime_contract", {})
    bounds = runtime_contract.get("bounds", {})
    height_cm = float(bounds.get("height_cm", 0.0))
    if (
        runtime_contract.get("bone_count") != 80
        or runtime_contract.get("actor_scale") != 1.0
        or not height_range[0] <= height_cm <= height_range[1]
        or bounds.get("height_passed") is not True
        or bounds.get("ground_passed") is not True
    ):
        raise HumanApartmentGateError(
            f"native Rocketbox adult height/runtime contract failed: {height_cm} cm"
        )
    animations = ue_manifest.get("content", {}).get("animations", {})
    if set(animations) != {"Walking", "Standing_Idle"}:
        raise HumanApartmentGateError(
            "native Rocketbox UE import lacks exact Walk/Idle actions"
        )
    blueprint = ue_manifest.get("content", {}).get("blueprint")
    if not isinstance(blueprint, str) or not blueprint:
        raise HumanApartmentGateError("native Rocketbox UE Blueprint is missing")

    return {
        "tag": tag,
        "asset_id": asset_id,
        "usage_scope": "research_candidate",
        "runtime_glb_sha256": runtime_sha256,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "ue_import_manifest_sha256": _sha256(ue_manifest_path),
        "ue_import_manifest_path": str(ue_manifest_path.resolve()),
        "bone_count": 80,
        "actor_scale": 1.0,
        "height_cm": height_cm,
        "height_range_cm": list(height_range),
        "walking_in_place": True,
        "dynamic_ground_snap_to_floor_required": True,
        "animations": animations,
        "blueprint": blueprint,
    }


def assert_human_apartment_ready(
    tag: str,
    *,
    stable_root: Path | None = None,
    formal_registry_root: Path | None = None,
    skip_review_gate: bool = False,
) -> dict:
    """Validate one technical-spike human before SPEAR/UE runtime rendering."""
    if tag not in ALLOWED_HUMAN_SPIKES:
        raise HumanApartmentGateError(f"unapproved humanoid spike tag: {tag!r}")
    if skip_review_gate:
        raise HumanApartmentGateError(
            "SPEAR_SKIP_REVIEW_GATE=1 cannot be used as humanoid smoke evidence"
        )

    stable_root = Path(stable_root or DEFAULT_STABLE_ROOT).resolve()
    if stable_root.name != "hy3d_rocketbox_template_fit_v1":
        raise HumanApartmentGateError(
            f"humanoid must come from stable-template root, got {stable_root}"
        )
    formal_registry_root = Path(
        formal_registry_root or DEFAULT_FORMAL_REGISTRY_ROOT
    ).resolve()
    if _registry_contains_tag(formal_registry_root, tag):
        raise HumanApartmentGateError(
            f"technical-spike humanoid appears in formal source registry: {tag}"
        )

    asset_id = ALLOWED_HUMAN_SPIKES[tag]
    asset_dir = (stable_root / asset_id).resolve()
    if asset_dir.parent != stable_root:
        raise HumanApartmentGateError("humanoid asset path escaped stable root")

    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = _load_json(ready_path, "stable-template readiness record")
    if ready.get("schema_version") != "hy3d_rocketbox_direct_attempt_ready_v1":
        raise HumanApartmentGateError("unexpected stable-template readiness schema")
    if ready.get("status") != "ready" or ready.get("asset_id") != asset_id:
        raise HumanApartmentGateError("stable-template readiness identity/status mismatch")

    bind_manifest_path = asset_dir / "bind_manifest.json"
    _verify_file_hash(
        bind_manifest_path,
        ready.get("bind_manifest_sha256", ""),
        "bind manifest",
    )
    bind_manifest = _load_json(bind_manifest_path, "bind manifest")
    if bind_manifest.get("asset_id") != asset_id:
        raise HumanApartmentGateError("bind manifest asset identity mismatch")
    if bind_manifest.get("binding_mode") != "stable_rocketbox_template_fit_v1":
        raise HumanApartmentGateError("runtime mesh is not the stable Rocketbox template")
    if bind_manifest.get("usage_scope") != "technical_spike_only":
        raise HumanApartmentGateError("Hunyuan-derived appearance lost technical scope")

    for key in ("pixel_qa", "bind_metrics", "bound_blend", "contact_sheet"):
        _verify_record(asset_dir, ready.get(key), key)
    for label, record in _walk_records(ready.get("glbs"), "glbs"):
        _verify_record(asset_dir, record, label)
    for label, record in _walk_records(ready.get("videos"), "videos"):
        _verify_record(asset_dir, record, label)
    _verify_file_hash(
        asset_dir / "review_manifest.json",
        ready.get("review_manifest_sha256", ""),
        "review manifest",
    )

    pixel_qa = _load_json(asset_dir / "pixel_qa.json", "pixel QA")
    if pixel_qa.get("decision") != "ready":
        raise HumanApartmentGateError("pixel QA is not ready")
    checks = pixel_qa.get("checks", {})
    if checks and not all(value is True for value in checks.values()):
        raise HumanApartmentGateError("pixel QA contains a failed visual check")
    review_manifest = _load_json(asset_dir / "review_manifest.json", "review manifest")
    if review_manifest.get("automatic_checks", {}).get("overall") != "passed":
        raise HumanApartmentGateError("automatic Blender review gate did not pass")

    ue_manifest_path = asset_dir / "ue_import_manifest.json"
    ue_manifest = _load_json(ue_manifest_path, "UE import manifest")
    if ue_manifest.get("schema") != "hy3d_rocketbox_ue_import_v1":
        raise HumanApartmentGateError("unexpected UE import manifest schema")
    if ue_manifest.get("tag") != tag or ue_manifest.get("asset_id") != asset_id:
        raise HumanApartmentGateError("UE import manifest identity mismatch")
    if ue_manifest.get("usage_scope") != "technical_spike_only":
        raise HumanApartmentGateError("UE import manifest lost technical scope")
    if ue_manifest.get("reload_verification", {}).get("status") != "passed":
        raise HumanApartmentGateError("UE import was not verified in a second process")
    if ue_manifest.get("runtime_contract", {}).get("bone_count") != 80:
        raise HumanApartmentGateError("UE runtime skeleton is not the sealed 80-bone rig")

    runtime_glb = (asset_dir / "ue_runtime.glb").resolve()
    if Path(ue_manifest.get("source_glb", "")).resolve() != runtime_glb:
        raise HumanApartmentGateError("UE import manifest points outside stable output")
    _verify_file_hash(
        runtime_glb,
        ue_manifest.get("source_glb_sha256", ""),
        "runtime GLB",
    )
    animations = ue_manifest.get("content", {}).get("animations", {})
    if set(animations) != {"Walking", "Standing_Idle"}:
        raise HumanApartmentGateError("UE import manifest lacks exact runtime actions")

    return {
        "tag": tag,
        "asset_id": asset_id,
        "asset_dir": str(asset_dir),
        "usage_scope": "technical_spike_only",
        "ready_record_sha256": _sha256(ready_path),
        "ue_import_manifest_sha256": _sha256(ue_manifest_path),
        "runtime_glb_sha256": _sha256(runtime_glb),
        "bone_count": 80,
        "animations": animations,
        "blueprint": ue_manifest.get("content", {}).get("blueprint"),
    }
