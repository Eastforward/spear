import hashlib
import json
from pathlib import Path

from tools.resolve_controlled_animal_apartment_manifests import resolve_manifests


def _write_manifest(path: Path, records: list[dict]) -> None:
    value = {
        "schema": "controlled_animal_walk_idle_apartment_specs_v1",
        "avatar_count": len(records),
        "clip_count": len(records) * 2,
        "records": records,
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _record(asset_id: str, root: Path, generation: str) -> dict:
    return {
        "actions": {
            "Idle": {
                "clip_id": f"{asset_id}_idle_{generation}",
                "output_dir": str(root / generation / "idle"),
                "spec": str(root / generation / "idle.json"),
            },
            "Walking": {
                "clip_id": f"{asset_id}_walking_{generation}",
                "output_dir": str(root / generation / "walking"),
                "spec": str(root / generation / "walking.json"),
            },
        },
        "asset_id": asset_id,
        "base_avatar_id": asset_id,
        "profile_schema_id": "dog_test_v1",
        "sampled_attributes": {"size": "medium"},
        "source_glb": {"path": "/tmp/test.glb", "sha256": "a" * 64},
        "tag": f"pixal_{asset_id}",
    }


def test_replacement_supersedes_only_matching_asset(tmp_path):
    base_path = tmp_path / "base.json"
    replacement_path = tmp_path / "replacement.json"
    output_path = tmp_path / "resolved.json"
    _write_manifest(
        base_path,
        [_record("dog_keep", tmp_path, "old"), _record("dog_replace", tmp_path, "old")],
    )
    _write_manifest(
        replacement_path, [_record("dog_replace", tmp_path, "calibrated")]
    )

    result = resolve_manifests(
        base_manifest=base_path,
        replacement_manifests=[replacement_path],
        output_path=output_path,
    )

    payload = json.loads(result.read_text())
    by_id = {item["asset_id"]: item for item in payload["records"]}
    assert payload["avatar_count"] == 2
    assert payload["clip_count"] == 4
    assert by_id["dog_keep"]["actions"]["Walking"]["clip_id"].endswith("old")
    assert by_id["dog_replace"]["actions"]["Walking"]["clip_id"].endswith(
        "calibrated"
    )
    assert payload["resolution"]["superseded_asset_ids"] == ["dog_replace"]
    expected = payload.pop("manifest_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
