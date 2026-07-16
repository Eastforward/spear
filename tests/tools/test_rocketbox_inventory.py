from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.rocketbox_inventory import (
    demographic_height_contract,
    discover_canonical_avatars,
    merge_blender_audits,
)


def _avatar(root: Path, category: str, name: str) -> None:
    export = root / "Assets" / "Avatars" / category / name / "Export"
    textures = export.parent / "Textures"
    export.mkdir(parents=True)
    textures.mkdir()
    (export / f"{name}.fbx").write_bytes(b"canonical")
    (export / f"{name}_facial.fbx").write_bytes(b"facial")
    (textures / "x_body_color.tga").write_bytes(b"texture")
    (export.parent / f"{name}.png").write_bytes(b"preview")


def test_discovery_uses_exact_folder_named_nonfacial_fbx_and_path_unique_ids(tmp_path):
    _avatar(tmp_path, "Adults", "Female_Party_01")
    _avatar(tmp_path, "Professions", "Female_Party_01")
    _avatar(tmp_path, "Children", "Male_Child_01")

    records = discover_canonical_avatars(tmp_path)

    assert [record["category"] for record in records] == [
        "Adults",
        "Children",
        "Professions",
    ]
    assert {record["base_avatar_id"] for record in records} == {
        "rocketbox_adults_female_party_01",
        "rocketbox_children_male_child_01",
        "rocketbox_professions_female_party_01",
    }
    assert all("_facial.fbx" not in record["fbx_path"] for record in records)
    assert records[1]["gender"] == "male"
    assert records[1]["demographic"] == "child"


@pytest.mark.parametrize(
    ("category", "height_cm", "expected"),
    [
        ("Adults", 183.1, "passed"),
        ("Professions", 149.0, "passed"),
        ("Children", 112.0, "passed"),
        ("Children", 181.0, "failed"),
        ("Adults", 250.0, "failed"),
    ],
)
def test_demographic_height_contract_preserves_scale_and_checks_room_headroom(
    category, height_cm, expected
):
    result = demographic_height_contract(
        category=category,
        authored_height_cm=height_cm,
        apartment_ceiling_cm=280.0,
    )

    assert result["actor_scale"] == 1.0
    assert result["status"] == expected
    assert result["authored_height_preserved"] is True
    assert result["mouth_audio_height_cm"] < height_cm
    assert result["ceiling_headroom_cm"] == pytest.approx(280.0 - height_cm)


def test_merge_blender_audits_requires_one_passed_measurement_per_avatar(tmp_path):
    records = [
        {
            "base_avatar_id": "rocketbox_adults_male_adult_01",
            "category": "Adults",
            "fbx_sha256": "a" * 64,
        }
    ]
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "rocketbox_blender_audit_shard_v1",
                "records": [
                    {
                        "base_avatar_id": "rocketbox_adults_male_adult_01",
                        "fbx_sha256": "a" * 64,
                        "status": "passed",
                        "authored_height_cm": 183.125,
                        "bounds_cm": {
                            "minimum": [-50.0, -20.0, -1.9],
                            "maximum": [50.0, 20.0, 181.2],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    merged = merge_blender_audits(records, [audit_path], 280.0)

    assert merged[0]["blender_audit"]["authored_height_cm"] == 183.125
    assert merged[0]["height_contract"]["status"] == "passed"


def test_merge_blender_audits_fails_closed_on_hash_mismatch(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "rocketbox_blender_audit_shard_v1",
                "records": [
                    {
                        "base_avatar_id": "rocketbox_adults_male_adult_01",
                        "fbx_sha256": "b" * 64,
                        "status": "passed",
                        "authored_height_cm": 183.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        merge_blender_audits(
            [
                {
                    "base_avatar_id": "rocketbox_adults_male_adult_01",
                    "category": "Adults",
                    "fbx_sha256": "a" * 64,
                }
            ],
            [audit_path],
            280.0,
        )
