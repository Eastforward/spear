import json
from pathlib import Path
import re

from tools import build_rocketbox_batch_video_index as video_index
from tools.build_rocketbox_batch_video_index import build_video_index


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_index_lists_every_avatar_and_marks_complete_pairs(tmp_path, monkeypatch):
    review_root = tmp_path / "review"
    inventory_path = tmp_path / "inventory.json"
    avatars = [
        {
            "base_avatar_id": "rocketbox_adults_female_adult_01",
            "category": "Adults",
            "gender": "female",
            "height_contract": {"authored_height_cm": 168.25},
        },
        {
            "base_avatar_id": "rocketbox_children_male_child_01",
            "category": "Children",
            "gender": "male",
            "height_contract": {"authored_height_cm": 143.0},
        },
    ]
    _write_json(
        inventory_path,
        {
            "schema_version": "rocketbox_human_inventory_v1",
            "avatars": avatars,
        },
    )

    records = []
    for avatar in avatars:
        avatar_id = avatar["base_avatar_id"]
        tag = f"{avatar_id}_original_ue_v1"
        records.append(
            {
                "base_avatar_id": avatar_id,
                "tag": tag,
                "actions": {
                    "Walking": {
                        "output_dir": str(review_root / "clips" / tag / "walking")
                    },
                    "Standing_Idle": {
                        "output_dir": str(review_root / "clips" / tag / "idle")
                    },
                },
            }
        )
    manifest_path = review_root / "batch_spec_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "rocketbox_batch_apartment_specs_v1",
            "avatar_count": 2,
            "clip_count": 4,
            "inventory": str(inventory_path),
            "records": records,
        },
    )

    complete_dir = Path(records[0]["actions"]["Walking"]["output_dir"])
    videos = complete_dir / "videos"
    videos.mkdir(parents=True)
    for name in (
        "apartment_v1_view0.mp4",
        "topdown_review.mp4",
        "side_by_side_review_annotated.mp4",
    ):
        (videos / name).write_bytes(b"video")

    document = tmp_path / "docs" / "rocketbox-videos.md"
    html = tmp_path / "docs" / "rocketbox-videos.html"
    monkeypatch.setattr(video_index, "AVENGINE_ROOT", tmp_path)
    summary = build_video_index(
        manifest_path,
        document,
        html_path=html,
        featured_manifest_path=None,
    )

    assert summary == {
        "avatar_count": 2,
        "clip_count": 4,
        "completed_clip_count": 1,
        "complete_pair_count": 0,
        "pending_clip_count": 3,
    }
    text = document.read_text(encoding="utf-8")
    assert "已完成：**1 / 4**" in text
    assert "完整 Walk/Idle 对：**0 / 2**" in text
    assert "## Adults · Female" in text
    assert "## Children · Male" in text
    assert "rocketbox_adults_female_adult_01" in text
    assert "[审核]" in text and "[主视图]" in text and "[Top-down]" in text
    assert "⏳ 待生成" in text
    assert "side_by_side_review_annotated.mp4" in text
    assert str(complete_dir.resolve()) in text
    assert "](../" not in text

    page = html.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="dataset" type="application/json">(.*?)</script>',
        page,
        re.DOTALL,
    )
    assert match is not None
    entries = json.loads(match.group(1))
    assert len(entries) == 1
    assert entries[0]["action"] == "Walking"
    assert entries[0]["featured"] is False
    assert set(entries[0]["media"]) == {"审核", "主视图", "Top-down"}
    assert all(
        item["absolute_path"].startswith(str(tmp_path))
        and item["url"].startswith("/")
        for item in entries[0]["media"].values()
    )


def test_index_rejects_manifest_inventory_identity_drift(tmp_path):
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": "rocketbox_human_inventory_v1",
            "avatars": [
                {
                    "base_avatar_id": "rocketbox_adults_male_adult_01",
                    "category": "Adults",
                    "gender": "male",
                    "height_contract": {"authored_height_cm": 180.0},
                }
            ],
        },
    )
    manifest = tmp_path / "review" / "batch_spec_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "rocketbox_batch_apartment_specs_v1",
            "avatar_count": 1,
            "clip_count": 2,
            "inventory": str(inventory),
            "records": [
                {
                    "base_avatar_id": "rocketbox_adults_female_adult_01",
                    "tag": "rocketbox_adults_female_adult_01_original_ue_v1",
                    "actions": {"Walking": {}, "Standing_Idle": {}},
                }
            ],
        },
    )

    try:
        build_video_index(manifest, tmp_path / "index.md")
    except RuntimeError as error:
        assert "identity set" in str(error)
    else:
        raise AssertionError("inventory/manifest drift was accepted")
