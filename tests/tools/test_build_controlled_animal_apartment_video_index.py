import json
from pathlib import Path

from tools.build_controlled_animal_apartment_video_index import build_video_index


def _write(path: Path, payload=b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


def test_index_requires_registry_and_links_complete_walk_idle_pair(tmp_path):
    root = tmp_path / "batch"
    asset_id = "cat_siamese_example"
    tag = f"pixal_{asset_id}"
    actions = {}
    for action in ("Walking", "Idle"):
        output = root / "clips" / tag / action.lower()
        spec = _write(
            root / "specs" / tag / f"{action.lower()}.json",
            {
                "sources": [
                    {
                        "tag": tag,
                        "asset_id": asset_id,
                        "asset_class": "animal",
                        "species": "cat",
                        "breed": "siamese",
                        "actor_scale": 0.09,
                        "sampled_attributes": {
                            "coat_color": "seal_point",
                            "size": "medium",
                        },
                    }
                ]
            },
        )
        for filename in (
            "side_by_side_review_annotated.mp4",
            "apartment_v1_view0.mp4",
            "topdown_review.mp4",
        ):
            _write(output / "videos" / filename)
        _write(output / "binaural.wav")
        _write(output / "binaural_source_schedule.json", {"sources": {tag: {}}})
        actions[action] = {
            "spec": str(spec),
            "clip_id": f"{tag}_{action.lower()}",
            "output_dir": str(output),
        }
    record = {
        "base_avatar_id": asset_id,
        "tag": tag,
        "actions": actions,
    }
    _write(
        root / "clips" / tag / "registry" / f"{tag}.json",
        {
            "usage_scope": "research_candidate",
            "formal_registry_promotion": False,
            "clips": {
                action: {"clip_id": actions[action]["clip_id"]}
                for action in actions
            },
        },
    )
    manifest = _write(
        root / "spec_manifest.json",
        {
            "schema": "controlled_animal_walk_idle_apartment_specs_v1",
            "avatar_count": 1,
            "clip_count": 2,
            "records": [record],
        },
    )
    document = tmp_path / "docs" / "animals.md"

    summary = build_video_index([manifest], document)

    text = document.read_text(encoding="utf-8")
    assert summary == {
        "animal_count": 1,
        "clip_count": 2,
        "completed_clip_count": 2,
        "complete_pair_count": 1,
    }
    assert "cat · siamese" in text
    assert "coat_color=seal_point" in text
    assert "[审核]" in text
    assert "[音频]" in text
    assert "[Registry]" in text
