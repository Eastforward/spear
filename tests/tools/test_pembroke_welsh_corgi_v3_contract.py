from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.quadruped_morphotype_guide import load_morphotype_guide_profile


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "quaternius_dog_short_leg_min_tail_side_clay_v2"
)
GUIDE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "quadruped_morphotype_guide_short_leg_min_tail_v2.json"
)
V1_PROVENANCE = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "quaternius_dog_short_leg_short_tail_side_clay_v1/provenance.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corgi_v3_frozen_reference_evidence_remains_hash_bound():
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    reference = provenance["reference"]
    reference_path = ROOT / reference["path"]
    guide = load_morphotype_guide_profile(GUIDE_PATH)

    assert provenance["reference_id"] == (
        "quaternius_dog_short_leg_min_tail_side_clay_v2"
    )
    assert reference_path == REFERENCE_ROOT / "frame_0000.png"
    assert _sha256(reference_path) == (
        "d85cfbd15b9a69394102cc8534fbb3326fb07099633f8cf9e0ae8186d281f025"
    )
    assert _sha256(reference_path) == reference["sha256"]
    assert reference_path.stat().st_size == reference["size_bytes"]
    assert provenance["method_revision"]["kind"] == (
        "new_render_only_morphotype_method_revision"
    )
    assert provenance["method_revision"]["triggered_by_rejected_candidate_sha256"] == (
        "d63814fbe4fe4f4951ced196d79d082616bba916a7c47d9d3ea33d347b7b3951"
    )
    assert guide.leg_length_ratio == 0.65
    assert guide.tail_length_ratio == 0.25


def test_corgi_v3_predecessor_reference_evidence_remains_unchanged():
    v1 = _load(V1_PROVENANCE)

    assert v1["reference_id"] == "quaternius_dog_short_leg_short_tail_side_clay_v1"
    assert v1["reference"]["sha256"] == (
        "ef883249cdbb484752607e8502071f05da88da53f975b56715d5fd2dd27c98b6"
    )
