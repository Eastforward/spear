from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest
from PIL import Image

from tools import controlled_source_asset_schema as contracts
from tools import publish_generated_animal_geometry_closure as publisher
from tools import repair_pixal_oriented_sheet_indices as repair

INSTANCE_ID = "dog_fixture_oriented_sheet_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(payload: bytearray, value: bytes) -> tuple[int, int]:
    while len(payload) % 4:
        payload.append(0)
    offset = len(payload)
    payload.extend(value)
    return offset, len(value)


def _glb_bytes(*, degenerate: bool, unpaired: bool = False, primitives: int = 1):
    positions = [
        (-0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (+0.0, 0.0, 0.0),
    ]
    triangles = [
        (4, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]
    if unpaired:
        triangles.pop()
    if degenerate:
        triangles.insert(2, (0, 4, 1))
    normals = [(0.0, 0.0, 1.0)] * len(positions)
    uvs = [(index / 4.0, index / 8.0) for index in range(len(positions))]
    binary = bytearray()
    views = []

    def view(value: bytes, *, target=None):
        offset, length = _append(binary, value)
        result = {"buffer": 0, "byteOffset": offset, "byteLength": length}
        if target is not None:
            result["target"] = target
        views.append(result)
        return len(views) - 1

    position_view = view(
        b"".join(struct.pack("<fff", *value) for value in positions),
        target=34962,
    )
    normal_view = view(
        b"".join(struct.pack("<fff", *value) for value in normals),
        target=34962,
    )
    uv_view = view(
        b"".join(struct.pack("<ff", *value) for value in uvs),
        target=34962,
    )
    index_view = view(
        b"".join(
            struct.pack("<H", value) for triangle in triangles for value in triangle
        ),
        target=34963,
    )
    image0_view = view(b"\x89PNG\r\nfixture_base_color")
    image1_view = view(b"RIFFfixture_orm_WEBP")
    while len(binary) % 4:
        binary.append(0)
    accessors = [
        {
            "bufferView": position_view,
            "componentType": 5126,
            "count": len(positions),
            "type": "VEC3",
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
        },
        {
            "bufferView": normal_view,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3",
        },
        {
            "bufferView": uv_view,
            "componentType": 5126,
            "count": len(uvs),
            "type": "VEC2",
        },
        {
            "bufferView": index_view,
            "componentType": 5123,
            "count": len(triangles) * 3,
            "type": "SCALAR",
        },
    ]
    primitive = {
        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
        "indices": 3,
        "material": 0,
        "mode": 4,
    }
    document = {
        "asset": {"version": "2.0", "generator": "oriented-sheet-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [copy.deepcopy(primitive) for _ in range(primitives)]}
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }
        ],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [
            {"bufferView": image0_view, "mimeType": "image/png"},
            {"bufferView": image1_view, "mimeType": "image/webp"},
        ],
    }
    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    total = 12 + 8 + len(json_payload) + 8 + len(binary)
    data = b"".join(
        (
            b"glTF",
            struct.pack("<II", 2, total),
            struct.pack("<II", len(json_payload), 0x4E4F534A),
            json_payload,
            struct.pack("<II", len(binary), 0x004E4942),
            bytes(binary),
        )
    )
    return data, triangles


def _rewrite_glb(data: bytes, mutate) -> bytes:
    document, binary = repair._parse_glb(data, "valid fixture")
    mutable_binary = bytearray(binary)
    mutate(document, mutable_binary)
    return repair._build_glb(document, bytes(mutable_binary))


def _glb_with_duplicate_json_key(data: bytes) -> bytes:
    document, binary = repair._parse_glb(data, "valid fixture")
    json_text = json.dumps(document, separators=(",", ":"))
    json_text = json_text.replace(
        '"asset":{"version":"2.0","generator":"oriented-sheet-test"}',
        (
            '"asset":{"version":"1.0"},'
            '"asset":{"version":"2.0","generator":"oriented-sheet-test"}'
        ),
        1,
    )
    assert json_text.count('"asset":') == 2
    payload = json_text.encode("utf-8")
    payload += b" " * ((-len(payload)) % 4)
    total = 12 + 8 + len(payload) + 8 + len(binary)
    return b"".join(
        (
            b"glTF",
            struct.pack("<II", 2, total),
            struct.pack("<II", len(payload), repair._JSON_CHUNK),
            payload,
            struct.pack("<II", len(binary), repair._BIN_CHUNK),
            binary,
        )
    )


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _record(path: Path, *, relative_to: Path | None = None):
    recorded = path.resolve()
    if relative_to is not None:
        recorded = recorded.relative_to(relative_to.resolve())
    return {
        "path": str(recorded),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _authorities(tmp_path: Path, source: Path):
    request_sha256 = "a" * 64
    source_record = {
        "path": str(source.resolve()),
        "sha256": _sha256(source),
        "bytes": source.stat().st_size,
    }
    pixal_manifest = _write_json(
        tmp_path / "pixal.manifest.json",
        {
            "backend": "pixal3d",
            "controlled_request": {
                "instance_id": INSTANCE_ID,
                "request_sha256": request_sha256,
            },
            "output": source_record,
        },
    )
    review_root = tmp_path / "static_review"
    review_dir = review_root / INSTANCE_ID
    views_dir = review_dir / "views"
    views_dir.mkdir(parents=True)
    reference_rgba = review_root / "reference_rgba.png"
    Image.new("RGBA", (16, 16), (180, 120, 80, 255)).save(reference_rgba)
    render_manifest = _write_json(
        views_dir / "render_manifest.json", {"status": "passed"}
    )
    contact_sheet = review_dir / "contact_sheet.png"
    contact_sheet.write_bytes(b"contact sheet")
    blender_log = review_dir / "blender.log"
    blender_log.write_bytes(b"blender log")
    views = {}
    for name in ("front", "back", "side", "top", "quarter"):
        view = views_dir / f"{name}.png"
        view.write_bytes(name.encode("utf-8"))
        views[name] = _record(view, relative_to=review_root)
    sampled_attributes = {
        "body_build": "standard",
        "coat_color": "red",
        "size": "medium",
    }
    target_physical_profile = {
        "control_attribute": "size",
        "measurement": "shoulder_height_cm",
    }
    review = {
        "schema": repair.static_reviews.REVIEW_SCHEMA,
        "instance_id": INSTANCE_ID,
        "request_sha256": request_sha256,
        "profile_schema_id": "dog_fixture_v1",
        "sampled_attributes": sampled_attributes,
        "target_physical_profile": target_physical_profile,
        "pixal_output": {
            "path": source.name,
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "reference_rgba": _record(reference_rgba),
        "render_manifest": _record(render_manifest, relative_to=review_root),
        "contact_sheet": _record(contact_sheet, relative_to=review_root),
        "blender_log": _record(blender_log, relative_to=review_root),
        "views": views,
        "automatic_checks": {"overall": "passed"},
    }
    review["review_sha256"] = repair._hash_without(review, "review_sha256")
    review_path = _write_json(review_dir / "static_review_manifest.json", review)
    review_batch = {
        "schema": repair.static_reviews.REVIEW_BATCH_SCHEMA,
        "status": "rendered_pending_visual_qa",
        "review_count": 1,
        "reviews": [
            {
                "instance_id": INSTANCE_ID,
                "review_sha256": review["review_sha256"],
                "review": _record(review_path, relative_to=review_root),
            }
        ],
        "automatic_checks": {"overall": "passed"},
    }
    review_batch["review_batch_sha256"] = repair._hash_without(
        review_batch, "review_batch_sha256"
    )
    review_batch_path = _write_json(
        review_root / "static_review_batch_manifest.json",
        review_batch,
    )

    decision = {
        "schema": repair.STATIC_DECISION_SCHEMA,
        "instance_id": INSTANCE_ID,
        "review_sha256": review["review_sha256"],
        "decision": repair.APPROVED,
        "state_classification": repair.RESEARCH,
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
        "checks": {name: True for name in repair.static_decisions.CHECK_FIELDS},
        "attribute_evidence": {
            "body_build": "passed_static_visual",
            "coat_color": "passed_static_visual",
            "size": "deferred_to_metric_3d",
        },
        "caveats": [],
        "notes": "Approved canonical fixture.",
        "review": {
            "path": str(review_path.resolve()),
            "sha256": _sha256(review_path),
            "size_bytes": review_path.stat().st_size,
        },
    }
    decision["decision_sha256"] = hashlib.sha256(
        contracts.canonical_json(decision).encode("utf-8")
    ).hexdigest()
    decision_root = tmp_path / "static_decision"
    decision_path = _write_json(
        decision_root / INSTANCE_ID / "static_decision.json", decision
    )
    decision_batch = {
        "schema": repair.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "static_review_batch": {
            "path": str(review_batch_path.resolve()),
            "sha256": _sha256(review_batch_path),
            "review_batch_sha256": review_batch["review_batch_sha256"],
        },
        "decision_count": 1,
        "approved_count": 1,
        "rejected_count": 0,
        "decisions": [
            {
                "instance_id": INSTANCE_ID,
                "decision": repair.APPROVED,
                "decision_sha256": decision["decision_sha256"],
                "record": _record(decision_path, relative_to=decision_root),
            }
        ],
        "automatic_checks": {"overall": "passed"},
    }
    decision_batch["decision_batch_sha256"] = repair._hash_without(
        decision_batch, "decision_batch_sha256"
    )
    decision_batch_path = _write_json(
        decision_root / "static_decision_batch_manifest.json",
        decision_batch,
    )
    return pixal_manifest, decision_batch_path, decision_path


def _run(tmp_path: Path, *, degenerate: bool, **glb_options):
    source = tmp_path / "source.glb"
    source_bytes, triangles = _glb_bytes(degenerate=degenerate, **glb_options)
    source.write_bytes(source_bytes)
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    output = tmp_path / "output.glb"
    manifest = tmp_path / "repair_manifest.json"
    published = repair.repair(
        instance_id=INSTANCE_ID,
        source_path=source,
        expected_source_sha256=_sha256(source),
        pixal_manifest_path=pixal_manifest,
        expected_pixal_manifest_sha256=_sha256(pixal_manifest),
        static_decision_batch_path=decision_batch,
        expected_static_decision_batch_sha256=_sha256(decision_batch),
        static_decision_path=decision,
        expected_static_decision_sha256=_sha256(decision),
        output_path=output,
        manifest_path=manifest,
    )
    return source, output, published, triangles


def test_filters_only_exact_position_degenerate_triangle(tmp_path):
    source, output, manifest_path, source_triangles = _run(tmp_path, degenerate=True)
    manifest = repair.validate_repair_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    source_document, source_binary = repair._parse_glb(
        source.read_bytes(), "source fixture"
    )
    output_document, output_binary = repair._parse_glb(
        output.read_bytes(), "output fixture"
    )
    source_index = repair._index_layout(source_document, source_binary, 3, "source")
    output_index = repair._index_layout(output_document, output_binary, 3, "output")
    output_triangles = repair._triplets(
        repair._unpack_indices(output_binary, output_index)
    )

    assert manifest["implementation_contract"] == repair.IMPLEMENTATION_CONTRACT
    assert manifest["mutation"]["removed_triangle_count"] == 1
    assert manifest["mutation"]["output_byte_identical_to_source"] is False
    assert output_triangles == [
        triangle for triangle in source_triangles if triangle != (0, 4, 1)
    ]
    start, end = source_index["start"], source_index["end"]
    assert source_binary[:start] + source_binary[end:] == (
        output_binary[:start] + output_binary[end:]
    )
    restored = copy.deepcopy(output_document)
    restored["accessors"][3]["count"] = source_document["accessors"][3]["count"]
    assert restored == source_document


def test_no_degenerate_triangles_publish_byte_identical_noop(tmp_path):
    source, output, manifest_path, _ = _run(tmp_path, degenerate=False)
    manifest = repair.validate_repair_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )

    assert output.read_bytes() == source.read_bytes()
    assert manifest["output"]["sha256"] == _sha256(source)
    assert manifest["mutation"]["authorized_json_changes"] == []
    assert manifest["decision"]["status"].startswith("passed_byte_identical_noop")


def test_accepts_pixel3d_ext_texture_webp_bindings(tmp_path):
    source = tmp_path / "source.glb"

    def use_pixel3d_webp_extension(document, _binary):
        document["extensionsUsed"] = ["EXT_texture_webp"]
        document["extensionsRequired"] = ["EXT_texture_webp"]
        document["textures"] = [
            {"extensions": {"EXT_texture_webp": {"source": 0}}},
            {"extensions": {"EXT_texture_webp": {"source": 1}}},
        ]
        for image in document["images"]:
            image["mimeType"] = "image/webp"

    source.write_bytes(
        _rewrite_glb(
            _glb_bytes(degenerate=False)[0],
            use_pixel3d_webp_extension,
        )
    )
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    output = tmp_path / "output.glb"

    repair.repair(
        instance_id=INSTANCE_ID,
        source_path=source,
        expected_source_sha256=_sha256(source),
        pixal_manifest_path=pixal_manifest,
        expected_pixal_manifest_sha256=_sha256(pixal_manifest),
        static_decision_batch_path=decision_batch,
        expected_static_decision_batch_sha256=_sha256(decision_batch),
        static_decision_path=decision,
        expected_static_decision_sha256=_sha256(decision),
        output_path=output,
        manifest_path=tmp_path / "repair_manifest.json",
    )
    assert publisher._glb_readback(output)["texture_count"] == 2


def test_accepts_adopted_copy_bound_by_manifest_hash_and_size(tmp_path):
    original = tmp_path / "original.glb"
    original.write_bytes(_glb_bytes(degenerate=False)[0])
    source = tmp_path / "adopted.glb"
    source.write_bytes(original.read_bytes())
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    payload = json.loads(pixal_manifest.read_text(encoding="utf-8"))
    payload["output"]["path"] = str(original.resolve())
    _write_json(pixal_manifest, payload)

    repair.repair(
        instance_id=INSTANCE_ID,
        source_path=source,
        expected_source_sha256=_sha256(source),
        pixal_manifest_path=pixal_manifest,
        expected_pixal_manifest_sha256=_sha256(pixal_manifest),
        static_decision_batch_path=decision_batch,
        expected_static_decision_batch_sha256=_sha256(decision_batch),
        static_decision_path=decision,
        expected_static_decision_sha256=_sha256(decision),
        output_path=tmp_path / "output.glb",
        manifest_path=tmp_path / "repair_manifest.json",
    )


def test_signed_zero_duplicate_positions_share_exact_identity(tmp_path):
    _, _, manifest_path, _ = _run(tmp_path, degenerate=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["topology"]["unique_exact_positions"] == 4
    assert manifest["topology"]["unpaired_oriented_edges"] == 0
    assert (
        manifest["checks"]["signed_zero_canonicalized_for_exact_position_identity"]
        is True
    )


def test_rejects_unpaired_oriented_occurrence_without_publishing(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(_glb_bytes(degenerate=False, unpaired=True)[0])
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    output = tmp_path / "output.glb"
    manifest = tmp_path / "repair_manifest.json"

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="not completely reverse-paired",
    ):
        repair.repair(
            instance_id=INSTANCE_ID,
            source_path=source,
            expected_source_sha256=_sha256(source),
            pixal_manifest_path=pixal_manifest,
            expected_pixal_manifest_sha256=_sha256(pixal_manifest),
            static_decision_batch_path=decision_batch,
            expected_static_decision_batch_sha256=_sha256(decision_batch),
            static_decision_path=decision,
            expected_static_decision_sha256=_sha256(decision),
            output_path=output,
            manifest_path=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


def test_rejects_multiple_primitives(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(_glb_bytes(degenerate=False, primitives=2)[0])
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="exactly one mesh primitive",
    ):
        repair.repair(
            instance_id=INSTANCE_ID,
            source_path=source,
            expected_source_sha256=_sha256(source),
            pixal_manifest_path=pixal_manifest,
            expected_pixal_manifest_sha256=_sha256(pixal_manifest),
            static_decision_batch_path=decision_batch,
            expected_static_decision_batch_sha256=_sha256(decision_batch),
            static_decision_path=decision,
            expected_static_decision_sha256=_sha256(decision),
            output_path=tmp_path / "output.glb",
            manifest_path=tmp_path / "repair_manifest.json",
        )


def test_rejects_self_hashed_approval_outside_canonical_batch(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(_glb_bytes(degenerate=False)[0])
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    fake = tmp_path / "standalone_self_hashed_approval.json"
    fake.write_bytes(decision.read_bytes())

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="not the selected canonical batch record",
    ):
        repair.repair(
            instance_id=INSTANCE_ID,
            source_path=source,
            expected_source_sha256=_sha256(source),
            pixal_manifest_path=pixal_manifest,
            expected_pixal_manifest_sha256=_sha256(pixal_manifest),
            static_decision_batch_path=decision_batch,
            expected_static_decision_batch_sha256=_sha256(decision_batch),
            static_decision_path=fake,
            expected_static_decision_sha256=_sha256(fake),
            output_path=tmp_path / "output.glb",
            manifest_path=tmp_path / "repair_manifest.json",
        )


def test_rejects_canonical_approval_with_any_false_visual_check(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(_glb_bytes(degenerate=False)[0])
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    decision_payload["checks"]["pose_riggable"] = False
    decision_payload["decision_sha256"] = repair._hash_without(
        decision_payload, "decision_sha256"
    )
    decision.write_text(
        json.dumps(decision_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_payload = json.loads(decision_batch.read_text(encoding="utf-8"))
    index = batch_payload["decisions"][0]
    index["decision_sha256"] = decision_payload["decision_sha256"]
    index["record"]["sha256"] = _sha256(decision)
    index["record"]["size_bytes"] = decision.stat().st_size
    batch_payload["decision_batch_sha256"] = repair._hash_without(
        batch_payload, "decision_batch_sha256"
    )
    decision_batch.write_text(
        json.dumps(batch_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="canonical raw static decision batch is invalid",
    ):
        repair.repair(
            instance_id=INSTANCE_ID,
            source_path=source,
            expected_source_sha256=_sha256(source),
            pixal_manifest_path=pixal_manifest,
            expected_pixal_manifest_sha256=_sha256(pixal_manifest),
            static_decision_batch_path=decision_batch,
            expected_static_decision_batch_sha256=_sha256(decision_batch),
            static_decision_path=decision,
            expected_static_decision_sha256=_sha256(decision),
            output_path=tmp_path / "output.glb",
            manifest_path=tmp_path / "repair_manifest.json",
        )


def test_manifest_rejects_pbr_readback_hash_upgrade(tmp_path):
    _, _, manifest_path, _ = _run(tmp_path, degenerate=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["readback"]["output_pbr_payload_sha256"] = "f" * 64

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="output readback identity",
    ):
        repair.validate_repair_manifest(manifest)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda topology: topology.update({"unsupported_claim": True}),
        lambda topology: topology.__setitem__(
            "directed_edge_occurrences",
            topology["directed_edge_occurrences"] + 1,
        ),
    ),
)
def test_manifest_rejects_noncanonical_topology_claims(tmp_path, mutate):
    _, _, manifest_path, _ = _run(tmp_path, degenerate=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest["topology"])
    manifest["readback"]["output_topology"] = copy.deepcopy(manifest["topology"])

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="oriented-sheet topology",
    ):
        repair.validate_repair_manifest(manifest)


def test_rejects_duplicate_keys_in_authority_json(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(_glb_bytes(degenerate=False)[0])
    pixal_manifest, decision_batch, decision = _authorities(tmp_path, source)
    text = pixal_manifest.read_text(encoding="utf-8")
    text = text.replace(
        '"backend": "pixal3d",',
        '"backend": "ignored",\n  "backend": "pixal3d",',
        1,
    )
    pixal_manifest.write_text(text, encoding="utf-8")

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="duplicate key 'backend'",
    ):
        repair.repair(
            instance_id=INSTANCE_ID,
            source_path=source,
            expected_source_sha256=_sha256(source),
            pixal_manifest_path=pixal_manifest,
            expected_pixal_manifest_sha256=_sha256(pixal_manifest),
            static_decision_batch_path=decision_batch,
            expected_static_decision_batch_sha256=_sha256(decision_batch),
            static_decision_path=decision,
            expected_static_decision_sha256=_sha256(decision),
            output_path=tmp_path / "output.glb",
            manifest_path=tmp_path / "repair_manifest.json",
        )


def test_rejects_duplicate_keys_in_glb_json_chunk():
    data = _glb_with_duplicate_json_key(_glb_bytes(degenerate=False)[0])

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="duplicate key 'asset'",
    ):
        repair._parse_glb(data, "duplicate-key fixture")


def test_buffer_view_cannot_consume_bin_chunk_padding():
    data = _rewrite_glb(
        _glb_bytes(degenerate=False)[0],
        lambda document, binary: (
            document["buffers"][0].update({"byteLength": len(binary) - 1}),
            document["bufferViews"][-1].update(
                {
                    "byteOffset": len(binary) - 1,
                    "byteLength": 1,
                }
            ),
        ),
    )
    document, binary = repair._parse_glb(data, "padding fixture")

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="bufferView bounds",
    ):
        repair._authenticate_primitive(document, binary)


def test_index_accessor_requires_absolute_component_alignment():
    def mutate(document, _binary):
        view = document["bufferViews"][3]
        view["byteOffset"] += 1
        view["byteLength"] -= 1

    data = _rewrite_glb(_glb_bytes(degenerate=False)[0], mutate)
    document, binary = repair._parse_glb(data, "misaligned-index fixture")

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="absolute offset",
    ):
        repair._authenticate_primitive(document, binary)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document, _binary: document["accessors"][0].update(
                {"min": [99.0, 0.0, 0.0]}
            ),
            "POSITION accessor min",
        ),
        (
            lambda document, _binary: document["accessors"][3].update(
                {"max": [99]}
            ),
            "indices accessor max",
        ),
        (
            lambda document, _binary: document["accessors"][1].update(
                {"componentType": 5123}
            ),
            "NORMAL must be",
        ),
        (
            lambda document, _binary: document["accessors"][2].update(
                {"count": document["accessors"][0]["count"] - 1}
            ),
            "TEXCOORD_0 count",
        ),
    ),
)
def test_rejects_false_extrema_or_invalid_normal_uv_contract(mutate, message):
    data = _rewrite_glb(_glb_bytes(degenerate=False)[0], mutate)
    document, binary = repair._parse_glb(data, "attribute fixture")

    with pytest.raises(repair.OrientedSheetRepairError, match=message):
        repair._authenticate_primitive(document, binary)


def test_filter_cannot_change_index_or_referenced_geometry_extrema():
    def mutate(document, binary):
        index_accessor = document["accessors"][3]
        view = document["bufferViews"][index_accessor["bufferView"]]
        start = view["byteOffset"] + index_accessor.get("byteOffset", 0)
        struct.pack_into("<H", binary, start, 0)

    data = _rewrite_glb(_glb_bytes(degenerate=True)[0], mutate)
    document, binary = repair._parse_glb(data, "extrema-change fixture")

    with pytest.raises(
        repair.OrientedSheetRepairError,
        match="would change accessor/geometry extrema",
    ):
        repair._authenticate_primitive(document, binary)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document, _binary: document["materials"][0][
                "pbrMetallicRoughness"
            ]["baseColorTexture"].update({"index": 2}),
            "baseColorTexture binding",
        ),
        (
            lambda document, _binary: document["textures"][1].update(
                {"source": 2}
            ),
            "image source",
        ),
        (
            lambda document, _binary: document["textures"][0].update(
                {"sampler": 0}
            ),
            "sampler",
        ),
        (
            lambda document, _binary: document["images"].pop(),
            "exactly two Pixel3D",
        ),
    ),
)
def test_rejects_invalid_pixel3d_pbr_reference_graph(mutate, message):
    data = _rewrite_glb(_glb_bytes(degenerate=False)[0], mutate)
    document, binary = repair._parse_glb(data, "PBR fixture")

    with pytest.raises(repair.OrientedSheetRepairError, match=message):
        repair._authenticate_primitive(document, binary)
