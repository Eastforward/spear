import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_new_review_tags_include_second_beagle_and_british_shorthair():
    import hy3d_generate_and_audit as h

    tags = {r["tag"] for r in h.NEW_RIGS}

    assert tags == {"dog_beagle_v2", "cat_british_shorthair_v2"}
    assert "dog_beagle" not in tags
    assert "cat_british_shorthair" not in tags


def test_review_instructions_describe_v2_rotation_flow(capsys):
    import hy3d_generate_and_audit as h

    h.PENDING_ROOT.mkdir(parents=True, exist_ok=True)
    h._print_review_instructions()
    out = capsys.readouterr().out

    assert "green HEAD" in out
    assert "blue UP" in out
    assert "red arrow" not in out


def test_drop_into_pending_preserves_textured_obj_and_diffuse(tmp_path, monkeypatch):
    import hy3d_generate_and_audit as h

    pending = tmp_path / "pending"
    work = tmp_path / "work"
    work.mkdir()
    textured = work / "hy3d_textured.obj"
    diffuse = work / "hy3d_diffuse.jpg"
    metallic = work / "hy3d_metallic.jpg"
    roughness = work / "hy3d_roughness.jpg"
    mtl = work / "hy3d_output_mesh.mtl"
    textured.write_text("mtllib hy3d_output_mesh.mtl\nv 0 0 0\n")
    mtl.write_text(
        "newmtl Material\n"
        "map_Kd hy3d_output_mesh.jpg\n"
        "map_Pm hy3d_output_mesh_metallic.jpg\n"
        "map_Pr hy3d_output_mesh_roughness.jpg\n"
    )
    diffuse.write_bytes(b"diffuse")
    metallic.write_bytes(b"metallic")
    roughness.write_bytes(b"roughness")
    monkeypatch.setattr(h, "PENDING_ROOT", pending)

    h._drop_into_pending("dog_beagle_v2", textured)

    tag_dir = pending / "dog_beagle_v2"
    assert (tag_dir / "mesh.obj").read_text() == "mtllib mesh.mtl\nv 0 0 0\n"
    assert "map_Kd hy3d_diffuse.jpg" in (tag_dir / "mesh.mtl").read_text()
    assert "map_Pm hy3d_metallic.jpg" in (tag_dir / "mesh.mtl").read_text()
    assert "map_Pr hy3d_roughness.jpg" in (tag_dir / "mesh.mtl").read_text()
    assert (tag_dir / "hy3d_diffuse.jpg").read_bytes() == b"diffuse"
    assert (tag_dir / "hy3d_metallic.jpg").read_bytes() == b"metallic"
    assert (tag_dir / "hy3d_roughness.jpg").read_bytes() == b"roughness"


def test_write_candidate_manifest_for_generated_rig(tmp_path, monkeypatch):
    import hy3d_generate_and_audit as h

    pending = tmp_path / "pending"
    monkeypatch.setattr(h, "PENDING_ROOT", pending)
    tag_dir = pending / "dog_beagle_v2"
    tag_dir.mkdir(parents=True)
    (tag_dir / "mesh.obj").write_text("v 0 0 0\n")
    rig = {"tag": "dog_beagle_v2", "species": "dog", "breed": "beagle", "seed": 4101}

    path = h._write_candidate_manifest_for_rig(rig, "a beagle dog prompt")

    assert path == tag_dir / "source_asset_candidate.json"
    manifest = json.loads(path.read_text())
    assert manifest["asset_id"] == "dog_beagle_0002"
    assert manifest["generation"]["positive_prompt"] == "a beagle dog prompt"
    assert manifest["review"]["overall_status"] == "needs_review"
