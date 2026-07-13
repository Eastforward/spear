from tools import prepare_controlled_animal_ue_imports as preparation


def test_transcode_command_is_explicit_and_no_replace_capable(tmp_path):
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    manifest = tmp_path / "manifest.json"

    command = preparation._transcode_command(source, output, manifest)

    assert command[0] == str(preparation.PYTHON)
    assert command[1] == str(preparation.TRANSCODER)
    assert command[command.index("--input") + 1] == str(source)
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--manifest") + 1] == str(manifest)


def test_ue_tags_are_derived_from_immutable_asset_ids():
    asset_id = "dog_golden_retriever_123456789abc"
    assert f"pixal_{asset_id}".startswith("pixal_dog_golden_retriever_")
