import io

from PIL import Image

from tools import transcode_glb_webp_to_png as transcode


def _webp_payload():
    image = Image.new("RGBA", (4, 3), (17, 34, 51, 255))
    output = io.BytesIO()
    image.save(output, format="WEBP", lossless=True)
    return output.getvalue()


def test_transcode_rewrites_webp_extension_without_touching_scene_graph():
    webp = _webp_payload()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(webp)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(webp)}],
        "images": [{"bufferView": 0, "mimeType": "image/webp"}],
        "textures": [{"extensions": {"EXT_texture_webp": {"source": 0}}}],
        "extensionsUsed": ["EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"],
        "nodes": [{"name": "unchanged"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    rewritten, binary, records = transcode.transcode(document, webp)

    assert rewritten["nodes"] == [{"name": "unchanged"}]
    assert rewritten["images"][0]["mimeType"] == "image/png"
    assert rewritten["textures"][0]["source"] == 0
    assert "extensions" not in rewritten["textures"][0]
    assert "extensionsUsed" not in rewritten
    assert "extensionsRequired" not in rewritten
    assert len(records) == 1
    view = rewritten["bufferViews"][rewritten["images"][0]["bufferView"]]
    png = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    with Image.open(io.BytesIO(png)) as opened:
        assert opened.format == "PNG"
        assert opened.size == (4, 3)


def test_exclusive_writer_never_replaces_existing_file(tmp_path):
    path = tmp_path / "artifact.bin"
    transcode._write_all_exclusive(path, b"first")

    try:
        transcode._write_all_exclusive(path, b"second")
    except FileExistsError:
        pass
    else:
        raise AssertionError("exclusive writer replaced an existing artifact")

    assert path.read_bytes() == b"first"
