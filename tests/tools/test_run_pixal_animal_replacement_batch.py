from pathlib import Path

from PIL import Image

from tools.run_pixal_animal_replacement_batch import normalize_reference


def test_large_animal_reference_is_bounded_without_mutating_source(tmp_path):
    source = tmp_path / "source.png"
    image = Image.new("RGBA", (1600, 800), (0, 0, 0, 0))
    for x in range(300, 1300):
        for y in range(150, 650):
            image.putpixel((x, y), (120, 80, 40, 255))
    image.save(source)
    before = source.read_bytes()
    destination = tmp_path / "normalized.png"

    evidence = normalize_reference(source, destination)

    assert source.read_bytes() == before
    assert evidence["normalization"].startswith("rgba_lanczos_contain")
    assert evidence["normalized_size"] == [1024, 1024]
    with Image.open(destination) as normalized:
        assert normalized.mode == "RGBA"
        assert normalized.size == (1024, 1024)
        assert normalized.getchannel("A").getextrema() == (0, 255)


def test_existing_1024_reference_is_used_byte_exactly(tmp_path):
    source = tmp_path / "source.png"
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    image.putpixel((512, 512), (1, 2, 3, 255))
    image.save(source)

    evidence = normalize_reference(source, tmp_path / "unused.png")

    assert evidence["pixal_input"]["path"] == str(source.resolve())
    assert evidence["source"]["sha256"] == evidence["pixal_input"]["sha256"]
    assert evidence["normalization"] == "none_existing_rgba_at_or_below_1024"
