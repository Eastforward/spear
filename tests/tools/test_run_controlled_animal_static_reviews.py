from PIL import Image

from tools import run_controlled_animal_static_reviews as reviews


def test_contact_sheet_contains_reference_and_five_views(tmp_path):
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (64, 64), (200, 100, 50, 255)).save(reference)
    view_root = tmp_path / "views"
    view_root.mkdir()
    for index, view in enumerate(reviews.VIEWS):
        Image.new("RGB", (48, 48), (index * 30, 20, 40)).save(
            view_root / f"{view}.png"
        )
    output = tmp_path / "contact.png"

    reviews.build_contact_sheet(reference, view_root, output)

    with Image.open(output) as opened:
        assert opened.mode == "RGB"
        assert opened.size == (960, 640)


def test_renderer_contract_requests_top_view():
    assert reviews.VIEWS == ("front", "back", "side", "top", "quarter")
