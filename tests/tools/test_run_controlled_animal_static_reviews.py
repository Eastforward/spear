from PIL import Image

from tools import run_controlled_animal_static_reviews as reviews


def test_published_review_copies_its_authenticated_reference(tmp_path):
    staging = tmp_path / "staging"
    destination = staging / "animal"
    destination.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(source)

    record = reviews._publish_reference(source, destination, staging)

    published = destination / "pixal_input_rgba.png"
    assert published.read_bytes() == source.read_bytes()
    assert record == reviews._relative(published, staging)


def test_contact_sheet_contains_route_aware_reference_and_five_views(
    tmp_path, monkeypatch
):
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (64, 64), (200, 100, 50, 255)).save(reference)
    view_root = tmp_path / "views"
    view_root.mkdir()
    for index, view in enumerate(reviews.VIEWS):
        Image.new("RGB", (48, 48), (index * 30, 20, 40)).save(
            view_root / f"{view}.png"
        )
    output = tmp_path / "contact.png"
    labels = []
    original_panel = reviews._panel

    def capture_label(image, label):
        labels.append(label)
        return original_panel(image, label)

    monkeypatch.setattr(reviews, "_panel", capture_label)

    reviews.build_contact_sheet(
        reference,
        view_root,
        output,
        reference_label="bounded shadow-cleaned input",
    )

    with Image.open(output) as opened:
        assert opened.mode == "RGB"
        assert opened.size == (960, 640)
    assert labels == ["bounded shadow-cleaned input", *reviews.VIEWS]


def test_renderer_contract_requests_top_view():
    assert reviews.VIEWS == ("front", "back", "side", "top", "quarter")
