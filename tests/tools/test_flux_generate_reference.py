import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "flux_generate_reference.py"


def _load_flux_module():
    spec = importlib.util.spec_from_file_location("flux_generate_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flux_human_prompt_template_has_no_pet_or_fur_words():
    flux = _load_flux_module()

    prompt = flux.build_prompt(
        "synthetic adult male speaker wearing a blue hoodie and dark jeans",
        template="human",
    )

    assert "blue hoodie" in prompt
    assert "full body" in prompt
    assert "plain white background" in prompt
    assert "photorealistic" in prompt
    assert "fur" not in prompt.lower()
    assert "pet" not in prompt.lower()


def test_flux_animal_template_preserves_existing_pet_bias():
    flux = _load_flux_module()

    prompt = flux.build_prompt("a beagle dog", template="animal")

    assert "detailed fur" in prompt
    assert "professional pet photography" in prompt
