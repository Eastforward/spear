from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_measure_generated_animal_emitter.py"
)


def test_measurement_emits_the_corrected_right_handed_v2_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"avengine_generated_animal_emitter_measurement_v2"' in text
    assert "+X\nforward, +Y up and +Z right" in text
    assert "forward cross up equals right" in text
    assert "avengine_generated_animal_emitter_measurement_v1" not in text
    assert "+Z left" not in text
