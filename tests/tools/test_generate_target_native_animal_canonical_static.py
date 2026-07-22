from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "generate_target_native_animal_canonical.py"
)


def test_target_native_canonical_uses_full_gpu_and_real_cfg() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '.to("cuda")' in source
    assert "negative_prompt_embeds=negative_embeds" in source
    assert "enable_model_cpu_offload" not in source
    assert "enable_sequential_cpu_offload" not in source
    assert "low_vram" not in source.lower()
    assert "device_map=" not in source


def test_target_native_canonical_keeps_pixel3d_blocked() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"pixel3d_authorized": False' in source
    assert '"status": "rendered_pending_project_owner_review"' in source
    assert "before_any_pixel3d_execution" in source


def test_target_native_canonical_separates_input_authority() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "breed_identity_proportions_silhouette_and_camera_authority" in source
    assert "real_photo_breed_appearance_evidence_only" in source
    assert "conditioning_images = [identity]" in source
    assert "image=conditioning_images" in source
    assert "if appearance is not None" in source
