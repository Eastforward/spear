"""Pure tests for bounded same-rig quadruped identity derivation contracts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import struct
import zlib

import pytest

from tools import bounded_quadruped_identity_contract as identity


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "pembroke_welsh_corgi_authored_v9_bounded_identity_plan_v1.json"
)


def plan_value():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def rgba8_png(width, height, pixels):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\x00" + bytes(pixels[row * width * 4 : (row + 1) * width * 4])
        for row in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_actual_v9_plan_is_non_executable_and_uses_two_explicit_masks():
    plan = identity.load_plan(plan_value())

    assert plan.source_sha256 == (
        "d0076b03c38a88ecbeda457aaa0a4651540db7bd661f33b50a41be8e3707d096"
    )
    assert len(plan.positive.indices) == 56
    assert len(plan.negative.indices) == 56
    assert set(plan.positive.indices).isdisjoint(plan.negative.indices)
    assert plan.positive.rotation_degrees == 70.0
    assert plan.negative.rotation_degrees == -70.0
    assert plan.removal_custom_shape_assignments == 34
    assert plan.motion_root_bone == "Bone"
    assert plan.roundtrip_canonical_precision_decimals == 3
    assert plan.roundtrip_animation_sample_phases == tuple(
        index / 80 for index in range(81)
    )
    assert plan.maximum_bind_matrix_semantic_delta == 0.001
    assert plan.maximum_sampled_skinned_world_delta == 0.001
    assert plan.maximum_animation_duration_delta_frames == 0.001
    assert plan.raw["execution_authorized"] is False
    assert plan.raw["formal_dataset_registration_authorized"] is False


def test_plan_rejects_approval_claims_overlapping_masks_and_unbounded_edits():
    changed = plan_value()
    changed["formal_dataset_registration_authorized"] = True
    with pytest.raises(identity.IdentityContractError, match="non-executable"):
        identity.load_plan(changed)

    changed = plan_value()
    changed["execution_authorized"] = True
    with pytest.raises(identity.IdentityContractError, match="non-executable"):
        identity.load_plan(changed)

    changed = plan_value()
    changed["ear_morph"]["negative_y"]["vertex_indices"][0] = changed["ear_morph"][
        "positive_y"
    ]["vertex_indices"][0]
    changed["ear_morph"]["negative_y"]["vertex_indices"].sort()
    with pytest.raises(identity.IdentityContractError, match="disjoint"):
        identity.load_plan(changed)

    changed = plan_value()
    changed["gates"]["maximum_moved_vertex_fraction"] = 0.5
    with pytest.raises(identity.IdentityContractError, match="\\[1e-06, 0.15\\]"):
        identity.load_plan(changed)

    changed = plan_value()
    changed["gates"]["roundtrip_animation_sample_phases"] = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    with pytest.raises(identity.IdentityContractError, match="uniform 81-phase"):
        identity.load_plan(changed)

    changed = plan_value()
    changed["gates"]["roundtrip_animation_sample_phases"][37] += 1.0e-5
    with pytest.raises(identity.IdentityContractError, match="uniform 81-phase"):
        identity.load_plan(changed)


def test_equal_opposite_rotations_keep_mirrored_points_mirrored():
    plan = identity.load_plan(plan_value())
    positive = (2.65, 1.1, 3.48)
    negative = (
        positive[0],
        2.0 * plan.symmetry_plane_y - positive[1],
        positive[2],
    )

    positive_output = identity.rotate_ear_point(positive, plan.positive)
    negative_output = identity.rotate_ear_point(negative, plan.negative)

    assert negative_output[0] == pytest.approx(positive_output[0])
    assert negative_output[1] == pytest.approx(
        2.0 * plan.symmetry_plane_y - positive_output[1]
    )
    assert negative_output[2] == pytest.approx(positive_output[2])


def test_profile_driven_coat_regions_are_deterministic_and_not_taxonomy_code():
    plan = identity.load_plan(plan_value())
    minimum = (0.0, -1.0, 0.0)
    maximum = (10.0, 1.0, 10.0)

    assert (
        identity.coat_role(
            (4.0, 0.6, 2.0),
            minimum,
            maximum,
            0.0,
            plan.coat_regions,
        )
        == "white"
    )
    assert (
        identity.coat_role(
            (4.0, 0.6, 6.5),
            minimum,
            maximum,
            0.0,
            plan.coat_regions,
        )
        == "base"
    )
    assert (
        identity.coat_role(
            (8.2, 0.0, 8.0),
            minimum,
            maximum,
            0.0,
            plan.coat_regions,
        )
        == "white"
    )


def test_srgb_plan_values_decode_as_exact_png_codes_without_double_gamma():
    plan = identity.load_plan(plan_value())
    base = identity.srgb_rgba8(plan.base_color)
    white = identity.srgb_rgba8(plan.white_color)
    payload = rgba8_png(2, 1, base + white)

    assert base == (140, 19, 9, 255)
    assert white == (230, 222, 199, 255)
    assert identity.png_rgba8_code_values(payload) == {base, white}

    corrupted = bytearray(payload)
    corrupted[payload.index(b"IDAT") + 4] ^= 1
    with pytest.raises(identity.IdentityContractError, match="CRC"):
        identity.png_rgba8_code_values(bytes(corrupted))


def execution_authorization(preflight_sha, plan_sha, source_sha):
    return {
        "schema": identity.AUTHORIZATION_SCHEMA,
        "decision": "authorized_for_new_bounded_research_candidate_output",
        "preflight_receipt_sha256": preflight_sha,
        "plan_sha256": plan_sha,
        "source_sha256": source_sha,
        "checks": {name: True for name in sorted(identity.AUTHORIZATION_CHECKS)},
        "notes": "Machine checks bind exact masks, removal target, and coat regions.",
        "authorization_basis": "scoped_task_continue_asset_identity_repair",
        "authorized_actor_type": "codex_task_agent",
        "authorized_output_scope": (
            "new_research_candidate_sealed_at_publication_only"
        ),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "user_approval_inferred": False,
    }


def test_realization_requires_task_scoped_machine_authorization():
    preflight_sha = "a" * 64
    plan_sha = "b" * 64
    source_sha = "c" * 64
    authorization = execution_authorization(preflight_sha, plan_sha, source_sha)

    identity.validate_execution_authorization(
        authorization,
        expected_preflight_sha256=preflight_sha,
        expected_plan_sha256=plan_sha,
        expected_source_sha256=source_sha,
    )

    changed = copy.deepcopy(authorization)
    changed["preflight_receipt_sha256"] = "d" * 64
    with pytest.raises(identity.IdentityContractError, match="binding"):
        identity.validate_execution_authorization(
            changed,
            expected_preflight_sha256=preflight_sha,
            expected_plan_sha256=plan_sha,
            expected_source_sha256=source_sha,
        )

    changed = copy.deepcopy(authorization)
    changed["checks"]["positive_mask_is_only_one_ear"] = False
    with pytest.raises(identity.IdentityContractError, match="every explicit"):
        identity.validate_execution_authorization(
            changed,
            expected_preflight_sha256=preflight_sha,
            expected_plan_sha256=plan_sha,
            expected_source_sha256=source_sha,
        )


def test_strict_json_rejects_duplicates_nonfinite_and_non_utf8():
    with pytest.raises(identity.IdentityContractError, match="duplicate"):
        identity.strict_json_loads(b'{"a":1,"a":2}')
    with pytest.raises(identity.IdentityContractError, match="non-finite"):
        identity.strict_json_loads(b'{"a":NaN}')
    with pytest.raises(identity.IdentityContractError, match="UTF-8|utf-8"):
        identity.strict_json_loads(b'{"a":"\xff"}')


def sealed_publication(output):
    publication = identity.open_secure_publication(output)
    publication.write_root("artifact.bin", b"bounded-output")
    publication.write_evidence("proof.json", b'{"proof":true}\n')
    records = publication.seal(
        expected_root_files={"artifact.bin"},
        expected_evidence_files={"proof.json"},
    )
    return publication, records


def test_publication_security_boundary_is_explicit_about_posix_same_uid_limit():
    boundary = identity.publication_security_boundary()

    assert (
        boundary["rename_semantics"][
            "rename_is_publication_boundary_not_readiness_claim"
        ]
        is True
    )
    assert boundary["posix_same_uid_limit"]["absolute_immutability_claimed"] is False
    assert (
        boundary["posix_same_uid_limit"][
            "malicious_same_uid_writer_can_mutate_after_verification"
        ]
        is True
    )
    assert (
        boundary["consumer_requirement"][
            "external_expected_manifest_raw_sha256_required"
        ]
        is True
    )
    assert (
        boundary["consumer_requirement"][
            "rehash_entire_declared_file_closure_before_use"
        ]
        is True
    )
    assert (
        boundary["parent_directory_policy"][
            "uid_matched_group_privacy_is_environment_assumption"
        ]
        is True
    )


def test_secure_publication_rejects_other_writable_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)

    with pytest.raises(identity.IdentityContractError, match="other-writable"):
        identity.open_secure_publication(parent / "candidate")


def test_secure_publication_accepts_uid_matched_private_group_parent(tmp_path):
    if os.geteuid() != os.getegid():
        pytest.skip("test requires the UID-matched private-group convention")
    parent = tmp_path / "private-group"
    parent.mkdir()
    parent.chmod(0o775)
    publication = identity.open_secure_publication(parent / "candidate")
    try:
        publication.cleanup()
        assert publication.quarantine_name is not None
        assert (parent / publication.quarantine_name).is_dir()
    finally:
        publication.close()


def test_secure_publication_is_sealed_and_published_no_replace(tmp_path):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    try:
        publication.publish(records)
        assert publication.published is True
        assert (output / "artifact.bin").read_bytes() == b"bounded-output"
        assert (output / "evidence/proof.json").read_bytes() == b'{"proof":true}\n'
        assert (output.stat().st_mode & 0o777) == 0o555
        assert ((output / "artifact.bin").stat().st_mode & 0o777) == 0o444
    finally:
        publication.close()


def test_secure_publication_rejects_output_created_after_preverify(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    original_rename = identity.SecurePublication._rename_no_replace

    def create_concurrent_then_rename(current):
        output.mkdir()
        (output / "owner").write_bytes(b"concurrent")
        original_rename(current)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_rename_no_replace",
        create_concurrent_then_rename,
    )
    try:
        with pytest.raises(identity.IdentityContractError, match="replace output"):
            publication.publish(records)
        publication.cleanup()
        assert (output / "owner").read_bytes() == b"concurrent"
        assert publication.quarantine_name is not None
    finally:
        publication.close()


def test_secure_publication_rejects_ancestor_symlink_swap(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    publication, records = sealed_publication(alias / "candidate")
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    try:
        with pytest.raises(identity.IdentityContractError, match="held directory"):
            publication.publish(records)
        publication.cleanup()
        assert not (first / "candidate").exists()
        assert not (second / "candidate").exists()
    finally:
        publication.close()


def test_secure_publication_rejects_output_parent_path_swap(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "candidate"
    publication, records = sealed_publication(output)
    moved_parent = tmp_path / "moved-parent"
    parent.rename(moved_parent)
    parent.mkdir()
    try:
        with pytest.raises(identity.IdentityContractError, match="held directory"):
            publication.publish(records)
        publication.cleanup()
        assert not (parent / "candidate").exists()
        assert not (moved_parent / "candidate").exists()
    finally:
        publication.close()


def test_secure_publication_detects_sealed_byte_swap(tmp_path):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    os.fchmod(publication.staging_fd, 0o700)
    os.chmod(
        "artifact.bin",
        0o600,
        dir_fd=publication.staging_fd,
        follow_symlinks=False,
    )
    descriptor = os.open(
        "artifact.bin",
        os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
        dir_fd=publication.staging_fd,
    )
    try:
        os.write(descriptor, b"attacker-bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(
        "artifact.bin",
        0o444,
        dir_fd=publication.staging_fd,
        follow_symlinks=False,
    )
    os.fchmod(publication.staging_fd, 0o555)
    try:
        with pytest.raises(identity.IdentityContractError, match="bytes changed"):
            publication.publish(records)
        publication.cleanup()
        assert not output.exists()
    finally:
        publication.close()


def test_secure_publication_reauthenticates_parent_mode(tmp_path):
    original_mode = tmp_path.stat().st_mode & 0o777
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    tmp_path.chmod(original_mode | 0o002)
    try:
        with pytest.raises(
            identity.IdentityContractError, match="owner, group, or mode"
        ):
            publication.publish(records)
    finally:
        tmp_path.chmod(original_mode)
    try:
        publication.cleanup()
        assert publication.quarantine_name is not None
        assert not output.exists()
    finally:
        publication.close()


def test_post_rename_verification_detects_tamper_and_retains_final(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    original_rename = identity.SecurePublication._rename_no_replace

    def tamper_between_verify_and_rename(current):
        os.fchmod(current.staging_fd, 0o700)
        os.chmod(
            "artifact.bin",
            0o600,
            dir_fd=current.staging_fd,
            follow_symlinks=False,
        )
        descriptor = os.open(
            "artifact.bin",
            os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
            dir_fd=current.staging_fd,
        )
        try:
            os.write(descriptor, b"tampered-data!")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(
            "artifact.bin",
            0o444,
            dir_fd=current.staging_fd,
            follow_symlinks=False,
        )
        os.fchmod(current.staging_fd, 0o555)
        original_rename(current)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_rename_no_replace",
        tamper_between_verify_and_rename,
    )
    try:
        with pytest.raises(identity.IdentityContractError, match="bytes changed"):
            publication.publish(records)
        assert publication.published is True
        publication.cleanup()
        assert (output / "artifact.bin").read_bytes() == b"tampered-data!"
        assert publication.quarantine_name is None
    finally:
        publication.close()


def test_post_rename_final_name_replacement_is_detected_and_retained(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    original_rename = identity.SecurePublication._rename_no_replace
    detached = tmp_path / "detached-published-candidate"
    victim = tmp_path / "external-victim"
    victim.mkdir()
    (victim / "sentinel").write_bytes(b"must survive")

    def rename_then_replace_final(current):
        original_rename(current)
        output.rename(detached)
        victim.rename(output)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_rename_no_replace",
        rename_then_replace_final,
    )
    try:
        with pytest.raises(
            identity.IdentityContractError,
            match="directory identity changed",
        ):
            publication.publish(records)
        assert publication.published is True
        assert (detached / "artifact.bin").read_bytes() == b"bounded-output"
        assert (output / "sentinel").read_bytes() == b"must survive"
        publication.cleanup()
        assert (detached / "artifact.bin").read_bytes() == b"bounded-output"
    finally:
        publication.close()


def test_post_rename_parent_alias_swap_is_detected_and_final_is_retained(
    tmp_path,
    monkeypatch,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    publication, records = sealed_publication(alias / "candidate")
    original_rename = identity.SecurePublication._rename_no_replace

    def rename_then_swap_alias(current):
        original_rename(current)
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_rename_no_replace",
        rename_then_swap_alias,
    )
    try:
        with pytest.raises(identity.IdentityContractError, match="held directory"):
            publication.publish(records)
        assert publication.published is True
        assert (first / "candidate/artifact.bin").read_bytes() == b"bounded-output"
        assert not (second / "candidate").exists()
        publication.cleanup()
        assert (first / "candidate/artifact.bin").read_bytes() == b"bounded-output"
    finally:
        publication.close()


@pytest.mark.parametrize(
    "mutation",
    ("mode", "nlink", "same_size_hash", "size", "closure"),
)
def test_post_rename_exact_closure_reverification_rejects_each_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    original_rename = identity.SecurePublication._rename_no_replace

    def rename_then_mutate(current):
        original_rename(current)
        artifact = output / "artifact.bin"
        if mutation == "mode":
            artifact.chmod(0o644)
        elif mutation == "nlink":
            os.link(artifact, tmp_path / "external-hardlink")
        elif mutation in {"same_size_hash", "size"}:
            original = artifact.read_bytes()
            artifact.chmod(0o644)
            artifact.write_bytes(
                b"x" * len(original) if mutation == "same_size_hash" else b"short"
            )
            artifact.chmod(0o444)
        else:
            output.chmod(0o700)
            (output / "unexpected").write_bytes(b"unknown")
            output.chmod(0o555)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_rename_no_replace",
        rename_then_mutate,
    )
    try:
        with pytest.raises(
            identity.IdentityContractError,
            match="identity changed|bytes changed|file set changed",
        ):
            publication.publish(records)
        assert publication.published is True
        publication.cleanup()
        assert output.is_dir()
    finally:
        publication.close()


def test_secure_cleanup_substitution_is_quarantined(tmp_path):
    output = tmp_path / "candidate"
    publication = identity.open_secure_publication(output)
    publication.write_root("artifact.bin", b"owned")
    staging = tmp_path / publication.staging_name
    moved = tmp_path / "moved-owned-staging"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("must survive", encoding="utf-8")
    staging.rename(moved)
    staging.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(
            identity.IdentityContractError, match="replacement retained"
        ):
            publication.cleanup()
        assert sentinel.read_text(encoding="utf-8") == "must survive"
        assert staging.is_symlink()
    finally:
        publication.close()
        staging.unlink()
        shutil.rmtree(moved)


def test_secure_cleanup_midrename_root_substitution_retains_every_inode(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication = identity.open_secure_publication(output)
    publication.write_root("artifact.bin", b"owned")
    staging = tmp_path / publication.staging_name
    detached_owned = tmp_path / "detached-owned-staging"
    external_victim = tmp_path / "external-victim"
    external_victim.mkdir()
    (external_victim / "sentinel").write_bytes(b"must survive cleanup")
    original_quarantine = identity.SecurePublication._quarantine_no_replace

    def replace_root_then_quarantine(current, quarantine_name):
        staging.rename(detached_owned)
        external_victim.rename(staging)
        original_quarantine(current, quarantine_name)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_quarantine_no_replace",
        replace_root_then_quarantine,
    )
    try:
        with pytest.raises(
            identity.IdentityContractError,
            match="moved a replacement",
        ):
            publication.cleanup()
        assert publication.quarantine_name is not None
        quarantine = tmp_path / publication.quarantine_name
        assert (quarantine / "sentinel").read_bytes() == b"must survive cleanup"
        assert (detached_owned / "artifact.bin").read_bytes() == b"owned"
        assert not staging.exists()
    finally:
        publication.close()


def test_secure_cleanup_unknown_entry_is_quarantined(tmp_path):
    output = tmp_path / "candidate"
    publication = identity.open_secure_publication(output)
    publication.write_root("artifact.bin", b"owned")
    descriptor = os.open(
        "unexpected",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=publication.staging_fd,
    )
    os.close(descriptor)
    staging = tmp_path / publication.staging_name
    try:
        publication.cleanup()
        assert publication.quarantine_name is not None
        quarantine = tmp_path / publication.quarantine_name
        assert not staging.exists()
        assert (quarantine / "unexpected").is_file()
    finally:
        publication.close()


def test_secure_cleanup_race_quarantines_external_victim_without_unlink(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication = identity.open_secure_publication(output)
    publication.write_root("artifact.bin", b"owned")
    victim = tmp_path / "external-victim"
    victim.write_bytes(b"must survive cleanup")
    original_quarantine = identity.SecurePublication._quarantine_no_replace

    def inject_victim_then_quarantine(current, quarantine_name):
        os.rename(
            victim,
            "raced-victim",
            dst_dir_fd=current.staging_fd,
        )
        original_quarantine(current, quarantine_name)

    monkeypatch.setattr(
        identity.SecurePublication,
        "_quarantine_no_replace",
        inject_victim_then_quarantine,
    )
    try:
        publication.cleanup()
        assert publication.quarantine_name is not None
        quarantine = tmp_path / publication.quarantine_name
        assert (quarantine / "raced-victim").read_bytes() == b"must survive cleanup"
    finally:
        publication.close()


def test_open_failure_quarantines_early_staging(tmp_path, monkeypatch):
    def fail_proc_fd_validation(_publication):
        raise identity.IdentityContractError("injected early validation failure")

    monkeypatch.setattr(
        identity.SecurePublication,
        "_require_proc_fd_paths",
        fail_proc_fd_validation,
    )
    with pytest.raises(identity.IdentityContractError, match="injected early"):
        identity.open_secure_publication(tmp_path / "candidate")

    quarantines = list(tmp_path.glob(".candidate.*.quarantine"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "evidence").is_dir()


def test_preconstruction_failure_midrename_substitution_retains_both_roots(
    tmp_path,
    monkeypatch,
):
    original_mkdir = identity.os.mkdir
    original_rename = identity._renameat2_noreplace
    detached_name = None

    def fail_evidence_mkdir(path, *args, **kwargs):
        if path == "evidence":
            raise OSError("injected evidence mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    def replace_root_before_early_quarantine(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal detached_name
        if destination_name.endswith(".quarantine"):
            detached_name = f"{source_name}.detached"
            os.rename(
                source_name,
                detached_name,
                src_dir_fd=source_fd,
                dst_dir_fd=source_fd,
            )
            original_mkdir(source_name, 0o700, dir_fd=source_fd)
            victim_fd = os.open(
                source_name,
                identity._directory_flags(),
                dir_fd=source_fd,
            )
            try:
                descriptor = os.open(
                    "external_victim",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=victim_fd,
                )
                try:
                    os.write(descriptor, b"must survive early cleanup")
                finally:
                    os.close(descriptor)
            finally:
                os.close(victim_fd)
        original_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(identity.os, "mkdir", fail_evidence_mkdir)
    monkeypatch.setattr(
        identity,
        "_renameat2_noreplace",
        replace_root_before_early_quarantine,
    )
    with pytest.raises(
        identity.IdentityContractError,
        match="early quarantine moved a replacement",
    ):
        identity.open_secure_publication(tmp_path / "candidate")

    assert detached_name is not None
    assert (tmp_path / detached_name).is_dir()
    quarantines = list(tmp_path.glob(".candidate.*.quarantine"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "external_victim").read_bytes() == (
        b"must survive early cleanup"
    )


def test_open_failure_quarantines_unknown_entry_without_deleting_it(
    tmp_path,
    monkeypatch,
):
    def inject_unknown_then_fail(publication):
        descriptor = os.open(
            "unexpected",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=publication.staging_fd,
        )
        try:
            os.write(descriptor, b"retain me")
        finally:
            os.close(descriptor)
        raise identity.IdentityContractError("injected early validation failure")

    monkeypatch.setattr(
        identity.SecurePublication,
        "_require_proc_fd_paths",
        inject_unknown_then_fail,
    )
    with pytest.raises(identity.IdentityContractError, match="injected early"):
        identity.open_secure_publication(tmp_path / "candidate")

    quarantines = list(tmp_path.glob(".candidate.*.quarantine"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "unexpected").read_bytes() == b"retain me"


def test_open_failure_retains_replacement_and_external_target(
    tmp_path,
    monkeypatch,
):
    moved = tmp_path / "moved-owned-staging"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_bytes(b"must survive")

    def replace_staging_then_fail(publication):
        staging = tmp_path / publication.staging_name
        staging.rename(moved)
        staging.symlink_to(external, target_is_directory=True)
        raise identity.IdentityContractError("injected early replacement")

    monkeypatch.setattr(
        identity.SecurePublication,
        "_require_proc_fd_paths",
        replace_staging_then_fail,
    )
    with pytest.raises(
        identity.IdentityContractError,
        match="replacement retained",
    ):
        identity.open_secure_publication(tmp_path / "candidate")

    replacement = next(tmp_path.glob(".candidate.*.staging"))
    assert replacement.is_symlink()
    assert sentinel.read_bytes() == b"must survive"
    assert moved.is_dir()
    replacement.unlink()
    shutil.rmtree(moved)


def test_post_publish_fsync_failure_never_deletes_final(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "candidate"
    publication, records = sealed_publication(output)
    original_fsync = os.fsync

    def fail_parent_fsync(descriptor):
        if descriptor == publication.parent_fd:
            raise OSError("injected parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    try:
        with pytest.raises(OSError, match="injected"):
            publication.publish(records)
        assert publication.published is True
        publication.cleanup()
        assert (output / "artifact.bin").read_bytes() == b"bounded-output"
    finally:
        publication.close()


def test_close_attempts_every_descriptor_and_is_idempotent_after_error(
    tmp_path,
    monkeypatch,
):
    publication = identity.open_secure_publication(tmp_path / "candidate")
    descriptors = {
        publication.evidence_fd,
        publication.staging_fd,
        publication.parent_fd,
    }
    injected_descriptor = publication.evidence_fd
    closed = []
    original_close = identity.os.close

    def close_then_report_one_error(descriptor):
        original_close(descriptor)
        closed.append(descriptor)
        if descriptor == injected_descriptor:
            raise OSError("injected close failure")

    monkeypatch.setattr(identity.os, "close", close_then_report_one_error)
    with pytest.raises(OSError, match="injected close failure"):
        publication.close()
    assert set(closed) == descriptors
    assert publication.evidence_fd == -1
    assert publication.staging_fd == -1
    assert publication.parent_fd == -1
    publication.close()
