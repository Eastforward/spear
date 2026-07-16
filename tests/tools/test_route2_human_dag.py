from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from tools import route2_human_dag as dag


def _immutable_evidence(root: Path, name: str, payload: str = "ok\n") -> Path:
    path = root / name
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o444)
    return path


def _finish_current_stage(run_root: Path, evidence_root: Path) -> dict:
    state = dag.validate_run(run_root)
    action = dag.next_action(state)
    assert action is not None
    evidence = _immutable_evidence(
        evidence_root,
        f"{action['case_id']}.{action['stage']}.{state['next_sequence']}.json",
    )
    dag.append_stage_event(
        run_root,
        case_id=action["case_id"],
        stage=action["stage"],
        status=(
            "agent_qa_passed_pending_user_acceptance"
            if action["stage"] == "qualified_candidate"
            else "succeeded"
        ),
        evidence_paths=[evidence],
    )
    return action


def test_fixed_case_order_and_stage_profiles_are_exact():
    assert dag.CASE_ORDER == (
        "rocketbox_male_adult_01",
        "rocketbox_female_adult_01",
        "tall_man",
        "short_woman",
        "glasses",
        "hat",
        "short_sleeve_color",
        "trousers",
        "shoes",
    )
    assert dag.ATTRIBUTE_CASES == dag.CASE_ORDER[2:]
    assert dag.BASE_FOR_CASE == {
        "rocketbox_male_adult_01": "rocketbox_male_adult_01",
        "rocketbox_female_adult_01": "rocketbox_female_adult_01",
        "tall_man": "rocketbox_male_adult_01",
        "short_woman": "rocketbox_female_adult_01",
        "glasses": "rocketbox_male_adult_01",
        "hat": "rocketbox_female_adult_01",
        "short_sleeve_color": "rocketbox_male_adult_01",
        "trousers": "rocketbox_female_adult_01",
        "shoes": "rocketbox_male_adult_01",
    }
    assert dag.BASE_STAGES == (
        "pixal_source",
        "tokenrig_static",
        "retarget_walk_idle",
        "dynamic_media",
        "agent_visual_qa",
        "qualified_candidate",
    )
    assert dag.ATTRIBUTE_STAGES == (
        "flux2_edit",
        "agent_2d_qa",
        "pixal3d",
        "tokenrig_static",
        "retarget_walk_idle",
        "dynamic_media",
        "agent_visual_qa",
        "qualified_candidate",
    )


def test_create_run_is_immutable_no_replace_and_starts_with_male(tmp_path):
    run_root = tmp_path / "run"
    manifest = dag.create_run(run_root)
    assert manifest == run_root / dag.RUN_MANIFEST_NAME
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o444
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == dag.RUN_SCHEMA
    assert payload["case_order"] == list(dag.CASE_ORDER)
    assert payload["user_acceptance"] == "pending_user_review"
    assert "user_approved" not in manifest.read_text(encoding="utf-8")
    state = dag.validate_run(run_root)
    assert dag.next_action(state) == {
        "case_id": "rocketbox_male_adult_01",
        "base_avatar_id": "rocketbox_male_adult_01",
        "stage": "pixal_source",
    }
    with pytest.raises(FileExistsError):
        dag.create_run(run_root)


def test_append_only_hash_chain_resumes_without_repeating_succeeded_stage(tmp_path):
    run_root = tmp_path / "run"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    dag.create_run(run_root)
    first = _finish_current_stage(run_root, evidence_root)
    assert first == {
        "case_id": "rocketbox_male_adult_01",
        "base_avatar_id": "rocketbox_male_adult_01",
        "stage": "pixal_source",
    }
    state = dag.validate_run(run_root)
    assert state["next_sequence"] == 2
    assert dag.next_action(state)["stage"] == "tokenrig_static"
    event = state["events"][0]
    assert event["sequence"] == 1
    assert event["previous_event_sha256"] is None
    assert event["status"] == "succeeded"
    assert stat.S_IMODE(Path(event["event_path"]).stat().st_mode) == 0o444


def test_event_and_bound_evidence_tampering_fail_closed(tmp_path):
    run_root = tmp_path / "run"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    dag.create_run(run_root)
    _finish_current_stage(run_root, evidence_root)
    state = dag.validate_run(run_root)
    evidence = Path(state["events"][0]["evidence"][0]["path"])
    evidence.chmod(0o644)
    evidence.write_text("tampered\n", encoding="utf-8")
    evidence.chmod(0o444)
    with pytest.raises(dag.DagError, match="evidence|changed|hash"):
        dag.validate_run(run_root)

    evidence.chmod(0o644)
    evidence.write_text("ok\n", encoding="utf-8")
    evidence.chmod(0o444)
    event_path = Path(state["events"][0]["event_path"])
    event_path.chmod(0o644)
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["stage"] = "qualified_candidate"
    event_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    event_path.chmod(0o444)
    with pytest.raises(dag.DagError, match="event|chain|changed|hash"):
        dag.validate_run(run_root)


def test_cannot_skip_stage_or_start_female_before_male_qualification(tmp_path):
    run_root = tmp_path / "run"
    evidence = _immutable_evidence(tmp_path, "evidence.json")
    dag.create_run(run_root)
    with pytest.raises(dag.DagError, match="expected|stage|serial"):
        dag.append_stage_event(
            run_root,
            case_id="rocketbox_female_adult_01",
            stage="pixal_source",
            status="succeeded",
            evidence_paths=[evidence],
        )
    with pytest.raises(dag.DagError, match="expected|stage|serial"):
        dag.append_stage_event(
            run_root,
            case_id="rocketbox_male_adult_01",
            stage="qualified_candidate",
            status="succeeded",
            evidence_paths=[evidence],
        )


def test_rejected_attribute_is_terminal_and_scheduler_continues(tmp_path):
    run_root = tmp_path / "run"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    dag.create_run(run_root)

    # Complete both base canaries and reach the first attribute.
    while True:
        action = dag.next_action(dag.validate_run(run_root))
        assert action is not None
        if action["case_id"] == "tall_man":
            break
        _finish_current_stage(run_root, evidence_root)

    assert action["stage"] == "flux2_edit"
    rejected = _immutable_evidence(evidence_root, "tall_man.rejected.json")
    dag.append_stage_event(
        run_root,
        case_id="tall_man",
        stage="flux2_edit",
        status="rejected",
        evidence_paths=[rejected],
        reason_code="flux2_non_target_drift",
    )
    state = dag.validate_run(run_root)
    assert state["cases"]["tall_man"]["terminal_status"] == "rejected"
    assert dag.next_action(state) == {
        "case_id": "short_woman",
        "base_avatar_id": "rocketbox_female_adult_01",
        "stage": "flux2_edit",
    }
    with pytest.raises(dag.DagError, match="expected|terminal|serial"):
        dag.append_stage_event(
            run_root,
            case_id="tall_man",
            stage="agent_2d_qa",
            status="succeeded",
            evidence_paths=[rejected],
        )


def test_qualified_terminal_requires_agent_pending_user_status(tmp_path):
    run_root = tmp_path / "run"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    dag.create_run(run_root)
    while True:
        state = dag.validate_run(run_root)
        action = dag.next_action(state)
        assert action is not None
        if action["stage"] == "qualified_candidate":
            break
        _finish_current_stage(run_root, evidence_root)
    evidence = _immutable_evidence(evidence_root, "male.qualified.json")
    with pytest.raises(dag.DagError, match="agent_qa_passed_pending_user_acceptance"):
        dag.append_stage_event(
            run_root,
            case_id=action["case_id"],
            stage=action["stage"],
            status="succeeded",
            evidence_paths=[evidence],
        )
    dag.append_stage_event(
        run_root,
        case_id=action["case_id"],
        stage=action["stage"],
        status="agent_qa_passed_pending_user_acceptance",
        evidence_paths=[evidence],
    )
    assert dag.next_action(dag.validate_run(run_root))["case_id"] == (
        "rocketbox_female_adult_01"
    )


def test_evidence_must_be_direct_readonly_regular_file(tmp_path):
    run_root = tmp_path / "run"
    dag.create_run(run_root)
    writable = tmp_path / "writable.json"
    writable.write_text("bad\n", encoding="utf-8")
    with pytest.raises(dag.DagError, match="0444|read-only|immutable"):
        dag.append_stage_event(
            run_root,
            case_id="rocketbox_male_adult_01",
            stage="pixal_source",
            status="succeeded",
            evidence_paths=[writable],
        )
    link = tmp_path / "link.json"
    target = _immutable_evidence(tmp_path, "target.json")
    link.symlink_to(target)
    with pytest.raises(dag.DagError, match="symlink|direct regular"):
        dag.append_stage_event(
            run_root,
            case_id="rocketbox_male_adult_01",
            stage="pixal_source",
            status="succeeded",
            evidence_paths=[link],
        )


def test_unowned_staging_file_is_not_silently_ignored_on_resume(tmp_path):
    run_root = tmp_path / "run"
    dag.create_run(run_root)
    stale = run_root / "events" / ".unexpected.staging"
    stale.write_text("partial\n", encoding="utf-8")
    with pytest.raises(dag.DagError, match="event|sequence|name"):
        dag.validate_run(run_root)


def test_male_rejection_blocks_female_and_is_explicit(tmp_path):
    run_root = tmp_path / "run"
    dag.create_run(run_root)
    evidence = _immutable_evidence(tmp_path, "male_static_rejected.json")
    dag.append_stage_event(
        run_root,
        case_id="rocketbox_male_adult_01",
        stage="pixal_source",
        status="rejected",
        evidence_paths=[evidence],
        reason_code="male_static_rejected",
    )
    state = dag.validate_run(run_root)
    assert dag.next_action(state) is None
    assert state["complete"] is False
    assert state["blocked_reason"] == "male_base_rejected"


def test_all_nine_qualified_results_form_a_complete_resumable_chain(tmp_path):
    run_root = tmp_path / "run"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    dag.create_run(run_root)
    while True:
        state = dag.validate_run(run_root)
        action = dag.next_action(state)
        if action is None:
            break
        _finish_current_stage(run_root, evidence_root)
    state = dag.validate_run(run_root)
    assert state["complete"] is True
    assert state["blocked_reason"] is None
    assert dag.next_action(state) is None
    assert len(state["events"]) == 68
    assert all(
        state["cases"][case_id]["terminal_status"]
        == "agent_qa_passed_pending_user_acceptance"
        for case_id in dag.CASE_ORDER
    )


def test_cli_create_status_and_append_have_unique_machine_sentinels(
    tmp_path, capsys
):
    run_root = tmp_path / "run"
    assert dag.main(["create", "--run-root", str(run_root)]) == 0
    created = capsys.readouterr().out.strip().splitlines()
    assert len(created) == 1
    assert created[0].startswith("ROUTE2_DAG_CREATED ")

    assert dag.main(["status", "--run-root", str(run_root)]) == 0
    status_lines = capsys.readouterr().out.strip().splitlines()
    assert status_lines[-1] == (
        "ROUTE2_DAG_NEXT case=rocketbox_male_adult_01 stage=pixal_source"
    )
    assert sum(line.startswith("ROUTE2_DAG_NEXT ") for line in status_lines) == 1

    evidence = _immutable_evidence(tmp_path, "cli_evidence.json")
    assert dag.main(
        [
            "append",
            "--run-root",
            str(run_root),
            "--case-id",
            "rocketbox_male_adult_01",
            "--stage",
            "pixal_source",
            "--status",
            "succeeded",
            "--evidence",
            str(evidence),
        ]
    ) == 0
    appended = capsys.readouterr().out.strip().splitlines()
    assert len(appended) == 1
    assert appended[0].startswith("ROUTE2_DAG_EVENT_APPENDED ")
    assert dag.next_action(dag.validate_run(run_root))["stage"] == "tokenrig_static"
