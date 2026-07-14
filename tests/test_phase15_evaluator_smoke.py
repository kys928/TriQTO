from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from triqto.evaluation import Phase15EvaluationConfig, load_phase15_result, run_phase15_evaluation


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("phase15_smoke") / "run"
    subprocess.run([sys.executable, "scripts/run_cpu_smoke_workflow.py", "--output", str(root)], check=True)
    return root


def _checkpoint(root: Path) -> Path:
    return next((root / "phase14" / "artifacts" / "checkpoints").glob("final-epoch-*.npz"))


def test_cpu_smoke_workflow_publishes_phase15_manifest_and_card(smoke_root: Path) -> None:
    result = load_phase15_result(smoke_root / "phase15")
    summary = result["summary"]
    card = result["card"]
    assert summary["evaluation_kind"] == "exact_fake_backend_axis_holdout"
    assert summary["claim_scope"].startswith("smoke engineering validation")
    assert summary["physical_hardware"] is False
    assert summary["topology_loss_weight"] == 0.0
    assert summary["test_row_count"] > 0
    assert summary["split_semantics"] == "exact_axis_holdout_test"
    assert card["label"] == "TriQTO CPU smoke / engineering validation"
    workflow = json.loads((smoke_root / "smoke_workflow_manifest.json").read_text())
    assert workflow["label"] == "smoke engineering validation"
    assert workflow["evidence_tier"] == "fake_backend_fixture"


def test_checkpoint_restore_and_deterministic_rerun(smoke_root: Path, tmp_path: Path) -> None:
    first = load_phase15_result(smoke_root / "phase15")["summary"]
    output = tmp_path / "phase15_rerun"
    rerun = run_phase15_evaluation(
        training_view_root=smoke_root / "phase12",
        training_root=smoke_root / "phase14",
        checkpoint=_checkpoint(smoke_root),
        output_root=output,
        config=Phase15EvaluationConfig(
            run_name="phase15_cpu_smoke",
            evaluation_kind="exact_fake_backend_axis_holdout",
            split="test",
            tasks=("diagnosis",),
            evidence_tier="fake_backend_fixture",
            baseline_ids=("trained_triqto", "random_control", "privileged_rule_only_control"),
            backend_holdout_config="configs/eval/phase15_backend_holdout.yaml",
            require_backend_holdout_audit=True,
        ),
        phase7_root=smoke_root / "phase7",
    )["summary"]
    assert rerun["phase15_run_id"] == first["phase15_run_id"]
    assert rerun["checkpoint_id"] == first["checkpoint_id"]
    assert rerun["metrics"] == first["metrics"]


def test_phase15_rejects_touched_test_metadata(smoke_root: Path, tmp_path: Path) -> None:
    copied = tmp_path / "phase14_touched"
    shutil.copytree(smoke_root / "phase14", copied)
    summary_path = copied / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["test_split_evaluated"] = True
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="test split was untouched"):
        run_phase15_evaluation(
            training_view_root=smoke_root / "phase12",
            training_root=copied,
            checkpoint=_checkpoint(smoke_root),
            output_root=tmp_path / "phase15_bad",
            config=Phase15EvaluationConfig(backend_holdout_config="configs/eval/phase15_backend_holdout.yaml", require_backend_holdout_audit=True),
            phase7_root=smoke_root / "phase7",
        )


def test_phase15_multitask_baselines_uncertainty_basis_and_identifiability(smoke_root: Path, tmp_path: Path) -> None:
    result = run_phase15_evaluation(
        training_view_root=smoke_root / "phase12",
        training_root=smoke_root / "phase14",
        checkpoint=_checkpoint(smoke_root),
        output_root=tmp_path / "phase15_multitask",
        config=Phase15EvaluationConfig(
            run_name="phase15_multitask_smoke",
            evaluation_kind="exact_fake_backend_axis_holdout",
            split="test",
            tasks=("diagnosis", "action_ranking"),
            evidence_tier="fake_backend_fixture",
            baseline_ids=("trained_triqto", "random_control", "privileged_rule_only_control"),
            stream_removals=("backend", "born"),
            backend_holdout_config="configs/eval/phase15_backend_holdout.yaml",
            require_backend_holdout_audit=True,
        ),
        phase7_root=smoke_root / "phase7",
    )["summary"]
    assert result["baseline_comparison_count"] == result["test_row_count"] * 3
    assert len(result["baseline_comparison_ids"]) == len(set(result["baseline_comparison_ids"]))
    assert result["basis_setting_counts"] == {"Z": result["test_row_count"]}
    assert result["identifiability"]["diagnosis_excluded_unidentifiable"] > 0
    assert result["uncertainty_head_diagnostics"]["count"] > 0
    assert result["softmax_confidence_diagnostics"]["separate_from_uncertainty_head"] is True
    assert all(row["label"] == "inference_sensitivity_analysis_not_causal_ablation" for row in result["sensitivity_analyses"])
    assert result["backend_holdout_audit"]["not_physical_hardware_generalization"] is True


def test_phase15_artifact_identity_atomic_publication_and_tamper_rejection(smoke_root: Path, tmp_path: Path) -> None:
    assert load_phase15_result(smoke_root / "phase15")["manifest"]["managed_files"] == ["phase15_summary.json", "phase15_card.json"]
    tampered = tmp_path / "tampered"
    shutil.copytree(smoke_root / "phase15", tampered)
    summary_path = tampered / "phase15_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics"]["mean_total"] += 1.0
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="summary content hash"):
        load_phase15_result(tampered)
    with pytest.raises(FileExistsError):
        run_phase15_evaluation(
            training_view_root=smoke_root / "phase12",
            training_root=smoke_root / "phase14",
            checkpoint=_checkpoint(smoke_root),
            output_root=smoke_root / "phase15",
            config=Phase15EvaluationConfig(),
            phase7_root=smoke_root / "phase7",
        )
