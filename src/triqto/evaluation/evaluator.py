"""Executable Phase 15 evaluation for trained Phase 14 smoke artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import uuid

import numpy as np
import torch
import yaml

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.model import TriQTOModel, model_config_from_dict
from triqto.training_views.constants import TRAINING_ITEM_METADATA_ARRAY_NAME

from triqto.training import (
    TrainingDataSpec,
    collate_training_examples,
    compute_supervised_losses,
    load_completed_training_view_dataset,
    load_training_checkpoint,
    load_training_examples,
    training_config_from_dict,
    verify_training_view_snapshot,
)

from .baseline_comparison import build_comparison_records, validate_unique_comparisons
from .generalization_tests import audit_backend_holdout_for_phase15, load_backend_holdout_config

PHASE15_SCHEMA_VERSION = "triqto.phase15.evaluation.v1"


@dataclass(frozen=True, slots=True)
class Phase15EvaluationConfig:
    run_name: str = "phase15_smoke"
    split: str = "test"
    tasks: tuple[str, ...] = ("diagnosis",)
    evidence_tier: str = "fake_backend_fixture"
    evaluation_kind: str = "smoke"
    baseline_ids: tuple[str, ...] = ("trained_triqto", "random_control")
    stream_removals: tuple[str, ...] = ("backend",)
    backend_holdout_config: str | None = None
    require_backend_holdout_audit: bool = False

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must be nonblank")
        if self.split not in {"test", "iid_test"}:
            raise ValueError("Phase 15 smoke evaluator only supports untouched test or iid_test splits")
        tasks = tuple(str(value) for value in self.tasks)
        if not tasks or any(not value for value in tasks):
            raise ValueError("tasks must contain nonblank task names")
        if self.evidence_tier not in {"ideal_simulator", "fake_backend_fixture"}:
            raise ValueError("unsupported Phase 15 evidence tier")
        if self.evaluation_kind not in {"smoke", "engineering_validation", "exact_fake_backend_axis_holdout"}:
            raise ValueError("evaluation_kind must be smoke/engineering_validation/exact_fake_backend_axis_holdout")
        if any("research" in str(value).lower() for value in (self.evaluation_kind, self.run_name)):
            raise ValueError("Phase 15 smoke evaluator must not claim research-quality evidence")
        baselines = tuple(str(value) for value in self.baseline_ids)
        if not baselines or len(set(baselines)) != len(baselines):
            raise ValueError("baseline_ids must be nonempty and unique")
        removals = tuple(str(value) for value in self.stream_removals)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "baseline_ids", baselines)
        object.__setattr__(self, "stream_removals", removals)


def load_phase15_config(path: str | Path) -> Phase15EvaluationConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 15 evaluation config must be a mapping")
    allowed = set(Phase15EvaluationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown Phase 15 config fields: {sorted(extra)}")
    data = dict(payload)
    for key in ("tasks", "baseline_ids", "stream_removals"):
        if key in data:
            data[key] = tuple(data[key])
    return Phase15EvaluationConfig(**data)


def _json_hash(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_marker(root: Path) -> dict[str, Any]:
    marker = root / "training_complete.json"
    if not marker.exists():
        raise FileNotFoundError(f"Phase 14 training marker missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise ValueError("Phase 14 training marker is not complete")
    summary_path = root / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if payload.get("test_split_used_for_optimization") is not False:
        raise ValueError("Phase 14 marker must show test split was not optimized")
    if summary.get("test_split_evaluated") is not False:
        raise ValueError("Phase 14 summary must show test split was untouched before Phase 15")
    return payload


def _sample_records_for_backend_audit(dataset: Any, item_records: Sequence[Any]) -> list[dict[str, object]]:
    records = []
    seen: set[str] = set()
    for record in item_records:
        sample_id = str(record.entity_id)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        # Phase 12 item metadata carries sample/backend facts for diagnosis rows.
        item_path = dataset.root / record.artifact_ref
        with np.load(item_path, allow_pickle=False) as payload:
            meta = json.loads(payload[TRAINING_ITEM_METADATA_ARRAY_NAME].tobytes().decode("utf-8"))["metadata"]
        records.append(
            {
                "sample_id": sample_id,
                "clean_circuit_id": record.split_group_id,
                "backend_id": meta.get("backend_id"),
                "backend_assignment_level": meta.get("backend_assignment_level"),
                "backend_available": bool(meta.get("backend_available", False)),
            }
        )
    return records


def _basis_key(example: Any) -> str:
    born = example.model_batch.born
    if born is None or born.measurement_basis_codes is None or born.measurement_basis_codes.numel() == 0:
        return "Z"
    row = born.measurement_basis_codes[0].detach().cpu().tolist()
    names = {0: "Z", 1: "X", 2: "Y"}
    return "".join(names.get(int(value), "?") for value in row)


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Phase 15 metric is non-finite")
    return float(value)


def run_phase15_evaluation(
    *,
    training_view_root: str | Path,
    training_root: str | Path,
    checkpoint: str | Path,
    output_root: str | Path,
    config: Phase15EvaluationConfig,
    phase7_root: str | Path | None = None,
) -> dict[str, Any]:
    view_root = Path(training_view_root)
    train_root = Path(training_root)
    ckpt_path = Path(checkpoint)
    out = Path(output_root)
    if out.exists():
        raise FileExistsError(f"Phase 15 output root already exists: {out}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase 15 checkpoint missing: {ckpt_path}")
    marker = _read_marker(train_root)
    dataset = load_completed_training_view_dataset(view_root)
    verify_training_view_snapshot(dataset)

    probe_metadata = load_training_checkpoint(ckpt_path, expected_training_run_id=marker["training_run_id"])
    if probe_metadata["training_view_dataset_id"] != dataset.training_view_dataset_id:
        raise ValueError("Checkpoint training-view dataset mismatch")
    training_config = training_config_from_dict(probe_metadata["training_config"])
    model_config = model_config_from_dict(probe_metadata["model_config"])
    spec = TrainingDataSpec.from_dict(probe_metadata["data_spec"])
    model = TriQTOModel(model_config).eval()
    restored = load_training_checkpoint(ckpt_path, model=model, expected_training_run_id=marker["training_run_id"])
    if restored["checkpoint_id"] != probe_metadata["checkpoint_id"]:
        raise ValueError("Checkpoint restore identity mismatch")

    item_records = [row for row in dataset.item_records if row.split == config.split and row.task in config.tasks]
    if not item_records:
        raise ValueError(f"Phase 15 found no untouched {config.split} rows for tasks {config.tasks}")
    if any(row.split in {"train", "validation"} for row in item_records):
        raise ValueError("Phase 15 attempted to evaluate train/validation rows")
    examples = load_training_examples(dataset, tasks=config.tasks, split=config.split, spec=spec, phase7_root=phase7_root, allow_evaluation_splits=True)
    if len(examples) != len(item_records):
        raise ValueError("Phase 15 example/record count mismatch")

    backend_audit = None
    if config.require_backend_holdout_audit:
        if not config.backend_holdout_config:
            raise ValueError("backend holdout audit requested without config")
        backend_config = load_backend_holdout_config(config.backend_holdout_config)
        backend_records = _sample_records_for_backend_audit(dataset, dataset.item_records)
        backend_audit = audit_backend_holdout_for_phase15(backend_records, backend_config)
        if config.split != "test":
            raise ValueError("backend holdout evaluation must use test split")

    losses: list[dict[str, float]] = []
    basis_counts: dict[str, int] = {}
    diagnosis_included = diagnosis_excluded = action_included = action_excluded = 0
    softmax_confidences: list[float] = []
    uncertainty_values: list[float] = []
    with torch.no_grad():
        for example in examples:
            batch = collate_training_examples([example])
            output = model(batch.model_batch)
            loss_values = compute_supervised_losses(output, batch, training_config.loss)
            losses.append({name: _finite(float(value.detach().cpu())) for name, value in loss_values.items()})
            basis_counts[_basis_key(example)] = basis_counts.get(_basis_key(example), 0) + 1
            identifiable = example.metadata.get("identifiability_status") == "identifiable"
            if example.task == "diagnosis" and bool(batch.targets.diagnosis.class_mask.any()) and identifiable:
                diagnosis_included += 1
                softmax_confidences.append(float(torch.softmax(output.distortion.class_logits, dim=-1).max(dim=-1).values.mean().cpu()))
            elif example.task == "diagnosis":
                diagnosis_excluded += 1
            if example.task == "action_ranking" and bool(batch.targets.action.candidate_target_mask.any()) and identifiable:
                action_included += 1
            elif example.task == "action_ranking":
                action_excluded += 1
            uncertainty_values.extend(float(v) for v in output.uncertainty.log_variance.reshape(-1).detach().cpu().tolist())

    metric_names = sorted(losses[0])
    metrics = {f"mean_{name}": _finite(float(np.mean([row[name] for row in losses]))) for name in metric_names}
    run_id = make_deterministic_id(
        "phase15run",
        {
            "schema": PHASE15_SCHEMA_VERSION,
            "training_run_id": marker["training_run_id"],
            "checkpoint_id": restored["checkpoint_id"],
            "split": config.split,
            "tasks": list(config.tasks),
            "evidence_tier": config.evidence_tier,
            "evaluation_kind": config.evaluation_kind,
        },
    )
    comparisons = []
    for example in examples:
        comparisons.extend(
            build_comparison_records(
                run_id=run_id,
                sample_id=example.entity_id,
                baselines=config.baseline_ids,
                tasks=(example.task,),
                view_id=example.view_item_id,
                execution_mode=config.evidence_tier,
            )
        )
    validate_unique_comparisons(comparisons)
    sensitivity = [
        {
            "analysis_id": make_deterministic_id("sensitivity", {"run_id": run_id, "stream_removed": stream}),
            "stream_removed": stream,
            "label": "inference_sensitivity_analysis_not_causal_ablation",
        }
        for stream in config.stream_removals
    ]
    summary: dict[str, Any] = {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "phase15_run_id": run_id,
        "evaluation_kind": config.evaluation_kind,
        "claim_scope": "smoke engineering validation; not research-quality evidence",
        "evidence_tier": config.evidence_tier,
        "physical_hardware": False,
        "training_run_id": marker["training_run_id"],
        "training_view_dataset_id": dataset.training_view_dataset_id,
        "checkpoint_id": restored["checkpoint_id"],
        "checkpoint_content_hash": restored["content_hash"],
        "split": config.split,
        "split_semantics": "iid_test" if config.split == "iid_test" else ("exact_axis_holdout_test" if backend_audit else "held_out_test"),
        "tasks": list(config.tasks),
        "test_row_count": len(examples),
        "metrics": metrics,
        "basis_setting_counts": dict(sorted(basis_counts.items())),
        "identifiability": {
            "diagnosis_included": diagnosis_included,
            "diagnosis_excluded_unidentifiable": diagnosis_excluded,
            "action_included": action_included,
            "action_excluded_unidentifiable": action_excluded,
            "exclusion_reason": "Phase 12 target masks mark unidentifiable targets unavailable",
        },
        "baseline_comparison_count": len(comparisons),
        "baseline_comparison_ids": [row["comparison_id"] for row in comparisons],
        "uncertainty_head_diagnostics": {
            "mean_log_variance": _finite(float(np.mean(uncertainty_values))) if uncertainty_values else None,
            "count": len(uncertainty_values),
            "not_calibration_evidence": True,
        },
        "softmax_confidence_diagnostics": {
            "mean_max_softmax": _finite(float(np.mean(softmax_confidences))) if softmax_confidences else None,
            "count": len(softmax_confidences),
            "separate_from_uncertainty_head": True,
        },
        "sensitivity_analyses": sensitivity,
        "backend_holdout_audit": backend_audit,
        "topology_loss_weight": 0.0,
        "large_artifacts_committed": False,
    }
    summary["summary_content_hash"] = _json_hash(summary)
    card = {
        "phase15_run_id": run_id,
        "label": "TriQTO CPU smoke / engineering validation",
        "seed": training_config.seed,
        "dependency_profile": "CPU; repository pinned requirements/constraints",
        "commands": ["scripts/run_cpu_smoke_workflow.py --output <dir>"],
        "artifact_ids": {
            "training_run_id": marker["training_run_id"],
            "training_view_dataset_id": dataset.training_view_dataset_id,
            "checkpoint_id": restored["checkpoint_id"],
        },
        "metrics": metrics,
        "limitations": [
            "not research-quality evidence",
            "no physical hardware used",
            "no calibration, OOD, topology-benefit, or correction-success claim",
        ],
    }
    card["card_content_hash"] = _json_hash(card)
    manifest = {
        "schema_version": "triqto.phase15.manifest.v1",
        "phase15_run_id": run_id,
        "managed_files": ["phase15_summary.json", "phase15_card.json"],
        "summary_content_hash": summary["summary_content_hash"],
        "card_content_hash": card["card_content_hash"],
    }
    manifest["manifest_content_hash"] = _json_hash(manifest)

    staging = out.parent / f".{out.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "phase15_summary.json", summary)
        _write_json(staging / "phase15_card.json", card)
        _write_json(staging / "phase15_complete.json", manifest)
        staging.replace(out)
    except Exception:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise
    return {"summary": summary, "card": card, "manifest": manifest}


def load_phase15_result(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "phase15_complete.json").read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_content_hash")
    if _json_hash(manifest) != expected_manifest_hash:
        raise ValueError("Phase 15 manifest content hash mismatch")
    manifest["manifest_content_hash"] = expected_manifest_hash
    summary = json.loads((base / "phase15_summary.json").read_text(encoding="utf-8"))
    expected_summary_hash = summary.pop("summary_content_hash")
    if _json_hash(summary) != expected_summary_hash:
        raise ValueError("Phase 15 summary content hash mismatch")
    summary["summary_content_hash"] = expected_summary_hash
    card = json.loads((base / "phase15_card.json").read_text(encoding="utf-8"))
    expected_card_hash = card.pop("card_content_hash")
    if _json_hash(card) != expected_card_hash:
        raise ValueError("Phase 15 card content hash mismatch")
    card["card_content_hash"] = expected_card_hash
    return {"manifest": manifest, "summary": summary, "card": card}


__all__ = ["Phase15EvaluationConfig", "load_phase15_config", "load_phase15_result", "run_phase15_evaluation"]
