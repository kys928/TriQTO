"""Strict artifacts and immutable publication for Phase 9 action datasets."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    json_copy,
    resolve_safe_file,
    strict_json_load,
    strict_json_loads,
    write_strict_json,
)
from triqto.storage.action_schema import (
    ActionCandidateRecordV1,
    ActionRolloutRecord,
)
from triqto.storage.manifest import ManifestReader, ManifestWriter

from .config import ActionEngineConfig, load_action_config, save_action_config
from .constants import (
    ACTION_ARTIFACT_SCHEMA_VERSION,
    ROLLOUT_ARTIFACT_SCHEMA_VERSION,
)
from .identities import (
    action_content_hash,
    circuit_semantic_hash,
    edit_payload,
    rollout_content_hash,
)
from .models import (
    ActionCandidate,
    ActionEdit,
    ActionEngineResult,
    ActionRollout,
    ActionWriteResult,
)
from .validators import (
    validate_action_candidate,
    validate_action_dataset_joins,
    validate_action_rollout,
)

_ROLLOUT_ARRAY_NAMES = {
    "metric_names",
    "baseline_metric_values",
    "candidate_metric_values",
    "improvement_values",
    "outcome_bitstrings",
    "exact_probabilities",
}
_ROLLOUT_METADATA_NAME = "rollout_metadata_json_utf8"


def _strict_json_text(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(
        json_copy(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_bytes_array(payload: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(
        _strict_json_text(payload).encode("utf-8"),
        dtype=np.uint8,
    ).copy()


def _decode_json_bytes(array: np.ndarray, name: str) -> dict[str, Any]:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 array")
    try:
        text = array.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    payload = strict_json_loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must decode to a JSON object")
    return dict(payload)


def _action_payload(candidate: ActionCandidate) -> dict[str, Any]:
    return {
        "artifact_schema_version": ACTION_ARTIFACT_SCHEMA_VERSION,
        "action_id": candidate.action_id,
        "sample_id": candidate.sample_id,
        "graph_pair_id": candidate.graph_pair_id,
        "source_circuit_id": candidate.source_circuit_id,
        "source_run_id": candidate.source_run_id,
        "distortion_id": candidate.distortion_id,
        "edits": [edit_payload(edit) for edit in candidate.edits],
        "generation_sources": list(candidate.generation_sources),
        "risk_score": candidate.risk_score,
        "metadata": candidate.metadata,
        "content_hash": action_content_hash(candidate),
    }


def save_action_artifact(
    candidate: ActionCandidate,
    config: ActionEngineConfig,
    path: str | Path,
) -> Path:
    """Persist one small action definition as strict canonical JSON."""
    validate_action_candidate(candidate, config, require_hash=True)
    target = Path(path)
    write_strict_json(target, _action_payload(candidate))
    return target


def load_action_artifact(
    path: str | Path,
    config: ActionEngineConfig,
    expected_content_hash: str | None = None,
) -> ActionCandidate:
    """Read and validate one action definition without permissive coercion."""
    payload_raw = strict_json_load(Path(path))
    payload = dict(payload_raw) if isinstance(payload_raw, Mapping) else None
    if payload is None:
        raise TypeError("Action artifact must contain a JSON object")
    expected_keys = {
        "artifact_schema_version",
        "action_id",
        "sample_id",
        "graph_pair_id",
        "source_circuit_id",
        "source_run_id",
        "distortion_id",
        "edits",
        "generation_sources",
        "risk_score",
        "metadata",
        "content_hash",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "Action artifact key mismatch; "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}"
        )
    if payload["artifact_schema_version"] != ACTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported action artifact schema version")
    edits_raw = payload["edits"]
    if not isinstance(edits_raw, list):
        raise TypeError("Action artifact edits must be a list")
    edits: list[ActionEdit] = []
    for index, item in enumerate(edits_raw):
        if not isinstance(item, Mapping) or set(item) != {
            "edit_type",
            "qubits",
            "magnitude",
        }:
            raise ValueError(f"Action artifact edit {index} has invalid fields")
        qubits = item["qubits"]
        if not isinstance(qubits, list):
            raise TypeError(f"Action artifact edit {index} qubits must be a list")
        edits.append(
            ActionEdit(
                edit_type=item["edit_type"],
                qubits=tuple(qubits),
                magnitude=item["magnitude"],
            )
        )
    sources = payload["generation_sources"]
    if not isinstance(sources, list):
        raise TypeError("Action artifact generation_sources must be a list")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("Action artifact metadata must be a mapping")
    candidate = ActionCandidate(
        action_id=payload["action_id"],
        sample_id=payload["sample_id"],
        graph_pair_id=payload["graph_pair_id"],
        source_circuit_id=payload["source_circuit_id"],
        source_run_id=payload["source_run_id"],
        distortion_id=payload["distortion_id"],
        edits=tuple(edits),
        generation_sources=tuple(sources),
        risk_score=payload["risk_score"],
        metadata=dict(metadata),
        content_hash=payload["content_hash"],
    )
    validate_action_candidate(candidate, config, require_hash=True)
    if (
        expected_content_hash is not None
        and candidate.content_hash != expected_content_hash
    ):
        raise ValueError("Action artifact content_hash does not match manifest")
    return candidate


def _qpy_module() -> Any:
    try:
        from qiskit import qpy
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Qiskit QPY support is required for Phase 9 persistence") from exc
    return qpy


def save_candidate_circuit(circuit: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        _qpy_module().dump(circuit, handle)
    return target


def load_candidate_circuit(
    path: str | Path,
    expected_circuit_hash: str | None = None,
) -> Any:
    target = Path(path)
    with target.open("rb") as handle:
        circuits = _qpy_module().load(handle)
    if len(circuits) != 1:
        raise ValueError("Candidate circuit QPY must contain exactly one circuit")
    circuit = circuits[0]
    actual_hash = circuit_semantic_hash(circuit)
    if expected_circuit_hash is not None and actual_hash != expected_circuit_hash:
        raise ValueError("Candidate circuit semantic hash does not match manifest")
    return circuit


def _rollout_metadata(rollout: ActionRollout) -> dict[str, Any]:
    return {
        "artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
        "rollout_id": rollout.rollout_id,
        "action_id": rollout.action_id,
        "sample_id": rollout.sample_id,
        "graph_pair_id": rollout.graph_pair_id,
        "candidate_circuit_id": rollout.candidate_circuit_id,
        "clean_target_run_id": rollout.clean_target_run_id,
        "scientific_config_id": rollout.scientific_config_id,
        "rank": rollout.rank,
        "reward": rollout.reward,
        "risk_score": rollout.risk_score,
        "dominates_baseline": rollout.dominates_baseline,
        "primary_metric_nonworsening": rollout.primary_metric_nonworsening,
        "selected": rollout.selected,
        "depth_delta": rollout.depth_delta,
        "gate_delta": rollout.gate_delta,
        "metadata": rollout.metadata,
        "content_hash": rollout_content_hash(rollout),
    }


def save_rollout_artifact(rollout: ActionRollout, path: str | Path) -> Path:
    validate_action_rollout(rollout, require_hash=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        metric_names=rollout.metric_names,
        baseline_metric_values=rollout.baseline_metric_values,
        candidate_metric_values=rollout.candidate_metric_values,
        improvement_values=rollout.improvement_values,
        outcome_bitstrings=rollout.outcome_bitstrings,
        exact_probabilities=rollout.exact_probabilities,
        **{
            _ROLLOUT_METADATA_NAME: _json_bytes_array(
                _rollout_metadata(rollout)
            )
        },
    )
    return target


def load_rollout_artifact(
    path: str | Path,
    candidate_circuit: Any,
    expected_content_hash: str | None = None,
) -> ActionRollout:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        expected_names = _ROLLOUT_ARRAY_NAMES | {_ROLLOUT_METADATA_NAME}
        if set(payload.files) != expected_names:
            raise ValueError(
                "Rollout artifact array-name mismatch; "
                f"missing={sorted(expected_names - set(payload.files))}, "
                f"unexpected={sorted(set(payload.files) - expected_names)}"
            )
        arrays = {name: payload[name].copy() for name in _ROLLOUT_ARRAY_NAMES}
        metadata = _decode_json_bytes(
            payload[_ROLLOUT_METADATA_NAME],
            _ROLLOUT_METADATA_NAME,
        )
    metadata_keys = {
        "artifact_schema_version",
        "rollout_id",
        "action_id",
        "sample_id",
        "graph_pair_id",
        "candidate_circuit_id",
        "clean_target_run_id",
        "scientific_config_id",
        "rank",
        "reward",
        "risk_score",
        "dominates_baseline",
        "primary_metric_nonworsening",
        "selected",
        "depth_delta",
        "gate_delta",
        "metadata",
        "content_hash",
    }
    if set(metadata) != metadata_keys:
        raise ValueError("Rollout artifact metadata-key mismatch")
    if metadata["artifact_schema_version"] != ROLLOUT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported rollout artifact schema version")
    if not isinstance(metadata["metadata"], Mapping):
        raise TypeError("Rollout artifact metadata.metadata must be a mapping")
    rollout = ActionRollout(
        rollout_id=metadata["rollout_id"],
        action_id=metadata["action_id"],
        sample_id=metadata["sample_id"],
        graph_pair_id=metadata["graph_pair_id"],
        candidate_circuit_id=metadata["candidate_circuit_id"],
        clean_target_run_id=metadata["clean_target_run_id"],
        scientific_config_id=metadata["scientific_config_id"],
        rank=metadata["rank"],
        reward=metadata["reward"],
        risk_score=metadata["risk_score"],
        metric_names=arrays["metric_names"],
        baseline_metric_values=arrays["baseline_metric_values"],
        candidate_metric_values=arrays["candidate_metric_values"],
        improvement_values=arrays["improvement_values"],
        outcome_bitstrings=arrays["outcome_bitstrings"],
        exact_probabilities=arrays["exact_probabilities"],
        dominates_baseline=metadata["dominates_baseline"],
        primary_metric_nonworsening=metadata[
            "primary_metric_nonworsening"
        ],
        selected=metadata["selected"],
        candidate_circuit=candidate_circuit,
        depth_delta=metadata["depth_delta"],
        gate_delta=metadata["gate_delta"],
        metadata=dict(metadata["metadata"]),
        content_hash=metadata["content_hash"],
    )
    validate_action_rollout(rollout, require_hash=True)
    if expected_content_hash is not None and rollout.content_hash != expected_content_hash:
        raise ValueError("Rollout artifact content_hash does not match manifest")
    return rollout


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _verify_result_sources(result: ActionEngineResult) -> None:
    actual_phase7 = snapshot_managed_files(
        result.phase7_source_root,
        tuple(entry.reference for entry in result.phase7_snapshot.entries),
    )
    if actual_phase7 != result.phase7_snapshot:
        raise RuntimeError("Phase 7 managed source files changed during Phase 9")
    actual_graph = snapshot_managed_files(
        result.graph_source_root,
        tuple(entry.reference for entry in result.graph_snapshot.entries),
    )
    if actual_graph != result.graph_snapshot:
        raise RuntimeError("Phase 8 managed source files changed during Phase 9")


def write_action_dataset(
    result: ActionEngineResult,
    output_root: str | Path,
) -> ActionWriteResult:
    """Publish a fully validated Phase 9 action dataset into a fresh root."""
    if not isinstance(result, ActionEngineResult):
        raise TypeError("result must be ActionEngineResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Action output root already exists: {output}")
    resolved_output = output.resolve()
    for source_name, source_root in (
        ("Phase 7", result.phase7_source_root),
        ("Phase 8", result.graph_source_root),
    ):
        resolved_source = Path(source_root).resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"Action output root must not be inside the {source_name} source root"
            )
    _verify_result_sources(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging}")

    try:
        for directory in (
            staging / "manifests",
            staging / "artifacts" / "actions",
            staging / "artifacts" / "circuits",
            staging / "artifacts" / "rollouts",
        ):
            directory.mkdir(parents=True, exist_ok=False)

        managed: list[str] = []
        save_action_config(result.config, staging / "action_config.json")
        managed.append("action_config.json")
        write_strict_json(staging / "action_summary.json", result.summary)
        managed.append("action_summary.json")

        rollout_by_action = {rollout.action_id: rollout for rollout in result.rollouts}
        for candidate in result.candidates:
            rollout = rollout_by_action[candidate.action_id]
            action_ref = f"artifacts/actions/{candidate.action_id}.json"
            circuit_ref = (
                f"artifacts/circuits/{rollout.candidate_circuit_id}.qpy"
            )
            save_action_artifact(candidate, result.config, staging / action_ref)
            save_candidate_circuit(rollout.candidate_circuit, staging / circuit_ref)
            managed.extend((action_ref, circuit_ref))
        for rollout in result.rollouts:
            rollout_ref = f"artifacts/rollouts/{rollout.rollout_id}.npz"
            save_rollout_artifact(rollout, staging / rollout_ref)
            managed.append(rollout_ref)

        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "action_candidate_manifest",
            result.candidate_records,
            overwrite=False,
        )
        writer.write_records(
            "action_rollout_manifest",
            result.rollout_records,
            overwrite=False,
        )
        managed.extend(
            (
                "manifests/action_candidate_manifest.parquet",
                "manifests/action_rollout_manifest.parquet",
            )
        )

        persisted_config = load_action_config(staging / "action_config.json")
        if persisted_config != result.config:
            raise ValueError("Persisted action config does not match conversion config")
        reader = ManifestReader(staging / "manifests")
        candidate_records = reader.read_typed_records(
            "action_candidate_manifest",
            ActionCandidateRecordV1,
        )
        rollout_records = reader.read_typed_records(
            "action_rollout_manifest",
            ActionRolloutRecord,
        )
        loaded_candidates: dict[str, ActionCandidate] = {}
        loaded_circuits: dict[str, Any] = {}
        for record in candidate_records:
            record.validate()
            candidate = load_action_artifact(
                resolve_safe_file(
                    staging,
                    record.action_ref,
                    f"ActionCandidateRecordV1 {record.action_id}.action_ref",
                ),
                persisted_config,
                record.content_hash,
            )
            circuit = load_candidate_circuit(
                resolve_safe_file(
                    staging,
                    record.circuit_ref,
                    f"ActionCandidateRecordV1 {record.action_id}.circuit_ref",
                ),
                record.circuit_hash,
            )
            if candidate.action_id in loaded_candidates:
                raise ValueError(f"Duplicate persisted action {candidate.action_id}")
            if record.candidate_circuit_id in loaded_circuits:
                raise ValueError(
                    f"Duplicate persisted candidate circuit {record.candidate_circuit_id}"
                )
            loaded_candidates[candidate.action_id] = candidate
            loaded_circuits[record.candidate_circuit_id] = circuit

        loaded_rollouts: dict[str, ActionRollout] = {}
        for record in rollout_records:
            record.validate()
            circuit = loaded_circuits.get(record.candidate_circuit_id)
            if circuit is None:
                raise ValueError(
                    f"Rollout {record.rollout_id} references missing candidate circuit"
                )
            rollout = load_rollout_artifact(
                resolve_safe_file(
                    staging,
                    record.rollout_ref,
                    f"ActionRolloutRecord {record.rollout_id}.rollout_ref",
                ),
                circuit,
                record.content_hash,
            )
            if rollout.rollout_id in loaded_rollouts:
                raise ValueError(f"Duplicate persisted rollout {rollout.rollout_id}")
            loaded_rollouts[rollout.rollout_id] = rollout

        validate_action_dataset_joins(
            candidate_records,
            rollout_records,
            candidates_by_id=loaded_candidates,
            rollouts_by_id=loaded_rollouts,
            config=persisted_config,
        )

        if len(set(managed)) != len(managed):
            raise ValueError("Managed Phase 9 file inventory contains duplicates")
        expected_before_marker = set(managed)
        actual_before_marker = _relative_file_set(staging)
        if actual_before_marker != expected_before_marker:
            raise ValueError(
                "Staging action dataset inventory mismatch; "
                f"missing={sorted(expected_before_marker - actual_before_marker)}, "
                f"unexpected={sorted(actual_before_marker - expected_before_marker)}"
            )

        managed_files = tuple(sorted([*managed, "action_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": (
                result.source_scientific_generation_id
            ),
            "graph_conversion_id": result.graph_conversion_id,
            "action_engine_id": result.action_engine_id,
            "operational_config_id": result.operational_config_id,
            "action_schema_id": result.action_schema_id,
            "candidate_count": len(result.candidates),
            "rollout_count": len(result.rollouts),
            "phase7_snapshot_hash": result.phase7_snapshot.aggregate_sha256,
            "graph_snapshot_hash": result.graph_snapshot.aggregate_sha256,
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "action_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed action file inventory does not match staging")
        persisted_marker = strict_json_load(staging / "action_complete.json")
        if persisted_marker != completion:
            raise ValueError("action_complete.json content mismatch")
        _verify_result_sources(result)

        if output.exists():
            raise FileExistsError(
                f"Action output root appeared during publication: {output}"
            )
        os.replace(staging, output)

        manifest_paths = tuple(
            sorted(
                (
                    output / "manifests" / "action_candidate_manifest.parquet",
                    output / "manifests" / "action_rollout_manifest.parquet",
                ),
                key=lambda path: path.as_posix(),
            )
        )
        artifact_paths = tuple(
            sorted(
                [
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ],
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                [output / reference for reference in managed_files],
                key=lambda path: path.as_posix(),
            )
        )
        return ActionWriteResult(
            output_root=output,
            action_complete_path=output / "action_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            candidate_count=len(result.candidates),
            rollout_count=len(result.rollouts),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "load_action_artifact",
    "load_candidate_circuit",
    "load_rollout_artifact",
    "save_action_artifact",
    "save_candidate_circuit",
    "save_rollout_artifact",
    "write_action_dataset",
]
