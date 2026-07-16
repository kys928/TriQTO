"""Compressed sharded persistence for large Phase 9 action datasets.

The legacy Phase 9 writer creates three files per candidate. Large campaigns can
therefore create millions of small files. This module preserves the exact logical
records and hashes while packing candidate, circuit, and rollout payloads into a
bounded deterministic set of ZIP shards.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import hashlib
from io import BytesIO
import os
from pathlib import Path
import shutil
from typing import Any
import uuid
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    json_copy,
    strict_json_load,
    strict_json_loads,
    write_strict_json,
)
from triqto.storage.action_schema import ActionCandidateRecordV1, ActionRolloutRecord
from triqto.storage.manifest import ManifestReader, ManifestWriter

from .artifacts import (
    _ROLLOUT_ARRAY_NAMES,
    _ROLLOUT_METADATA_NAME,
    _action_payload,
    _decode_json_bytes,
    _json_bytes_array,
    _qpy_module,
    _rollout_metadata,
)
from .config import ActionEngineConfig, load_action_config, save_action_config
from .constants import ACTION_ARTIFACT_SCHEMA_VERSION, ROLLOUT_ARTIFACT_SCHEMA_VERSION
from .identities import circuit_semantic_hash
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

DEFAULT_ACTION_SHARD_COUNT = 256
_SHARD_SEPARATOR = "#"


def action_shard_reference(
    sample_id: str,
    shard_count: int = DEFAULT_ACTION_SHARD_COUNT,
) -> str:
    """Return the deterministic archive path for one sample."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be nonblank")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer and not bool")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    bucket = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:4],
        "big",
    )
    bucket %= shard_count
    return f"artifacts/shards/action-shard-{bucket:03d}.zip"


def _action_member(action_id: str) -> str:
    return f"actions/{action_id}.json"


def _circuit_member(candidate_circuit_id: str) -> str:
    return f"circuits/{candidate_circuit_id}.qpy"


def _rollout_member(rollout_id: str) -> str:
    return f"rollouts/{rollout_id}.npz"


def sharded_member_reference(archive_reference: str, member: str) -> str:
    """Return a unique manifest reference for a member inside one ZIP shard."""
    if not isinstance(archive_reference, str) or not archive_reference.endswith(".zip"):
        raise ValueError("archive_reference must be a .zip path")
    if not isinstance(member, str) or not member or member.startswith("/"):
        raise ValueError("member must be a nonempty relative path")
    if _SHARD_SEPARATOR in archive_reference or _SHARD_SEPARATOR in member:
        raise ValueError("sharded references must not contain '#' in path components")
    return f"{archive_reference}{_SHARD_SEPARATOR}{member}"


def split_sharded_reference(reference: str) -> tuple[str, str] | None:
    """Split a member-qualified shard reference, returning None for legacy files."""
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference must be nonblank")
    if _SHARD_SEPARATOR not in reference:
        return None
    archive_reference, member = reference.split(_SHARD_SEPARATOR, 1)
    if not archive_reference.endswith(".zip") or not member or member.startswith("/"):
        raise ValueError(f"Malformed sharded artifact reference: {reference}")
    return archive_reference, member


def archive_reference(reference: str) -> str:
    """Return the physical file path represented by a manifest reference."""
    split = split_sharded_reference(reference)
    return reference if split is None else split[0]


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    import json

    return (
        json.dumps(
            json_copy(dict(payload)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _circuit_bytes(circuit: Any) -> bytes:
    handle = BytesIO()
    _qpy_module().dump(circuit, handle)
    return handle.getvalue()


def _rollout_bytes(rollout: ActionRollout) -> bytes:
    handle = BytesIO()
    np.savez_compressed(
        handle,
        metric_names=rollout.metric_names,
        baseline_metric_values=rollout.baseline_metric_values,
        candidate_metric_values=rollout.candidate_metric_values,
        improvement_values=rollout.improvement_values,
        outcome_bitstrings=rollout.outcome_bitstrings,
        exact_probabilities=rollout.exact_probabilities,
        **{_ROLLOUT_METADATA_NAME: _json_bytes_array(_rollout_metadata(rollout))},
    )
    return handle.getvalue()


def write_deterministic_member(
    archive: ZipFile,
    member: str,
    data: bytes,
    *,
    compress_type: int,
    compresslevel: int | None = None,
) -> None:
    """Write a reproducible ZIP member with a fixed timestamp and permissions."""
    info = ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compress_type
    info.create_system = 3
    info.external_attr = 0o600 << 16
    archive.writestr(
        info,
        data,
        compress_type=compress_type,
        compresslevel=compresslevel,
    )


def write_candidate_bundle(
    archive: ZipFile,
    candidate: ActionCandidate,
    rollout: ActionRollout,
    config: ActionEngineConfig,
) -> tuple[str, str, str]:
    """Validate and immediately serialize one candidate/rollout bundle."""
    validate_action_candidate(candidate, config, require_hash=True)
    validate_action_rollout(rollout, require_hash=True)
    action_name = _action_member(candidate.action_id)
    circuit_name = _circuit_member(rollout.candidate_circuit_id)
    rollout_name = _rollout_member(rollout.rollout_id)
    write_deterministic_member(
        archive,
        action_name,
        _strict_json_bytes(_action_payload(candidate)),
        compress_type=ZIP_DEFLATED,
        compresslevel=6,
    )
    write_deterministic_member(
        archive,
        circuit_name,
        _circuit_bytes(rollout.candidate_circuit),
        compress_type=ZIP_DEFLATED,
        compresslevel=3,
    )
    write_deterministic_member(
        archive,
        rollout_name,
        _rollout_bytes(rollout),
        compress_type=ZIP_STORED,
    )
    return action_name, circuit_name, rollout_name


def _candidate_from_payload(
    payload_raw: Any,
    config: ActionEngineConfig,
    expected_content_hash: str,
) -> ActionCandidate:
    if not isinstance(payload_raw, Mapping):
        raise TypeError("Action shard member must contain a JSON object")
    payload = dict(payload_raw)
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
        raise ValueError("Action shard member key mismatch")
    if payload["artifact_schema_version"] != ACTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported action artifact schema version")
    edits_raw = payload["edits"]
    if not isinstance(edits_raw, list):
        raise TypeError("Action shard edits must be a list")
    edits: list[ActionEdit] = []
    for index, item in enumerate(edits_raw):
        if not isinstance(item, Mapping) or set(item) != {
            "edit_type",
            "qubits",
            "magnitude",
        }:
            raise ValueError(f"Action shard edit {index} has invalid fields")
        qubits = item["qubits"]
        if not isinstance(qubits, list):
            raise TypeError(f"Action shard edit {index} qubits must be a list")
        edits.append(
            ActionEdit(
                edit_type=item["edit_type"],
                qubits=tuple(qubits),
                magnitude=item["magnitude"],
            )
        )
    sources = payload["generation_sources"]
    metadata = payload["metadata"]
    if not isinstance(sources, list):
        raise TypeError("Action shard generation_sources must be a list")
    if not isinstance(metadata, Mapping):
        raise TypeError("Action shard metadata must be a mapping")
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
    if candidate.content_hash != expected_content_hash:
        raise ValueError("Action shard content_hash does not match manifest")
    return candidate


def _rollout_from_bytes(
    data: bytes,
    candidate_circuit: Any,
    expected_content_hash: str,
) -> ActionRollout:
    with np.load(BytesIO(data), allow_pickle=False) as payload:
        expected_names = _ROLLOUT_ARRAY_NAMES | {_ROLLOUT_METADATA_NAME}
        if set(payload.files) != expected_names:
            raise ValueError("Rollout shard array-name mismatch")
        arrays = {name: payload[name].copy() for name in _ROLLOUT_ARRAY_NAMES}
        metadata = _decode_json_bytes(
            payload[_ROLLOUT_METADATA_NAME],
            _ROLLOUT_METADATA_NAME,
        )
    expected_metadata_keys = {
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
    if set(metadata) != expected_metadata_keys:
        raise ValueError("Rollout shard metadata-key mismatch")
    if metadata["artifact_schema_version"] != ROLLOUT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported rollout artifact schema version")
    if not isinstance(metadata["metadata"], Mapping):
        raise TypeError("Rollout shard metadata.metadata must be a mapping")
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
        primary_metric_nonworsening=metadata["primary_metric_nonworsening"],
        selected=metadata["selected"],
        candidate_circuit=candidate_circuit,
        depth_delta=metadata["depth_delta"],
        gate_delta=metadata["gate_delta"],
        metadata=dict(metadata["metadata"]),
        content_hash=metadata["content_hash"],
    )
    validate_action_rollout(rollout, require_hash=True)
    if rollout.content_hash != expected_content_hash:
        raise ValueError("Rollout shard content_hash does not match manifest")
    return rollout


class ShardedActionReader:
    """Read many logical artifacts while keeping each physical ZIP open once."""

    def __init__(self, root: str | Path, config: ActionEngineConfig) -> None:
        self.root = Path(root)
        self.config = config
        self._archives: dict[Path, ZipFile] = {}

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "ShardedActionReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _archive(self, reference: str) -> ZipFile:
        physical_reference = archive_reference(reference)
        path = (self.root / physical_reference).resolve()
        root = self.root.resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"unsafe or missing action shard reference: {reference}")
        archive = self._archives.get(path)
        if archive is None:
            archive = ZipFile(path, "r")
            self._archives[path] = archive
        return archive

    @staticmethod
    def _member(reference: str, fallback: str) -> str:
        split = split_sharded_reference(reference)
        return fallback if split is None else split[1]

    def load_candidate(
        self,
        reference: str,
        action_id: str,
        expected_content_hash: str,
    ) -> ActionCandidate:
        member = self._member(reference, _action_member(action_id))
        data = self._archive(reference).read(member)
        return _candidate_from_payload(
            strict_json_loads(data.decode("utf-8")),
            self.config,
            expected_content_hash,
        )

    def load_circuit(
        self,
        reference: str,
        candidate_circuit_id: str,
        expected_circuit_hash: str,
    ) -> Any:
        member = self._member(reference, _circuit_member(candidate_circuit_id))
        data = self._archive(reference).read(member)
        circuits = _qpy_module().load(BytesIO(data))
        if len(circuits) != 1:
            raise ValueError("Candidate circuit shard member must contain one circuit")
        circuit = circuits[0]
        if circuit_semantic_hash(circuit) != expected_circuit_hash:
            raise ValueError("Candidate circuit shard hash does not match manifest")
        return circuit

    def load_rollout(
        self,
        reference: str,
        rollout_id: str,
        candidate_circuit: Any,
        expected_content_hash: str,
    ) -> ActionRollout:
        member = self._member(reference, _rollout_member(rollout_id))
        data = self._archive(reference).read(member)
        return _rollout_from_bytes(data, candidate_circuit, expected_content_hash)


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


def write_sharded_action_dataset(
    result: ActionEngineResult,
    output_root: str | Path,
    *,
    shard_count: int = DEFAULT_ACTION_SHARD_COUNT,
) -> ActionWriteResult:
    """Publish an in-memory Phase 9 result using bounded compressed artifacts."""
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

    rollout_by_action = {rollout.action_id: rollout for rollout in result.rollouts}
    candidate_records = []
    for record in result.candidate_records:
        archive_ref = action_shard_reference(record.sample_id, shard_count)
        candidate_records.append(
            replace(
                record,
                action_ref=sharded_member_reference(
                    archive_ref,
                    _action_member(record.action_id),
                ),
                circuit_ref=sharded_member_reference(
                    archive_ref,
                    _circuit_member(record.candidate_circuit_id),
                ),
            )
        )
    rollout_records = []
    for record in result.rollout_records:
        archive_ref = action_shard_reference(record.sample_id, shard_count)
        rollout_records.append(
            replace(
                record,
                rollout_ref=sharded_member_reference(
                    archive_ref,
                    _rollout_member(record.rollout_id),
                ),
            )
        )

    try:
        (staging / "manifests").mkdir(parents=True, exist_ok=False)
        (staging / "artifacts" / "shards").mkdir(parents=True, exist_ok=False)
        managed: set[str] = set()
        save_action_config(result.config, staging / "action_config.json")
        write_strict_json(staging / "action_summary.json", result.summary)
        managed.update(("action_config.json", "action_summary.json"))

        candidates_by_shard: dict[str, list[ActionCandidate]] = defaultdict(list)
        for candidate in result.candidates:
            candidates_by_shard[
                action_shard_reference(candidate.sample_id, shard_count)
            ].append(candidate)

        for reference in sorted(candidates_by_shard):
            path = staging / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            expected_members: set[str] = set()
            with ZipFile(path, "w", allowZip64=True) as archive:
                for candidate in sorted(
                    candidates_by_shard[reference],
                    key=lambda item: item.action_id,
                ):
                    rollout = rollout_by_action[candidate.action_id]
                    expected_members.update(
                        write_candidate_bundle(
                            archive,
                            candidate,
                            rollout,
                            result.config,
                        )
                    )
            with ZipFile(path, "r") as archive:
                actual_members = set(archive.namelist())
                if actual_members != expected_members:
                    raise ValueError(
                        f"Action shard inventory mismatch for {reference}: "
                        f"missing={sorted(expected_members - actual_members)}, "
                        f"unexpected={sorted(actual_members - expected_members)}"
                    )
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"Corrupt action shard member: {bad_member}")
            managed.add(reference)

        writer = ManifestWriter(staging / "manifests")
        writer.write_records("action_candidate_manifest", candidate_records)
        writer.write_records("action_rollout_manifest", rollout_records)
        managed.update(
            (
                "manifests/action_candidate_manifest.parquet",
                "manifests/action_rollout_manifest.parquet",
            )
        )

        persisted_config = load_action_config(staging / "action_config.json")
        reader = ManifestReader(staging / "manifests")
        persisted_candidates = reader.read_typed_records(
            "action_candidate_manifest",
            ActionCandidateRecordV1,
        )
        persisted_rollouts = reader.read_typed_records(
            "action_rollout_manifest",
            ActionRolloutRecord,
        )
        if persisted_config != result.config:
            raise ValueError("Persisted action config does not match conversion config")
        validate_action_dataset_joins(
            persisted_candidates,
            persisted_rollouts,
            candidates_by_id={item.action_id: item for item in result.candidates},
            rollouts_by_id={item.rollout_id: item for item in result.rollouts},
            config=persisted_config,
        )

        if _relative_file_set(staging) != set(managed):
            raise ValueError("Staging sharded action dataset inventory mismatch")
        managed_files = tuple(sorted([*managed, "action_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": result.source_scientific_generation_id,
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
        if strict_json_load(staging / "action_complete.json") != completion:
            raise ValueError("action_complete.json content mismatch")
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed sharded action inventory mismatch")
        _verify_result_sources(result)
        os.replace(staging, output)

        manifest_paths = tuple(
            output / reference
            for reference in sorted(
                reference
                for reference in managed_files
                if reference.startswith("manifests/")
            )
        )
        artifact_paths = tuple(
            output / reference
            for reference in sorted(
                reference
                for reference in managed_files
                if reference.startswith("artifacts/")
            )
        )
        written_paths = tuple(output / reference for reference in managed_files)
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
    "DEFAULT_ACTION_SHARD_COUNT",
    "ShardedActionReader",
    "action_shard_reference",
    "archive_reference",
    "sharded_member_reference",
    "split_sharded_reference",
    "write_candidate_bundle",
    "write_deterministic_member",
    "write_sharded_action_dataset",
]
