"""Fully streaming, resumable Phase 9 action generation and persistence.

Work is partitioned into deterministic storage shards. Each worker processes one
shard sequentially and writes every candidate bundle immediately, so candidate
circuits and rollout arrays never accumulate for the full dataset. Completed
shards are checkpointed and reused after interruption.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable
import uuid
from zipfile import ZipFile

from triqto.graph.utils import strict_json_load, write_strict_json
from triqto.storage.action_schema import ActionCandidateRecordV1, ActionRolloutRecord
from triqto.storage.manifest import ManifestReader, ManifestWriter

from .candidates import generate_action_candidates
from .config import ActionEngineConfig, load_action_config, save_action_config
from .identities import (
    action_engine_id,
    action_operational_config_id,
    action_schema_id,
)
from .models import ActionWriteResult
from .rollout_runner import run_action_rollouts
from .sharded_artifacts import (
    DEFAULT_ACTION_SHARD_COUNT,
    action_shard_reference,
    sharded_member_reference,
    write_candidate_bundle,
)
from .source import load_action_engine_sources, verify_action_source_snapshots
from .validators import validate_action_dataset_joins

ProgressCallback = Callable[[dict[str, Any]], None]
_STREAM_STATE_SCHEMA = "triqto.phase9.streaming_state.v1"
_SHARD_CHECKPOINT_SCHEMA = "triqto.phase9.streaming_shard.v1"


def _default_workers() -> int:
    count = os.cpu_count() or 1
    return max(1, min(8, count))


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    write_strict_json(temporary, payload)
    os.replace(temporary, path)


def _selected_action_kind(candidate: Any) -> str:
    if not candidate.edits:
        return "no_op"
    if len(candidate.edits) > 1:
        return "composite"
    return candidate.edits[0].edit_type


def _candidate_record(
    candidate: Any,
    rollout: Any,
    archive_ref: str,
) -> ActionCandidateRecordV1:
    circuit_hash = rollout.metadata.get("candidate_circuit_hash")
    if not isinstance(circuit_hash, str):
        raise ValueError(
            f"Rollout {rollout.rollout_id} is missing candidate_circuit_hash"
        )
    record = ActionCandidateRecordV1(
        action_id=candidate.action_id,
        sample_id=candidate.sample_id,
        graph_pair_id=candidate.graph_pair_id,
        source_circuit_id=candidate.source_circuit_id,
        source_run_id=candidate.source_run_id,
        distortion_id=candidate.distortion_id,
        candidate_circuit_id=rollout.candidate_circuit_id,
        generation_sources=list(candidate.generation_sources),
        action_ref=sharded_member_reference(
            archive_ref,
            f"actions/{candidate.action_id}.json",
        ),
        circuit_ref=sharded_member_reference(
            archive_ref,
            f"circuits/{rollout.candidate_circuit_id}.qpy",
        ),
        content_hash=candidate.content_hash,
        circuit_hash=circuit_hash,
        edit_count=len(candidate.edits),
        validity_mask=True,
        risk_score=candidate.risk_score,
        metadata={
            "phase": 9,
            "candidate_generation_is_not_a_learned_policy": True,
            "storage_layout": "zip_shard_member",
        },
    )
    record.validate()
    return record


def _rollout_record(rollout: Any, archive_ref: str) -> ActionRolloutRecord:
    record = ActionRolloutRecord(
        rollout_id=rollout.rollout_id,
        action_id=rollout.action_id,
        sample_id=rollout.sample_id,
        graph_pair_id=rollout.graph_pair_id,
        candidate_circuit_id=rollout.candidate_circuit_id,
        clean_target_run_id=rollout.clean_target_run_id,
        scientific_config_id=rollout.scientific_config_id,
        rollout_ref=sharded_member_reference(
            archive_ref,
            f"rollouts/{rollout.rollout_id}.npz",
        ),
        content_hash=rollout.content_hash,
        rank=rollout.rank,
        reward=rollout.reward,
        risk_score=rollout.risk_score,
        dominates_baseline=rollout.dominates_baseline,
        primary_metric_nonworsening=rollout.primary_metric_nonworsening,
        selected=rollout.selected,
        metadata={
            "exact_born_recovery": bool(
                rollout.metadata.get("exact_born_recovery", False)
            ),
            "validation_mode": "ideal_statevector",
            "phase": 9,
            "storage_layout": "zip_shard_member",
        },
    )
    record.validate()
    return record


def _merge_parquet_parts(parts: list[Path], output: Path) -> int:
    """Merge validated Parquet parts one row group at a time."""
    if not parts:
        raise ValueError("At least one manifest part is required")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required for streaming manifest merge") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    schema = None
    row_count = 0
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch])
                if schema is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(
                        output,
                        schema,
                        compression="snappy",
                        use_dictionary=True,
                    )
                elif table.schema != schema:
                    table = table.cast(schema, safe=True)
                writer.write_table(table)
                row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if row_count <= 0 or not output.is_file():
        raise ValueError(f"Merged manifest is empty: {output}")
    return row_count


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required for streaming manifests") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def _checkpoint_paths(staging: Path, bucket: int) -> dict[str, Path]:
    suffix = f"{bucket:03d}"
    return {
        "checkpoint": staging / "checkpoints" / f"shard-{suffix}.json",
        "archive": staging / "artifacts" / "shards" / f"action-shard-{suffix}.zip",
        "candidate_part": (
            staging
            / "manifest_parts"
            / "candidates"
            / f"part-{suffix}.parquet"
        ),
        "rollout_part": (
            staging
            / "manifest_parts"
            / "rollouts"
            / f"part-{suffix}.parquet"
        ),
    }


def _read_valid_checkpoint(
    staging: Path,
    bucket: int,
    expected_sample_ids: list[str],
) -> dict[str, Any] | None:
    paths = _checkpoint_paths(staging, bucket)
    checkpoint_path = paths["checkpoint"]
    if not checkpoint_path.is_file():
        return None
    payload_raw = strict_json_load(checkpoint_path)
    if not isinstance(payload_raw, dict):
        raise TypeError(f"Malformed shard checkpoint: {checkpoint_path}")
    payload = dict(payload_raw)
    if payload.get("schema") != _SHARD_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported shard checkpoint: {checkpoint_path}")
    if payload.get("bucket") != bucket:
        raise ValueError(f"Shard checkpoint bucket mismatch: {checkpoint_path}")
    if payload.get("sample_ids") != expected_sample_ids:
        raise ValueError(f"Shard checkpoint sample inventory mismatch: {checkpoint_path}")
    for name in ("archive", "candidate_part", "rollout_part"):
        expected_ref = paths[name].relative_to(staging).as_posix()
        if payload.get(name) != expected_ref or not paths[name].is_file():
            raise FileNotFoundError(
                f"Completed shard checkpoint is missing {name}: {checkpoint_path}"
            )
    with ZipFile(paths["archive"], "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(
                f"Completed shard {bucket:03d} contains corrupt member {bad_member}"
            )
        if len(archive.namelist()) != int(payload["candidate_count"]) * 3:
            raise ValueError(f"Completed shard member count mismatch: {checkpoint_path}")
    if _parquet_row_count(paths["candidate_part"]) != payload["candidate_count"]:
        raise ValueError(f"Candidate manifest part count mismatch: {checkpoint_path}")
    if _parquet_row_count(paths["rollout_part"]) != payload["rollout_count"]:
        raise ValueError(f"Rollout manifest part count mismatch: {checkpoint_path}")
    return payload


def _remove_uncheckpointed_outputs(staging: Path, bucket: int) -> None:
    paths = _checkpoint_paths(staging, bucket)
    for name in ("archive", "candidate_part", "rollout_part"):
        path = paths[name]
        partial = path.with_name(f".{path.name}.partial")
        if partial.exists():
            partial.unlink()
        if path.exists():
            path.unlink()


def _state_identity(
    *,
    sources: Any,
    config: ActionEngineConfig,
    shard_count: int,
) -> dict[str, Any]:
    return {
        "schema": _STREAM_STATE_SCHEMA,
        "source_scientific_generation_id": (
            sources.phase7.source_scientific_generation_id
        ),
        "graph_conversion_id": sources.graph.completion_marker["graph_conversion_id"],
        "action_engine_id": action_engine_id(
            sources.phase7.source_scientific_generation_id,
            sources.graph.completion_marker["graph_conversion_id"],
            config,
        ),
        "operational_config_id": action_operational_config_id(config),
        "action_schema_id": action_schema_id(),
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "sample_count": len(sources.phase7.samples),
        "shard_count": shard_count,
    }


def _action_write_result_from_marker(
    root: Path,
    marker: dict[str, Any],
) -> ActionWriteResult:
    managed_files = tuple(marker["managed_files"])
    return ActionWriteResult(
        output_root=root,
        action_complete_path=root / "action_complete.json",
        manifest_paths=tuple(
            root / reference
            for reference in managed_files
            if reference.startswith("manifests/")
        ),
        artifact_paths=tuple(
            root / reference
            for reference in managed_files
            if reference.startswith("artifacts/")
        ),
        written_paths=tuple(root / reference for reference in managed_files),
        managed_files=managed_files,
        candidate_count=int(marker["candidate_count"]),
        rollout_count=int(marker["rollout_count"]),
    )


def _publish_ready_staging(
    staging: Path,
    output: Path,
    identity: dict[str, Any],
) -> ActionWriteResult | None:
    """Publish a fully finalized staging root left by a hard interruption."""
    marker_path = staging / "action_complete.json"
    if not marker_path.is_file():
        return None
    marker_raw = strict_json_load(marker_path)
    if not isinstance(marker_raw, dict):
        raise TypeError(f"Malformed ready Phase 9 marker: {marker_path}")
    marker = dict(marker_raw)
    expected = {
        "source_scientific_generation_id": identity[
            "source_scientific_generation_id"
        ],
        "graph_conversion_id": identity["graph_conversion_id"],
        "action_engine_id": identity["action_engine_id"],
        "operational_config_id": identity["operational_config_id"],
        "action_schema_id": identity["action_schema_id"],
        "phase7_snapshot_hash": identity["phase7_snapshot_hash"],
        "graph_snapshot_hash": identity["graph_snapshot_hash"],
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise ValueError(
                f"Ready Phase 9 staging marker {name} does not match current sources"
            )
    managed = marker.get("managed_files")
    if not isinstance(managed, list) or any(
        not isinstance(reference, str) or not reference
        for reference in managed
    ):
        raise TypeError("Ready Phase 9 managed_files must be nonblank strings")
    for directory in (
        staging / "checkpoints",
        staging / "manifest_parts",
    ):
        if directory.exists():
            shutil.rmtree(directory)
    state_path = staging / "stream_state.json"
    if state_path.exists():
        state_path.unlink()
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if actual != set(managed):
        raise ValueError(
            "Ready Phase 9 staging inventory mismatch; "
            f"missing={sorted(set(managed) - actual)}, "
            f"unexpected={sorted(actual - set(managed))}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Action output root already exists: {output}")
    os.replace(staging, output)
    return _action_write_result_from_marker(output, marker)


def _prepare_staging(
    staging: Path,
    identity: dict[str, Any],
) -> None:
    state_path = staging / "stream_state.json"
    if staging.exists():
        if not state_path.is_file():
            if any(staging.iterdir()):
                raise FileExistsError(
                    f"Unrecognized nonempty Phase 9 streaming staging root: {staging}"
                )
        else:
            existing = strict_json_load(state_path)
            if existing != identity:
                raise ValueError(
                    "Phase 9 streaming checkpoint identity does not match current "
                    "sources/configuration"
                )
            return
    staging.mkdir(parents=True, exist_ok=True)
    for directory in (
        staging / "checkpoints",
        staging / "artifacts" / "shards",
        staging / "manifest_parts" / "candidates",
        staging / "manifest_parts" / "rollouts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(state_path, identity)


def _aggregate_summary(
    checkpoints: list[dict[str, Any]],
    *,
    sources: Any,
    config: ActionEngineConfig,
    workers: int,
    shard_count: int,
) -> dict[str, Any]:
    totals = Counter()
    selected_types = Counter()
    candidate_distribution = Counter()
    selected_reward_sum = 0.0
    selected_reward_count = 0
    for checkpoint in checkpoints:
        shard_summary = checkpoint["summary"]
        for name in (
            "source_sample_count",
            "candidate_count",
            "rollout_count",
            "selected_action_count",
            "oracle_candidate_count",
            "blind_candidate_count",
            "no_op_candidate_count",
            "nonworsening_rollout_count",
            "improving_rollout_count",
            "exact_born_recovery_count",
            "selected_no_op_count",
        ):
            totals[name] += int(shard_summary[name])
        selected_types.update(shard_summary["selected_action_type_counts"])
        candidate_distribution.update(
            {
                int(count): int(frequency)
                for count, frequency in shard_summary[
                    "candidate_count_distribution"
                ].items()
            }
        )
        selected_reward_sum += float(shard_summary["selected_reward_sum"])
        selected_reward_count += int(shard_summary["selected_reward_count"])

    engine_id = action_engine_id(
        sources.phase7.source_scientific_generation_id,
        sources.graph.completion_marker["graph_conversion_id"],
        config,
    )
    return {
        "source_scientific_generation_id": (
            sources.phase7.source_scientific_generation_id
        ),
        "graph_conversion_id": sources.graph.completion_marker["graph_conversion_id"],
        "action_engine_id": engine_id,
        "operational_config_id": action_operational_config_id(config),
        "action_schema_id": action_schema_id(),
        "source_sample_count": totals["source_sample_count"],
        "candidate_count": totals["candidate_count"],
        "rollout_count": totals["rollout_count"],
        "selected_action_count": totals["selected_action_count"],
        "oracle_candidate_count": totals["oracle_candidate_count"],
        "blind_candidate_count": totals["blind_candidate_count"],
        "no_op_candidate_count": totals["no_op_candidate_count"],
        "nonworsening_rollout_count": totals["nonworsening_rollout_count"],
        "improving_rollout_count": totals["improving_rollout_count"],
        "exact_born_recovery_count": totals["exact_born_recovery_count"],
        "selected_no_op_count": totals["selected_no_op_count"],
        "selected_action_type_counts": dict(sorted(selected_types.items())),
        "candidate_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(candidate_distribution.items())
        },
        "mean_selected_reward": (
            selected_reward_sum / selected_reward_count
            if selected_reward_count
            else 0.0
        ),
        "phase7_managed_file_count": len(
            sources.phase7.source_snapshot.entries
        ),
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_managed_file_count": len(sources.graph.snapshot.entries),
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "source_immutability_verified": True,
        "validation_mode": "ideal_statevector",
        "learned_policy_present": False,
        "parallel_workers": workers,
        "streaming_bounded_memory": True,
        "artifact_shard_count": len(checkpoints),
        "configured_shard_count": shard_count,
        "resume_checkpoint_granularity": "shard",
        "schema_versions": {
            "action": config.schema_version,
            "graph": sources.graph.config.schema_version,
            "phase7": sources.phase7.generation_config.schema_version,
        },
    }


def build_and_write_action_dataset_streaming(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    output_root: str | Path,
    config: ActionEngineConfig | None = None,
    *,
    workers: int | None = None,
    shard_count: int = DEFAULT_ACTION_SHARD_COUNT,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 100,
) -> ActionWriteResult:
    """Generate, validate, stream, checkpoint, and publish Phase 9."""
    action_config = config or ActionEngineConfig()
    if not isinstance(action_config, ActionEngineConfig):
        raise TypeError("config must be ActionEngineConfig or None")
    requested_workers = (
        _default_workers()
        if workers is None
        else _strict_positive_int(workers, "workers")
    )
    shard_count = _strict_positive_int(shard_count, "shard_count")
    progress_every = _strict_positive_int(progress_every, "progress_every")

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Action output root already exists: {output}")
    sources = load_action_engine_sources(phase7_source_root, graph_source_root)
    verify_action_source_snapshots(sources)
    resolved_output = output.resolve()
    for source_name, source_root in (
        ("Phase 7", sources.phase7.source_root),
        ("Phase 8", sources.graph.root),
    ):
        resolved_source = Path(source_root).resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"Action output root must not be inside the {source_name} source root"
            )

    staging = output.with_name(f".{output.name}.streaming")
    identity = _state_identity(
        sources=sources,
        config=action_config,
        shard_count=shard_count,
    )
    ready = _publish_ready_staging(staging, output, identity)
    if ready is not None:
        verify_action_source_snapshots(sources)
        return ready
    _prepare_staging(staging, identity)

    phase7 = sources.phase7
    graph = sources.graph
    distortions = {
        record.distortion_id: record
        for record in phase7.distortions
    }
    if len(distortions) != len(phase7.distortions):
        raise ValueError("Duplicate Phase 7 distortion_id")

    samples_by_bucket: dict[int, list[Any]] = defaultdict(list)
    archive_by_bucket: dict[int, str] = {}
    for sample in sorted(phase7.samples, key=lambda item: item.sample_id):
        reference = action_shard_reference(sample.sample_id, shard_count)
        bucket = int(Path(reference).stem.rsplit("-", 1)[1])
        samples_by_bucket[bucket].append(sample)
        archive_by_bucket[bucket] = reference
    if not samples_by_bucket:
        raise ValueError("Phase 9 requires at least one source sample")

    completed_checkpoints: dict[int, dict[str, Any]] = {}
    for bucket, samples in sorted(samples_by_bucket.items()):
        expected_ids = [sample.sample_id for sample in samples]
        checkpoint = _read_valid_checkpoint(staging, bucket, expected_ids)
        if checkpoint is not None:
            completed_checkpoints[bucket] = checkpoint
        else:
            _remove_uncheckpointed_outputs(staging, bucket)

    worker_count = min(requested_workers, len(samples_by_bucket))
    resumed_sample_count = sum(
        int(checkpoint["summary"]["source_sample_count"])
        for checkpoint in completed_checkpoints.values()
    )
    completed_samples = resumed_sample_count
    candidate_count = sum(
        int(checkpoint["candidate_count"])
        for checkpoint in completed_checkpoints.values()
    )
    started = time.monotonic()
    progress_lock = threading.Lock()
    last_reported = completed_samples

    def report_progress(sample_delta: int, candidate_delta: int) -> None:
        nonlocal completed_samples, candidate_count, last_reported
        callback_payload = None
        with progress_lock:
            completed_samples += sample_delta
            candidate_count += candidate_delta
            if progress_callback is not None and (
                completed_samples == len(phase7.samples)
                or completed_samples - last_reported >= progress_every
            ):
                elapsed = max(time.monotonic() - started, 1e-9)
                newly_completed = max(
                    completed_samples - resumed_sample_count,
                    0,
                )
                rate = newly_completed / elapsed
                remaining = len(phase7.samples) - completed_samples
                callback_payload = {
                    "phase": 9,
                    "completed_samples": completed_samples,
                    "total_samples": len(phase7.samples),
                    "candidate_count": candidate_count,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": rate,
                    "eta_seconds": (
                        remaining / rate if rate > 0.0 else None
                    ),
                    "workers": worker_count,
                    "completed_shards": len(completed_checkpoints),
                    "total_shards": len(samples_by_bucket),
                    "resumed_samples": resumed_sample_count,
                }
                last_reported = completed_samples
        if callback_payload is not None:
            progress_callback(callback_payload)

    def process_shard(bucket: int, samples: list[Any]) -> dict[str, Any]:
        paths = _checkpoint_paths(staging, bucket)
        archive_ref = archive_by_bucket[bucket]
        archive_partial = paths["archive"].with_name(
            f".{paths['archive'].name}.partial"
        )
        candidate_records: list[ActionCandidateRecordV1] = []
        rollout_records: list[ActionRolloutRecord] = []
        expected_members: set[str] = set()
        summary = Counter()
        selected_types = Counter()
        candidate_distribution = Counter()
        selected_reward_sum = 0.0
        selected_reward_count = 0

        paths["archive"].parent.mkdir(parents=True, exist_ok=True)
        if archive_partial.exists():
            archive_partial.unlink()
        with ZipFile(archive_partial, "w", allowZip64=True) as archive:
            for sample in samples:
                try:
                    graph_pair_record = graph.pair_records_by_sample_id[
                        sample.sample_id
                    ]
                    distortion = distortions[sample.distortion_id]
                    distorted_circuit = phase7.circuits_by_id[
                        sample.distorted_circuit_id
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"Phase 9 source join failed for sample {sample.sample_id}"
                    ) from exc

                sample_candidates = generate_action_candidates(
                    sample=sample,
                    graph_pair_record=graph_pair_record,
                    distortion=distortion,
                    distorted_circuit=distorted_circuit,
                    config=action_config,
                )
                sample_rollouts = run_action_rollouts(
                    distorted_circuit=distorted_circuit,
                    clean_target_run_id=sample.clean_run_id,
                    clean_probabilities=phase7.probabilities_by_run_id[
                        sample.clean_run_id
                    ],
                    distorted_probabilities=phase7.probabilities_by_run_id[
                        sample.distorted_run_id
                    ],
                    candidates=sample_candidates,
                    config=action_config,
                )
                rollout_by_action = {
                    rollout.action_id: rollout
                    for rollout in sample_rollouts
                }
                sample_candidate_records = []
                for candidate in sorted(
                    sample_candidates,
                    key=lambda item: item.action_id,
                ):
                    rollout = rollout_by_action[candidate.action_id]
                    expected_members.update(
                        write_candidate_bundle(
                            archive,
                            candidate,
                            rollout,
                            action_config,
                        )
                    )
                    sample_candidate_records.append(
                        _candidate_record(candidate, rollout, archive_ref)
                    )
                sample_rollout_records = [
                    _rollout_record(rollout, archive_ref)
                    for rollout in sorted(
                        sample_rollouts,
                        key=lambda item: (
                            item.sample_id,
                            item.rank,
                            item.action_id,
                        ),
                    )
                ]
                validate_action_dataset_joins(
                    sample_candidate_records,
                    sample_rollout_records,
                    candidates_by_id={
                        item.action_id: item
                        for item in sample_candidates
                    },
                    rollouts_by_id={
                        item.rollout_id: item
                        for item in sample_rollouts
                    },
                    source_samples=[sample],
                    graph_pair_records=[graph_pair_record],
                    config=action_config,
                )
                candidate_records.extend(sample_candidate_records)
                rollout_records.extend(sample_rollout_records)

                selected_rollout = next(
                    rollout
                    for rollout in sample_rollouts
                    if rollout.selected
                )
                selected_candidate = next(
                    candidate
                    for candidate in sample_candidates
                    if candidate.action_id == selected_rollout.action_id
                )
                summary["source_sample_count"] += 1
                summary["candidate_count"] += len(sample_candidates)
                summary["rollout_count"] += len(sample_rollouts)
                summary["selected_action_count"] += 1
                summary["oracle_candidate_count"] += sum(
                    "oracle_inverse" in candidate.generation_sources
                    for candidate in sample_candidates
                )
                summary["blind_candidate_count"] += sum(
                    "blind_physics_prior" in candidate.generation_sources
                    for candidate in sample_candidates
                )
                summary["no_op_candidate_count"] += sum(
                    not candidate.edits
                    for candidate in sample_candidates
                )
                summary["nonworsening_rollout_count"] += sum(
                    rollout.primary_metric_nonworsening
                    for rollout in sample_rollouts
                )
                summary["improving_rollout_count"] += sum(
                    rollout.dominates_baseline
                    for rollout in sample_rollouts
                )
                summary["exact_born_recovery_count"] += sum(
                    bool(rollout.metadata.get("exact_born_recovery", False))
                    for rollout in sample_rollouts
                )
                selected_kind = _selected_action_kind(selected_candidate)
                selected_types[selected_kind] += 1
                if selected_kind == "no_op":
                    summary["selected_no_op_count"] += 1
                candidate_distribution[len(sample_candidates)] += 1
                selected_reward_sum += float(selected_rollout.reward)
                selected_reward_count += 1
                report_progress(1, len(sample_candidates))

                del (
                    rollout_by_action,
                    sample_candidate_records,
                    sample_rollout_records,
                    sample_candidates,
                    sample_rollouts,
                )

        os.replace(archive_partial, paths["archive"])
        with ZipFile(paths["archive"], "r") as archive:
            actual_members = set(archive.namelist())
            if actual_members != expected_members:
                raise ValueError(
                    f"Streaming action shard {bucket:03d} inventory mismatch"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"Streaming action shard {bucket:03d} corrupt member {bad_member}"
                )

        candidate_root = paths["candidate_part"].parent
        rollout_root = paths["rollout_part"].parent
        candidate_name = paths["candidate_part"].stem
        rollout_name = paths["rollout_part"].stem
        ManifestWriter(candidate_root).write_records(
            candidate_name,
            candidate_records,
        )
        ManifestWriter(rollout_root).write_records(
            rollout_name,
            rollout_records,
        )
        persisted_candidates = ManifestReader(candidate_root).read_typed_records(
            candidate_name,
            ActionCandidateRecordV1,
        )
        persisted_rollouts = ManifestReader(rollout_root).read_typed_records(
            rollout_name,
            ActionRolloutRecord,
        )
        validate_action_dataset_joins(
            persisted_candidates,
            persisted_rollouts,
            config=action_config,
        )

        checkpoint = {
            "schema": _SHARD_CHECKPOINT_SCHEMA,
            "bucket": bucket,
            "sample_ids": [sample.sample_id for sample in samples],
            "archive": paths["archive"].relative_to(staging).as_posix(),
            "candidate_part": (
                paths["candidate_part"].relative_to(staging).as_posix()
            ),
            "rollout_part": (
                paths["rollout_part"].relative_to(staging).as_posix()
            ),
            "candidate_count": len(candidate_records),
            "rollout_count": len(rollout_records),
            "summary": {
                **{
                    name: int(summary[name])
                    for name in (
                        "source_sample_count",
                        "candidate_count",
                        "rollout_count",
                        "selected_action_count",
                        "oracle_candidate_count",
                        "blind_candidate_count",
                        "no_op_candidate_count",
                        "nonworsening_rollout_count",
                        "improving_rollout_count",
                        "exact_born_recovery_count",
                        "selected_no_op_count",
                    )
                },
                "selected_action_type_counts": dict(
                    sorted(selected_types.items())
                ),
                "candidate_count_distribution": {
                    str(count): frequency
                    for count, frequency in sorted(
                        candidate_distribution.items()
                    )
                },
                "selected_reward_sum": selected_reward_sum,
                "selected_reward_count": selected_reward_count,
            },
        }
        _atomic_json(paths["checkpoint"], checkpoint)
        return checkpoint

    missing = [
        (bucket, samples)
        for bucket, samples in sorted(samples_by_bucket.items())
        if bucket not in completed_checkpoints
    ]
    if missing:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="triqto-p9-stream",
        ) as pool:
            future_by_bucket = {
                pool.submit(process_shard, bucket, samples): bucket
                for bucket, samples in missing
            }
            for future in as_completed(future_by_bucket):
                bucket = future_by_bucket[future]
                try:
                    checkpoint = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Phase 9 streaming worker failed for shard {bucket:03d}; "
                        f"completed shard checkpoints remain in {staging}"
                    ) from exc
                completed_checkpoints[bucket] = checkpoint

    checkpoints = [
        completed_checkpoints[bucket]
        for bucket in sorted(samples_by_bucket)
    ]
    if len(checkpoints) != len(samples_by_bucket):
        raise RuntimeError("Not all Phase 9 shards completed")

    candidate_parts = [
        staging / checkpoint["candidate_part"]
        for checkpoint in checkpoints
    ]
    rollout_parts = [
        staging / checkpoint["rollout_part"]
        for checkpoint in checkpoints
    ]
    manifest_root = staging / "manifests"
    candidate_rows = _merge_parquet_parts(
        candidate_parts,
        manifest_root / "action_candidate_manifest.parquet",
    )
    rollout_rows = _merge_parquet_parts(
        rollout_parts,
        manifest_root / "action_rollout_manifest.parquet",
    )

    summary = _aggregate_summary(
        checkpoints,
        sources=sources,
        config=action_config,
        workers=worker_count,
        shard_count=shard_count,
    )
    if candidate_rows != summary["candidate_count"]:
        raise ValueError("Merged candidate manifest row count mismatch")
    if rollout_rows != summary["rollout_count"]:
        raise ValueError("Merged rollout manifest row count mismatch")
    if summary["source_sample_count"] != len(phase7.samples):
        raise ValueError("Streaming Phase 9 sample coverage mismatch")

    save_action_config(action_config, staging / "action_config.json")
    write_strict_json(staging / "action_summary.json", summary)
    if load_action_config(staging / "action_config.json") != action_config:
        raise ValueError("Persisted action config mismatch")

    managed = {
        "action_config.json",
        "action_summary.json",
        "manifests/action_candidate_manifest.parquet",
        "manifests/action_rollout_manifest.parquet",
        *(
            checkpoint["archive"]
            for checkpoint in checkpoints
        ),
    }
    completion = {
        "complete": True,
        "source_scientific_generation_id": (
            phase7.source_scientific_generation_id
        ),
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": summary["action_engine_id"],
        "operational_config_id": summary["operational_config_id"],
        "action_schema_id": summary["action_schema_id"],
        "candidate_count": summary["candidate_count"],
        "rollout_count": summary["rollout_count"],
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
        "managed_files": sorted([*managed, "action_complete.json"]),
    }
    write_strict_json(staging / "action_complete.json", completion)

    shutil.rmtree(staging / "checkpoints")
    shutil.rmtree(staging / "manifest_parts")
    (staging / "stream_state.json").unlink()

    actual_files = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if actual_files != set(completion["managed_files"]):
        raise ValueError(
            "Committed streaming action inventory mismatch; "
            f"missing={sorted(set(completion['managed_files']) - actual_files)}, "
            f"unexpected={sorted(actual_files - set(completion['managed_files']))}"
        )
    if strict_json_load(staging / "action_complete.json") != completion:
        raise ValueError("action_complete.json content mismatch")
    verify_action_source_snapshots(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Action output root appeared during publication: {output}")
    os.replace(staging, output)

    managed_files = tuple(completion["managed_files"])
    artifact_paths = tuple(
        output / reference
        for reference in managed_files
        if reference.startswith("artifacts/")
    )
    manifest_paths = tuple(
        output / reference
        for reference in managed_files
        if reference.startswith("manifests/")
    )
    return ActionWriteResult(
        output_root=output,
        action_complete_path=output / "action_complete.json",
        manifest_paths=manifest_paths,
        artifact_paths=artifact_paths,
        written_paths=tuple(output / reference for reference in managed_files),
        managed_files=managed_files,
        candidate_count=summary["candidate_count"],
        rollout_count=summary["rollout_count"],
    )


__all__ = ["build_and_write_action_dataset_streaming"]
