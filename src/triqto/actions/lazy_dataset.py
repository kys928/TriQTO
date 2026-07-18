"""Lazy, resumable access to completed Phase 9 action datasets.

The ordinary completed-action loader hydrates every candidate, circuit, and rollout into
Python. That is appropriate for small baseline jobs but catastrophically expensive for
multi-million-action Phase 15.6 datasets. This module validates the immutable Phase 9
control plane, builds a durable SQLite manifest index in bounded Parquet batches, and
hydrates only the sample currently needed by Phase 11 or Phase 12.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import threading
import time
from typing import Any, Generic, TypeVar
import uuid

from triqto.graph import SourceFileSnapshot, snapshot_managed_files
from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
    write_strict_json,
)
from triqto.storage import ActionCandidateRecordV1, ActionRolloutRecord

from .artifacts import load_action_artifact, load_candidate_circuit, load_rollout_artifact
from .config import ActionEngineConfig, load_action_config
from .identities import (
    action_engine_id,
    action_operational_config_id,
    action_schema_id,
)
from .sharded_artifacts import (
    ShardedActionReader,
    archive_reference,
    split_sharded_reference,
)

_INDEX_SCHEMA = "triqto.phase9.lazy_action_index.v1"
_DEFAULT_BATCH_SIZE = 50_000
_ACTION_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "action_engine_id",
    "operational_config_id",
    "action_schema_id",
    "candidate_count",
    "rollout_count",
    "phase7_snapshot_hash",
    "graph_snapshot_hash",
    "managed_files",
}
_REQUIRED_MANAGED = {
    "action_config.json",
    "action_summary.json",
    "action_complete.json",
    "manifests/action_candidate_manifest.parquet",
    "manifests/action_rollout_manifest.parquet",
}
_CONTROL_FILES = tuple(sorted(_REQUIRED_MANAGED))
T = TypeVar("T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. TriQTO production is Linux.
    return float(value) / (1024.0 * 1024.0)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        write_strict_json(temporary, dict(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _log(label: str, message: str) -> None:
    print(f"[{label}][lazy-source] {message}", flush=True)


def _file_inventory(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _artifact_inventory_hash(root: Path, managed_files: Sequence[str]) -> str:
    """Hash artifact names and stable filesystem metadata without reading all ZIP bytes.

    Every artifact opened later is still checked against its manifest content/circuit hash.
    This fast inventory guard detects replacement, removal, truncation, or timestamp changes
    before index reuse without rereading a ~10 GiB Phase 9 archive universe.
    """
    digest = hashlib.sha256()
    for reference in managed_files:
        if not reference.startswith("artifacts/"):
            continue
        path = resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
        stat = path.stat()
        digest.update(reference.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _strict_count(marker: Mapping[str, Any], name: str) -> int:
    value = marker.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"action_complete.json {name} must be a nonnegative integer")
    return value


def _resume_mode() -> str:
    value = os.environ.get("TRIQTO_RESUME_MODE", "strict")
    if value not in {"strict", "repair", "off"}:
        raise ValueError("TRIQTO_RESUME_MODE must be strict, repair, or off")
    return value


def _quarantine(root: Path, reason: str) -> None:
    if not root.exists():
        return
    destination = root.parent / (
        f"{root.name}.quarantine-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    root.rename(destination)
    _atomic_json(destination / "quarantine_reason.json", {"reason": reason, "time": _utc_now()})


def _validate_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonblank text")
    return value


def _validate_hash(value: Any, name: str) -> str:
    text = _validate_text(value, name)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{name} must be a sha256 content hash")
    return text


def _safe_bool(value: Any, name: str) -> int:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return int(value)


def _safe_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be integer and not bool")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _safe_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=120000")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS index_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            action_id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            graph_pair_id TEXT NOT NULL,
            source_circuit_id TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            distortion_id TEXT NOT NULL,
            candidate_circuit_id TEXT NOT NULL UNIQUE,
            generation_sources_json TEXT NOT NULL,
            action_ref TEXT NOT NULL UNIQUE,
            circuit_ref TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            circuit_hash TEXT NOT NULL,
            edit_count INTEGER NOT NULL,
            validity_mask INTEGER NOT NULL,
            risk_score REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rollouts (
            rollout_id TEXT PRIMARY KEY,
            action_id TEXT NOT NULL UNIQUE REFERENCES candidates(action_id),
            sample_id TEXT NOT NULL,
            graph_pair_id TEXT NOT NULL,
            candidate_circuit_id TEXT NOT NULL,
            clean_target_run_id TEXT NOT NULL,
            scientific_config_id TEXT NOT NULL,
            rollout_ref TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            rank_value INTEGER NOT NULL,
            reward REAL NOT NULL,
            risk_score REAL NOT NULL,
            dominates_baseline INTEGER NOT NULL,
            primary_metric_nonworsening INTEGER NOT NULL,
            selected INTEGER NOT NULL
        );
        """
    )
    connection.commit()


def _state_int(connection: sqlite3.Connection, key: str) -> int:
    row = connection.execute("SELECT value FROM index_state WHERE key=?", (key,)).fetchone()
    return 0 if row is None else int(row["value"])


def _set_state(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO index_state(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _write_progress(root: Path, **payload: Any) -> None:
    _atomic_json(
        root / "progress.json",
        {"schema": _INDEX_SCHEMA, "updated_at": _utc_now(), **payload},
    )


def _batch_log(
    label: str,
    kind: str,
    batch_index: int,
    total_batches: int,
    rows_done: int,
    total_rows: int,
    started: float,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = rows_done / elapsed
    remaining = max(total_rows - rows_done, 0)
    eta = remaining / rate if rate > 0.0 else 0.0
    _log(
        label,
        f"index {kind} batch {batch_index}/{total_batches} | "
        f"rows={rows_done:,}/{total_rows:,} | rate={rate:,.0f}/s | "
        f"elapsed={elapsed/60.0:.1f}m | ETA≈{eta/60.0:.1f}m | RSS≈{_rss_gib():.2f} GiB",
    )


def _candidate_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    sources = row.get("generation_sources")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(item, str) or not item for item in sources
    ):
        raise ValueError("candidate generation_sources must be a nonempty string list")
    return (
        _validate_text(row.get("action_id"), "candidate.action_id"),
        _validate_text(row.get("sample_id"), "candidate.sample_id"),
        _validate_text(row.get("graph_pair_id"), "candidate.graph_pair_id"),
        _validate_text(row.get("source_circuit_id"), "candidate.source_circuit_id"),
        _validate_text(row.get("source_run_id"), "candidate.source_run_id"),
        _validate_text(row.get("distortion_id"), "candidate.distortion_id"),
        _validate_text(row.get("candidate_circuit_id"), "candidate.candidate_circuit_id"),
        json.dumps(sources, sort_keys=True, separators=(",", ":")),
        _validate_text(row.get("action_ref"), "candidate.action_ref"),
        _validate_text(row.get("circuit_ref"), "candidate.circuit_ref"),
        _validate_hash(row.get("content_hash"), "candidate.content_hash"),
        _validate_hash(row.get("circuit_hash"), "candidate.circuit_hash"),
        _safe_int(row.get("edit_count"), "candidate.edit_count"),
        _safe_bool(row.get("validity_mask"), "candidate.validity_mask"),
        _safe_float(row.get("risk_score"), "candidate.risk_score"),
    )


def _rollout_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _validate_text(row.get("rollout_id"), "rollout.rollout_id"),
        _validate_text(row.get("action_id"), "rollout.action_id"),
        _validate_text(row.get("sample_id"), "rollout.sample_id"),
        _validate_text(row.get("graph_pair_id"), "rollout.graph_pair_id"),
        _validate_text(row.get("candidate_circuit_id"), "rollout.candidate_circuit_id"),
        _validate_text(row.get("clean_target_run_id"), "rollout.clean_target_run_id"),
        _validate_text(row.get("scientific_config_id"), "rollout.scientific_config_id"),
        _validate_text(row.get("rollout_ref"), "rollout.rollout_ref"),
        _validate_hash(row.get("content_hash"), "rollout.content_hash"),
        _safe_int(row.get("rank"), "rollout.rank", positive=True),
        _safe_float(row.get("reward"), "rollout.reward"),
        _safe_float(row.get("risk_score"), "rollout.risk_score"),
        _safe_bool(row.get("dominates_baseline"), "rollout.dominates_baseline"),
        _safe_bool(
            row.get("primary_metric_nonworsening"),
            "rollout.primary_metric_nonworsening",
        ),
        _safe_bool(row.get("selected"), "rollout.selected"),
    )


def _ingest_manifest(
    *,
    label: str,
    root: Path,
    connection: sqlite3.Connection,
    manifest_path: Path,
    kind: str,
    expected_rows: int,
    batch_size: int,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency controlled by project
        raise RuntimeError("pyarrow is required for lazy Phase 9 indexing") from exc

    if kind == "candidates":
        columns = [
            "action_id", "sample_id", "graph_pair_id", "source_circuit_id",
            "source_run_id", "distortion_id", "candidate_circuit_id",
            "generation_sources", "action_ref", "circuit_ref", "content_hash",
            "circuit_hash", "edit_count", "validity_mask", "risk_score",
        ]
        sql = (
            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        converter = _candidate_tuple
    elif kind == "rollouts":
        columns = [
            "rollout_id", "action_id", "sample_id", "graph_pair_id",
            "candidate_circuit_id", "clean_target_run_id", "scientific_config_id",
            "rollout_ref", "content_hash", "rank", "reward", "risk_score",
            "dominates_baseline", "primary_metric_nonworsening", "selected",
        ]
        sql = "INSERT INTO rollouts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        converter = _rollout_tuple
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown manifest kind {kind}")

    parquet = pq.ParquetFile(manifest_path)
    if parquet.metadata is None or parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"{kind} Parquet row count does not match completion marker")
    total_batches = max(1, math.ceil(expected_rows / batch_size))
    completed_batches = _state_int(connection, f"{kind}_batches")
    rows_done = _state_int(connection, f"{kind}_rows")
    started = time.monotonic()
    _log(
        label,
        f"indexing {kind}: {expected_rows:,} rows in {total_batches} bounded batches; "
        f"resume_at_batch={completed_batches + 1}",
    )
    _write_progress(
        root,
        stage=f"index_{kind}",
        batch=completed_batches,
        total_batches=total_batches,
        rows=rows_done,
        total_rows=expected_rows,
    )
    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns),
        start=1,
    ):
        if batch_index <= completed_batches:
            continue
        values = [converter(row) for row in batch.to_pylist()]
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(sql, values)
            rows_done += len(values)
            _set_state(connection, f"{kind}_batches", batch_index)
            _set_state(connection, f"{kind}_rows", rows_done)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        _write_progress(
            root,
            stage=f"index_{kind}",
            batch=batch_index,
            total_batches=total_batches,
            rows=rows_done,
            total_rows=expected_rows,
        )
        _batch_log(
            label, kind, batch_index, total_batches, rows_done, expected_rows, started
        )
    if rows_done != expected_rows:
        raise ValueError(f"indexed {kind} row count mismatch: {rows_done} != {expected_rows}")


def _finalize_index(
    label: str,
    root: Path,
    connection: sqlite3.Connection,
    candidate_count: int,
    rollout_count: int,
    source_sample_ids: set[str],
) -> None:
    if _state_int(connection, "ready") == 1:
        return
    _log(label, "finalizing SQLite indexes and validating all manifest joins")
    _write_progress(root, stage="finalize_index", detail="creating SQL indexes")
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_candidates_sample ON candidates(sample_id, action_id);
        CREATE INDEX IF NOT EXISTS idx_rollouts_sample ON rollouts(sample_id, rank_value, action_id);
        CREATE INDEX IF NOT EXISTS idx_rollouts_candidate_circuit ON rollouts(candidate_circuit_id);
        """
    )
    actual_candidates = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    actual_rollouts = int(connection.execute("SELECT COUNT(*) FROM rollouts").fetchone()[0])
    if actual_candidates != candidate_count or actual_rollouts != rollout_count:
        raise ValueError("lazy action index count mismatch")
    mismatch = connection.execute(
        """
        SELECT COUNT(*) FROM rollouts r
        LEFT JOIN candidates c ON c.action_id=r.action_id
        WHERE c.action_id IS NULL OR c.sample_id<>r.sample_id
           OR c.graph_pair_id<>r.graph_pair_id
           OR c.candidate_circuit_id<>r.candidate_circuit_id
        """
    ).fetchone()[0]
    if mismatch:
        raise ValueError(f"lazy action index contains {mismatch} candidate/rollout join mismatches")
    missing_rollouts = connection.execute(
        """
        SELECT COUNT(*) FROM candidates c
        LEFT JOIN rollouts r ON r.action_id=c.action_id
        WHERE r.action_id IS NULL
        """
    ).fetchone()[0]
    if missing_rollouts:
        raise ValueError(f"lazy action index has {missing_rollouts} candidates without rollouts")
    sample_rows = connection.execute(
        """
        SELECT sample_id, COUNT(*) AS n, MIN(rank_value) AS min_rank,
               MAX(rank_value) AS max_rank, SUM(selected) AS selected_count,
               SUM(CASE WHEN selected=1 AND rank_value=1 THEN 1 ELSE 0 END) AS selected_rank1
        FROM rollouts GROUP BY sample_id
        """
    ).fetchall()
    observed_samples = {str(row["sample_id"]) for row in sample_rows}
    if observed_samples != source_sample_ids:
        raise ValueError("lazy action index sample coverage does not match Phase 7")
    for row in sample_rows:
        count = int(row["n"])
        if int(row["min_rank"]) != 1 or int(row["max_rank"]) != count:
            raise ValueError(f"sample {row['sample_id']} rollout ranks are not contiguous")
        if int(row["selected_count"]) != 1 or int(row["selected_rank1"]) != 1:
            raise ValueError(f"sample {row['sample_id']} does not have one selected rank-one action")
    _set_state(connection, "ready", 1)
    connection.commit()
    _write_progress(
        root,
        stage="ready",
        candidate_count=candidate_count,
        rollout_count=rollout_count,
        sample_count=len(source_sample_ids),
    )
    _log(
        label,
        f"lazy index ready | candidates={candidate_count:,} | rollouts={rollout_count:,} | "
        f"samples={len(source_sample_ids):,} | RSS≈{_rss_gib():.2f} GiB",
    )


def _candidate_record(row: sqlite3.Row) -> ActionCandidateRecordV1:
    record = ActionCandidateRecordV1(
        action_id=row["action_id"],
        sample_id=row["sample_id"],
        graph_pair_id=row["graph_pair_id"],
        source_circuit_id=row["source_circuit_id"],
        source_run_id=row["source_run_id"],
        distortion_id=row["distortion_id"],
        candidate_circuit_id=row["candidate_circuit_id"],
        generation_sources=list(json.loads(row["generation_sources_json"])),
        action_ref=row["action_ref"],
        circuit_ref=row["circuit_ref"],
        content_hash=row["content_hash"],
        circuit_hash=row["circuit_hash"],
        edit_count=int(row["edit_count"]),
        validity_mask=bool(row["validity_mask"]),
        risk_score=float(row["risk_score"]),
        metadata={"lazy_index": True},
    )
    record.validate()
    return record


def _rollout_record(row: sqlite3.Row) -> ActionRolloutRecord:
    record = ActionRolloutRecord(
        rollout_id=row["rollout_id"],
        action_id=row["action_id"],
        sample_id=row["sample_id"],
        graph_pair_id=row["graph_pair_id"],
        candidate_circuit_id=row["candidate_circuit_id"],
        clean_target_run_id=row["clean_target_run_id"],
        scientific_config_id=row["scientific_config_id"],
        rollout_ref=row["rollout_ref"],
        content_hash=row["rollout_content_hash"],
        rank=int(row["rank_value"]),
        reward=float(row["reward"]),
        risk_score=float(row["rollout_risk_score"]),
        dominates_baseline=bool(row["dominates_baseline"]),
        primary_metric_nonworsening=bool(row["primary_metric_nonworsening"]),
        selected=bool(row["selected"]),
        metadata={"lazy_index": True},
    )
    record.validate()
    return record


@dataclass(slots=True)
class _HydratedSample:
    sample_id: str
    candidates_by_id: dict[str, Any]
    circuits_by_id: dict[str, Any]
    rollouts_by_id: dict[str, Any]
    rollouts_by_sample_id: dict[str, tuple[Any, ...]]


class _SqlMapping(Mapping[str, T], Generic[T]):
    def __init__(self, owner: "LazyActionDataset", kind: str) -> None:
        self.owner = owner
        self.kind = kind

    def __len__(self) -> int:
        return self.owner.candidate_count if self.kind in {"candidate", "circuit", "action_sample"} else self.owner.rollout_count

    def __iter__(self) -> Iterator[str]:
        if self.kind == "candidate":
            query = "SELECT action_id AS key FROM candidates ORDER BY action_id"
        elif self.kind == "circuit":
            query = "SELECT candidate_circuit_id AS key FROM candidates ORDER BY candidate_circuit_id"
        elif self.kind == "rollout_record":
            query = "SELECT action_id AS key FROM rollouts ORDER BY action_id"
        elif self.kind == "rollout_id":
            query = "SELECT rollout_id AS key FROM rollouts ORDER BY rollout_id"
        elif self.kind == "action_sample":
            query = "SELECT action_id AS key FROM candidates ORDER BY action_id"
        else:
            query = "SELECT DISTINCT sample_id AS key FROM candidates ORDER BY sample_id"
        with _connect(self.owner.db_path) as connection:
            for row in connection.execute(query):
                yield str(row["key"])

    def __getitem__(self, key: str) -> T:
        if self.kind == "sample_rollouts":
            return self.owner._hydrate_sample(key).rollouts_by_sample_id[key]  # type: ignore[return-value]
        if self.kind in {"candidate", "circuit", "action_sample", "rollout_record"}:
            with _connect(self.owner.db_path) as connection:
                row = connection.execute(
                    "SELECT sample_id FROM candidates WHERE action_id=?", (key,)
                ).fetchone()
            if row is None:
                raise KeyError(key)
            if self.kind == "action_sample":
                return str(row["sample_id"])  # type: ignore[return-value]
            if self.kind == "rollout_record":
                return self.owner.rollout_record(key)  # type: ignore[return-value]
            hydrated = self.owner._hydrate_sample(str(row["sample_id"]))
            if self.kind == "candidate":
                return hydrated.candidates_by_id[key]  # type: ignore[return-value]
            circuit_id = self.owner.candidate_record(key).candidate_circuit_id
            return hydrated.circuits_by_id[circuit_id]  # type: ignore[return-value]
        if self.kind == "rollout_id":
            with _connect(self.owner.db_path) as connection:
                row = connection.execute(
                    "SELECT action_id,sample_id FROM rollouts WHERE rollout_id=?", (key,)
                ).fetchone()
            if row is None:
                raise KeyError(key)
            hydrated = self.owner._hydrate_sample(str(row["sample_id"]))
            return hydrated.rollouts_by_id[key]  # type: ignore[return-value]
        raise KeyError(key)


class _RecordSequence(Sequence[Any]):
    def __init__(self, owner: "LazyActionDataset", kind: str) -> None:
        self.owner = owner
        self.kind = kind

    def __len__(self) -> int:
        return self.owner.candidate_count if self.kind == "candidate" else self.owner.rollout_count

    def __getitem__(self, index: int) -> Any:
        if not isinstance(index, int):
            raise TypeError("lazy record sequence only supports integer indexing")
        count = len(self)
        normalized = index + count if index < 0 else index
        if normalized < 0 or normalized >= count:
            raise IndexError(index)
        table = "candidates" if self.kind == "candidate" else "rollouts"
        order = "action_id" if self.kind == "candidate" else "rollout_id"
        with _connect(self.owner.db_path) as connection:
            row = connection.execute(
                f"SELECT * FROM {table} ORDER BY {order} LIMIT 1 OFFSET ?",
                (normalized,),
            ).fetchone()
        if row is None:
            raise IndexError(index)
        return _candidate_record(row) if self.kind == "candidate" else self.owner._rollout_record_from_plain(row)


class LazyActionDataset:
    """Completed Phase 9 dataset with bounded per-sample artifact hydration."""

    is_lazy = True

    def __init__(
        self,
        *,
        root: Path,
        config: ActionEngineConfig,
        completion_marker: dict[str, Any],
        summary: dict[str, Any],
        managed_files: tuple[str, ...],
        snapshot: SourceFileSnapshot,
        artifact_inventory_hash: str,
        db_path: Path,
        label: str,
    ) -> None:
        self.root = root
        self.config = config
        self.completion_marker = completion_marker
        self.summary = summary
        self.managed_files = managed_files
        self.snapshot = snapshot
        self.artifact_inventory_hash = artifact_inventory_hash
        self.db_path = db_path
        self.label = label
        self.candidate_count = int(completion_marker["candidate_count"])
        self.rollout_count = int(completion_marker["rollout_count"])
        self._local = threading.local()
        self.candidates_by_id: Mapping[str, Any] = _SqlMapping(self, "candidate")
        self.circuits_by_id: Mapping[str, Any] = _SqlMapping(self, "circuit")
        self.rollouts_by_id: Mapping[str, Any] = _SqlMapping(self, "rollout_id")
        self.rollouts_by_sample_id: Mapping[str, tuple[Any, ...]] = _SqlMapping(self, "sample_rollouts")
        self.candidate_records_by_action_id: Mapping[str, Any] = _CandidateRecordMapping(self)
        self.rollout_records_by_action_id: Mapping[str, Any] = _SqlMapping(self, "rollout_record")
        self.action_to_sample: Mapping[str, str] = _SqlMapping(self, "action_sample")
        self.candidate_records: Sequence[Any] = _RecordSequence(self, "candidate")
        self.rollout_records: Sequence[Any] = _RecordSequence(self, "rollout")

    def verify_source(self) -> None:
        current_snapshot = snapshot_managed_files(self.root, _CONTROL_FILES)
        if current_snapshot != self.snapshot:
            raise RuntimeError("Phase 9 control files changed during lazy consumption")
        current_inventory = _artifact_inventory_hash(self.root, self.managed_files)
        if current_inventory != self.artifact_inventory_hash:
            raise RuntimeError("Phase 9 artifact inventory changed during lazy consumption")

    def action_ids_for_sample(self, sample_id: str) -> tuple[str, ...]:
        with _connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT action_id FROM candidates WHERE sample_id=? ORDER BY action_id",
                (sample_id,),
            ).fetchall()
        if not rows:
            raise KeyError(sample_id)
        return tuple(str(row["action_id"]) for row in rows)

    def candidate_record(self, action_id: str) -> ActionCandidateRecordV1:
        with _connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE action_id=?", (action_id,)
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return _candidate_record(row)

    def _rollout_record_from_plain(self, row: sqlite3.Row) -> ActionRolloutRecord:
        payload = dict(row)
        payload["rollout_content_hash"] = payload.pop("content_hash")
        payload["rollout_risk_score"] = payload.pop("risk_score")
        return _rollout_record(_DictRow(payload))

    def rollout_record(self, action_id: str) -> ActionRolloutRecord:
        with _connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT rollout_id,action_id,sample_id,graph_pair_id,candidate_circuit_id,
                       clean_target_run_id,scientific_config_id,rollout_ref,
                       content_hash AS rollout_content_hash,rank_value,reward,
                       risk_score AS rollout_risk_score,dominates_baseline,
                       primary_metric_nonworsening,selected
                FROM rollouts WHERE action_id=?
                """,
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return _rollout_record(row)

    def _hydrate_sample(self, sample_id: str) -> _HydratedSample:
        cached = getattr(self._local, "sample", None)
        if isinstance(cached, _HydratedSample) and cached.sample_id == sample_id:
            return cached
        with _connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT c.*,
                       r.rollout_id,r.rollout_ref,
                       r.content_hash AS rollout_content_hash,
                       r.clean_target_run_id,r.scientific_config_id,r.rank_value,
                       r.reward,r.risk_score AS rollout_risk_score,
                       r.dominates_baseline,r.primary_metric_nonworsening,r.selected
                FROM candidates c JOIN rollouts r ON r.action_id=c.action_id
                WHERE c.sample_id=? ORDER BY c.action_id
                """,
                (sample_id,),
            ).fetchall()
        if not rows:
            raise KeyError(sample_id)
        total = len(rows)
        started = time.monotonic()
        _log(self.label, f"hydrating sample={sample_id} | bundles={total} | bounded one-sample load")
        _write_progress(
            self.db_path.parent,
            stage="hydrate_sample",
            sample_id=sample_id,
            completed=0,
            total=total,
        )
        candidates: dict[str, Any] = {}
        circuits: dict[str, Any] = {}
        rollouts: dict[str, Any] = {}
        with ShardedActionReader(self.root, self.config) as shard_reader:
            for index, row in enumerate(rows, start=1):
                candidate_record = _candidate_record(row)
                rollout_record = _rollout_record(row)
                action_path, action_sharded = _artifact_path(
                    self.root, candidate_record.action_ref
                )
                circuit_path, circuit_sharded = _artifact_path(
                    self.root, candidate_record.circuit_ref
                )
                rollout_path, rollout_sharded = _artifact_path(
                    self.root, rollout_record.rollout_ref
                )
                candidate = (
                    shard_reader.load_candidate(
                        candidate_record.action_ref,
                        candidate_record.action_id,
                        candidate_record.content_hash,
                    )
                    if action_sharded
                    else load_action_artifact(
                        action_path, self.config, candidate_record.content_hash
                    )
                )
                circuit = (
                    shard_reader.load_circuit(
                        candidate_record.circuit_ref,
                        candidate_record.candidate_circuit_id,
                        candidate_record.circuit_hash,
                    )
                    if circuit_sharded
                    else load_candidate_circuit(
                        circuit_path, candidate_record.circuit_hash
                    )
                )
                rollout = (
                    shard_reader.load_rollout(
                        rollout_record.rollout_ref,
                        rollout_record.rollout_id,
                        circuit,
                        rollout_record.content_hash,
                    )
                    if rollout_sharded
                    else load_rollout_artifact(
                        rollout_path, circuit, rollout_record.content_hash
                    )
                )
                if candidate.sample_id != sample_id or rollout.sample_id != sample_id:
                    raise ValueError("hydrated Phase 9 artifact sample identity mismatch")
                candidates[candidate.action_id] = candidate
                circuits[rollout.candidate_circuit_id] = circuit
                rollouts[rollout.rollout_id] = rollout
                if index == total or index % 50 == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = index / elapsed
                    eta = (total - index) / rate if rate > 0 else 0.0
                    _log(
                        self.label,
                        f"sample={sample_id} artifacts {index}/{total} | "
                        f"rate={rate:.1f}/s | ETA≈{eta/60.0:.1f}m | RSS≈{_rss_gib():.2f} GiB",
                    )
                    _write_progress(
                        self.db_path.parent,
                        stage="hydrate_sample",
                        sample_id=sample_id,
                        completed=index,
                        total=total,
                    )
        ordered = tuple(sorted(rollouts.values(), key=lambda item: (item.rank, item.action_id)))
        hydrated = _HydratedSample(
            sample_id=sample_id,
            candidates_by_id=candidates,
            circuits_by_id=circuits,
            rollouts_by_id=rollouts,
            rollouts_by_sample_id={sample_id: ordered},
        )
        self._local.sample = hydrated
        _write_progress(self.db_path.parent, stage="ready", last_hydrated_sample=sample_id)
        _log(
            self.label,
            f"sample={sample_id} hydration complete in {(time.monotonic()-started)/60.0:.2f}m; "
            "full Phase 9 universe was never materialized",
        )
        return hydrated


class _CandidateRecordMapping(Mapping[str, ActionCandidateRecordV1]):
    def __init__(self, owner: LazyActionDataset) -> None:
        self.owner = owner

    def __len__(self) -> int:
        return self.owner.candidate_count

    def __iter__(self) -> Iterator[str]:
        with _connect(self.owner.db_path) as connection:
            for row in connection.execute("SELECT action_id FROM candidates ORDER BY action_id"):
                yield str(row["action_id"])

    def __getitem__(self, key: str) -> ActionCandidateRecordV1:
        return self.owner.candidate_record(key)


class _DictRow(dict[str, Any]):
    pass


def _artifact_path(root: Path, reference: str) -> tuple[Path, bool]:
    physical = archive_reference(reference)
    path = resolve_safe_file(root, physical, f"action artifact {reference}")
    return path, split_sharded_reference(reference) is not None or reference.endswith(".zip")


def load_lazy_action_dataset(
    action_root: str | Path,
    *,
    phase7: Any,
    graph: Any,
    checkpoint_root: str | Path,
    label: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> LazyActionDataset:
    """Validate Phase 9 and return a durable, per-sample lazy action source."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("lazy action index batch_size must be a positive integer")
    root = Path(action_root)
    index_root = Path(checkpoint_root)
    mode = _resume_mode()
    _log(label, f"opening Phase 9 lazily | root={root} | resume_mode={mode}")
    if not root.is_dir():
        raise FileNotFoundError(f"Phase 9 action root does not exist: {root}")
    marker_raw = strict_json_load(root / "action_complete.json")
    marker = dict(require_mapping(marker_raw, "action_complete.json"))
    if set(marker) != _ACTION_MARKER_KEYS or marker.get("complete") is not True:
        raise ValueError("Phase 9 action completion marker is malformed")
    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError("action_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    missing = _REQUIRED_MANAGED - set(managed_files)
    if missing:
        raise ValueError(f"Phase 9 managed inventory misses {sorted(missing)}")
    _log(label, f"validating {len(managed_files)} managed paths without hydrating artifacts")
    for reference in managed_files:
        resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
    actual = _file_inventory(root)
    if actual != set(managed_files):
        raise ValueError("Phase 9 managed file inventory mismatch")
    config = load_action_config(root / "action_config.json")
    expected_engine = action_engine_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        config,
    )
    expected = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": expected_engine,
        "operational_config_id": action_operational_config_id(config),
        "action_schema_id": action_schema_id(),
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
    }
    for name, value in expected.items():
        require_nonblank(marker.get(name), f"action_complete.json {name}")
        if marker.get(name) != value:
            raise ValueError(f"action_complete.json {name} mismatch")
    summary = dict(require_mapping(strict_json_load(root / "action_summary.json"), "action_summary.json"))
    candidate_count = _strict_count(marker, "candidate_count")
    rollout_count = _strict_count(marker, "rollout_count")
    _log(label, "hashing only Phase 9 control files and manifests for immutable provenance")
    control_snapshot = snapshot_managed_files(root, _CONTROL_FILES)
    inventory_hash = _artifact_inventory_hash(root, managed_files)
    identity = {
        "schema": _INDEX_SCHEMA,
        "action_engine_id": marker["action_engine_id"],
        "control_snapshot_hash": control_snapshot.aggregate_sha256,
        "artifact_inventory_hash": inventory_hash,
        "candidate_count": candidate_count,
        "rollout_count": rollout_count,
        "batch_size": batch_size,
    }
    identity_path = index_root / "identity.json"
    if mode == "off" and index_root.exists():
        shutil.rmtree(index_root)
    if identity_path.is_file():
        existing = strict_json_load(identity_path)
        if existing != identity:
            if mode == "repair":
                _quarantine(index_root, "lazy action index identity mismatch")
            else:
                raise RuntimeError(
                    "lazy Phase 9 index identity mismatch; use --resume-mode repair to quarantine it"
                )
    elif index_root.exists() and any(index_root.iterdir()):
        if mode == "repair":
            _quarantine(index_root, "lazy action index lacks identity marker")
        else:
            raise RuntimeError(
                "partial lazy Phase 9 index lacks identity marker; use --resume-mode repair"
            )
    index_root.mkdir(parents=True, exist_ok=True)
    if not identity_path.is_file():
        _atomic_json(identity_path, identity)
    db_path = index_root / "action_index.sqlite3"
    with _connect(db_path) as connection:
        _create_schema(connection)
        _ingest_manifest(
            label=label,
            root=index_root,
            connection=connection,
            manifest_path=root / "manifests" / "action_candidate_manifest.parquet",
            kind="candidates",
            expected_rows=candidate_count,
            batch_size=batch_size,
        )
        _ingest_manifest(
            label=label,
            root=index_root,
            connection=connection,
            manifest_path=root / "manifests" / "action_rollout_manifest.parquet",
            kind="rollouts",
            expected_rows=rollout_count,
            batch_size=batch_size,
        )
        _finalize_index(
            label,
            index_root,
            connection,
            candidate_count,
            rollout_count,
            {sample.sample_id for sample in phase7.samples},
        )
    return LazyActionDataset(
        root=root,
        config=config,
        completion_marker=marker,
        summary=summary,
        managed_files=managed_files,
        snapshot=control_snapshot,
        artifact_inventory_hash=inventory_hash,
        db_path=db_path,
        label=label,
    )


__all__ = ["LazyActionDataset", "load_lazy_action_dataset"]
