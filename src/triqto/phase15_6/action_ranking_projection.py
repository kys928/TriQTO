"""Lossless Phase 12 action-ranking projection over immutable Phase 9 shards.

The canonical action-ranking view needs candidate edits plus a small set of rollout
metadata fields.  It does not consume candidate QPY circuits or the large metric,
bitstring, and probability arrays stored in each rollout NPZ.  This module therefore
projects exactly the required fields while preserving every candidate and every source
reference.

For sharded Phase 9 data, rollout NPZ members are ZIP_STORED inside the outer shard.
A bounded seekable view opens the nested NPZ in place, so only its tiny metadata NPY
member is decompressed.  Candidate action JSON remains fully decoded and validated;
candidate circuits are never reconstructed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BufferedReader, BytesIO, RawIOBase
import json
import math
from pathlib import Path
import sqlite3
import struct
import time
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from triqto.actions.artifacts import (
    _ROLLOUT_ARRAY_NAMES,
    _ROLLOUT_METADATA_NAME,
    _decode_json_bytes,
    load_action_artifact,
)
from triqto.actions.constants import ROLLOUT_ARTIFACT_SCHEMA_VERSION
from triqto.actions.sharded_artifacts import (
    _candidate_from_payload,
    archive_reference,
    split_sharded_reference,
)
from triqto.graph.utils import resolve_safe_file, strict_json_loads
from triqto.training_views.action_ranking_view import (
    build_action_ranking_items as _canonical_action_ranking_builder,
)
from triqto.training_views.base_view import (
    graph_structure_arrays,
    make_training_item,
    unicode_array,
)
from triqto.training_views.context import ViewBuildContext
from triqto.training_views.models import TrainingViewItem


_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_SIGNATURE = 0x04034B50
_EXPECTED_NPZ_MEMBERS = {
    *(f"{name}.npy" for name in _ROLLOUT_ARRAY_NAMES),
    f"{_ROLLOUT_METADATA_NAME}.npy",
}
_EXPECTED_ROLLOUT_METADATA_KEYS = {
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


@dataclass(frozen=True, slots=True)
class ActionRankingProjection:
    """The exact Phase 9 fields consumed by one Phase 12 action-ranking row."""

    action_id: str
    edits: tuple[Any, ...]
    generation_sources: tuple[str, ...]
    risk_score: float
    rank: int
    reward: float
    selected: bool
    dominates_baseline: bool
    primary_metric_nonworsening: bool
    depth_delta: int
    gate_delta: int
    circuit_ref: str
    action_ref: str
    rollout_ref: str


class _StoredZipMemberView(RawIOBase):
    """Seekable bounded view over one uncompressed member of a physical ZIP file."""

    def __init__(self, archive_path: Path, info: ZipInfo) -> None:
        if info.compress_type != ZIP_STORED:
            raise ValueError("bounded ZIP member view requires ZIP_STORED content")
        self._handle = archive_path.open("rb")
        self._length = int(info.file_size)
        self._position = 0
        self._handle.seek(info.header_offset)
        raw = self._handle.read(_LOCAL_FILE_HEADER.size)
        if len(raw) != _LOCAL_FILE_HEADER.size:
            self._handle.close()
            raise ValueError("truncated outer ZIP local-file header")
        (
            signature,
            _version,
            flags,
            compression,
            _mtime,
            _mdate,
            _crc,
            _compressed_size,
            _file_size,
            name_length,
            extra_length,
        ) = _LOCAL_FILE_HEADER.unpack(raw)
        if signature != _LOCAL_FILE_SIGNATURE:
            self._handle.close()
            raise ValueError("invalid outer ZIP local-file signature")
        if flags & 0x1:
            self._handle.close()
            raise ValueError("encrypted Phase 9 shard members are unsupported")
        if compression != ZIP_STORED:
            self._handle.close()
            raise ValueError("outer rollout NPZ member is not stored")
        self._start = info.header_offset + _LOCAL_FILE_HEADER.size + name_length + extra_length

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._position + offset
        elif whence == 2:
            target = self._length + offset
        else:
            raise ValueError(f"unsupported seek whence {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = min(target, self._length)
        return self._position

    def readinto(self, buffer: Any) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed member view")
        remaining = self._length - self._position
        if remaining <= 0:
            return 0
        count = min(len(buffer), remaining)
        self._handle.seek(self._start + self._position)
        data = self._handle.read(count)
        size = len(data)
        buffer[:size] = data
        self._position += size
        return size

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


class _ProjectionArtifactReader:
    def __init__(self, root: Path, config: Any) -> None:
        self.root = root
        self.config = config
        self._archives: dict[Path, ZipFile] = {}

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "_ProjectionArtifactReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _archive(self, reference: str) -> tuple[Path, ZipFile]:
        physical = archive_reference(reference)
        path = resolve_safe_file(self.root, physical, f"action shard {reference}")
        archive = self._archives.get(path)
        if archive is None:
            archive = ZipFile(path, "r")
            self._archives[path] = archive
        return path, archive

    @staticmethod
    def _member(reference: str, fallback: str) -> str:
        split = split_sharded_reference(reference)
        return fallback if split is None else split[1]

    @staticmethod
    def _is_sharded(reference: str) -> bool:
        return split_sharded_reference(reference) is not None or reference.endswith(".zip")

    def load_candidate(self, row: sqlite3.Row) -> Any:
        reference = str(row["action_ref"])
        if not self._is_sharded(reference):
            path = resolve_safe_file(self.root, reference, f"action artifact {reference}")
            return load_action_artifact(path, self.config, str(row["content_hash"]))
        _path, archive = self._archive(reference)
        member = self._member(reference, f"actions/{row['action_id']}.json")
        payload = strict_json_loads(archive.read(member).decode("utf-8"))
        return _candidate_from_payload(payload, self.config, str(row["content_hash"]))

    def load_rollout_metadata(self, row: sqlite3.Row) -> dict[str, Any]:
        reference = str(row["rollout_ref"])
        if not self._is_sharded(reference):
            path = resolve_safe_file(self.root, reference, f"rollout artifact {reference}")
            with ZipFile(path, "r") as npz_archive:
                return _decode_rollout_metadata_member(npz_archive)

        archive_path, archive = self._archive(reference)
        member = self._member(reference, f"rollouts/{row['rollout_id']}.npz")
        info = archive.getinfo(member)
        if info.compress_type == ZIP_STORED:
            raw_view = _StoredZipMemberView(archive_path, info)
            buffered = BufferedReader(raw_view)
            try:
                with ZipFile(buffered, "r") as npz_archive:
                    return _decode_rollout_metadata_member(npz_archive)
            finally:
                buffered.close()

        # Compatibility fallback for legacy shards whose outer rollout member was compressed.
        with ZipFile(BytesIO(archive.read(member)), "r") as npz_archive:
            return _decode_rollout_metadata_member(npz_archive)


def _decode_rollout_metadata_member(npz_archive: ZipFile) -> dict[str, Any]:
    names = set(npz_archive.namelist())
    if names != _EXPECTED_NPZ_MEMBERS:
        raise ValueError(
            "Rollout projection NPZ member mismatch; "
            f"missing={sorted(_EXPECTED_NPZ_MEMBERS - names)}, "
            f"unexpected={sorted(names - _EXPECTED_NPZ_MEMBERS)}"
        )
    raw = npz_archive.read(f"{_ROLLOUT_METADATA_NAME}.npy")
    array = np.load(BytesIO(raw), allow_pickle=False)
    metadata = _decode_json_bytes(array, _ROLLOUT_METADATA_NAME)
    if set(metadata) != _EXPECTED_ROLLOUT_METADATA_KEYS:
        raise ValueError("Rollout projection metadata-key mismatch")
    if metadata["artifact_schema_version"] != ROLLOUT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported rollout artifact schema version")
    if not isinstance(metadata["metadata"], Mapping):
        raise TypeError("Rollout projection metadata.metadata must be a mapping")
    return metadata


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be integer and not bool")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _validate_candidate_against_row(candidate: Any, row: sqlite3.Row) -> None:
    expected = {
        "action_id": str(row["action_id"]),
        "sample_id": str(row["sample_id"]),
        "graph_pair_id": str(row["graph_pair_id"]),
        "source_circuit_id": str(row["source_circuit_id"]),
        "source_run_id": str(row["source_run_id"]),
        "distortion_id": str(row["distortion_id"]),
    }
    for name, value in expected.items():
        if getattr(candidate, name) != value:
            raise ValueError(f"Projected candidate {name} does not match manifest index")
    if len(candidate.edits) != int(row["edit_count"]):
        raise ValueError("Projected candidate edit count does not match manifest index")
    if float(candidate.risk_score) != float(row["candidate_risk_score"]):
        raise ValueError("Projected candidate risk does not match manifest index")
    indexed_sources = tuple(json.loads(str(row["generation_sources_json"])))
    if tuple(candidate.generation_sources) != indexed_sources:
        raise ValueError("Projected candidate generation sources do not match manifest index")


def _validate_rollout_metadata(metadata: Mapping[str, Any], row: sqlite3.Row) -> None:
    identity_fields = {
        "rollout_id": "rollout_id",
        "action_id": "action_id",
        "sample_id": "sample_id",
        "graph_pair_id": "graph_pair_id",
        "candidate_circuit_id": "candidate_circuit_id",
        "clean_target_run_id": "clean_target_run_id",
        "scientific_config_id": "scientific_config_id",
        "content_hash": "rollout_content_hash",
    }
    for metadata_name, row_name in identity_fields.items():
        if metadata[metadata_name] != str(row[row_name]):
            raise ValueError(
                f"Projected rollout {metadata_name} does not match manifest index"
            )
    exact_fields = {
        "rank": int(row["rank_value"]),
        "reward": float(row["reward"]),
        "risk_score": float(row["rollout_risk_score"]),
        "dominates_baseline": bool(row["dominates_baseline"]),
        "primary_metric_nonworsening": bool(row["primary_metric_nonworsening"]),
        "selected": bool(row["selected"]),
    }
    for name, expected in exact_fields.items():
        if metadata[name] != expected:
            raise ValueError(f"Projected rollout {name} does not match manifest index")
    _require_integer(metadata["rank"], "rollout.rank")
    _require_finite_number(metadata["reward"], "rollout.reward")
    _require_finite_number(metadata["risk_score"], "rollout.risk_score")
    _require_bool(metadata["dominates_baseline"], "rollout.dominates_baseline")
    _require_bool(
        metadata["primary_metric_nonworsening"],
        "rollout.primary_metric_nonworsening",
    )
    _require_bool(metadata["selected"], "rollout.selected")
    _require_integer(metadata["depth_delta"], "rollout.depth_delta")
    _require_integer(metadata["gate_delta"], "rollout.gate_delta")


def _projection_rows(action: Any, sample_id: str) -> list[ActionRankingProjection]:
    connection = sqlite3.connect(action.db_path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=120000")
    try:
        rows = connection.execute(
            """
            SELECT c.action_id,c.sample_id,c.graph_pair_id,c.source_circuit_id,
                   c.source_run_id,c.distortion_id,c.candidate_circuit_id,
                   c.generation_sources_json,c.action_ref,c.circuit_ref,
                   c.content_hash,c.edit_count,c.risk_score AS candidate_risk_score,
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
    finally:
        connection.close()
    if not rows:
        raise KeyError(sample_id)

    started = time.monotonic()
    result: list[ActionRankingProjection] = []
    with _ProjectionArtifactReader(action.root, action.config) as reader:
        for index, row in enumerate(rows, start=1):
            candidate = reader.load_candidate(row)
            metadata = reader.load_rollout_metadata(row)
            _validate_candidate_against_row(candidate, row)
            _validate_rollout_metadata(metadata, row)
            result.append(
                ActionRankingProjection(
                    action_id=str(row["action_id"]),
                    edits=tuple(candidate.edits),
                    generation_sources=tuple(candidate.generation_sources),
                    risk_score=float(candidate.risk_score),
                    rank=int(metadata["rank"]),
                    reward=float(metadata["reward"]),
                    selected=bool(metadata["selected"]),
                    dominates_baseline=bool(metadata["dominates_baseline"]),
                    primary_metric_nonworsening=bool(
                        metadata["primary_metric_nonworsening"]
                    ),
                    depth_delta=int(metadata["depth_delta"]),
                    gate_delta=int(metadata["gate_delta"]),
                    circuit_ref=str(row["circuit_ref"]),
                    action_ref=str(row["action_ref"]),
                    rollout_ref=str(row["rollout_ref"]),
                )
            )
            if index == len(rows) or index % 50 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = index / elapsed
                print(
                    "[Phase 12][ranking-projection] "
                    f"sample={sample_id} candidates={index}/{len(rows)} | "
                    f"rate={rate:.1f}/s | circuits_skipped={index} | "
                    "rollout_arrays_skipped=true",
                    flush=True,
                )
    return result


def build_action_ranking_items_projected(
    context: ViewBuildContext,
) -> list[TrainingViewItem]:
    """Build canonical action-ranking items from a metadata-only lazy projection."""
    action = context.sources.action
    if not (
        getattr(action, "is_lazy", False)
        and hasattr(action, "db_path")
        and hasattr(action, "root")
    ):
        return _canonical_action_ranking_builder(context)

    task = "action_ranking"
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        pair_record = context.pair_records_by_sample_id.get(sample.sample_id)
        if pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        distorted_graph = context.sources.graph.graphs_by_id[pair_record.distorted_graph_id]
        graph_record = context.graph_records_by_id[pair_record.distorted_graph_id]
        projections = _projection_rows(action, sample.sample_id)
        if len(projections) > context.config.max_candidates_per_item:
            raise RuntimeError(
                f"Sample {sample.sample_id} has {len(projections)} candidates, exceeding "
                f"max_candidates_per_item={context.config.max_candidates_per_item}"
            )

        candidate_ids: list[str] = []
        features: list[list[float]] = []
        ranks: list[int] = []
        rewards: list[float] = []
        selected: list[bool] = []
        dominates: list[bool] = []
        nonworsening: list[bool] = []
        privileged: list[bool] = []
        edit_ptr = [0]
        edit_types: list[str] = []
        edit_magnitudes: list[float] = []
        edit_qubit_ptr = [0]
        edit_qubits: list[int] = []
        source_refs: list[tuple[str, str, str]] = [
            ("phase8", "provenance", graph_record.graph_ref),
            ("phase8", "provenance", pair_record.pair_ref),
        ]
        for projection in projections:
            is_privileged = "oracle_inverse" in projection.generation_sources
            candidate_ids.append(projection.action_id)
            features.append(
                [
                    float(len(projection.edits)),
                    projection.risk_score,
                    float(projection.depth_delta),
                    float(projection.gate_delta),
                    float(len(projection.edits) == 0),
                ]
            )
            ranks.append(projection.rank)
            rewards.append(projection.reward)
            selected.append(projection.selected)
            dominates.append(projection.dominates_baseline)
            nonworsening.append(projection.primary_metric_nonworsening)
            privileged.append(is_privileged)
            for edit in projection.edits:
                edit_types.append(edit.edit_type)
                edit_magnitudes.append(float(edit.magnitude))
                edit_qubits.extend(int(qubit) for qubit in edit.qubits)
                edit_qubit_ptr.append(len(edit_qubits))
            edit_ptr.append(len(edit_types))
            source_refs.extend(
                (
                    ("phase9", "input", projection.circuit_ref),
                    ("phase9", "provenance", projection.action_ref),
                    ("phase9", "target_provenance", projection.rollout_ref),
                )
            )
        if not candidate_ids:
            raise ValueError(f"Sample {sample.sample_id} has no action candidates")
        if sum(selected) != 1:
            raise ValueError(
                f"Sample {sample.sample_id} must have exactly one selected action target"
            )

        arrays = graph_structure_arrays(distorted_graph)
        arrays.update(
            {
                "action_candidate_ids": unicode_array(candidate_ids),
                "action_candidate_feature_names": unicode_array(
                    (
                        "edit_count",
                        "risk_score",
                        "depth_delta",
                        "gate_delta",
                        "is_no_op",
                    )
                ),
                "action_candidate_features": np.asarray(features, dtype=np.float64),
                "action_edit_ptr": np.asarray(edit_ptr, dtype=np.int64),
                "action_edit_types": unicode_array(edit_types),
                "action_edit_magnitudes": np.asarray(
                    edit_magnitudes,
                    dtype=np.float64,
                ),
                "action_edit_qubit_ptr": np.asarray(edit_qubit_ptr, dtype=np.int64),
                "action_edit_qubits": np.asarray(edit_qubits, dtype=np.int64),
                "action_target_rank": np.asarray(ranks, dtype=np.int64),
                "action_target_reward": np.asarray(rewards, dtype=np.float64),
                "action_target_selected_mask": np.asarray(selected, dtype=np.bool_),
                "action_target_dominates_baseline_mask": np.asarray(
                    dominates,
                    dtype=np.bool_,
                ),
                "action_target_primary_metric_nonworsening_mask": np.asarray(
                    nonworsening,
                    dtype=np.bool_,
                ),
                "action_privileged_oracle_mask": np.asarray(
                    privileged,
                    dtype=np.bool_,
                ),
            }
        )
        items.append(
            make_training_item(
                dataset_id=context.dataset_id,
                view_id=view_id,
                task=task,
                split=context.sample_splits[sample.sample_id],
                split_group_id=context.sample_split_groups[sample.sample_id],
                entity_id=sample.sample_id,
                input_available=(True, True, False),
                target_available=(True, True, True),
                arrays=arrays,
                source_refs=source_refs,
                hilbert_available=False,
                topology_available=False,
                privileged_target_available=any(privileged),
                metadata={
                    "sample_id": sample.sample_id,
                    "graph_pair_id": pair_record.graph_pair_id,
                    "candidate_order": "sorted_action_id_not_target_rank",
                    "candidate_count": len(candidate_ids),
                    "metric_context_available": False,
                    "clean_target_metrics_are_inputs": False,
                    "rollout_artifacts_are_target_provenance_only": True,
                    "generation_sources_excluded_from_candidate_inputs": True,
                    "privileged_oracle_candidates_retained_with_explicit_mask": True,
                    "identifiability_status": sample.metadata.get(
                        "identifiability_status"
                    ),
                    "identifiability_reason": sample.metadata.get(
                        "identifiability_reason"
                    ),
                    "diagnosis_supervision_mask": sample.metadata.get(
                        "diagnosis_supervision_mask"
                    ),
                    "action_supervision_mask": sample.metadata.get(
                        "action_supervision_mask"
                    ),
                    "hardware_data": False,
                },
                max_source_refs=context.config.max_source_refs_per_item,
            )
        )
    return items


def install_action_ranking_projection() -> None:
    """Install the projection builder into the resumable Phase 12 task dispatcher."""
    from . import resumable_phase12

    resumable_phase12.build_action_ranking_items = build_action_ranking_items_projected


__all__ = [
    "ActionRankingProjection",
    "build_action_ranking_items_projected",
    "install_action_ranking_projection",
]
