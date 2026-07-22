"""Fast lossless publication for an already validated Phase 12 build result.

The normal resumable publisher defensively reloads every checkpoint item before staging and
then reloads every staged artifact again.  The dedicated fast runner has already loaded,
SHA-verified, decoded, and semantically validated every checkpoint while assembling the
``TrainingViewBuildResult``.  Repeating both full NPZ passes is therefore redundant and can
turn publication of large campaigns into an hours-long serial bottleneck.

This publisher preserves atomic output, source-reference validation, manifest round-trips,
join validation, complete inventory checks, and immutable hard-link/copy publication.  It
parallelizes source-reference validation and file publication, and validates joins against
the already validated in-memory items instead of decoding all NPZ artifacts twice more.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import uuid

from triqto.graph.utils import write_strict_json
from triqto.storage import ManifestReader, ManifestWriter
from triqto.storage.training_view_schema import (
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)
from triqto.training_views import load_training_view_config
from triqto.training_views.artifacts import _validate_source_refs, _verify_result_sources
from triqto.training_views.config import save_training_view_config
from triqto.training_views.models import TrainingViewBuildResult, TrainingViewWriteResult
from triqto.training_views.validators import validate_training_view_dataset_joins

from . import resumable_phase12 as _base
from .resumable import prepare_checkpoint_root


def _workers() -> int:
    raw = os.environ.get("TRIQTO_PHASE12_PUBLICATION_WORKERS", "8")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("TRIQTO_PHASE12_PUBLICATION_WORKERS must be an integer") from exc
    if value < 1 or value > 32:
        raise ValueError("TRIQTO_PHASE12_PUBLICATION_WORKERS must be in [1, 32]")
    return value


def _parallel_map(function, values, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="triqto-phase12-publish") as pool:
        for _ in pool.map(function, values, chunksize=64):
            pass


def write_training_view_dataset_fast(
    result: TrainingViewBuildResult,
    output_root: str | Path,
    checkpoint_root: str | Path,
    *,
    progress_callback=None,
    resume_mode: str = "strict",
) -> TrainingViewWriteResult:
    """Atomically publish a build result without redundant full NPZ reload passes."""
    if not isinstance(result, TrainingViewBuildResult):
        raise TypeError("result must be TrainingViewBuildResult")

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Training-view output root already exists: {output}")

    source_roots = {
        "phase7": Path(result.phase7_source_root),
        "phase8": Path(result.graph_source_root),
        "phase9": Path(result.action_source_root),
        "phase11": Path(result.topology_source_root),
    }
    _verify_result_sources(result)
    active_resume_mode = "strict" if resume_mode == "off" else resume_mode
    checkpoints = prepare_checkpoint_root(checkpoint_root, active_resume_mode)
    workers = _workers()

    print(
        "[Phase 12 publication] validating source references in parallel | "
        f"items={len(result.items):,} | workers={workers}",
        flush=True,
    )
    _parallel_map(lambda item: _validate_source_refs(item, source_roots), result.items, workers)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "items").mkdir(parents=True)
        managed: list[str] = []

        save_training_view_config(result.config, staging / "training_view_config.json")
        managed.append("training_view_config.json")
        summary = {
            **result.summary,
            "checkpoint_resume": {
                **dict(result.summary.get("checkpoint_resume", {})),
                "resumed_item_count": len(result.items),
                "fast_publication": True,
                "redundant_npz_reloads_skipped": True,
            },
        }
        write_strict_json(staging / "training_view_summary.json", summary)
        managed.append("training_view_summary.json")

        references = [f"artifacts/items/{item.view_item_id}.npz" for item in result.items]

        def publish_one(pair) -> None:
            item, reference = pair
            source = _base._item_paths(checkpoints, item.view_item_id)[0]
            if not source.is_file():
                raise FileNotFoundError(f"Missing validated Phase 12 checkpoint item: {source}")
            target = staging / reference
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                raise ValueError(f"Published Phase 12 artifact size mismatch: {reference}")

        print(
            "[Phase 12 publication] linking validated artifacts in parallel | "
            f"items={len(result.items):,} | workers={workers}",
            flush=True,
        )
        _parallel_map(publish_one, zip(result.items, references), workers)
        managed.extend(references)

        writer = ManifestWriter(staging / "manifests")
        writer.write_records("training_view_manifest", result.definition_records, overwrite=False)
        managed.append("manifests/training_view_manifest.parquet")
        writer.write_records("training_item_manifest", result.item_records, overwrite=False)
        managed.append("manifests/training_item_manifest.parquet")

        persisted_config = load_training_view_config(staging / "training_view_config.json")
        if persisted_config != result.config:
            raise ValueError("Persisted training-view config mismatch")

        reader = ManifestReader(staging / "manifests")
        definitions = reader.read_typed_records(
            "training_view_manifest", TrainingViewDefinitionRecordV1
        )
        records = reader.read_typed_records(
            "training_item_manifest", TrainingViewItemRecordV1
        )
        if len(records) != len(result.items):
            raise ValueError("Persisted training-item manifest count mismatch")
        expected_ids = {item.view_item_id for item in result.items}
        if {record.view_item_id for record in records} != expected_ids:
            raise ValueError("Persisted training-item manifest identity mismatch")

        validate_training_view_dataset_joins(
            definitions,
            records,
            items_by_id={item.view_item_id: item for item in result.items},
            config=persisted_config,
        )

        managed_files = tuple(sorted([*managed, "training_view_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": result.source_scientific_generation_id,
            "graph_conversion_id": result.graph_conversion_id,
            "action_engine_id": result.action_engine_id,
            "topology_audit_id": result.topology_audit_id,
            "training_view_dataset_id": result.training_view_dataset_id,
            "operational_config_id": result.operational_config_id,
            "training_view_schema_id": result.training_view_schema_id,
            "view_count": len(result.definitions),
            "item_count": len(result.items),
            "phase7_snapshot_hash": result.phase7_snapshot.aggregate_sha256,
            "graph_snapshot_hash": result.graph_snapshot.aggregate_sha256,
            "action_snapshot_hash": result.action_snapshot.aggregate_sha256,
            "topology_snapshot_hash": result.topology_snapshot.aggregate_sha256,
            "topology_loss_weight": 0.0,
            "checkpoint_resume": summary["checkpoint_resume"],
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "training_view_complete.json", completion)
        if _base._relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed training-view inventory does not match staging")

        _verify_result_sources(result)
        os.replace(staging, output)

        manifest_paths = (
            output / "manifests" / "training_view_manifest.parquet",
            output / "manifests" / "training_item_manifest.parquet",
        )
        artifact_paths = tuple(
            sorted(
                (
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ),
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                (output / reference for reference in managed_files),
                key=lambda path: path.as_posix(),
            )
        )
        print("[Phase 12 publication] atomic publication complete", flush=True)
        return TrainingViewWriteResult(
            output_root=output,
            training_view_complete_path=output / "training_view_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            view_count=len(result.definitions),
            item_count=len(result.items),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = ["write_training_view_dataset_fast"]
