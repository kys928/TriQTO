"""Split-safe Phase 11 topology attachment for Phase 12 model-ready artifacts.

This module augments an immutable Phase 12 model-ready product with validated
Phase 11 topology feature vectors. It never mutates Phase 11, Phase 12, or the
source model-ready product. A new content-addressed output directory is staged,
validated, and published atomically.

Scientific boundaries:
* ``lambda_top`` remains exactly zero.
* Topology is an auxiliary input, never a supervised target.
* Cross-split Phase 11 cohorts remain audit-only.
* Action-neighborhood topology is never enabled for action-ranking or
  Born-prediction heads because it contains exact rollout/Born evidence.
* Hardware-masked attachment is disabled by default and can only be requested
  when the Phase 11 audit was generated with ``include_hilbert=false``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "triqto.phase11_phase12.topology_attachment.v1"
ATTACHMENT_VERSION = "1.0.0"
MODEL_INPUT_ARRAYS = (
    "x_topology_features",
    "x_topology_alignment_features",
    "x_topology_parameter_features",
    "x_topology_born_features",
)
FEATURE_FAMILIES = {
    "topology": ("topology_feature_names", "topology_feature_values"),
    "alignment": ("alignment_feature_names", "alignment_feature_values"),
    "parameter": ("parameter_feature_names", "parameter_feature_values"),
    "born": ("born_feature_names", "born_feature_values"),
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got {raw!r}")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class TopologyAttachmentConfig:
    phase11_root: Path
    phase12_root: Path
    model_ready_root: Path
    output_root: Path
    progress_every: int = 500
    clip_value: float = 10.0
    strict: bool = True
    copy_mode: str = "hardlink"
    attach_hardware_masked: bool = False
    enable_joint_diagnosis: bool = True
    dry_run: bool = False

    @staticmethod
    def from_environment() -> "TopologyAttachmentConfig":
        workspace = Path(
            os.environ.get(
                "TRIQTO_WORKSPACE",
                "/workspace/triqto-data/phase15_6_pilot_v2",
            )
        ).expanduser().resolve()
        model_ready_raw = os.environ.get("TRIQTO_MODEL_READY_ROOT")
        if not model_ready_raw:
            raise ValueError("TRIQTO_MODEL_READY_ROOT is required")
        copy_mode = os.environ.get(
            "TRIQTO_TOPOLOGY_ATTACH_COPY_MODE", "hardlink"
        ).strip().lower()
        if copy_mode not in {"hardlink", "copy"}:
            raise ValueError(
                "TRIQTO_TOPOLOGY_ATTACH_COPY_MODE must be hardlink or copy"
            )
        return TopologyAttachmentConfig(
            phase11_root=Path(
                os.environ.get(
                    "TRIQTO_PHASE11_ROOT", workspace / "data" / "phase11"
                )
            ).expanduser().resolve(),
            phase12_root=Path(
                os.environ.get(
                    "TRIQTO_PHASE12_ROOT", workspace / "data" / "phase12"
                )
            ).expanduser().resolve(),
            model_ready_root=Path(model_ready_raw).expanduser().resolve(),
            output_root=Path(
                os.environ.get(
                    "TRIQTO_TOPOLOGY_OUTPUT_ROOT",
                    workspace / "data" / "phase12_model_ready_topology",
                )
            ).expanduser().resolve(),
            progress_every=_env_int(
                "TRIQTO_TOPOLOGY_ATTACH_PROGRESS_EVERY", 500, 1
            ),
            clip_value=_env_float("TRIQTO_TOPOLOGY_ATTACH_CLIP", 10.0, 0.1),
            strict=_env_bool("TRIQTO_TOPOLOGY_ATTACH_STRICT", True),
            copy_mode=copy_mode,
            attach_hardware_masked=_env_bool(
                "TRIQTO_TOPOLOGY_ATTACH_HARDWARE", False
            ),
            enable_joint_diagnosis=_env_bool(
                "TRIQTO_TOPOLOGY_ENABLE_JOINT_DIAGNOSIS", True
            ),
            dry_run=_env_bool("TRIQTO_TOPOLOGY_ATTACH_DRY_RUN", False),
        )

    def serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "phase11_root",
            "phase12_root",
            "model_ready_root",
            "output_root",
        ):
            payload[key] = Path(payload[key]).as_posix()
        payload["lambda_top"] = 0.0
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            ).encode("utf-8")
        )
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _decode_json_utf8(array: np.ndarray) -> Any:
    raw = np.asarray(array, dtype=np.uint8).tobytes()
    return json.loads(raw.decode("utf-8"))


def _unicode_array(values: Sequence[str]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{max(1, width)}")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temp)
        os.replace(temp, path)
        return digest
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _safe_resolve(root: Path, reference: str) -> Path:
    ref = Path(str(reference))
    candidates = [ref] if ref.is_absolute() else [root / ref, root.parent / ref]
    root_resolved = root.resolve()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            if not ref.is_absolute():
                continue
        if resolved.is_file():
            return resolved
    rendered = "\n  ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(
        f"unable to resolve {reference!r}; tried:\n  {rendered}"
    )


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _hardlink_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path, mode: str) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _hardlink_or_copy(path, target, mode)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; install repository dependencies with "
            "python -m pip install -r requirements.txt -c constraints/cpu.txt"
        ) from exc
    return pa, pq


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    _pa, pq = _require_pyarrow()
    return pq.read_table(path).to_pylist()


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    pa, pq = _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    temp = path.with_name(path.name + ".tmp")
    pq.write_table(table, temp, compression="zstd")
    os.replace(temp, path)


def _find_first_string_by_key(value: Any, keys: set[str]) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                str(key).lower() in keys
                and isinstance(child, str)
                and child
            ):
                return child
        for child in value.values():
            found = _find_first_string_by_key(child, keys)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            found = _find_first_string_by_key(child, keys)
            if found is not None:
                return found
    return None


@dataclass(slots=True)
class GroupFeatures:
    group_id: str
    group_kind: str
    artifact_ref: str
    content_hash: str
    member_entity_ids: tuple[str, ...]
    missing_member_ids: tuple[str, ...]
    source_splits: tuple[str, ...]
    source_split_group_ids: tuple[str, ...]
    status: str
    hardware_safe: bool
    manifold_available_mask: np.ndarray
    names: dict[str, np.ndarray]
    values: dict[str, np.ndarray]

    @property
    def attachable(self) -> bool:
        return self.status == "attachable_same_split"


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    name: str
    transform: str
    center: float
    scale: float
    finite_count: int
    positive_infinity_count: int


def _action_neighborhood_sample_id(
    metadata: Mapping[str, Any],
    group_key: Any,
    known_entities: set[str],
) -> str | None:
    found = _find_first_string_by_key(
        metadata, {"sample_id", "source_sample_id", "entity_id"}
    )
    if found in known_entities:
        return found
    if isinstance(group_key, str):
        if group_key in known_entities:
            return group_key
        try:
            parsed = json.loads(group_key)
        except (ValueError, TypeError):
            parsed = None
        found = _find_first_string_by_key(
            parsed, {"sample_id", "source_sample_id", "entity_id"}
        )
        if found in known_entities:
            return found
    return None


def classify_group_membership(
    *,
    group_kind: str,
    metadata: Mapping[str, Any],
    group_key: Any,
    point_ids: Sequence[str],
    entity_split: Mapping[str, str],
    entity_split_group: Mapping[str, str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    """Resolve a Phase 11 group to model-ready entities and classify split safety."""
    known = set(entity_split)
    if group_kind == "action_neighborhood":
        sample_id = _action_neighborhood_sample_id(metadata, group_key, known)
        members = () if sample_id is None else (sample_id,)
        missing = (
            ()
            if sample_id is not None
            else ("unresolved_action_neighborhood_sample",)
        )
    else:
        unique_points = tuple(sorted({str(value) for value in point_ids}))
        members = tuple(value for value in unique_points if value in known)
        missing = tuple(value for value in unique_points if value not in known)

    splits = tuple(sorted({entity_split[value] for value in members}))
    split_groups = tuple(
        sorted({entity_split_group[value] for value in members})
    )
    if missing:
        status = "audit_only_unresolved_members"
    elif not members:
        status = "audit_only_no_model_ready_members"
    elif len(splits) != 1:
        status = "audit_only_cross_split"
    else:
        status = "attachable_same_split"
    return members, missing, splits, split_groups, status


def _load_phase11_config(phase11_root: Path) -> dict[str, Any]:
    for name in ("topology_config.json", "config/topology_config.json"):
        path = phase11_root / name
        if path.is_file():
            return _read_json(path)
    raise FileNotFoundError("Phase 11 topology_config.json is missing")


def _load_phase11_groups(
    config: TopologyAttachmentConfig,
    model_rows: list[dict[str, Any]],
) -> tuple[
    list[GroupFeatures],
    dict[str, list[GroupFeatures]],
    dict[str, Any],
]:
    manifest_path = (
        config.phase11_root
        / "manifests"
        / "topology_group_manifest.parquet"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = _read_parquet_rows(manifest_path)

    entity_split: dict[str, str] = {}
    entity_split_group: dict[str, str] = {}
    for row in model_rows:
        entity_id = str(row["entity_id"])
        split = str(row["split"])
        split_group = str(row["split_group_id"])
        previous = entity_split.setdefault(entity_id, split)
        previous_group = entity_split_group.setdefault(
            entity_id, split_group
        )
        if previous != split or previous_group != split_group:
            raise ValueError(
                f"entity {entity_id} appears in multiple split identities"
            )

    phase11_config = _load_phase11_config(config.phase11_root)
    include_hilbert = phase11_config.get("include_hilbert")
    if not isinstance(include_hilbert, bool):
        raise ValueError(
            "Phase 11 topology_config.json must declare include_hilbert as bool"
        )
    hardware_safe = not include_hilbert

    groups: list[GroupFeatures] = []
    entity_groups: dict[str, list[GroupFeatures]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        group_id = str(row["topology_group_id"])
        group_kind = str(row["group_kind"])
        artifact_ref = str(row["artifact_ref"])
        path = _safe_resolve(config.phase11_root, artifact_ref)
        expected_hash = str(row.get("content_hash") or "")
        payload = _load_npz(path)
        if "topology_metadata_json_utf8" not in payload:
            raise ValueError(
                f"{group_id} lacks topology_metadata_json_utf8"
            )
        metadata = _decode_json_utf8(
            payload["topology_metadata_json_utf8"]
        )
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"{group_id} topology metadata must be an object"
            )
        if (
            config.strict
            and expected_hash
            and str(metadata.get("content_hash") or "") != expected_hash
        ):
            raise ValueError(
                f"Phase 11 logical content hash mismatch for {group_id}"
            )
        point_ids = [
            str(value)
            for value in np.asarray(payload.get("point_ids", [])).tolist()
        ]
        members, missing, splits, split_groups, status = (
            classify_group_membership(
                group_kind=group_kind,
                metadata=metadata,
                group_key=row.get("group_key"),
                point_ids=point_ids,
                entity_split=entity_split,
                entity_split_group=entity_split_group,
            )
        )

        names: dict[str, np.ndarray] = {}
        values: dict[str, np.ndarray] = {}
        for family, (names_key, values_key) in FEATURE_FAMILIES.items():
            if names_key not in payload or values_key not in payload:
                raise ValueError(
                    f"{group_id} lacks {names_key}/{values_key}"
                )
            family_names = np.asarray(payload[names_key]).reshape(-1)
            family_values = np.asarray(
                payload[values_key], dtype=np.float64
            ).reshape(-1)
            if family_names.size != family_values.size:
                raise ValueError(
                    f"{group_id} {family} feature names/values length mismatch"
                )
            names[family] = family_names.copy()
            values[family] = family_values.copy()

        manifold_mask = np.asarray(
            payload.get("manifold_available_mask", []), dtype=bool
        ).reshape(-1)
        group = GroupFeatures(
            group_id=group_id,
            group_kind=group_kind,
            artifact_ref=artifact_ref,
            content_hash=expected_hash,
            member_entity_ids=members,
            missing_member_ids=missing,
            source_splits=splits,
            source_split_group_ids=split_groups,
            status=status,
            hardware_safe=hardware_safe,
            manifold_available_mask=manifold_mask,
            names=names,
            values=values,
        )
        groups.append(group)
        status_counts[status] += 1
        kind_counts[group_kind] += 1
        if group.attachable:
            for entity_id in members:
                entity_groups[entity_id].append(group)

        if index % config.progress_every == 0 or index == len(rows):
            print(
                f"  Phase 11 groups: {index:,}/{len(rows):,}",
                flush=True,
            )

    audit = {
        "group_count": len(groups),
        "status_counts": dict(sorted(status_counts.items())),
        "group_kind_counts": dict(sorted(kind_counts.items())),
        "phase11_include_hilbert": include_hilbert,
        "hardware_safe": hardware_safe,
    }
    return groups, entity_groups, audit


def _select_primary_groups(
    entity_groups: Mapping[str, list[GroupFeatures]],
    *,
    strict: bool,
) -> tuple[
    dict[str, GroupFeatures],
    dict[str, list[GroupFeatures]],
    dict[str, Any],
]:
    primary: dict[str, GroupFeatures] = {}
    cohorts: dict[str, list[GroupFeatures]] = {}
    missing_action = 0
    duplicate_action = 0
    for entity_id, groups in sorted(entity_groups.items()):
        actions = sorted(
            (
                group
                for group in groups
                if group.group_kind == "action_neighborhood"
            ),
            key=lambda group: group.group_id,
        )
        cohort_groups = sorted(
            (
                group
                for group in groups
                if group.group_kind != "action_neighborhood"
            ),
            key=lambda group: group.group_id,
        )
        cohorts[entity_id] = cohort_groups
        if not actions:
            missing_action += 1
            continue
        if len(actions) > 1:
            duplicate_action += 1
            if strict:
                raise ValueError(
                    f"entity {entity_id} has {len(actions)} "
                    "action-neighborhood topology groups"
                )
        primary[entity_id] = actions[0]
    return primary, cohorts, {
        "entities_with_primary_action_topology": len(primary),
        "entities_without_primary_action_topology": missing_action,
        "entities_with_duplicate_action_topology": duplicate_action,
        "entities_with_same_split_cohort_context": sum(
            bool(value) for value in cohorts.values()
        ),
    }


def _is_count_like(name: str) -> bool:
    lowered = name.lower()
    tokens = (
        "count",
        "essential",
        "finite_feature",
        "betti",
        "point_count",
        "dimension",
    )
    return any(token in lowered for token in tokens)


def fit_feature_scaler(
    name: str, values: Sequence[float]
) -> FeatureScaler:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    positive_inf = int(np.isposinf(array).sum())
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return FeatureScaler(
            name, "robust", 0.0, 1.0, 0, positive_inf
        )
    transform = (
        "log1p_robust"
        if _is_count_like(name) and np.all(finite >= 0.0)
        else "robust"
    )
    working = (
        np.log1p(finite) if transform == "log1p_robust" else finite
    )
    q25, median, q75 = np.quantile(working, [0.25, 0.5, 0.75])
    scale = float(q75 - q25)
    if not math.isfinite(scale) or scale <= 1.0e-12:
        mad = float(np.median(np.abs(working - median)))
        scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return FeatureScaler(
        name=name,
        transform=transform,
        center=float(median),
        scale=float(scale),
        finite_count=int(finite.size),
        positive_infinity_count=positive_inf,
    )


def _canonical_feature_names(
    groups: Iterable[GroupFeatures],
) -> dict[str, tuple[str, ...]]:
    canonical: dict[str, tuple[str, ...]] = {}
    for group in groups:
        for family in FEATURE_FAMILIES:
            names = tuple(
                str(value) for value in group.names[family].tolist()
            )
            existing = canonical.setdefault(family, names)
            if existing != names:
                raise ValueError(
                    f"Phase 11 {family} feature-name mapping drift "
                    f"in group {group.group_id}"
                )
    return canonical


def fit_train_only_scalers(
    primary_groups: Mapping[str, GroupFeatures],
    entity_split: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    unique_train_groups = {
        group.group_id: group
        for entity_id, group in primary_groups.items()
        if entity_split.get(entity_id) == "train"
    }
    if not unique_train_groups:
        raise ValueError(
            "no train-attached topology groups are available for scaler fitting"
        )
    canonical_names = _canonical_feature_names(
        unique_train_groups.values()
    )
    result: dict[str, Any] = {}
    for family, names in canonical_names.items():
        matrix = np.vstack(
            [
                group.values[family]
                for group in unique_train_groups.values()
            ]
        )
        columns = [
            asdict(fit_feature_scaler(name, matrix[:, index]))
            for index, name in enumerate(names)
        ]
        result[family] = {
            "fit_partition": "train",
            "fit_unique_group_count": len(unique_train_groups),
            "feature_names": list(names),
            "columns": columns,
        }
    return result, canonical_names


def transform_feature_vector(
    values: np.ndarray,
    scalers: Mapping[str, Any],
    *,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    columns = list(scalers.get("columns", []))
    if raw.size != len(columns):
        raise ValueError(
            f"feature width {raw.size} does not match "
            f"scaler width {len(columns)}"
        )
    finite_mask = np.isfinite(raw)
    positive_inf_mask = np.isposinf(raw)
    negative_inf_mask = np.isneginf(raw)
    transformed = np.zeros(raw.shape, dtype=np.float32)
    for index, column in enumerate(columns):
        if not finite_mask[index]:
            continue
        value = float(raw[index])
        if column["transform"] == "log1p_robust":
            value = math.log1p(max(value, 0.0))
        elif column["transform"] != "robust":
            raise ValueError(
                f"unsupported topology transform {column['transform']!r}"
            )
        scaled = (
            value - float(column["center"])
        ) / float(column["scale"])
        transformed[index] = np.float32(
            np.clip(scaled, -clip_value, clip_value)
        )
    return (
        transformed,
        finite_mask,
        positive_inf_mask,
        negative_inf_mask,
    )


def build_topology_model_arrays(
    group: GroupFeatures,
    scalers: Mapping[str, Any],
    canonical_names: Mapping[str, tuple[str, ...]],
    *,
    clip_value: float,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "x_topology_source_available_mask": np.asarray(
            True, dtype=np.bool_
        ),
        "x_topology_materialized_mask": np.asarray(
            True, dtype=np.bool_
        ),
        "x_topology_available_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_manifold_available_mask": (
            group.manifold_available_mask.astype(np.bool_)
        ),
    }
    output_prefix = {
        "topology": "x_topology",
        "alignment": "x_topology_alignment",
        "parameter": "x_topology_parameter",
        "born": "x_topology_born",
    }
    for family, prefix in output_prefix.items():
        observed_names = tuple(
            str(value) for value in group.names[family].tolist()
        )
        expected_names = canonical_names[family]
        if observed_names != expected_names:
            raise ValueError(
                f"{group.group_id} {family} feature mapping "
                "differs from train mapping"
            )
        transformed, finite, positive_inf, negative_inf = (
            transform_feature_vector(
                group.values[family],
                scalers[family],
                clip_value=clip_value,
            )
        )
        arrays[f"{prefix}_feature_names"] = _unicode_array(
            expected_names
        )
        arrays[f"{prefix}_features"] = transformed
        arrays[f"{prefix}_feature_mask"] = finite.astype(np.bool_)
        arrays[f"{prefix}_positive_infinity_mask"] = (
            positive_inf.astype(np.bool_)
        )
        arrays[f"{prefix}_negative_infinity_mask"] = (
            negative_inf.astype(np.bool_)
        )
    return arrays


def _set_named_mask(
    arrays: dict[str, np.ndarray],
    *,
    names_key: str,
    mask_key: str,
    name: str,
    enabled: bool,
) -> None:
    if names_key not in arrays or mask_key not in arrays:
        return
    names = [
        str(value)
        for value in np.asarray(arrays[names_key]).reshape(-1).tolist()
    ]
    mask = np.asarray(
        arrays[mask_key], dtype=np.bool_
    ).reshape(-1).copy()
    if mask.size != len(names):
        raise ValueError(
            f"{mask_key} length does not match {names_key}"
        )
    if name in names:
        mask[names.index(name)] = enabled
        arrays[mask_key] = mask


def _set_head_stream_mask(
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
    enabled_heads: set[str],
) -> None:
    names_key = f"y_{prefix}_head_names"
    groups_key = f"y_{prefix}_head_input_group_names"
    mask_key = f"y_{prefix}_head_input_mask"
    if (
        names_key not in arrays
        or groups_key not in arrays
        or mask_key not in arrays
    ):
        return
    heads = [
        str(value)
        for value in np.asarray(arrays[names_key]).reshape(-1).tolist()
    ]
    groups = [
        str(value)
        for value in np.asarray(arrays[groups_key]).reshape(-1).tolist()
    ]
    mask = np.asarray(arrays[mask_key], dtype=np.bool_).copy()
    if mask.shape != (len(heads), len(groups)):
        raise ValueError(f"{mask_key} shape does not match names")
    if "topology" not in groups:
        return
    topology_index = groups.index("topology")
    mask[:, topology_index] = False
    for head in enabled_heads:
        if head in heads:
            mask[heads.index(head), topology_index] = True
    arrays[mask_key] = mask


def attach_topology_to_item(
    source_arrays: Mapping[str, np.ndarray],
    topology_arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    attach_hardware: bool,
    enable_joint_diagnosis: bool,
) -> dict[str, np.ndarray]:
    """Return a new item payload with topology and conservative masks."""
    arrays = {
        name: np.asarray(value).copy()
        for name, value in source_arrays.items()
    }
    for name, value in topology_arrays.items():
        arrays[name] = np.asarray(value).copy()

    _set_named_mask(
        arrays,
        names_key="x_input_group_names",
        mask_key="x_input_group_available_mask",
        name="topology",
        enabled=True,
    )
    if task == "joint_multitask":
        enabled = {"topology_audit"}
        if enable_joint_diagnosis:
            enabled.add("diagnosis")
        _set_head_stream_mask(
            arrays, prefix="joint", enabled_heads=enabled
        )
    elif task == "hardware_masked":
        if not attach_hardware:
            raise ValueError(
                "hardware_masked topology attachment was not enabled"
            )
        _set_head_stream_mask(
            arrays,
            prefix="hardware",
            enabled_heads={"diagnosis"},
        )
    else:
        raise ValueError(
            f"topology attachment is not permitted for task {task!r}"
        )

    forbidden_targets = [
        name for name in arrays if name.startswith("y_topology")
    ]
    if forbidden_targets:
        raise ValueError(
            f"topology targets must remain absent: {forbidden_targets}"
        )
    return arrays


def _model_entity_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    entity_split: dict[str, str] = {}
    entity_split_group: dict[str, str] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        split = str(row["split"])
        split_group = str(row["split_group_id"])
        if (
            entity_id in entity_split
            and entity_split[entity_id] != split
        ):
            raise ValueError(
                f"entity {entity_id} crosses model-ready splits"
            )
        if (
            entity_id in entity_split_group
            and entity_split_group[entity_id] != split_group
        ):
            raise ValueError(
                f"entity {entity_id} crosses model-ready split groups"
            )
        entity_split[entity_id] = split
        entity_split_group[entity_id] = split_group
    return entity_split, entity_split_group


def _target_task_allowed(
    task: str,
    *,
    hardware_safe: bool,
    config: TopologyAttachmentConfig,
) -> bool:
    if task == "joint_multitask":
        return True
    if task == "hardware_masked":
        return config.attach_hardware_masked and hardware_safe
    return False


def _source_identity(
    config: TopologyAttachmentConfig,
) -> dict[str, Any]:
    required = {
        "phase11_complete": (
            config.phase11_root / "topology_complete.json"
        ),
        "phase11_manifest": (
            config.phase11_root
            / "manifests"
            / "topology_group_manifest.parquet"
        ),
        "phase12_complete": (
            config.phase12_root / "training_view_complete.json"
        ),
        "model_ready_complete": (
            config.model_ready_root / "preprocessed_complete.json"
        ),
        "model_ready_manifest": (
            config.model_ready_root
            / "manifests"
            / "processed_item_manifest.parquet"
        ),
    }
    identity: dict[str, Any] = {}
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        identity[name] = {
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
        }
    return identity


def _validate_roots(config: TopologyAttachmentConfig) -> None:
    for root in (
        config.phase11_root,
        config.phase12_root,
        config.model_ready_root,
    ):
        if not root.is_dir():
            raise NotADirectoryError(root)
    output = config.output_root.resolve()
    for source in (
        config.phase11_root,
        config.phase12_root,
        config.model_ready_root,
    ):
        if output == source.resolve() or _path_inside(output, source):
            raise ValueError(
                "topology output must not be a source root or its child"
            )
    config.output_root.mkdir(parents=True, exist_ok=True)


def _group_audit_rows(
    groups: Sequence[GroupFeatures],
) -> list[dict[str, Any]]:
    return [
        {
            "topology_group_id": group.group_id,
            "group_kind": group.group_kind,
            "artifact_ref": group.artifact_ref,
            "content_hash": group.content_hash,
            "status": group.status,
            "attachable": group.attachable,
            "hardware_safe": group.hardware_safe,
            "member_count": len(group.member_entity_ids),
            "missing_member_count": len(group.missing_member_ids),
            "member_entity_ids_json": json.dumps(
                list(group.member_entity_ids)
            ),
            "missing_member_ids_json": json.dumps(
                list(group.missing_member_ids)
            ),
            "source_splits_json": json.dumps(
                list(group.source_splits)
            ),
            "source_split_group_ids_json": json.dumps(
                list(group.source_split_group_ids)
            ),
        }
        for group in groups
    ]


def _entity_audit_rows(
    entity_split: Mapping[str, str],
    entity_split_group: Mapping[str, str],
    primary: Mapping[str, GroupFeatures],
    cohorts: Mapping[str, list[GroupFeatures]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in sorted(entity_split):
        group = primary.get(entity_id)
        cohort_groups = cohorts.get(entity_id, [])
        rows.append(
            {
                "entity_id": entity_id,
                "split": entity_split[entity_id],
                "split_group_id": entity_split_group[entity_id],
                "primary_topology_group_id": (
                    None if group is None else group.group_id
                ),
                "primary_group_kind": (
                    None if group is None else group.group_kind
                ),
                "topology_available": group is not None,
                "hardware_safe": (
                    False if group is None else group.hardware_safe
                ),
                "same_split_cohort_group_count": len(cohort_groups),
                "same_split_cohort_group_ids_json": json.dumps(
                    [value.group_id for value in cohort_groups]
                ),
                "cohort_policy": (
                    "audit_context_only_not_dense_model_input"
                ),
            }
        )
    return rows


def _update_model_input_contract(
    path: Path, config: TopologyAttachmentConfig
) -> None:
    payload = _read_json(path) if path.is_file() else {}
    payload["topology_attachment"] = {
        "schema_version": SCHEMA_VERSION,
        "lambda_top": 0.0,
        "feature_arrays": list(MODEL_INPUT_ARRAYS),
        "feature_map": "versioned in manifests/topology_scalers.json",
        "scaler_policy": (
            "fit on unique train-attached Phase 11 "
            "action-neighborhood groups only"
        ),
        "cross_split_policy": "audit_only",
        "cohort_policy": (
            "same-split cohorts are recorded for audit but are not "
            "collapsed into the dense model input"
        ),
        "head_policy": {
            "joint_multitask.diagnosis": (
                config.enable_joint_diagnosis
            ),
            "joint_multitask.action_ranking": False,
            "joint_multitask.born_prediction": False,
            "joint_multitask.topology_audit": True,
            "hardware_masked": config.attach_hardware_masked,
        },
        "leakage_boundary": (
            "action-neighborhood topology contains exact candidate "
            "rollout/Born evidence; it is forbidden for action-ranking "
            "and Born-prediction heads"
        ),
        "hardware_boundary": (
            "disabled by default; when explicitly enabled, Phase 11 "
            "must declare include_hilbert=false"
        ),
    }
    _write_json(path, payload)


def _completion_payload(
    *,
    config: TopologyAttachmentConfig,
    source_identity: Mapping[str, Any],
    run_id: str,
    processed_manifest_hash: str,
    scalers_hash: str,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "attachment_version": ATTACHMENT_VERSION,
        "run_id": run_id,
        "generated_at": _utc_now(),
        "lambda_top": 0.0,
        "source_identity": source_identity,
        "processed_item_manifest_sha256": processed_manifest_hash,
        "topology_scalers_sha256": scalers_hash,
        "counts": dict(counts),
        "scientific_boundaries": {
            "cross_split_groups_audit_only": True,
            "topology_supervised_target_present": False,
            "action_head_topology_enabled": False,
            "born_prediction_head_topology_enabled": False,
            "hardware_attachment_requested": (
                config.attach_hardware_masked
            ),
        },
    }


def attach_phase11_topology(
    config: TopologyAttachmentConfig,
) -> dict[str, Any]:
    """Run the complete immutable topology attachment stage."""
    _validate_roots(config)
    started = time.monotonic()
    source_identity = _source_identity(config)
    model_manifest_path = (
        config.model_ready_root
        / "manifests"
        / "processed_item_manifest.parquet"
    )
    model_rows = _read_parquet_rows(model_manifest_path)
    if not model_rows:
        raise ValueError("model-ready manifest is empty")
    entity_split, entity_split_group = _model_entity_maps(model_rows)

    print("Loading and classifying Phase 11 topology groups", flush=True)
    groups, entity_groups, phase11_audit = _load_phase11_groups(
        config, model_rows
    )
    primary, cohorts, entity_resolution_audit = (
        _select_primary_groups(entity_groups, strict=config.strict)
    )
    scalers, canonical_names = fit_train_only_scalers(
        primary, entity_split
    )

    config_payload = config.serializable()
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "attachment_version": ATTACHMENT_VERSION,
        "source_identity": source_identity,
        "config": config_payload,
        "topology_feature_map": {
            family: list(names)
            for family, names in canonical_names.items()
        },
    }
    run_id = (
        f"phase12_topology_{_sha256_json(identity_payload)[:24]}"
    )
    final_root = config.output_root / run_id
    if final_root.exists():
        marker = final_root / "topology_attachment_complete.json"
        if marker.is_file():
            payload = _read_json(marker)
            if payload.get("complete") is True:
                return {
                    "status": "already_complete",
                    "output_root": final_root,
                    **payload,
                }
        raise FileExistsError(
            f"incomplete or conflicting output already exists: {final_root}"
        )

    if (
        config.attach_hardware_masked
        and phase11_audit["phase11_include_hilbert"]
    ):
        raise ValueError(
            "hardware-masked topology attachment is forbidden because "
            "Phase 11 include_hilbert=true"
        )

    plan = {
        "run_id": run_id,
        "output_root": final_root.as_posix(),
        "model_item_count": len(model_rows),
        "unique_entity_count": len(entity_split),
        "primary_topology_entity_count": len(primary),
        "phase11_audit": phase11_audit,
        "entity_resolution_audit": entity_resolution_audit,
        "config": config_payload,
    }
    if config.dry_run:
        return {"status": "dry_run", "plan": plan}

    staging = config.output_root / (
        f".{run_id}.staging-{os.getpid()}-{int(time.time())}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    try:
        print("Copying immutable model-ready product", flush=True)
        _copy_tree(
            config.model_ready_root, staging, config.copy_mode
        )

        updated_rows: list[dict[str, Any]] = []
        attached_by_task: Counter[str] = Counter()
        unavailable_by_task: Counter[str] = Counter()
        source_hash_mismatches = 0
        for index, row in enumerate(model_rows, start=1):
            updated = dict(row)
            entity_id = str(row["entity_id"])
            task = str(row["task"])
            group = primary.get(entity_id)
            allowed = group is not None and _target_task_allowed(
                task,
                hardware_safe=(
                    False if group is None else group.hardware_safe
                ),
                config=config,
            )
            updated.update(
                {
                    "topology_attachment_status": "not_attached",
                    "topology_primary_group_id": (
                        None if group is None else group.group_id
                    ),
                    "topology_hardware_safe": (
                        False if group is None else group.hardware_safe
                    ),
                    "topology_feature_dim": (
                        0
                        if group is None
                        else int(group.values["topology"].size)
                    ),
                    "topology_alignment_feature_dim": (
                        0
                        if group is None
                        else int(group.values["alignment"].size)
                    ),
                    "topology_parameter_feature_dim": (
                        0
                        if group is None
                        else int(group.values["parameter"].size)
                    ),
                    "topology_born_feature_dim": (
                        0
                        if group is None
                        else int(group.values["born"].size)
                    ),
                    "topology_same_split_cohort_group_count": len(
                        cohorts.get(entity_id, [])
                    ),
                }
            )
            if allowed and group is not None:
                source_path = _safe_resolve(
                    config.model_ready_root,
                    str(row["artifact_ref"]),
                )
                if config.strict and row.get("content_hash"):
                    observed = _sha256_file(source_path)
                    if observed != str(row["content_hash"]):
                        source_hash_mismatches += 1
                        raise ValueError(
                            "model-ready source content hash mismatch: "
                            f"{entity_id}/{task}"
                        )
                destination = _safe_resolve(
                    staging, str(row["artifact_ref"])
                )
                source_arrays = _load_npz(source_path)
                topology_arrays = build_topology_model_arrays(
                    group,
                    scalers,
                    canonical_names,
                    clip_value=config.clip_value,
                )
                augmented = attach_topology_to_item(
                    source_arrays,
                    topology_arrays,
                    task=task,
                    attach_hardware=(task == "hardware_masked"),
                    enable_joint_diagnosis=(
                        config.enable_joint_diagnosis
                    ),
                )
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "attachment_version": ATTACHMENT_VERSION,
                    "phase11_topology_group_id": group.group_id,
                    "phase11_group_kind": group.group_kind,
                    "phase11_artifact_ref": group.artifact_ref,
                    "source_model_ready_content_hash": str(
                        row.get("content_hash") or ""
                    ),
                    "lambda_top": 0.0,
                    "topology_supervised_target_present": False,
                    "action_head_topology_enabled": False,
                    "born_prediction_head_topology_enabled": False,
                    "same_split_cohort_group_ids": [
                        value.group_id
                        for value in cohorts.get(entity_id, [])
                    ],
                    "cohort_policy": (
                        "audit_context_only_not_dense_model_input"
                    ),
                }
                augmented[
                    "topology_attachment_metadata_json_utf8"
                ] = np.frombuffer(
                    _canonical_json_bytes(metadata), dtype=np.uint8
                )
                output_hash = _atomic_save_npz(
                    destination, augmented
                )
                updated["content_hash"] = output_hash
                updated["topology_available_mask"] = True
                updated["topology_attachment_status"] = "attached"
                attached_by_task[task] += 1
            else:
                unavailable_by_task[task] += 1
            updated_rows.append(updated)

            if (
                index % config.progress_every == 0
                or index == len(model_rows)
            ):
                print(
                    f"  model-ready items: {index:,}/{len(model_rows):,} "
                    f"attached={sum(attached_by_task.values()):,}",
                    flush=True,
                )

        manifest_path = (
            staging
            / "manifests"
            / "processed_item_manifest.parquet"
        )
        _write_parquet_rows(manifest_path, updated_rows)
        _write_parquet_rows(
            staging / "manifests" / "topology_group_audit.parquet",
            _group_audit_rows(groups),
        )
        _write_parquet_rows(
            staging / "manifests" / "topology_entity_manifest.parquet",
            _entity_audit_rows(
                entity_split,
                entity_split_group,
                primary,
                cohorts,
            ),
        )
        _write_json(
            staging
            / "manifests"
            / "topology_attachment_config.json",
            config_payload,
        )
        _write_json(
            staging / "manifests" / "topology_scalers.json",
            scalers,
        )
        _write_json(
            staging / "reports" / "topology_source_audit.json",
            phase11_audit,
        )
        _write_json(
            staging
            / "reports"
            / "topology_entity_resolution_audit.json",
            entity_resolution_audit,
        )
        _update_model_input_contract(
            staging / "manifests" / "model_input_contract.json",
            config,
        )

        counts = {
            "source_model_items": len(model_rows),
            "published_model_items": len(updated_rows),
            "unique_entities": len(entity_split),
            "entities_with_primary_topology": len(primary),
            "phase11_groups": len(groups),
            "attachable_phase11_groups": sum(
                group.attachable for group in groups
            ),
            "audit_only_phase11_groups": sum(
                not group.attachable for group in groups
            ),
            "attached_model_items": sum(attached_by_task.values()),
            "joint_multitask_attached": attached_by_task.get(
                "joint_multitask", 0
            ),
            "hardware_masked_attached": attached_by_task.get(
                "hardware_masked", 0
            ),
            "source_hash_mismatches": source_hash_mismatches,
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "attachment_version": ATTACHMENT_VERSION,
            "run_id": run_id,
            "generated_at": _utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "counts": counts,
            "attached_by_task": dict(
                sorted(attached_by_task.items())
            ),
            "not_attached_by_task": dict(
                sorted(unavailable_by_task.items())
            ),
            "phase11_audit": phase11_audit,
            "entity_resolution_audit": entity_resolution_audit,
            "lambda_top": 0.0,
            "topology_target_present": False,
        }
        _write_json(
            staging
            / "reports"
            / "topology_attachment_report.json",
            report,
        )
        summary_lines = [
            "# TriQTO Phase 11 → Phase 12 topology attachment",
            "",
            f"- Model-ready source items: **{len(model_rows):,}**",
            f"- Unique entities: **{len(entity_split):,}**",
            f"- Phase 11 groups audited: **{len(groups):,}**",
            (
                "- Entities with primary action-neighborhood topology: "
                f"**{len(primary):,}**"
            ),
            (
                "- Model-ready items with materialized topology: "
                f"**{sum(attached_by_task.values()):,}**"
            ),
            (
                "- Joint multitask items attached: "
                f"**{attached_by_task.get('joint_multitask', 0):,}**"
            ),
            (
                "- Hardware-masked items attached: "
                f"**{attached_by_task.get('hardware_masked', 0):,}**"
            ),
            "- Cross-split cohorts: **audit-only**",
            "- Topology loss weight: **0.0**",
            "- Topology supervised targets: **none**",
            (
                "- Action-ranking and Born-prediction heads: "
                "**topology forbidden**"
            ),
            "",
            (
                "Dense inputs are fitted with train-only robust scalers. "
                "Same-split cohort groups are"
            ),
            (
                "retained in the audit manifests but are not collapsed "
                "into the dense per-entity input."
            ),
        ]
        _write_text(
            staging
            / "reports"
            / "topology_attachment_summary.md",
            "\n".join(summary_lines),
        )

        processed_manifest_hash = _sha256_file(manifest_path)
        scalers_path = (
            staging / "manifests" / "topology_scalers.json"
        )
        completion = _completion_payload(
            config=config,
            source_identity=source_identity,
            run_id=run_id,
            processed_manifest_hash=processed_manifest_hash,
            scalers_hash=_sha256_file(scalers_path),
            counts=counts,
        )
        _write_json(
            staging / "topology_attachment_complete.json",
            completion,
        )
        _write_json(
            staging / "preprocessed_complete.json", completion
        )

        if _source_identity(config) != source_identity:
            raise RuntimeError(
                "one or more immutable source identity files changed "
                "during attachment"
            )
        os.replace(staging, final_root)
        return {
            "status": "complete",
            "output_root": final_root,
            **completion,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ATTACHMENT_VERSION",
    "SCHEMA_VERSION",
    "FeatureScaler",
    "GroupFeatures",
    "TopologyAttachmentConfig",
    "attach_phase11_topology",
    "attach_topology_to_item",
    "build_topology_model_arrays",
    "classify_group_membership",
    "fit_feature_scaler",
    "fit_train_only_scalers",
    "transform_feature_vector",
]
