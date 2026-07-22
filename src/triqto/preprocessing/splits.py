"""Hierarchical group-aware challenge splits with independent leakage verification."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping

from .config import SplitConfig
from .records import LeakageRelation, ProcessedSample, SplitAssignment, SplitStatus


@dataclass(slots=True)
class SplitBuildResult:
    assignments: list[SplitAssignment]
    statuses: list[SplitStatus]
    leakage_violations: dict[str, list[dict[str, Any]]]


def _stable_unit(*, key: Any, seed: int, namespace: str) -> float:
    payload = json.dumps(
        {"key": key, "seed": seed, "namespace": namespace},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _partition_from_unit(value: float, config: SplitConfig) -> str:
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _parse_timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _parameter_region(sample: ProcessedSample, bins: int) -> str | None:
    if not sample.parameter_bindings_canonical:
        return None
    name = sorted(sample.parameter_bindings_canonical)[0]
    value = float(sample.parameter_bindings_canonical[name])
    unit = ((value + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi)
    index = min(bins - 1, int(math.floor(unit * bins)))
    return f"{name}:region:{index:02d}/{bins:02d}"


def _distortion_strength_bin(sample: ProcessedSample) -> str | None:
    raw = sample.provenance.get("distortion_strength")
    if raw is None:
        return sample.severity
    try:
        value = abs(float(raw))
    except (TypeError, ValueError):
        return None
    if value == 0.0:
        return "zero"
    exponent = math.floor(math.log10(value))
    return f"10^{exponent}"


def _distortion_combination(sample: ProcessedSample) -> str:
    raw = sample.provenance.get("distortion_components")
    if isinstance(raw, (list, tuple)) and raw:
        return "+".join(sorted(str(item) for item in raw))
    return sample.intervention_label


def _layout_identity(sample: ProcessedSample) -> str:
    layout = sample.hardware_context.get("layout")
    if layout:
        return json.dumps(layout, sort_keys=True, separators=(",", ":"), default=str)
    return sample.hashes.labeled_graph_hash


def _backend(sample: ProcessedSample) -> str | None:
    value = sample.hardware_context.get("backend_name")
    return None if value in (None, "") else str(value)


def _calibration_group(sample: ProcessedSample) -> str | None:
    for key in ("calibration_window_id", "calibration_snapshot_id", "backend_run_id"):
        value = sample.hardware_context.get(key)
        if value not in (None, ""):
            return str(value)
    timestamp = _parse_timestamp(sample.hardware_context.get("calibration_timestamp"))
    return None if timestamp is None else f"timestamp:{timestamp:.6f}"


def _strata(sample: ProcessedSample, fields: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        if field == "family":
            value = sample.family
        elif field == "n_qubits":
            value = sample.n_qubits
        elif field == "intervention_label":
            value = sample.intervention_label
        elif field == "severity":
            value = sample.severity
        else:
            value = getattr(sample, field, None)
        values.append(f"{field}={value}")
    return tuple(values)


def _group_members(
    samples: list[ProcessedSample],
    key_fn: Callable[[ProcessedSample], Any],
) -> tuple[dict[str, list[ProcessedSample]], list[str]]:
    groups: dict[str, list[ProcessedSample]] = defaultdict(list)
    missing: list[str] = []
    for sample in samples:
        raw_key = key_fn(sample)
        if raw_key is None or raw_key == "":
            missing.append(sample.sample_id)
            continue
        key = json.dumps(raw_key, sort_keys=True, separators=(",", ":"), default=str)
        groups[key].append(sample)
    for members in groups.values():
        members.sort(key=lambda item: item.sample_id)
    return dict(groups), sorted(missing)


def _greedy_stratified_group_assignment(
    groups: Mapping[str, list[ProcessedSample]],
    *,
    split_name: str,
    config: SplitConfig,
) -> dict[str, str]:
    partitions = ("train", "validation", "test")
    target_fraction = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    total = sum(len(members) for members in groups.values())
    target_count = {name: total * target_fraction[name] for name in partitions}
    global_strata: Counter[str] = Counter()
    group_strata: dict[str, Counter[str]] = {}
    for group_id, members in groups.items():
        counter: Counter[str] = Counter()
        for sample in members:
            counter.update(_strata(sample, config.stratification_fields))
        group_strata[group_id] = counter
        global_strata.update(counter)
    target_strata = {
        partition: {
            key: count * target_fraction[partition]
            for key, count in global_strata.items()
        }
        for partition in partitions
    }
    assigned_counts = Counter({partition: 0 for partition in partitions})
    assigned_strata = {
        partition: Counter() for partition in partitions
    }
    assignments: dict[str, str] = {}
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            -len(groups[group_id]),
            _stable_unit(key=group_id, seed=config.seed, namespace=split_name),
            group_id,
        ),
    )
    for index, group_id in enumerate(ordered_groups):
        group_size = len(groups[group_id])
        remaining_groups = len(ordered_groups) - index
        empty_partitions = [partition for partition in partitions if assigned_counts[partition] == 0]
        candidate_partitions = partitions
        if empty_partitions and remaining_groups <= len(empty_partitions):
            candidate_partitions = tuple(empty_partitions)
        scored: list[tuple[float, float, str]] = []
        for partition in candidate_partitions:
            projected_count = assigned_counts[partition] + group_size
            size_error = abs(projected_count - target_count[partition]) / max(1.0, total)
            stratum_error = 0.0
            for key, increment in group_strata[group_id].items():
                projected = assigned_strata[partition][key] + increment
                stratum_error += abs(projected - target_strata[partition][key]) / max(
                    1.0, float(global_strata[key])
                )
            stratum_error /= max(1, len(group_strata[group_id]))
            overfill = max(0.0, projected_count - target_count[partition]) / max(1.0, total)
            tie = _stable_unit(
                key={"group": group_id, "partition": partition},
                seed=config.seed,
                namespace=split_name,
            )
            cost = size_error + 0.35 * stratum_error + 0.5 * overfill
            scored.append((cost, tie, partition))
        _, _, selected = min(scored)
        assignments[group_id] = selected
        assigned_counts[selected] += group_size
        assigned_strata[selected].update(group_strata[group_id])
    return assignments


def _temporal_assignment(
    groups: Mapping[str, list[ProcessedSample]],
    config: SplitConfig,
) -> dict[str, str]:
    decorated: list[tuple[float, str]] = []
    for group_id, members in groups.items():
        times = [
            _parse_timestamp(member.hardware_context.get("calibration_timestamp"))
            for member in members
        ]
        finite = [value for value in times if value is not None]
        if not finite:
            raise ValueError(f"temporal group {group_id} has no calibration timestamp")
        decorated.append((min(finite), group_id))
    decorated.sort(key=lambda item: (item[0], item[1]))
    total = sum(len(groups[group_id]) for _, group_id in decorated)
    train_cut = total * config.train_fraction
    validation_cut = total * (config.train_fraction + config.validation_fraction)
    seen = 0
    assignments: dict[str, str] = {}
    for _, group_id in decorated:
        midpoint = seen + 0.5 * len(groups[group_id])
        if midpoint < train_cut:
            partition = "train"
        elif midpoint < validation_cut:
            partition = "validation"
        else:
            partition = "test"
        assignments[group_id] = partition
        seen += len(groups[group_id])
    return assignments


def _split_specifications(config: SplitConfig) -> dict[str, tuple[str, str, Callable[[ProcessedSample], Any], bool]]:
    return {
        "grouped_baseline": (
            "Clean-circuit grouped baseline without parent-child leakage.",
            "clean_circuit_id",
            lambda sample: sample.clean_circuit_id,
            False,
        ),
        "held_out_circuit_instance": (
            "Unseen clean circuit instances while circuit families may remain known.",
            "clean_circuit_instance",
            lambda sample: sample.clean_circuit_id,
            False,
        ),
        "held_out_parameter_region": (
            "Disjoint circular parameter regions with periodic boundary handling.",
            "circular_parameter_region",
            lambda sample: _parameter_region(sample, config.parameter_region_bins),
            False,
        ),
        "held_out_distortion_strength": (
            "Unseen distortion strength or observed severity bins.",
            "distortion_strength_bin",
            _distortion_strength_bin,
            False,
        ),
        "held_out_distortion_combination": (
            "Unseen composed distortion classes.",
            "distortion_combination",
            _distortion_combination,
            False,
        ),
        "held_out_circuit_family": (
            "Entire circuit families absent from training.",
            "circuit_family",
            lambda sample: sample.family,
            False,
        ),
        "held_out_layout_identity": (
            "Unseen labeled layouts or qubit mappings with familiar structures permitted.",
            "labeled_layout_identity",
            _layout_identity,
            False,
        ),
        "held_out_layout_structure": (
            "Genuinely unseen interaction-graph structure rather than qubit renaming.",
            "structural_graph_identity",
            lambda sample: sample.hashes.structural_graph_hash,
            False,
        ),
        "held_out_qubit_count": (
            "Unseen qubit counts for variable-size model evaluation.",
            "qubit_count",
            lambda sample: sample.n_qubits,
            False,
        ),
        "temporal_calibration": (
            "Earlier calibration periods train and later drift periods test.",
            "calibration_window",
            _calibration_group,
            True,
        ),
        "held_out_backend": (
            "Entire backend or hardware-graph contexts absent from training.",
            "backend",
            _backend,
            False,
        ),
    }


def _relation_is_forbidden(relation: LeakageRelation, split_name: str) -> bool:
    evidence = relation.evidence
    if evidence.get("forbid_cross_split") is True:
        return True
    if split_name == "temporal_calibration" and evidence.get(
        "forbid_cross_split_in_temporal_split"
    ):
        return True
    if split_name == "held_out_circuit_instance" and evidence.get(
        "forbid_cross_split_in_instance_split"
    ):
        return True
    if split_name in {"held_out_layout_identity", "held_out_layout_structure"} and evidence.get(
        "forbid_cross_split_in_symmetry_strict_split"
    ):
        return True
    return False


def verify_split(
    *,
    split_name: str,
    samples: list[ProcessedSample],
    assignments: list[SplitAssignment],
    leakage_relations: list[LeakageRelation],
) -> list[dict[str, Any]]:
    accepted_ids = {sample.sample_id for sample in samples if sample.accepted}
    assigned: dict[str, str] = {}
    violations: list[dict[str, Any]] = []
    for record in assignments:
        if record.split_name != split_name:
            continue
        if record.sample_id in assigned:
            violations.append(
                {
                    "kind": "duplicate_assignment",
                    "sample_id": record.sample_id,
                }
            )
        assigned[record.sample_id] = record.partition
        if record.partition not in {"train", "validation", "test"}:
            violations.append(
                {
                    "kind": "invalid_partition",
                    "sample_id": record.sample_id,
                    "partition": record.partition,
                }
            )
    missing = sorted(accepted_ids - assigned.keys())
    extra = sorted(assigned.keys() - accepted_ids)
    if missing:
        violations.append({"kind": "missing_assignments", "sample_ids": missing})
    if extra:
        violations.append({"kind": "unknown_assignments", "sample_ids": extra})
    for relation in leakage_relations:
        if not _relation_is_forbidden(relation, split_name):
            continue
        partitions = {
            assigned[sample_id]
            for sample_id in relation.member_sample_ids
            if sample_id in assigned
        }
        if len(partitions) > 1:
            violations.append(
                {
                    "kind": "forbidden_relation_crossing",
                    "relation_type": relation.relation_type,
                    "relation_id": relation.relation_id,
                    "partitions": sorted(partitions),
                    "member_sample_ids": list(relation.member_sample_ids),
                }
            )
    return violations


def build_challenge_splits(
    samples: list[ProcessedSample],
    leakage_relations: list[LeakageRelation],
    config: SplitConfig,
) -> SplitBuildResult:
    accepted = sorted(
        (sample for sample in samples if sample.accepted),
        key=lambda item: item.sample_id,
    )
    if not accepted:
        raise ValueError("cannot build splits without accepted samples")
    specifications = _split_specifications(config)
    all_assignments: list[SplitAssignment] = []
    statuses: list[SplitStatus] = []
    violations_by_split: dict[str, list[dict[str, Any]]] = {}

    for split_name in config.challenge_splits:
        if split_name not in specifications:
            statuses.append(
                SplitStatus(
                    split_name=split_name,
                    status="skipped",
                    scientific_purpose="Unknown configured challenge split.",
                    reason="unsupported_split_name",
                    assignment_count=0,
                    partition_counts={},
                    leakage_passed=False,
                )
            )
            continue
        purpose, policy, key_fn, temporal = specifications[split_name]
        groups, missing = _group_members(accepted, key_fn)
        if missing:
            statuses.append(
                SplitStatus(
                    split_name=split_name,
                    status="skipped",
                    scientific_purpose=purpose,
                    reason=f"missing required grouping evidence for {len(missing)} samples",
                    assignment_count=0,
                    partition_counts={},
                    leakage_passed=False,
                )
            )
            continue
        if len(groups) < config.minimum_group_count:
            statuses.append(
                SplitStatus(
                    split_name=split_name,
                    status="skipped",
                    scientific_purpose=purpose,
                    reason=(
                        f"requires at least {config.minimum_group_count} groups; "
                        f"found {len(groups)}"
                    ),
                    assignment_count=0,
                    partition_counts={},
                    leakage_passed=False,
                )
            )
            continue
        try:
            group_partitions = (
                _temporal_assignment(groups, config)
                if temporal
                else _greedy_stratified_group_assignment(
                    groups,
                    split_name=split_name,
                    config=config,
                )
            )
        except ValueError as exc:
            statuses.append(
                SplitStatus(
                    split_name=split_name,
                    status="skipped",
                    scientific_purpose=purpose,
                    reason=str(exc),
                    assignment_count=0,
                    partition_counts={},
                    leakage_passed=False,
                )
            )
            continue
        split_assignments: list[SplitAssignment] = []
        for group_id, members in sorted(groups.items()):
            partition = group_partitions[group_id]
            for sample in members:
                split_assignments.append(
                    SplitAssignment(
                        split_name=split_name,
                        sample_id=sample.sample_id,
                        partition=partition,
                        split_group_id=group_id,
                        grouping_policy=policy,
                        stratification_summary={
                            field: (
                                sample.family
                                if field == "family"
                                else sample.n_qubits
                                if field == "n_qubits"
                                else sample.intervention_label
                                if field == "intervention_label"
                                else sample.severity
                                if field == "severity"
                                else getattr(sample, field, None)
                            )
                            for field in config.stratification_fields
                        },
                    )
                )
        violations = verify_split(
            split_name=split_name,
            samples=accepted,
            assignments=split_assignments,
            leakage_relations=leakage_relations,
        )
        violations_by_split[split_name] = violations
        counts = Counter(item.partition for item in split_assignments)
        status = "valid" if not violations and len(counts) == 3 else "invalid"
        reason = None
        if violations:
            reason = f"{len(violations)} leakage or integrity violations"
        elif len(counts) < 3:
            reason = "one or more partitions are empty"
        statuses.append(
            SplitStatus(
                split_name=split_name,
                status=status,
                scientific_purpose=purpose,
                reason=reason,
                assignment_count=len(split_assignments),
                partition_counts=dict(sorted(counts.items())),
                leakage_passed=not violations,
            )
        )
        if status == "valid":
            all_assignments.extend(split_assignments)
    return SplitBuildResult(
        assignments=sorted(
            all_assignments,
            key=lambda item: (item.split_name, item.partition, item.sample_id),
        ),
        statuses=sorted(statuses, key=lambda item: item.split_name),
        leakage_violations=violations_by_split,
    )


def verify_saved_split_directory(
    preprocessing_root: str | Path,
    *,
    split_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Independently verify persisted split assignments and leakage relations.

    This verifier intentionally reads only published preprocessing artifacts.  It
    does not trust the in-memory builder result that originally produced them.
    """
    root = Path(preprocessing_root).expanduser().resolve()
    completion_path = root / "preprocessing_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing completion marker: {completion_path}")
    import json
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise ValueError("Preprocessing completion marker is not complete")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas and pyarrow are required to verify persisted splits") from exc

    assignments_path = root / "splits" / "assignments.parquet"
    samples_path = root / "validated" / "accepted_samples.parquet"
    relations_path = root / "groups" / "leakage_relations.parquet"
    for path in (assignments_path, samples_path, relations_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required split-verification artifact is missing: {path}")

    assignments = pd.read_parquet(assignments_path).to_dict(orient="records")
    samples = pd.read_parquet(samples_path).to_dict(orient="records")
    relations = pd.read_parquet(relations_path).to_dict(orient="records")
    accepted_ids = {str(row["sample_id"]) for row in samples if bool(row.get("accepted", True))}
    requested = set(split_names or ())
    available = sorted({str(row["split_name"]) for row in assignments})
    if requested:
        missing = requested.difference(available)
        if missing:
            raise ValueError(f"Unknown persisted split names: {sorted(missing)}")
        available = [name for name in available if name in requested]

    result: dict[str, Any] = {}
    overall_valid = True
    for split_name in available:
        rows = [row for row in assignments if str(row["split_name"]) == split_name]
        seen: dict[str, str] = {}
        violations: list[str] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            partition = str(row["partition"])
            if partition not in {"train", "validation", "test"}:
                violations.append(f"invalid partition {partition!r} for {sample_id}")
            previous = seen.setdefault(sample_id, partition)
            if previous != partition:
                violations.append(
                    f"sample {sample_id} appears in both {previous} and {partition}"
                )
        missing_ids = accepted_ids.difference(seen)
        extra_ids = set(seen).difference(accepted_ids)
        if missing_ids:
            violations.append(f"{len(missing_ids)} accepted samples are unassigned")
        if extra_ids:
            violations.append(f"{len(extra_ids)} unknown samples are assigned")

        for relation in relations:
            evidence = relation.get("evidence") or {}
            if isinstance(evidence, dict) and evidence.get("forbid_cross_split") is False:
                continue
            members_raw = relation.get("member_sample_ids") or []
            if hasattr(members_raw, "tolist"):
                members_raw = members_raw.tolist()
            members = [str(value) for value in members_raw if str(value) in seen]
            partitions = {seen[value] for value in members}
            if len(partitions) > 1:
                violations.append(
                    "leakage relation "
                    f"{relation.get('relation_id')} ({relation.get('relation_type')}) "
                    f"crosses {sorted(partitions)}"
                )

        partition_counts = Counter(seen.values())
        if set(partition_counts) != {"train", "validation", "test"}:
            violations.append("one or more partitions are empty")
        valid = not violations
        overall_valid = overall_valid and valid
        result[split_name] = {
            "valid": valid,
            "assignment_count": len(rows),
            "partition_counts": dict(sorted(partition_counts.items())),
            "violations": violations,
        }

    return {
        "status": "valid" if overall_valid else "invalid",
        "preprocessing_root": root.as_posix(),
        "verified_splits": result,
    }
