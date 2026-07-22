"""Machine-readable and human-readable preprocessing reports."""
from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
import json
from typing import Any, Iterable, Mapping

from .records import DuplicateRelation, OutlierRecord, ProcessedSample, SplitAssignment, SplitStatus


def build_health_report(samples: list[ProcessedSample]) -> dict[str, Any]:
    accepted = [sample for sample in samples if sample.accepted]
    quarantined = [sample for sample in samples if not sample.accepted]
    findings = Counter(finding.rule_id for sample in samples for finding in sample.findings)
    dispositions = Counter(finding.disposition for sample in samples for finding in sample.findings)
    sources = Counter(sample.source_type for sample in samples)
    repairs = sum(1 for sample in samples for finding in sample.findings if finding.repair_applied)
    return {
        "record_count": len(samples),
        "accepted_count": len(accepted),
        "quarantine_count": len(quarantined),
        "acceptance_rate": len(accepted) / len(samples) if samples else 0.0,
        "repair_count": repairs,
        "finding_counts_by_rule": dict(sorted(findings.items())),
        "finding_counts_by_disposition": dict(sorted(dispositions.items())),
        "source_type_counts": dict(sorted(sources.items())),
        "quarantine_reasons": dict(sorted(Counter(sample.quarantine_reason for sample in quarantined).items())),
    }


def build_duplicate_report(relations: list[DuplicateRelation]) -> dict[str, Any]:
    counts = Counter(relation.relation_type for relation in relations)
    multiplicities: dict[str, list[int]] = defaultdict(list)
    for relation in relations:
        multiplicities[relation.relation_type].append(relation.multiplicity)
    return {
        "group_count": len(relations),
        "groups_by_type": dict(sorted(counts.items())),
        "multiplicity_summary": {
            relation_type: {
                "minimum": min(values), "maximum": max(values), "mean": sum(values) / len(values)
            }
            for relation_type, values in sorted(multiplicities.items())
        },
        "same_born_different_state_groups": counts.get("same_born_different_state", 0),
    }


def build_label_audit_report(samples: list[ProcessedSample]) -> dict[str, Any]:
    accepted = [sample for sample in samples if sample.accepted]
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in accepted:
        matrix[sample.intervention_label][sample.observed_effect_label] += 1
    return {
        "confusion_matrix": {
            injected: dict(sorted(counter.items())) for injected, counter in sorted(matrix.items())
        },
        "ambiguous_count": sum(sample.observed_effect_ambiguous for sample in accepted),
        "negligible_count": sum(sample.observed_effect_label == "negligible" for sample in accepted),
        "unobservable_in_available_basis_count": sum(
            sample.observed_effect_label == "unobservable_in_available_basis" for sample in accepted
        ),
    }


def _distribution(rows: Iterable[ProcessedSample], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sample in rows:
        if field == "family": value = sample.family
        elif field == "n_qubits": value = sample.n_qubits
        elif field == "intervention_label": value = sample.intervention_label
        elif field == "severity": value = sample.severity
        elif field == "measurement_basis": value = sample.measurement_basis
        elif field == "source_type": value = sample.source_type
        elif field == "layout_structure": value = sample.hashes.structural_graph_hash
        elif field == "shot_count": value = sample.shot_count
        elif field == "hilbert_available": value = sample.masks.get("hilbert_available", False)
        elif field == "backend": value = sample.hardware_context.get("backend_name")
        elif field == "calibration_period": value = sample.hardware_context.get("calibration_window_id")
        else: value = getattr(sample, field, None)
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def build_distribution_report(
    samples: list[ProcessedSample], assignments: list[SplitAssignment]
) -> dict[str, Any]:
    by_id = {sample.sample_id: sample for sample in samples if sample.accepted}
    grouped: dict[str, dict[str, list[ProcessedSample]]] = defaultdict(lambda: defaultdict(list))
    for assignment in assignments:
        sample = by_id.get(assignment.sample_id)
        if sample is not None:
            grouped[assignment.split_name][assignment.partition].append(sample)
    fields = (
        "family", "n_qubits", "intervention_label", "severity", "measurement_basis",
        "source_type", "layout_structure", "shot_count", "hilbert_available", "backend",
        "calibration_period",
    )
    return {
        split_name: {
            partition: {
                "count": len(rows),
                "distributions": {field: _distribution(rows, field) for field in fields},
            }
            for partition, rows in sorted(partitions.items())
        }
        for split_name, partitions in sorted(grouped.items())
    }


def build_outlier_report(records: list[OutlierRecord]) -> dict[str, Any]:
    counts = Counter((record.view, record.method) for record in records if record.is_outlier)
    review_queue = sorted(
        (record for record in records if record.is_outlier),
        key=lambda item: (-(float(item.score) if item.score is not None else -1.0), item.view, item.method, item.sample_id),
    )
    return {
        "tag_only": True,
        "outlier_counts": {f"{view}:{method}": count for (view, method), count in sorted(counts.items())},
        "review_queue": [record.to_dict() for record in review_queue[:500]],
    }


def build_leakage_report(
    statuses: list[SplitStatus], violations: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        "splits": [status.to_dict() for status in statuses],
        "violations": {split_name: rows for split_name, rows in sorted(violations.items())},
        "all_valid_generated_splits_passed": all(
            status.leakage_passed for status in statuses if status.status == "valid"
        ),
    }


def render_html_report(title: str, sections: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    for name, payload in sections.items():
        formatted = json.dumps(payload, sort_keys=True, indent=2, default=str)
        blocks.append(f"<section><h2>{escape(name)}</h2><pre>{escape(formatted)}</pre></section>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem;"
        "border-radius:.5rem}h1,h2{line-height:1.2}</style></head><body>"
        f"<h1>{escape(title)}</h1>{''.join(blocks)}</body></html>"
    )
