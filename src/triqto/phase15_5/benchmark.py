"""Grouped test benchmarking for the Phase 15.5 operational policy."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
import math, random
import numpy as np
from triqto.data_generation import derive_child_seed
from .config import Phase155Config
from .policy import FAMILY_NAMES, PolicyDataset

def _group_indices(dataset: PolicyDataset, split: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (group_id, row_split) in enumerate(zip(dataset.group_ids, dataset.splits, strict=True)):
        if row_split == split:
            groups[group_id].append(index)
    return dict(sorted(groups.items()))

def _heuristic_choice(dataset: PolicyDataset, indices: list[int], metadata_by_id: Mapping[str, Mapping[str, Any]]) -> int:
    available = [index for index in indices if dataset.available_mask[index]]
    family = int(dataset.family_ids[available[0]])
    if family == 0:
        for index in available:
            if metadata_by_id[dataset.candidate_ids[index]]["metadata"].get("basis") == "Z":
                return index
        return available[0]
    features = dataset.candidate_features
    def score(index: int) -> float:
        row = features[index]
        return float(-row[9] - 0.5 * row[10] - 2.0 * row[11] - 0.5 * row[12])
    return max(available, key=lambda index: (score(index), dataset.candidate_ids[index]))

def _bootstrap_ci(values_by_group: Mapping[str, float], *, replicates: int, confidence: float, seed: int) -> tuple[float, float]:
    keys = sorted(values_by_group)
    if not keys:
        raise ValueError("bootstrap requires at least one split group")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        estimates.append(float(sum(values_by_group[key] for key in sampled) / len(sampled)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(estimates) - 1, int(math.floor(alpha * len(estimates)))))
    high_index = max(0, min(len(estimates) - 1, int(math.ceil((1.0 - alpha) * len(estimates))) - 1))
    return estimates[low_index], estimates[high_index]

def _benchmark(
    dataset: PolicyDataset,
    scores: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    config: Phase155Config,
) -> dict[str, Any]:
    metadata_by_id = {str(value["candidate_id"]): value for value in metadata_rows}
    groups = _group_indices(dataset, "test")
    if not groups:
        raise ValueError("Phase 15.5 benchmark requires untouched test groups")
    methods = ("trained_policy", "random_control", "no_op_control", "family_heuristic", "oracle_upper_bound")
    utilities: dict[str, list[float]] = {name: [] for name in methods}
    regrets: dict[str, list[float]] = {name: [] for name in methods}
    matches: dict[str, list[float]] = {name: [] for name in methods}
    per_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_split_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures: list[dict[str, Any]] = []
    for group_id, indices in groups.items():
        available = [index for index in indices if dataset.available_mask[index]]
        oracle = max(available, key=lambda index: (dataset.utilities[index], dataset.candidate_ids[index]))
        trained = max(available, key=lambda index: (scores[index], dataset.candidate_ids[index]))
        rng = random.Random(derive_child_seed(config.seed, "phase155_random_baseline", {"group_id": group_id}))
        random_choice = available[rng.randrange(len(available))]
        noop = next((index for index in available if metadata_by_id[dataset.candidate_ids[index]]["metadata"].get("kind") == "no_op"), available[0])
        heuristic = _heuristic_choice(dataset, indices, metadata_by_id)
        choices = {
            "trained_policy": trained,
            "random_control": random_choice,
            "no_op_control": noop,
            "family_heuristic": heuristic,
            "oracle_upper_bound": oracle,
        }
        oracle_utility = float(dataset.utilities[oracle])
        family = FAMILY_NAMES[int(dataset.family_ids[oracle])]
        split_group = dataset.split_group_ids[oracle]
        for method, index in choices.items():
            utility = float(dataset.utilities[index])
            regret = oracle_utility - utility
            utilities[method].append(utility)
            regrets[method].append(regret)
            matches[method].append(float(index == oracle))
            per_family[family][f"{method}:utility"].append(utility)
            per_family[family][f"{method}:regret"].append(regret)
            per_split_group[split_group][method].append(utility)
        if choices["trained_policy"] != oracle:
            failures.append(
                {
                    "group_id": group_id,
                    "family": family,
                    "split_group_id": split_group,
                    "selected_candidate_id": dataset.candidate_ids[trained],
                    "oracle_candidate_id": dataset.candidate_ids[oracle],
                    "selected_utility": float(dataset.utilities[trained]),
                    "oracle_utility": oracle_utility,
                    "regret": oracle_utility - float(dataset.utilities[trained]),
                }
            )
    failures.sort(key=lambda value: (-value["regret"], value["group_id"]))
    aggregate: dict[str, Any] = {}
    for method in methods:
        group_means = {key: float(np.mean(values[method])) for key, values in per_split_group.items() if method in values}
        low, high = _bootstrap_ci(group_means, replicates=config.bootstrap_replicates, confidence=config.confidence_level, seed=derive_child_seed(config.seed, "phase155_bootstrap", {"method": method}))
        aggregate[method] = {
            "mean_selected_utility": float(np.mean(utilities[method])),
            "mean_regret": float(np.mean(regrets[method])),
            "oracle_match_rate": float(np.mean(matches[method])),
            "mean_selected_utility_ci": [low, high],
            "group_count": len(utilities[method]),
        }
    family_summary: dict[str, Any] = {}
    for family, values in sorted(per_family.items()):
        family_summary[family] = {
            method: {
                "mean_selected_utility": float(np.mean(values[f"{method}:utility"])),
                "mean_regret": float(np.mean(values[f"{method}:regret"])),
                "group_count": len(values[f"{method}:utility"]),
            }
            for method in methods
        }
    return {
        "evaluation_kind": "grouped_phase12_test_split_offline_noisy_simulation",
        "split_semantics": "clean_circuit_grouped_phase12_test; not OOD unless separately configured and audited",
        "aggregate": aggregate,
        "by_family": family_summary,
        "trained_minus_controls": {
            name: aggregate["trained_policy"]["mean_selected_utility"] - aggregate[name]["mean_selected_utility"]
            for name in ("random_control", "no_op_control", "family_heuristic")
        },
        "failure_cases": failures[:20],
        "test_group_count": len(groups),
        "physical_hardware": False,
        "research_quality_claim": False,
        "calibrated_uncertainty_claim": False,
        "broad_ood_claim": False,
        "correction_success_claim": False,
        "topology_loss_weight": 0.0,
    }

def _policy_dataset_arrays(dataset: PolicyDataset) -> dict[str, np.ndarray]:
    width = lambda values: max(1, *(len(value) for value in values))
    return {
        "candidate_ids": np.asarray(dataset.candidate_ids, dtype=f"<U{width(dataset.candidate_ids)}"),
        "group_ids": np.asarray(dataset.group_ids, dtype=f"<U{width(dataset.group_ids)}"),
        "split_group_ids": np.asarray(dataset.split_group_ids, dtype=f"<U{width(dataset.split_group_ids)}"),
        "splits": np.asarray(dataset.splits, dtype=f"<U{width(dataset.splits)}"),
        "family_ids": dataset.family_ids,
        "context_features": dataset.context_features,
        "candidate_features": dataset.candidate_features,
        "utilities": dataset.utilities,
        "available_mask": dataset.available_mask,
    }
