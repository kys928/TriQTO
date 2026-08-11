#!/usr/bin/env python3
"""Step 4.1: correlation-recovery audit for the hardware-valid B_delta core.

Step 4 v1 froze and failed the one-local-Pauli core because GHZ-family
mechanism separation lives in joint bitstring structure. This follow-up keeps
that failure immutable and tests a pre-frozen ladder of bounded summaries
derived from the same X/Y/Z sampled bitstrings, with no additional basis
programs:

1. local one-body Pauli expectation deltas;
2. local + global same-basis parity deltas;
3. local + all same-basis two-body correlation deltas;
4. local + two-body + global parity deltas.

No QPU is used. Exact statevectors are privileged audit-only machinery used to
regenerate the already frozen Step 3.5 matched simulator contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
STEP35_PATH = HERE / "audit_generalized_bdelta_identifiability.py"
STEP35_NAME = "triqto_step35_for_step4_correlation"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/bdelta_correlation_recovery_audit.json"
DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/step4_1_bdelta_correlation_recovery"
)
SCHEMA = "triqto.v0_2.bdelta_correlation_recovery_audit_result.v1"


def load_step35():
    spec = importlib.util.spec_from_file_location(STEP35_NAME, STEP35_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Step 3.5 runner: {STEP35_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[STEP35_NAME] = module
    spec.loader.exec_module(module)
    return module


S35 = load_step35()
V3 = S35.V3
V2 = S35.V2
BASE = S35.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def resolve_product(args: argparse.Namespace) -> Path:
    if args.product_dir is not None:
        root = args.product_dir.expanduser().resolve()
    else:
        pointer = args.product_parent.expanduser().resolve() / "current_product.json"
        root = Path(str(BASE.read_json(pointer)["product_dir"])).expanduser().resolve()
    if not (root / "generation_complete.json").is_file():
        raise RuntimeError(f"incomplete source product: {root}")
    return root


def same_basis_summary(probabilities: np.ndarray, n_qubits: int) -> dict[str, np.ndarray | float]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    indices = np.arange(p.size, dtype=np.int64)
    local = BASE.pauli_expectations(p, n_qubits)
    parity_sign = np.ones(p.size, dtype=np.float64)
    for qubit in range(n_qubits):
        parity_sign *= np.where((indices & (1 << qubit)) == 0, 1.0, -1.0)
    global_parity = float(np.sum(parity_sign * p))
    pairwise = []
    for left in range(n_qubits):
        left_sign = np.where((indices & (1 << left)) == 0, 1.0, -1.0)
        for right in range(left + 1, n_qubits):
            right_sign = np.where((indices & (1 << right)) == 0, 1.0, -1.0)
            pairwise.append(float(np.sum(left_sign * right_sign * p)))
    return {
        "local": np.asarray(local, dtype=np.float64),
        "global_parity": global_parity,
        "pairwise": np.asarray(pairwise, dtype=np.float64),
    }


def summaries_for_state(state: np.ndarray, n_qubits: int) -> dict[str, dict[str, Any]]:
    return {
        basis: same_basis_summary(BASE.measurement_probabilities(state, basis), n_qubits)
        for basis in ("X", "Y", "Z")
    }


def delta_summary(clean: Mapping[str, Mapping[str, Any]], distorted: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for basis in ("X", "Y", "Z"):
        output[basis] = {
            "local": np.asarray(distorted[basis]["local"]) - np.asarray(clean[basis]["local"]),
            "global_parity": float(distorted[basis]["global_parity"]) - float(clean[basis]["global_parity"]),
            "pairwise": np.asarray(distorted[basis]["pairwise"]) - np.asarray(clean[basis]["pairwise"]),
        }
    return output


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def variant_component_values(summary: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for basis in ("X", "Y", "Z"):
        if bool(variant["include_local"]):
            values.append(0.5 * rms(np.asarray(summary[basis]["local"])))
        if bool(variant["include_pairwise"]):
            values.append(0.5 * rms(np.asarray(summary[basis]["pairwise"])))
        if bool(variant["include_global_parity"]):
            values.append(0.5 * abs(float(summary[basis]["global_parity"])))
    return values


def variant_score(summary: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Any]) -> float:
    values = variant_component_values(summary, variant)
    if not values:
        raise ValueError("variant has no enabled components")
    return float(np.mean(values))


def pair_difference(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for basis in ("X", "Y", "Z"):
        output[basis] = {
            "local": np.asarray(left[basis]["local"]) - np.asarray(right[basis]["local"]),
            "global_parity": float(left[basis]["global_parity"]) - float(right[basis]["global_parity"]),
            "pairwise": np.asarray(left[basis]["pairwise"]) - np.asarray(right[basis]["pairwise"]),
        }
    return output


def phenotype(clean: np.ndarray, distorted: np.ndarray, config: Mapping[str, Any]) -> str:
    p = config["phenomenology"]
    decomposition = BASE.overlap_decomposition(clean, distorted, epsilon=float(p["epsilon"]))
    return BASE.phenotype(
        decomposition,
        negligible_floor=float(p["negligible_overlap_loss_floor"]),
        dominance_ratio=float(p["strong_dominance_ratio"]),
    )


def build_clean_contexts(root: Path, config: Mapping[str, Any], progress_every: int) -> dict[str, dict[str, Any]]:
    source = config["source"]
    rows = V2.load_rows(root, int(source["expected_source_examples"]))
    overlap_max = float(source["state_overlap_loss_max"])
    amp_max = float(source["aligned_amplitude_max_abs_error_max"])
    clean_contexts: dict[str, dict[str, Any]] = {}
    valid = 0
    for index, row in enumerate(rows, start=1):
        info = V3.replay_source(
            root,
            row,
            overlap_loss_max=overlap_max,
            amplitude_error_max=amp_max,
        )
        if not bool(info["valid"]):
            raise RuntimeError(f"source replay invalid for {info['entity_id']}")
        records = S35.canonical_unitary_records(info["records"], int(info["removed_index"]))
        S35.validate_clean_reconstruction(
            info,
            records,
            int(info["n_qubits"]),
            overlap_max,
            amp_max,
        )
        clean_hash = S35.clean_unitary_hash(records)
        payload = {
            "clean_unitary_hash": clean_hash,
            "family": info["family"],
            "n_qubits": int(info["n_qubits"]),
            "phase_sensitive_family": bool(info["phase_sensitive_family"]),
        }
        clean_id = BASE.sha256_bytes(BASE.canonical_json(payload).encode("utf-8"))
        if clean_id not in clean_contexts:
            clean_contexts[clean_id] = {
                "clean_context_id": clean_id,
                "records": records,
                "clean_state": np.asarray(info["clean_state"], dtype=np.complex128),
                "family": str(info["family"]),
                "n_qubits": int(info["n_qubits"]),
                "phase_sensitive_family": bool(info["phase_sensitive_family"]),
            }
        valid += 1
        if progress_every > 0 and index % min(progress_every, 25) == 0:
            print(f"Source replay validated {index}/{len(rows)} examples", flush=True)
    if valid != int(source["expected_source_examples"]):
        raise RuntimeError("source replay count mismatch")
    if len(clean_contexts) != int(source["expected_clean_circuits"]):
        raise RuntimeError(
            f"clean circuit count mismatch: {len(clean_contexts)} != {source['expected_clean_circuits']}"
        )
    return clean_contexts


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pair_count": 0,
            "strong_pair_fraction": float("nan"),
            "numerical_collision_fraction": float("nan"),
            "median_pair_separation_score": float("nan"),
            "median_relative_separation": float("nan"),
        }
    return {
        "pair_count": len(rows),
        "strong_pair_fraction": float(np.mean([float(bool(r["strong_pair"])) for r in rows])),
        "numerical_collision_fraction": float(np.mean([float(bool(r["numerical_collision"])) for r in rows])),
        "median_pair_separation_score": float(np.median([float(r["pair_separation_score"]) for r in rows])),
        "median_relative_separation": float(np.median([float(r["relative_separation"]) for r in rows])),
    }


def grouped_bootstrap(rows: Sequence[Mapping[str, Any]], repeats: int, seed: int) -> list[float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["clean_context_id"])].append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for key in chosen for row in groups[str(key)]]
        values[i] = np.mean([float(bool(r["strong_pair"])) for r in sample])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def variant_decision(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    policy = config["selection_policy"]
    effectful = [r for r in rows if bool(r["effectful_pair"])]
    nn = [
        r for r in effectful
        if not bool(r["terminal_insertion"]) and int(r["affected_qubit"]) != 0
    ]
    overall = summarize(effectful)
    subset = summarize(nn)

    dims = ("family", "pair_type", "n_qubits", "strength_key", "insertion_depth_bin")
    strata = []
    min_pairs = int(policy["minimum_eligible_stratum_pairs"])
    for dim in dims:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in effectful:
            grouped[str(row[dim])].append(row)
        for value, part in sorted(grouped.items()):
            stats = summarize(part)
            strata.append({
                "stratum_type": dim,
                "stratum_value": value,
                "eligible": len(part) >= min_pairs,
                **stats,
            })
    eligible = [s for s in strata if bool(s["eligible"])]
    minimum_eligible = min((float(s["strong_pair_fraction"]) for s in eligible), default=float("nan"))
    ghz = next(
        (s for s in strata if s["stratum_type"] == "family" and s["stratum_value"] == "ghz"),
        None,
    )
    ghz_fraction = float(ghz["strong_pair_fraction"]) if ghz is not None else float("nan")
    passed = bool(
        float(overall["strong_pair_fraction"]) >= float(policy["minimum_effectful_pair_strong_fraction"])
        and float(subset["strong_pair_fraction"]) >= float(policy["minimum_effectful_nonterminal_non_q0_strong_fraction"])
        and minimum_eligible >= float(policy["minimum_every_eligible_stratum_strong_fraction"])
        and ghz_fraction >= float(policy["minimum_ghz_strong_fraction"])
    )
    return {
        "pass": passed,
        "effectful": overall,
        "effectful_nonterminal_non_q0": subset,
        "minimum_eligible_stratum_strong_fraction": minimum_eligible,
        "ghz_strong_pair_fraction": ghz_fraction,
        "strata": strata,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = BASE.read_json(config_path)
    if config.get("schema") != "triqto.v0_2.bdelta_correlation_recovery_audit.v1":
        raise RuntimeError("unexpected Step 4.1 config schema")

    root = resolve_product(args)
    identity = {
        "config_sha256": BASE.sha256_file(config_path),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        "source_generation_complete_sha256": BASE.sha256_file(root / "generation_complete.json"),
        "step35_runner_sha256": BASE.sha256_file(STEP35_PATH),
    }
    audit_id = "audit_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing audit: {output_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    clean_contexts = build_clean_contexts(root, config, args.progress_every)

    factor = config["generalized_factorial"]
    target_fractions = [float(v) for v in factor["depth_target_fractions"]]
    strengths = [float(v) for v in factor["strengths"]]
    mechanisms = tuple(str(v) for v in factor["mechanisms"])
    if mechanisms != S35.MECHANISMS:
        raise RuntimeError("mechanism set differs from Step 3.5")

    variants = list(config["variants"])
    pair_cfg = config["pairwise_thresholds"]
    raw_min = float(pair_cfg["minimum_raw_separation"])
    rel_min = float(pair_cfg["minimum_relative_separation"])
    collision_max = float(pair_cfg["numerical_collision_score_max"])
    epsilon = 1e-12
    all_pair_rows: list[dict[str, Any]] = []
    context_count = 0

    for clean_id, context in sorted(clean_contexts.items()):
        records = context["records"]
        n_qubits = int(context["n_qubits"])
        clean_final = np.asarray(context["clean_state"], dtype=np.complex128)
        clean_summary = summaries_for_state(clean_final, n_qubits)
        boundaries = S35.depth_boundaries(len(records), target_fractions)

        for qubit in range(n_qubits):
            for boundary in boundaries:
                boundary_rank = int(boundary["boundary_rank"])
                prefix = S35.prefix_state(records, n_qubits, boundary_rank)
                suffix = S35.suffix_operator(records, n_qubits, boundary_rank)
                for strength in strengths:
                    matched_id = S35.generated_context_id(clean_id, qubit, boundary_rank, strength)
                    generated: dict[str, dict[str, Any]] = {}
                    for mechanism in mechanisms:
                        axis = BASE.axis_for_mechanism(mechanism)
                        rotated = BASE.apply_single_qubit_matrix(
                            prefix,
                            qubit,
                            BASE.rotation_matrix(axis, strength),
                        )
                        distorted = S35.propagate_suffix(rotated, suffix)
                        absolute = summaries_for_state(distorted, n_qubits)
                        delta = delta_summary(clean_summary, absolute)
                        generated[mechanism] = {
                            "absolute": absolute,
                            "delta": delta,
                            "phenotype": phenotype(clean_final, distorted, config),
                        }

                    for left, right, pair_type in S35.PAIR_TYPES:
                        left_item = generated[left]
                        right_item = generated[right]
                        difference = pair_difference(
                            left_item["absolute"], right_item["absolute"]
                        )
                        effectful = (
                            left_item["phenotype"] != "negligible"
                            and right_item["phenotype"] != "negligible"
                        )
                        for variant in variants:
                            left_signal = variant_score(left_item["delta"], variant)
                            right_signal = variant_score(right_item["delta"], variant)
                            score = variant_score(difference, variant)
                            relative = 2.0 * score / (left_signal + right_signal + epsilon)
                            all_pair_rows.append({
                                "variant": variant["name"],
                                "clean_context_id": clean_id,
                                "matched_context_id": matched_id,
                                "family": context["family"],
                                "n_qubits": n_qubits,
                                "affected_qubit": qubit,
                                "qubit_position_class": S35.qubit_position_class(qubit, n_qubits),
                                "insertion_depth_bin": boundary["depth_bin"],
                                "terminal_insertion": bool(boundary["terminal_insertion"]),
                                "strength_key": f"{strength:.12g}",
                                "pair_type": pair_type,
                                "left_mechanism": left,
                                "right_mechanism": right,
                                "left_phenotype": left_item["phenotype"],
                                "right_phenotype": right_item["phenotype"],
                                "effectful_pair": effectful,
                                "left_signal_score": left_signal,
                                "right_signal_score": right_signal,
                                "pair_separation_score": score,
                                "relative_separation": relative,
                                "numerical_collision": score <= collision_max,
                                "strong_pair": bool(score >= raw_min and relative >= rel_min),
                            })
                    context_count += 1
                    if args.progress_every > 0 and context_count % args.progress_every == 0:
                        print(f"Audited {context_count} generalized correlation contexts", flush=True)

    expected_contexts = int(config["source"]["expected_generalized_contexts"])
    if context_count != expected_contexts:
        raise RuntimeError(f"context count mismatch: {context_count} != {expected_contexts}")

    variant_results: dict[str, Any] = {}
    for index, variant in enumerate(variants):
        name = str(variant["name"])
        rows = [r for r in all_pair_rows if r["variant"] == name]
        result = variant_decision(rows, config)
        effectful_rows = [r for r in rows if bool(r["effectful_pair"])]
        result["effectful"]["strong_pair_fraction_ci95"] = grouped_bootstrap(
            effectful_rows,
            int(config["selection_policy"]["bootstrap_repeats"]),
            int(config["selection_policy"]["bootstrap_seed"]) + index,
        )
        variant_results[name] = result

    selected = None
    for variant in variants:
        name = str(variant["name"])
        if bool(variant_results[name]["pass"]):
            selected = name
            break
    decision = (
        "CORRELATION_CORE_RECOVERED"
        if selected is not None
        else "HARDWARE_VALID_CORRELATION_SUMMARIES_INSUFFICIENT"
    )

    summary_rows = []
    for variant in variants:
        name = str(variant["name"])
        result = variant_results[name]
        summary_rows.append({
            "variant": name,
            "pass": result["pass"],
            "effectful_pair_count": result["effectful"]["pair_count"],
            "effectful_strong_pair_fraction": result["effectful"]["strong_pair_fraction"],
            "effectful_ci_low": result["effectful"]["strong_pair_fraction_ci95"][0],
            "effectful_ci_high": result["effectful"]["strong_pair_fraction_ci95"][1],
            "nonterminal_non_q0_pair_count": result["effectful_nonterminal_non_q0"]["pair_count"],
            "nonterminal_non_q0_strong_pair_fraction": result["effectful_nonterminal_non_q0"]["strong_pair_fraction"],
            "minimum_eligible_stratum_strong_fraction": result["minimum_eligible_stratum_strong_fraction"],
            "ghz_strong_pair_fraction": result["ghz_strong_pair_fraction"],
        })

    BASE.atomic_csv(staging / "correlation_pair_metrics.csv", all_pair_rows)
    BASE.atomic_csv(staging / "correlation_variant_summary.csv", summary_rows)
    BASE.atomic_json(staging / "variant_results.json", variant_results)
    BASE.atomic_json(staging / "decision.json", {
        "schema": SCHEMA,
        "decision": decision,
        "selected_smallest_passing_variant": selected,
        "variant_order": [v["name"] for v in variants],
        "same_basis_programs_only": True,
        "additional_basis_programs_beyond_step4_v1": 0,
        "hardware_executed": False,
        "classifier_trained": False,
    })
    complete = {
        "schema": SCHEMA,
        "status": "AUDIT_COMPLETE",
        "audit_id": audit_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "selected_smallest_passing_variant": selected,
        "source_root": str(root),
        "clean_circuit_count": len(clean_contexts),
        "generalized_context_count": context_count,
        "pair_rows_across_variants": len(all_pair_rows),
        "identity": identity,
        "hardware_executed": False,
        "classifier_trained": False,
        "historical_v0_1_test_accessed": False,
        "spent_confirmatory_cohort_accessed": False,
    }
    BASE.atomic_json(staging / "audit_complete.json", complete)
    os.replace(staging, output_root)

    print("\nTRIQTO STEP 4.1 B_DELTA CORRELATION-RECOVERY AUDIT COMPLETE\n")
    print(f"Decision: {decision}")
    print(f"Selected smallest passing variant: {selected}")
    for row in summary_rows:
        print(
            f"{row['variant']}: overall={row['effectful_strong_pair_fraction']:.4f} "
            f"nonterminal+non-q0={row['nonterminal_non_q0_strong_pair_fraction']:.4f} "
            f"min-stratum={row['minimum_eligible_stratum_strong_fraction']:.4f} "
            f"GHZ={row['ghz_strong_pair_fraction']:.4f} pass={row['pass']}"
        )
    print("Additional basis programs beyond Step 4 v1: 0")
    print("Hardware executed: NO")
    print("Classifier trained: NO")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
