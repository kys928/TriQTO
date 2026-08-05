#!/usr/bin/env python3
"""TriQTO v0.2 relational-observable phase/amplitude probe experiment.

Development-only comparison on the already-opened 280-entity pilot:

B_absolute
    Prior clean-graph/backend + distorted Z/X/Y observable regime.

B_delta
    B_absolute plus clean Z/X/Y probabilities, clean-to-distorted probability
    deltas, clean/distorted expectation deltas, and per-basis TV, Hellinger,
    and Jensen-Shannon distances.

C_summary
    B_delta plus compressed Hilbert summaries and aggregate aligned-state
    diagnostics. No elementwise statevector amplitudes are exposed.

C_full
    C_summary plus raw clean/distorted/aligned-delta statevector components.

The clean observable panel is derived from the exact frozen clean statevectors
produced by the verified circuit reconstruction stage. Statevectors are used
only to derive exact observables for B_delta; they are not model inputs there.

Because the existing validation cohort has already informed this experiment,
all results are development diagnostics, not final confirmatory evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "train_phase_amplitude_identifiability_probes.py"
BASE_MODULE_NAME = "triqto_v0_2_phase_amplitude_probe_base"

SCHEMA = "triqto.v0_2.phase_amplitude_relational_observable_probe.v1"
REGIMES = ("B_absolute", "B_delta", "C_summary", "C_full")
MODELS = ("linear", "nonlinear")
NONLINEAR_GRID = tuple(
    {"C": c_value, "gamma": gamma}
    for c_value in (0.1, 1.0, 10.0)
    for gamma in ("scale", 0.01, 0.1)
)
ALIGNED_SUMMARY_NAMES = (
    "aligned_delta_l1",
    "aligned_delta_l2",
    "aligned_delta_linf",
    "aligned_delta_mean_abs",
    "aligned_delta_std_abs",
    "aligned_delta_real_mean",
    "aligned_delta_real_std",
    "aligned_delta_real_min",
    "aligned_delta_real_max",
    "aligned_delta_imag_mean",
    "aligned_delta_imag_std",
    "aligned_delta_imag_min",
    "aligned_delta_imag_max",
)
VERIFY_ATOL = 1.0e-10
AUDIT: dict[str, Any] = {
    "verified_entities": set(),
    "max_distorted_z_probability_abs_error": 0.0,
    "max_distorted_x_probability_abs_error": 0.0,
    "max_distorted_y_probability_abs_error": 0.0,
    "max_clean_x_expectation_abs_error": 0.0,
    "max_clean_y_expectation_abs_error": 0.0,
    "max_clean_z_expectation_abs_error": 0.0,
    "max_distorted_x_expectation_abs_error": 0.0,
    "max_distorted_y_expectation_abs_error": 0.0,
    "max_distorted_z_expectation_abs_error": 0.0,
}


def load_base():
    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base evaluator from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    output = vector.copy()
    step = 1 << qubit
    period = step << 1
    for start in range(0, vector.size, period):
        for offset in range(step):
            zero_index = start + offset
            one_index = zero_index + step
            zero = vector[zero_index]
            one = vector[one_index]
            output[zero_index] = gate[0, 0] * zero + gate[0, 1] * one
            output[one_index] = gate[1, 0] * zero + gate[1, 1] * one
    return output


def basis_probabilities(
    state: np.ndarray,
    n_qubits: int,
    basis: str,
) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    expected = 1 << n_qubits
    if vector.size != expected:
        raise ValueError(
            f"State dimension {vector.size} does not match {n_qubits} qubits"
        )
    norm = float(np.vdot(vector, vector).real)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"Statevector norm is {norm}, not 1")

    hadamard = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    s_dagger = np.asarray(
        [[1.0, 0.0], [0.0, -1.0j]],
        dtype=np.complex128,
    )

    rotated = vector.copy()
    if basis == "Z":
        pass
    elif basis == "X":
        for qubit in range(n_qubits):
            rotated = apply_single_qubit_gate(rotated, hadamard, qubit)
    elif basis == "Y":
        for qubit in range(n_qubits):
            rotated = apply_single_qubit_gate(rotated, s_dagger, qubit)
            rotated = apply_single_qubit_gate(rotated, hadamard, qubit)
    else:
        raise ValueError(f"Unsupported basis: {basis}")

    probability = np.abs(rotated) ** 2
    probability /= float(np.sum(probability))
    return probability.astype(np.float64)


def pauli_expectations(
    probability: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    expected = 1 << n_qubits
    if values.size != expected:
        raise ValueError("Probability dimension does not match qubit count")
    indices = np.arange(expected, dtype=np.int64)
    output = np.empty(n_qubits, dtype=np.float64)
    for qubit in range(n_qubits):
        signs = 1.0 - 2.0 * ((indices >> qubit) & 1)
        output[qubit] = float(np.dot(values, signs))
    return output


def total_variation(clean: np.ndarray, distorted: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(clean - distorted)))


def hellinger_distance(clean: np.ndarray, distorted: np.ndarray) -> float:
    return math.sqrt(
        0.5
        * float(
            np.sum(
                (
                    np.sqrt(np.clip(clean, 0.0, None))
                    - np.sqrt(np.clip(distorted, 0.0, None))
                )
                ** 2
            )
        )
    )


def jensen_shannon_distance(clean: np.ndarray, distorted: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    distorted = np.asarray(distorted, dtype=np.float64)
    midpoint = 0.5 * (clean + distorted)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0.0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    divergence = 0.5 * kl(clean, midpoint) + 0.5 * kl(distorted, midpoint)
    return math.sqrt(max(0.0, divergence))


def state_from_arrays(value: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    real = np.asarray(value[f"c__{prefix}_statevector_real"], dtype=np.float64)
    imag = np.asarray(value[f"c__{prefix}_statevector_imag"], dtype=np.float64)
    return real + 1.0j * imag


def max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def pad(value: np.ndarray, length: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size > length:
        raise ValueError(f"Vector length {flat.size} exceeds schema {length}")
    output = np.zeros(length, dtype=np.float64)
    output[: flat.size] = flat
    return output


@lru_cache(maxsize=None)
def derived_observables(
    artifact_path: str,
    n_qubits: int,
) -> dict[str, np.ndarray | float]:
    with np.load(artifact_path, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}

    clean_state = state_from_arrays(value, "clean")
    distorted_state = state_from_arrays(value, "distorted")
    dimension = 1 << n_qubits

    result: dict[str, np.ndarray | float] = {}
    for basis in ("Z", "X", "Y"):
        clean_probability = basis_probabilities(clean_state, n_qubits, basis)
        distorted_probability = basis_probabilities(
            distorted_state,
            n_qubits,
            basis,
        )
        clean_expectation = pauli_expectations(clean_probability, n_qubits)
        distorted_expectation = pauli_expectations(
            distorted_probability,
            n_qubits,
        )

        result[f"clean_{basis}_probability"] = clean_probability
        result[f"distorted_{basis}_probability"] = distorted_probability
        result[f"delta_{basis}_probability"] = (
            distorted_probability - clean_probability
        )
        result[f"clean_{basis}_expectation"] = clean_expectation
        result[f"distorted_{basis}_expectation"] = distorted_expectation
        result[f"delta_{basis}_expectation"] = (
            distorted_expectation - clean_expectation
        )
        result[f"{basis}_tv"] = total_variation(
            clean_probability,
            distorted_probability,
        )
        result[f"{basis}_hellinger"] = hellinger_distance(
            clean_probability,
            distorted_probability,
        )
        result[f"{basis}_js_distance"] = jensen_shannon_distance(
            clean_probability,
            distorted_probability,
        )

    stored_z = np.zeros(dimension, dtype=np.float64)
    bitstrings = np.asarray(
        value["a__x_born_input_outcome_bitstrings"]
    ).astype(str)
    probabilities = np.asarray(
        value["a__x_born_input_probabilities"],
        dtype=np.float64,
    )
    for bits, probability in zip(bitstrings, probabilities, strict=True):
        stored_z[int(bits, 2)] = probability

    comparisons = {
        "max_distorted_z_probability_abs_error": max_abs_error(
            result["distorted_Z_probability"],
            stored_z,
        ),
        "max_distorted_x_probability_abs_error": max_abs_error(
            result["distorted_X_probability"],
            value["b__distorted_x_probabilities"],
        ),
        "max_distorted_y_probability_abs_error": max_abs_error(
            result["distorted_Y_probability"],
            value["b__distorted_y_probabilities"],
        ),
        "max_clean_x_expectation_abs_error": max_abs_error(
            result["clean_X_expectation"],
            value["c__clean_x_expectations"],
        ),
        "max_clean_y_expectation_abs_error": max_abs_error(
            result["clean_Y_expectation"],
            value["c__clean_y_expectations"],
        ),
        "max_clean_z_expectation_abs_error": max_abs_error(
            result["clean_Z_expectation"],
            value["c__clean_z_expectations"],
        ),
        "max_distorted_x_expectation_abs_error": max_abs_error(
            result["distorted_X_expectation"],
            value["b__distorted_x_expectations"],
        ),
        "max_distorted_y_expectation_abs_error": max_abs_error(
            result["distorted_Y_expectation"],
            value["b__distorted_y_expectations"],
        ),
        "max_distorted_z_expectation_abs_error": max_abs_error(
            result["distorted_Z_expectation"],
            value["c__distorted_z_expectations"],
        ),
    }
    for name, error in comparisons.items():
        AUDIT[name] = max(float(AUDIT[name]), error)
        if error > VERIFY_ATOL:
            raise ValueError(
                f"Observable derivation mismatch for {artifact_path}: "
                f"{name}={error:.3e}"
            )
    entity_id = str(np.asarray(value["entity_id"]).reshape(-1)[0])
    AUDIT["verified_entities"].add(entity_id)
    return result


def relational_increment(item: Any, schema: Any) -> np.ndarray:
    derived = derived_observables(str(item.artifact), item.n_qubits)
    pieces: list[np.ndarray] = []
    distances: list[float] = []
    for basis in ("Z", "X", "Y"):
        pieces.extend(
            [
                pad(
                    np.asarray(derived[f"clean_{basis}_probability"]),
                    schema.max_state_dim,
                ),
                pad(
                    np.asarray(derived[f"delta_{basis}_probability"]),
                    schema.max_state_dim,
                ),
                pad(
                    np.asarray(derived[f"clean_{basis}_expectation"]),
                    schema.max_qubits,
                ),
                pad(
                    np.asarray(derived[f"delta_{basis}_expectation"]),
                    schema.max_qubits,
                ),
            ]
        )
        if basis == "Z":
            pieces.append(
                pad(
                    np.asarray(derived["distorted_Z_expectation"]),
                    schema.max_qubits,
                )
            )
        distances.extend(
            [
                float(derived[f"{basis}_tv"]),
                float(derived[f"{basis}_hellinger"]),
                float(derived[f"{basis}_js_distance"]),
            ]
        )
    return np.concatenate((*pieces, np.asarray(distances, dtype=np.float64)))


def hilbert_summary(value: Mapping[str, np.ndarray], schema: Any) -> np.ndarray:
    names = np.asarray(value["c__hilbert_summary_names"]).astype(str).tolist()
    values = np.asarray(
        value["c__hilbert_summary_values"],
        dtype=np.float64,
    ).tolist()
    mapping = dict(zip(names, values, strict=True))
    return np.asarray(
        [mapping.get(name, 0.0) for name in schema.hilbert_names],
        dtype=np.float64,
    )


def aligned_state_summary(value: Mapping[str, np.ndarray]) -> np.ndarray:
    real = np.asarray(value["c__aligned_state_delta_real"], dtype=np.float64)
    imag = np.asarray(value["c__aligned_state_delta_imag"], dtype=np.float64)
    magnitude = np.sqrt(real * real + imag * imag)
    return np.asarray(
        [
            np.sum(magnitude),
            np.linalg.norm(magnitude),
            np.max(magnitude),
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(real),
            np.std(real),
            np.min(real),
            np.max(real),
            np.mean(imag),
            np.std(imag),
            np.min(imag),
            np.max(imag),
        ],
        dtype=np.float64,
    )


def raw_state_increment(
    value: Mapping[str, np.ndarray],
    schema: Any,
) -> np.ndarray:
    return np.concatenate(
        tuple(
            pad(value[name], schema.max_state_dim)
            for name in (
                "c__clean_statevector_real",
                "c__clean_statevector_imag",
                "c__distorted_statevector_real",
                "c__distorted_statevector_imag",
                "c__aligned_state_delta_real",
                "c__aligned_state_delta_imag",
            )
        )
    )


def feature_vector(
    base: Any,
    item: Any,
    schema: Any,
    regime: str,
) -> np.ndarray:
    absolute = base.vector(item, schema, "B")
    if regime == "B_absolute":
        return absolute

    value = base.arrays(item)
    delta = np.concatenate((absolute, relational_increment(item, schema)))
    if regime == "B_delta":
        return delta

    summary = np.concatenate(
        (
            delta,
            hilbert_summary(value, schema),
            aligned_state_summary(value),
        )
    )
    if regime == "C_summary":
        return summary
    if regime == "C_full":
        return np.concatenate(
            (
                summary,
                raw_state_increment(value, schema),
            )
        )
    raise ValueError(regime)


def feature_matrix(
    base: Any,
    examples: Sequence[Any],
    schema: Any,
    regime: str,
) -> np.ndarray:
    vectors = [feature_vector(base, item, schema, regime) for item in examples]
    widths = {vector.size for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"Variable feature widths for {regime}: {widths}")
    matrix = np.vstack(vectors).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {regime}")
    return matrix


def nonlinear_estimator(
    parameters: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    gamma_value = parameters["gamma"]
    gamma: str | float = (
        str(gamma_value)
        if isinstance(gamma_value, str)
        else float(gamma_value)
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.995, svd_solver="full")),
            (
                "model",
                SVC(
                    C=float(parameters["C"]),
                    kernel="rbf",
                    gamma=gamma,
                    shrinking=True,
                    tol=1.0e-5,
                    cache_size=512.0,
                    max_iter=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def regime_status(results: Mapping[str, Any], regime: str) -> str:
    values = {results[regime][model]["status"] for model in MODELS}
    if values == {"strong"}:
        return "strong"
    if values == {"weak"}:
        return "weak"
    return "mixed_or_intermediate"


def development_decision(statuses: Mapping[str, str]) -> dict[str, Any]:
    if statuses["B_delta"] == "strong":
        code = "OBSERVABLE_RELATIONAL_EVIDENCE_SUFFICIENT"
        interpretation = (
            "B_delta is strong: ordinary Z/X/Y observables become sufficient "
            "when represented relative to the clean circuit."
        )
    elif statuses["C_summary"] == "strong":
        code = "COMPRESSED_HILBERT_DIAGNOSTICS_SUFFICIENT"
        interpretation = (
            "B_delta is not strong while C_summary is strong: compressed "
            "Hilbert diagnostics add decisive information without raw amplitudes."
        )
    elif statuses["C_full"] == "strong":
        code = "RAW_HILBERT_COMPONENTS_ADD_DECISIVE_SIGNAL"
        interpretation = (
            "Only C_full is strong: raw statevector components add decisive "
            "signal beyond observables and compressed Hilbert diagnostics."
        )
    else:
        code = "INCONCLUSIVE_OR_MIXED"
        interpretation = (
            "No nested regime reached a unanimous strong status across both "
            "probe families. Treat the pattern as a development diagnostic."
        )
    return {
        "status": code,
        "interpretation": interpretation,
        "regime_statuses": dict(statuses),
        "confirmatory": False,
        "reason_not_confirmatory": (
            "The 80-entity validation cohort already informed this experiment."
        ),
    }


def main() -> None:
    config = parse_args()
    base = load_base()
    original_estimator = base.estimator
    base.NONLINEAR_GRID = NONLINEAR_GRID

    def estimator(
        model: str,
        parameters: Mapping[str, Any],
        seed: int,
    ) -> Pipeline:
        if model == "nonlinear":
            return nonlinear_estimator(parameters, seed)
        return original_estimator(model, parameters, seed)

    base.estimator = estimator
    warnings.filterwarnings("error", category=ConvergenceWarning)

    root = config.product_dir.expanduser().resolve()
    complete = base.read_json(root / "generation_complete.json")
    if complete.get("test_split_accessed") is not False:
        raise RuntimeError("Pilot did not certify historical-test isolation")

    examples = base.load_examples(root)
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    train_y = np.asarray([item.y for item in train], dtype=np.int64)
    validation_y = np.asarray(
        [item.y for item in validation],
        dtype=np.int64,
    )
    train_groups = np.asarray([item.group for item in train], dtype=object)
    validation_groups = np.asarray(
        [item.group for item in validation],
        dtype=object,
    )
    if set(train_groups) & set(validation_groups):
        raise RuntimeError("Train/validation group overlap")

    schema = base.fit_schema(train)
    protocol = {
        "schema": SCHEMA,
        "product_id": str(complete["product_id"]),
        "product_manifest_sha256": str(complete["manifest_sha256"]),
        "script_sha256": base.file_hash(Path(__file__).resolve()),
        "base_evaluator_sha256": base.file_hash(BASE_PATH),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "evaluation_role": "development_diagnostic_not_confirmatory",
        "regimes": list(REGIMES),
        "nested_regime_order": list(REGIMES),
        "models": list(MODELS),
        "model_seeds": list(base.MODEL_SEEDS),
        "cv_seed": base.CV_SEED,
        "cv_folds": config.cv_folds,
        "shuffle_repeats": config.shuffle_repeats,
        "bootstrap_repeats": config.bootstrap_repeats,
        "ece_bins": config.ece_bins,
        "linear_grid": list(base.LINEAR_GRID),
        "nonlinear_grid": list(NONLINEAR_GRID),
        "strong_thresholds": base.STRONG,
        "js_distance_convention": (
            "sqrt(Jensen-Shannon divergence), natural logarithm"
        ),
        "b_delta_statevector_policy": (
            "Frozen statevectors are used only to derive exact clean/distorted "
            "Z/X/Y observables. No amplitudes enter B_absolute or B_delta."
        ),
        "c_summary_raw_amplitudes_exposed": False,
        "c_full_raw_amplitudes_exposed": True,
        "historical_v0_1_test_accessed": False,
    }
    protocol_id = base.text_hash(base.canon(protocol))
    output = (
        root
        / "reports"
        / "relational_observable_probes"
        / f"reldelta_{protocol_id.removeprefix('sha256:')[:20]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol["protocol_id"] = protocol_id
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and base.read_json(protocol_path) != protocol:
        raise RuntimeError("Protocol directory contains different content")
    base.atomic_json(protocol_path, protocol)

    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for regime in REGIMES:
        if config.progress:
            print(f"Extracting {regime}", flush=True)
        features[regime] = (
            feature_matrix(base, train, schema, regime),
            feature_matrix(base, validation, schema, regime),
        )

    if len(AUDIT["verified_entities"]) != len(examples):
        raise RuntimeError(
            f"Observable audit covered {len(AUDIT['verified_entities'])}/"
            f"{len(examples)} entities"
        )
    observable_audit = {
        key: (len(value) if key == "verified_entities" else float(value))
        for key, value in AUDIT.items()
    }
    observable_audit["verification_atol"] = VERIFY_ATOL
    observable_audit["status"] = "PASS"
    base.atomic_json(
        output / "observable_derivation_audit.json",
        observable_audit,
    )

    splits = base.cv_splits(train_y, train_groups, config.cv_folds)
    results: dict[str, Any] = {}
    validation_probabilities: dict[tuple[str, str], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []

    for regime_index, regime in enumerate(REGIMES):
        results[regime] = {}
        train_x, validation_x = features[regime]
        for model_index, model_name in enumerate(MODELS):
            prefix = f"[{regime}/{model_name}]" if config.progress else ""
            parameters, tuning_rows = base.tune(
                model_name,
                train_x,
                train_y,
                train_groups,
                splits,
                progress_prefix=prefix,
            )
            _, probability, calibration = base.calibrated_ensemble(
                model_name,
                parameters,
                train_x,
                train_y,
                validation_x,
                splits,
            )
            validation_probabilities[(regime, model_name)] = probability
            row = base.metrics(validation_y, probability, config.ece_bins)
            ci = base.bootstrap_ci(
                validation_y,
                probability,
                validation_groups,
                config.bootstrap_repeats,
                config.ece_bins,
                base.BOOT_SEED + regime_index * 100 + model_index,
            )
            controls = base.label_shuffle_controls(
                model_name,
                train_x,
                train_y,
                train_groups,
                validation_x,
                validation_y,
                config.shuffle_repeats,
                config.cv_folds,
                prefix,
            )
            shuffle_ba = [item["balanced_accuracy"] for item in controls]
            status = base.model_status(row, ci)
            results[regime][model_name] = {
                "best_parameters": parameters,
                "tuning": tuning_rows,
                "calibration": calibration,
                "validation_metrics": row,
                "bootstrap_95_ci": ci,
                "status": status,
                "label_shuffle_controls": controls,
                "label_shuffle_summary": {
                    "mean_balanced_accuracy": float(np.mean(shuffle_ba)),
                    "max_balanced_accuracy": float(np.max(shuffle_ba)),
                    "empirical_p_value": float(
                        (
                            1
                            + sum(
                                value >= row["balanced_accuracy"]
                                for value in shuffle_ba
                            )
                        )
                        / (1 + len(shuffle_ba))
                    ),
                },
            }
            metric_rows.append(
                {
                    "regime": regime,
                    "model": model_name,
                    "status": status,
                    **row,
                    **{
                        f"{name}_ci_low": interval["low"]
                        for name, interval in ci.items()
                    },
                    **{
                        f"{name}_ci_high": interval["high"]
                        for name, interval in ci.items()
                    },
                }
            )
            for index, item in enumerate(validation):
                prediction_rows.append(
                    {
                        "entity_id": item.entity_id,
                        "split_group_id": item.group,
                        "coarse_label": item.coarse_label,
                        "raw_label": item.raw_label,
                        "family": item.family,
                        "strength": item.strength,
                        "affected_qubit": item.affected_qubit,
                        "n_qubits": item.n_qubits,
                        "phase_sensitive_family": item.phase_sensitive,
                        "regime": regime,
                        "model": model_name,
                        "true_binary_label": int(validation_y[index]),
                        "amplitude_probability": float(probability[index]),
                        "predicted_binary_label": int(probability[index] >= 0.5),
                    }
                )
            for stratum in base.stratified(
                validation,
                validation_y,
                probability,
            ):
                stratified_rows.append(
                    {
                        "regime": regime,
                        "model": model_name,
                        **stratum,
                    }
                )

    statuses = {
        regime: regime_status(results, regime)
        for regime in REGIMES
    }
    comparisons = (
        ("B_delta_minus_B_absolute", "B_absolute", "B_delta"),
        ("C_summary_minus_B_delta", "B_delta", "C_summary"),
        ("C_full_minus_C_summary", "C_summary", "C_full"),
    )
    paired: dict[str, Any] = {}
    for model_index, model_name in enumerate(MODELS):
        paired[model_name] = {}
        for comparison_index, (name, left, right) in enumerate(comparisons):
            paired[model_name][name] = base.paired_delta(
                validation_y,
                validation_probabilities[(left, model_name)],
                validation_probabilities[(right, model_name)],
                validation_groups,
                config.bootstrap_repeats,
                base.BOOT_SEED
                + 3000
                + model_index * 100
                + comparison_index,
            )

    decision = development_decision(statuses)
    report = {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "product_id": complete["product_id"],
        "evaluation_role": "development_diagnostic_not_confirmatory",
        "historical_v0_1_test_accessed": False,
        "train_count": len(train),
        "validation_count": len(validation),
        "feature_dimensions": {
            regime: int(features[regime][0].shape[1])
            for regime in REGIMES
        },
        "aligned_summary_names": list(ALIGNED_SUMMARY_NAMES),
        "observable_derivation_audit": observable_audit,
        "results": results,
        "paired_bootstrap_deltas": paired,
        "decision": decision,
    }
    base.atomic_json(output / "results.json", report)
    base.atomic_json(output / "decision.json", decision)
    base.write_csv(output / "metrics.csv", metric_rows)
    base.write_csv(output / "predictions.csv", prediction_rows)
    base.write_csv(output / "stratified_metrics.csv", stratified_rows)
    base.atomic_json(
        output / "complete.json",
        {
            "schema": SCHEMA,
            "protocol_id": protocol_id,
            "product_id": complete["product_id"],
            "status": "COMPLETE",
            "results_sha256": base.file_hash(output / "results.json"),
            "decision_sha256": base.file_hash(output / "decision.json"),
            "observable_audit_sha256": base.file_hash(
                output / "observable_derivation_audit.json"
            ),
            "confirmatory": False,
            "historical_v0_1_test_accessed": False,
        },
    )

    print()
    print("=" * 78)
    print("TRIQTO RELATIONAL-OBSERVABLE DEVELOPMENT PROBE COMPLETE")
    print("=" * 78)
    for regime in REGIMES:
        print(f"{regime}: {statuses[regime]}")
        for model_name in MODELS:
            row = results[regime][model_name]["validation_metrics"]
            print(
                f"  {model_name}: "
                f"BA={row['balanced_accuracy']:.4f}, "
                f"macro-F1={row['macro_f1']:.4f}, "
                f"phase recall={row['phase_recall']:.4f}, "
                f"amplitude recall={row['amplitude_recall']:.4f}, "
                f"AUROC={row['auroc']:.4f}, "
                f"ECE={row['ece']:.4f}"
            )
    print()
    print(f"Decision: {decision['status']}")
    print(decision["interpretation"])
    print("Confirmatory: NO — this validation cohort informed the design.")
    print(f"Output: {output}")
    print("The historical v0.1 test split was not accessed.")


if __name__ == "__main__":
    main()
