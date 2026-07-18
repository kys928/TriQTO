"""Configuration contract for deterministic offline dataset preprocessing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import MISSINGNESS_STATUSES, PREPROCESSING_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class NumericalTolerances:
    probability_sum_warning: float = 1e-10
    probability_sum_repair: float = 1e-8
    probability_negative_repair: float = 1e-14
    state_norm_warning: float = 1e-10
    state_norm_repair: float = 1e-8
    hermiticity_warning: float = 1e-10
    hermiticity_repair: float = 1e-8
    trace_warning: float = 1e-10
    trace_repair: float = 1e-8
    psd_eigenvalue_warning: float = -1e-10
    psd_eigenvalue_failure: float = -1e-7
    hash_rounding_decimals: int = 14
    comparison_atol: float = 1e-10
    comparison_rtol: float = 1e-9


@dataclass(frozen=True, slots=True)
class CanonicalizationConfig:
    angle_interval: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    angle_period: float = 6.283185307179586
    gate_alias_map: dict[str, str] = field(
        default_factory=lambda: {
            "cnot": "cx",
            "CNOT": "cx",
            "CX": "cx",
            "measure_all": "measure",
        }
    )
    basis_alias_map: dict[str, str] = field(
        default_factory=lambda: {
            "computational": "Z",
            "z": "Z",
            "Z": "Z",
            "x": "X",
            "X": "X",
            "y": "Y",
            "Y": "Y",
        }
    )
    bit_order: str = "qiskit_msb_left"
    graph_method: str = "wl_bucket_plus_exact_isomorphism"
    state_global_phase_epsilon: float = 1e-12
    floating_point_format: str = ".17g"
    barrier_semantically_irrelevant: bool = False


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    allowed_labels: tuple[str, ...] = ()
    repair_small_numerical_drift: bool = True
    validate_cptp: bool = True
    validate_layout: bool = True
    validate_backend_basis: bool = True
    quarantine_unknown_bit_order: bool = True


@dataclass(frozen=True, slots=True)
class EffectConfig:
    hilbert_metrics: tuple[str, ...] = (
        "infidelity",
        "fubini_study",
        "pure_trace_distance",
    )
    born_metrics: tuple[str, ...] = (
        "hellinger",
        "jensen_shannon_distance",
        "total_variation",
        "fisher_rao",
    )
    combined_distance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "hilbert": 0.35,
            "born": 0.35,
            "graph": 0.15,
            "parameter": 0.15,
            "metadata": 0.0,
        }
    )
    severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "negligible_max": 1e-6,
            "weak_max": 1e-3,
            "moderate_max": 5e-2,
            "strong_max": 2.5e-1,
        }
    )
    phase_sensitive_threshold: float = 1e-4
    born_effect_threshold: float = 1e-4
    layout_depth_threshold: float = 0.05


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    group_by_base_circuit: bool = True
    group_by_target: bool = True
    group_by_trajectory: bool = True
    group_by_parameter_neighbourhood: bool = True
    group_by_calibration_window: bool = True
    group_by_symmetry: bool = True
    group_by_family_for_selected_splits: bool = True
    calibration_window_seconds: int = 86_400
    hard_negative_max_pairs_per_category: int = 10_000


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 20260718
    challenge_splits: tuple[str, ...] = (
        "grouped_baseline",
        "held_out_circuit_instance",
        "held_out_parameter_region",
        "held_out_distortion_strength",
        "held_out_distortion_combination",
        "held_out_circuit_family",
        "held_out_layout_identity",
        "held_out_layout_structure",
        "held_out_qubit_count",
        "temporal_calibration",
        "held_out_backend",
    )
    minimum_group_count: int = 3
    stratification_fields: tuple[str, ...] = (
        "family",
        "n_qubits",
        "intervention_label",
        "severity",
    )
    parameter_region_bins: int = 8


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    enabled: bool = True
    methods: tuple[str, ...] = ("mad", "iqr", "nearest_neighbor")
    mad_threshold: float = 6.0
    iqr_multiplier: float = 3.0
    nearest_neighbor_quantile: float = 0.995
    tag_only: bool = True


@dataclass(frozen=True, slots=True)
class BalancingConfig:
    enabled: bool = True
    dimensions: tuple[str, ...] = (
        "family",
        "n_qubits",
        "intervention_label",
        "severity",
        "measurement_basis",
        "source_type",
    )
    method: str = "effective_number"
    beta: float = 0.999
    clipping: tuple[float, float] = (0.25, 8.0)


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    enabled: bool = True
    method_by_family: dict[str, str] = field(
        default_factory=lambda: {
            "angles": "sincos",
            "gate_errors": "log1p_robust",
            "coherence_times": "log_robust",
            "probabilities": "identity",
            "distances": "identity",
            "topology_lifetimes": "robust",
            "topology_counts": "log1p_robust",
            "depth_counts": "log1p_robust",
        }
    )


@dataclass(frozen=True, slots=True)
class ReportConfig:
    html: bool = True
    json: bool = True
    parquet: bool = True
    plots: bool = False


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    schema_version: str = PREPROCESSING_SCHEMA_VERSION
    preprocessing_version: str = "1.0.0"
    random_seed: int = 20260718
    numerical_tolerances: NumericalTolerances = field(default_factory=NumericalTolerances)
    canonicalization: CanonicalizationConfig = field(default_factory=CanonicalizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    effects: EffectConfig = field(default_factory=EffectConfig)
    missingness_statuses: tuple[str, ...] = MISSINGNESS_STATUSES
    forbid_implicit_zero_fill: bool = True
    preserve_duplicate_multiplicity: bool = True
    canonical_representative_policy: str = "lexicographically_smallest_sample_id"
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)

    def validate(self) -> None:
        if self.schema_version != PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preprocessing schema_version {self.schema_version!r}; "
                f"expected {PREPROCESSING_SCHEMA_VERSION!r}"
            )
        if not self.preprocessing_version.strip():
            raise ValueError("preprocessing_version must be nonblank")
        fractions = (
            self.splits.train_fraction,
            self.splits.validation_fraction,
            self.splits.test_fraction,
        )
        if any(value <= 0.0 or value >= 1.0 for value in fractions):
            raise ValueError("split fractions must each be in (0, 1)")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to exactly 1 within tolerance")
        if self.canonicalization.angle_period <= 0.0:
            raise ValueError("angle_period must be positive")
        low, high = self.canonicalization.angle_interval
        if not low < high:
            raise ValueError("angle_interval must be increasing")
        if abs((high - low) - self.canonicalization.angle_period) > 1e-12:
            raise ValueError("angle_interval width must match angle_period")
        if self.numerical_tolerances.hash_rounding_decimals < 0:
            raise ValueError("hash_rounding_decimals must be nonnegative")
        if self.grouping.calibration_window_seconds <= 0:
            raise ValueError("calibration_window_seconds must be positive")
        if self.splits.parameter_region_bins < 2:
            raise ValueError("parameter_region_bins must be at least 2")
        missing = set(self.missingness_statuses)
        required = set(MISSINGNESS_STATUSES)
        if not required.issubset(missing):
            raise ValueError(
                f"missingness_statuses must contain {sorted(required - missing)}"
            )
        weights = self.effects.combined_distance_weights
        if any(float(value) < 0.0 for value in weights.values()):
            raise ValueError("combined effect weights must be nonnegative")
        if sum(float(value) for value in weights.values()) <= 0.0:
            raise ValueError("at least one combined effect weight must be positive")
        lower, upper = self.balancing.clipping
        if lower <= 0.0 or upper < lower:
            raise ValueError("balancing clipping must be positive and ordered")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return default
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("expected a list or tuple")
    return tuple(value)


def preprocessing_config_from_dict(payload: Mapping[str, Any]) -> PreprocessingConfig:
    root = dict(payload)
    num = _mapping(root.get("numerical_tolerances"), "numerical_tolerances")
    can = _mapping(root.get("canonicalization"), "canonicalization")
    val = _mapping(root.get("validation"), "validation")
    eff = _mapping(root.get("effects"), "effects")
    grp = _mapping(root.get("grouping"), "grouping")
    spl = _mapping(root.get("splits"), "splits")
    out = _mapping(root.get("outliers"), "outliers")
    bal = _mapping(root.get("balancing"), "balancing")
    sca = _mapping(root.get("scaling"), "scaling")
    rep = _mapping(root.get("reports"), "reports")

    defaults = PreprocessingConfig()
    config = PreprocessingConfig(
        schema_version=str(root.get("schema_version", defaults.schema_version)),
        preprocessing_version=str(
            root.get("preprocessing_version", defaults.preprocessing_version)
        ),
        random_seed=int(root.get("random_seed", defaults.random_seed)),
        numerical_tolerances=NumericalTolerances(**num),
        canonicalization=CanonicalizationConfig(
            angle_interval=tuple(can.get("angle_interval", defaults.canonicalization.angle_interval)),
            angle_period=float(can.get("angle_period", defaults.canonicalization.angle_period)),
            gate_alias_map=dict(can.get("gate_alias_map", defaults.canonicalization.gate_alias_map)),
            basis_alias_map=dict(can.get("basis_alias_map", defaults.canonicalization.basis_alias_map)),
            bit_order=str(can.get("bit_order", defaults.canonicalization.bit_order)),
            graph_method=str(can.get("graph_method", defaults.canonicalization.graph_method)),
            state_global_phase_epsilon=float(
                can.get(
                    "state_global_phase_epsilon",
                    defaults.canonicalization.state_global_phase_epsilon,
                )
            ),
            floating_point_format=str(
                can.get("floating_point_format", defaults.canonicalization.floating_point_format)
            ),
            barrier_semantically_irrelevant=bool(
                can.get(
                    "barrier_semantically_irrelevant",
                    defaults.canonicalization.barrier_semantically_irrelevant,
                )
            ),
        ),
        validation=ValidationConfig(
            allowed_labels=_tuple(val.get("allowed_labels"), defaults.validation.allowed_labels),
            repair_small_numerical_drift=bool(
                val.get(
                    "repair_small_numerical_drift",
                    defaults.validation.repair_small_numerical_drift,
                )
            ),
            validate_cptp=bool(val.get("validate_cptp", defaults.validation.validate_cptp)),
            validate_layout=bool(val.get("validate_layout", defaults.validation.validate_layout)),
            validate_backend_basis=bool(
                val.get("validate_backend_basis", defaults.validation.validate_backend_basis)
            ),
            quarantine_unknown_bit_order=bool(
                val.get(
                    "quarantine_unknown_bit_order",
                    defaults.validation.quarantine_unknown_bit_order,
                )
            ),
        ),
        effects=EffectConfig(
            hilbert_metrics=_tuple(eff.get("hilbert_metrics"), defaults.effects.hilbert_metrics),
            born_metrics=_tuple(eff.get("born_metrics"), defaults.effects.born_metrics),
            combined_distance_weights=dict(
                eff.get(
                    "combined_distance_weights",
                    defaults.effects.combined_distance_weights,
                )
            ),
            severity_thresholds=dict(
                eff.get("severity_thresholds", defaults.effects.severity_thresholds)
            ),
            phase_sensitive_threshold=float(
                eff.get(
                    "phase_sensitive_threshold",
                    defaults.effects.phase_sensitive_threshold,
                )
            ),
            born_effect_threshold=float(
                eff.get("born_effect_threshold", defaults.effects.born_effect_threshold)
            ),
            layout_depth_threshold=float(
                eff.get("layout_depth_threshold", defaults.effects.layout_depth_threshold)
            ),
        ),
        missingness_statuses=_tuple(
            root.get("missingness_statuses"), defaults.missingness_statuses
        ),
        forbid_implicit_zero_fill=bool(
            root.get("forbid_implicit_zero_fill", defaults.forbid_implicit_zero_fill)
        ),
        preserve_duplicate_multiplicity=bool(
            root.get(
                "preserve_duplicate_multiplicity",
                defaults.preserve_duplicate_multiplicity,
            )
        ),
        canonical_representative_policy=str(
            root.get(
                "canonical_representative_policy",
                defaults.canonical_representative_policy,
            )
        ),
        grouping=GroupingConfig(
            group_by_base_circuit=bool(
                grp.get("group_by_base_circuit", defaults.grouping.group_by_base_circuit)
            ),
            group_by_target=bool(grp.get("group_by_target", defaults.grouping.group_by_target)),
            group_by_trajectory=bool(
                grp.get("group_by_trajectory", defaults.grouping.group_by_trajectory)
            ),
            group_by_parameter_neighbourhood=bool(
                grp.get(
                    "group_by_parameter_neighbourhood",
                    defaults.grouping.group_by_parameter_neighbourhood,
                )
            ),
            group_by_calibration_window=bool(
                grp.get(
                    "group_by_calibration_window",
                    defaults.grouping.group_by_calibration_window,
                )
            ),
            group_by_symmetry=bool(
                grp.get("group_by_symmetry", defaults.grouping.group_by_symmetry)
            ),
            group_by_family_for_selected_splits=bool(
                grp.get(
                    "group_by_family_for_selected_splits",
                    defaults.grouping.group_by_family_for_selected_splits,
                )
            ),
            calibration_window_seconds=int(
                grp.get(
                    "calibration_window_seconds",
                    defaults.grouping.calibration_window_seconds,
                )
            ),
            hard_negative_max_pairs_per_category=int(
                grp.get(
                    "hard_negative_max_pairs_per_category",
                    defaults.grouping.hard_negative_max_pairs_per_category,
                )
            ),
        ),
        splits=SplitConfig(
            train_fraction=float(spl.get("train_fraction", defaults.splits.train_fraction)),
            validation_fraction=float(
                spl.get("validation_fraction", defaults.splits.validation_fraction)
            ),
            test_fraction=float(spl.get("test_fraction", defaults.splits.test_fraction)),
            seed=int(spl.get("seed", defaults.splits.seed)),
            challenge_splits=_tuple(
                spl.get("challenge_splits"), defaults.splits.challenge_splits
            ),
            minimum_group_count=int(
                spl.get("minimum_group_count", defaults.splits.minimum_group_count)
            ),
            stratification_fields=_tuple(
                spl.get("stratification_fields"), defaults.splits.stratification_fields
            ),
            parameter_region_bins=int(
                spl.get("parameter_region_bins", defaults.splits.parameter_region_bins)
            ),
        ),
        outliers=OutlierConfig(
            enabled=bool(out.get("enabled", defaults.outliers.enabled)),
            methods=_tuple(out.get("methods"), defaults.outliers.methods),
            mad_threshold=float(out.get("mad_threshold", defaults.outliers.mad_threshold)),
            iqr_multiplier=float(
                out.get("iqr_multiplier", defaults.outliers.iqr_multiplier)
            ),
            nearest_neighbor_quantile=float(
                out.get(
                    "nearest_neighbor_quantile",
                    defaults.outliers.nearest_neighbor_quantile,
                )
            ),
            tag_only=bool(out.get("tag_only", defaults.outliers.tag_only)),
        ),
        balancing=BalancingConfig(
            enabled=bool(bal.get("enabled", defaults.balancing.enabled)),
            dimensions=_tuple(bal.get("dimensions"), defaults.balancing.dimensions),
            method=str(bal.get("method", defaults.balancing.method)),
            beta=float(bal.get("beta", defaults.balancing.beta)),
            clipping=tuple(bal.get("clipping", defaults.balancing.clipping)),
        ),
        scaling=ScalingConfig(
            enabled=bool(sca.get("enabled", defaults.scaling.enabled)),
            method_by_family=dict(
                sca.get("method_by_family", defaults.scaling.method_by_family)
            ),
        ),
        reports=ReportConfig(
            html=bool(rep.get("html", defaults.reports.html)),
            json=bool(rep.get("json", defaults.reports.json)),
            parquet=bool(rep.get("parquet", defaults.reports.parquet)),
            plots=bool(rep.get("plots", defaults.reports.plots)),
        ),
    )
    config.validate()
    return config


def load_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise TypeError("preprocessing config root must be a mapping")
    return preprocessing_config_from_dict(payload)


def preprocessing_config_to_dict(config: PreprocessingConfig) -> dict[str, Any]:
    config.validate()
    return asdict(config)


def save_preprocessing_config(config: PreprocessingConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(preprocessing_config_to_dict(config), sort_keys=True),
        encoding="utf-8",
    )
    return target
