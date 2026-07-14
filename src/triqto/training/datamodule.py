"""Phase 12 → Phase 13 deterministic tensor adapter and budget batching."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from triqto.graph.utils import resolve_safe_file
from triqto.model import (
    ACTION_EDIT_TYPES,
    DISTORTION_LABELS,
    ActionCandidateTensorBatch,
    BornTensorBatch,
    DenseFeatureBatch,
    GraphTensorBatch,
    HilbertTensorBatch,
    OutcomeQueryTensorBatch,
    ParameterTensorBatch,
    TriQTOBatch,
    TriQTOModelConfig,
)
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER
from triqto.training_views.artifacts import load_training_view_item_artifact

from .config import TrainingConfig
from .constants import (
    ACTION_EDIT_TYPE_MAP,
    DISTORTION_TO_COARSE_LABEL,
    INPUT_GROUP_TO_STREAM,
    PHASE12_TO_MODEL_HEAD,
    TRAINING_ADAPTER_VERSION,
)
from .models import (
    ActionTargets,
    BornTargets,
    CompletedTrainingViewDataset,
    DiagnosisTargets,
    GeometryTargets,
    SupervisedBatch,
    TrainingDataSpec,
    TrainingExample,
    TrainingTargets,
)

_ACTION_FEATURE_NAMES = (
    "edit_count",
    "risk_score",
    "depth_delta",
    "gate_delta",
    "is_no_op",
)


def _np(item: Any, name: str, *, required: bool = True) -> np.ndarray | None:
    value = item.arrays.get(name)
    if value is None:
        if required:
            raise ValueError(f"Training item {item.view_item_id} is missing array {name}")
        return None
    if not isinstance(value, np.ndarray) or value.dtype.kind == "O":
        raise TypeError(f"Training item array {name} must be a non-object NumPy array")
    return value


def _float_tensor(value: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).clone()


def _long_tensor(value: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.long).clone()


def _bool_tensor(value: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.bool).clone()


def _bitstrings_to_tensors(bitstrings: Sequence[str]) -> tuple[Tensor, Tensor]:
    if not bitstrings:
        return torch.zeros((0, 0), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.bool)
    widths = [len(value) for value in bitstrings]
    if any(width <= 0 for width in widths):
        raise ValueError("Outcome bitstrings must be nonblank")
    maximum = max(widths)
    bits = torch.zeros((len(bitstrings), maximum), dtype=torch.float32)
    mask = torch.zeros((len(bitstrings), maximum), dtype=torch.bool)
    for row, value in enumerate(bitstrings):
        if any(character not in {"0", "1"} for character in value):
            raise ValueError(f"Invalid outcome bitstring {value!r}")
        # Preserve Phase 8/12 textual order exactly; the model treats positions as
        # ordered logical output coordinates and does not reinterpret endianness.
        width = len(value)
        bits[row, :width] = torch.tensor([int(character) for character in value], dtype=torch.float32)
        mask[row, :width] = True
    return bits, mask


def _graph_batch_from_arrays(arrays: dict[str, np.ndarray]) -> GraphTensorBatch:
    required = (
        "graph_node_features",
        "graph_edge_index",
        "graph_edge_features",
        "graph_edge_event_index",
        "graph_gate_features",
        "graph_gate_qubit_ptr",
        "graph_gate_qubit_indices",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"Graph anchor is missing arrays: {missing}")
    node = _float_tensor(arrays["graph_node_features"])
    gate = _float_tensor(arrays["graph_gate_features"])
    return GraphTensorBatch(
        node_features=node,
        edge_index=_long_tensor(arrays["graph_edge_index"]),
        edge_features=_float_tensor(arrays["graph_edge_features"]),
        edge_event_index=_long_tensor(arrays["graph_edge_event_index"]),
        gate_features=gate,
        gate_qubit_ptr=_long_tensor(arrays["graph_gate_qubit_ptr"]),
        gate_qubit_indices=_long_tensor(arrays["graph_gate_qubit_indices"]),
        node_batch=torch.zeros(node.shape[0], dtype=torch.long),
        gate_batch=torch.zeros(gate.shape[0], dtype=torch.long),
        graph_count=1,
    )


def _parameter_batch(arrays: dict[str, np.ndarray]) -> ParameterTensorBatch | None:
    values = arrays.get("graph_parameter_values")
    if values is None or values.size == 0:
        return None
    sin = arrays.get("graph_parameter_sin")
    cos = arrays.get("graph_parameter_cos")
    if sin is None or cos is None:
        raise ValueError("Parameter values require sine and cosine arrays")
    tensor = _float_tensor(values.reshape(-1))
    return ParameterTensorBatch(
        values=tensor,
        sin=_float_tensor(sin.reshape(-1)),
        cos=_float_tensor(cos.reshape(-1)),
        batch_index=torch.zeros(tensor.numel(), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def _born_batch(arrays: dict[str, np.ndarray], prefix: str) -> BornTensorBatch | None:
    names = arrays.get(f"{prefix}_outcome_bitstrings")
    probabilities = arrays.get(f"{prefix}_probabilities")
    if names is None and probabilities is None:
        return None
    if names is None or probabilities is None:
        raise ValueError(f"Incomplete {prefix} Born arrays")
    values = [str(value) for value in names.tolist()]
    bits, mask = _bitstrings_to_tensors(values)
    probs = _float_tensor(probabilities.reshape(-1))
    return BornTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=mask,
        probabilities=probs,
        batch_index=torch.zeros(probs.numel(), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def _outcome_queries(arrays: dict[str, np.ndarray]) -> OutcomeQueryTensorBatch | None:
    names = arrays.get("born_target_outcome_bitstrings")
    if names is None:
        return None
    values = [str(value) for value in names.tolist()]
    bits, mask = _bitstrings_to_tensors(values)
    return OutcomeQueryTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=mask,
        batch_index=torch.zeros(len(values), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def _phase7_input_ref(item: Any) -> str | None:
    datasets = [str(value) for value in _np(item, "source_dataset_names").tolist()]
    usages = [str(value) for value in _np(item, "source_usage_names").tolist()]
    refs = [str(value) for value in _np(item, "source_refs").tolist()]
    found = [
        reference
        for dataset, usage, reference in zip(datasets, usages, refs, strict=True)
        if dataset == "phase7" and usage == "input"
    ]
    if len(found) > 1:
        raise ValueError(f"Item {item.view_item_id} has multiple Phase 7 input refs")
    return found[0] if found else None


def _hilbert_batch(item: Any, phase7_root: Path | None, n_qubits: int) -> tuple[HilbertTensorBatch | None, Tensor | None]:
    reference = _phase7_input_ref(item)
    if reference is None:
        return None, None
    if phase7_root is None:
        raise ValueError("Hilbert-enabled Phase 12 items require --phase7-root")
    path = resolve_safe_file(phase7_root, reference, f"Hilbert input for {item.view_item_id}")
    state = np.load(path, allow_pickle=False)
    if not isinstance(state, np.ndarray) or state.ndim != 1 or state.dtype.kind != "c":
        raise TypeError("Phase 7 Hilbert input must be a one-dimensional complex array")
    if state.size != 2**n_qubits:
        raise ValueError(
            f"Hilbert state length {state.size} does not equal 2**{n_qubits}"
        )
    if not np.isfinite(state.real).all() or not np.isfinite(state.imag).all():
        raise ValueError("Hilbert state contains non-finite values")
    norm = float(np.vdot(state, state).real)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Hilbert state is not normalized: norm²={norm}")
    amplitude = torch.from_numpy(np.stack((state.real, state.imag), axis=1)).to(torch.float32)
    strings = [format(index, f"0{n_qubits}b") for index in range(state.size)]
    bits, mask = _bitstrings_to_tensors(strings)
    return (
        HilbertTensorBatch(
            amplitudes_real_imag=amplitude,
            basis_bits=bits,
            basis_bit_mask=mask,
            batch_index=torch.zeros(state.size, dtype=torch.long),
            available_mask=torch.tensor([True]),
        ),
        amplitude,
    )


def _topology_pairs(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in sorted(arrays):
        if not name.startswith("topology_") or not name.endswith("_feature_names"):
            continue
        value_name = name[:-5] + "values"
        values = arrays.get(value_name)
        names = arrays[name]
        if values is None:
            raise ValueError(f"Topology names {name} have no matching values")
        if names.ndim != 1 or values.ndim != 1 or names.size != values.size:
            raise ValueError(f"Topology arrays {name}/{value_name} are inconsistent")
        namespace = name.removeprefix("topology_").removesuffix("_feature_names") or "global"
        for feature, value in zip(names.tolist(), values.tolist(), strict=True):
            key = f"{namespace}:{feature}"
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Topology feature {key} is non-finite")
            if key in result:
                raise ValueError(f"Duplicate qualified topology feature {key}")
            result[key] = numeric
    return result


def _topology_batch(arrays: dict[str, np.ndarray], spec: TrainingDataSpec) -> DenseFeatureBatch | None:
    pairs = _topology_pairs(arrays)
    if not pairs:
        return None
    unknown = set(pairs) - set(spec.topology_feature_names)
    if unknown:
        raise ValueError(f"Validation item contains unseen topology features: {sorted(unknown)}")
    vector = np.zeros(spec.topology_input_dim, dtype=np.float32)
    position = {name: index for index, name in enumerate(spec.topology_feature_names)}
    for name, value in pairs.items():
        index = position[name]
        if spec.normalize_topology_features:
            value = (value - spec.topology_feature_mean[index]) / spec.topology_feature_std[index]
        vector[index] = value
    return DenseFeatureBatch(
        features=torch.from_numpy(vector).reshape(1, -1),
        available_mask=torch.tensor([True]),
    )


def _backend_batch(arrays: dict[str, np.ndarray], spec: TrainingDataSpec) -> DenseFeatureBatch | None:
    features = arrays.get("backend_features")
    available = arrays.get("backend_available_mask")
    if features is None and available is None:
        return None
    if features is None or available is None:
        raise ValueError("Incomplete backend arrays")
    names = tuple(str(value) for value in arrays.get("backend_feature_names", np.asarray([], dtype="<U1")).tolist())
    if names != spec.backend_feature_names:
        raise ValueError("Backend feature schema does not match training-only adapter spec")
    matrix = np.asarray(features, dtype=np.float32).copy()
    if spec.normalize_backend_features:
        mean = np.asarray(spec.backend_feature_mean, dtype=np.float32)
        std = np.asarray(spec.backend_feature_std, dtype=np.float32)
        matrix = (matrix - mean.reshape(1, -1)) / std.reshape(1, -1)
    tensor = _float_tensor(matrix)
    mask = _bool_tensor(np.asarray(available, dtype=np.bool_).reshape(-1))
    if tensor.shape != (1, 16) or mask.shape != (1,):
        raise ValueError("Backend feature tensors must have shape (1, 16) and mask shape (1,)")
    if not bool(mask[0]):
        if bool((tensor != 0).any()):
            raise ValueError("Unavailable backend feature row must be zero")
        return None
    return DenseFeatureBatch(features=tensor, available_mask=mask)


def _action_batches(
    arrays: dict[str, np.ndarray],
    spec: TrainingDataSpec,
    n_qubits: int,
) -> tuple[ActionCandidateTensorBatch | None, ActionTargets]:
    features = arrays.get("action_candidate_features")
    if features is None:
        empty_long = torch.zeros(0, dtype=torch.long)
        empty_float = torch.zeros(0, dtype=torch.float32)
        empty_bool = torch.zeros(0, dtype=torch.bool)
        return None, ActionTargets(
            rank=empty_long,
            reward=empty_float,
            selected_mask=empty_bool,
            candidate_target_mask=empty_bool,
            privileged_oracle_mask=empty_bool,
            candidate_batch=empty_long,
        )
    names = tuple(str(value) for value in arrays["action_candidate_feature_names"].tolist())
    if names != spec.action_feature_names:
        raise ValueError(f"Action feature names mismatch: {names}")
    matrix = np.asarray(features, dtype=np.float32).copy()
    if spec.normalize_action_features:
        matrix = (matrix - np.asarray(spec.action_feature_mean, dtype=np.float32)) / np.asarray(spec.action_feature_std, dtype=np.float32)
    count = matrix.shape[0]
    edit_ptr = np.asarray(arrays["action_edit_ptr"], dtype=np.int64)
    edit_types = [str(value) for value in arrays["action_edit_types"].tolist()]
    magnitudes = np.asarray(arrays["action_edit_magnitudes"], dtype=np.float32)
    qubit_ptr = np.asarray(arrays["action_edit_qubit_ptr"], dtype=np.int64)
    qubits = np.asarray(arrays["action_edit_qubits"], dtype=np.int64)
    if edit_ptr.shape != (count + 1,) or qubit_ptr.shape != (len(edit_types) + 1,):
        raise ValueError("Action edit pointer shapes are inconsistent")
    type_ids: list[int] = []
    expanded_magnitudes: list[float] = []
    normalized_qubits: list[float] = []
    edit_candidate: list[int] = []
    type_position = {name: index for index, name in enumerate(ACTION_EDIT_TYPES)}
    denominator = max(n_qubits - 1, 1)
    for candidate in range(count):
        for edit_index in range(int(edit_ptr[candidate]), int(edit_ptr[candidate + 1])):
            raw_type = edit_types[edit_index]
            mapped = ACTION_EDIT_TYPE_MAP.get(raw_type)
            if mapped is None or mapped not in type_position:
                raise ValueError(f"Unsupported Phase 12 action edit type {raw_type!r}")
            operands = qubits[int(qubit_ptr[edit_index]) : int(qubit_ptr[edit_index + 1])]
            if operands.size == 0:
                raise ValueError("Non-no-op action edit must reference at least one qubit")
            for qubit in operands.tolist():
                if qubit < 0 or qubit >= n_qubits:
                    raise ValueError(f"Action edit qubit {qubit} is out of range")
                type_ids.append(type_position[mapped])
                expanded_magnitudes.append(float(magnitudes[edit_index]))
                normalized_qubits.append(float(qubit) / denominator)
                edit_candidate.append(candidate)
    candidate_batch = torch.zeros(count, dtype=torch.long)
    candidate_mask = torch.ones(count, dtype=torch.bool)
    model = ActionCandidateTensorBatch(
        candidate_features=torch.from_numpy(matrix),
        candidate_batch=candidate_batch,
        candidate_available_mask=candidate_mask,
        edit_type_ids=torch.tensor(type_ids, dtype=torch.long),
        edit_magnitudes=torch.tensor(expanded_magnitudes, dtype=torch.float32),
        edit_qubit_positions=torch.tensor(normalized_qubits, dtype=torch.float32),
        edit_candidate_index=torch.tensor(edit_candidate, dtype=torch.long),
    )
    target = ActionTargets(
        rank=_long_tensor(arrays["action_target_rank"].reshape(-1)),
        reward=_float_tensor(arrays["action_target_reward"].reshape(-1)),
        selected_mask=_bool_tensor(arrays["action_target_selected_mask"].reshape(-1)),
        candidate_target_mask=torch.ones(count, dtype=torch.bool),
        privileged_oracle_mask=_bool_tensor(arrays["action_privileged_oracle_mask"].reshape(-1)),
        candidate_batch=candidate_batch,
    )
    return model, target


def _empty_diagnosis(n_nodes: int) -> DiagnosisTargets:
    return DiagnosisTargets(
        class_index=torch.zeros(1, dtype=torch.long),
        class_mask=torch.zeros(1, dtype=torch.bool),
        strength=torch.zeros(1),
        strength_mask=torch.zeros(1, dtype=torch.bool),
        affected_qubit=torch.zeros(n_nodes),
        affected_qubit_mask=torch.zeros(n_nodes, dtype=torch.bool),
    )


def _diagnosis_targets(arrays: dict[str, np.ndarray], n_nodes: int) -> DiagnosisTargets:
    raw = arrays.get("diagnosis_distortion_type")
    if raw is None:
        return _empty_diagnosis(n_nodes)
    raw_name = str(raw.reshape(-1)[0])
    coarse = DISTORTION_TO_COARSE_LABEL.get(raw_name)
    if coarse is None:
        raise ValueError(f"No versioned coarse-label mapping for distortion {raw_name!r}")
    label_index = DISTORTION_LABELS.index(coarse)
    affected = _bool_tensor(arrays["diagnosis_affected_qubit_mask"].reshape(-1))
    if affected.numel() != n_nodes:
        raise ValueError("Diagnosis affected-qubit target does not match graph nodes")
    return DiagnosisTargets(
        class_index=torch.tensor([label_index], dtype=torch.long),
        class_mask=torch.tensor([True]),
        strength=_float_tensor(arrays["diagnosis_strength"].reshape(-1)),
        strength_mask=_bool_tensor(arrays["diagnosis_strength_available_mask"].reshape(-1)),
        affected_qubit=affected.to(torch.float32),
        affected_qubit_mask=torch.ones(n_nodes, dtype=torch.bool),
    )


def _born_targets(arrays: dict[str, np.ndarray]) -> BornTargets:
    values = arrays.get("born_target_probabilities")
    if values is None:
        return BornTargets(
            probabilities=torch.zeros(0),
            outcome_batch=torch.zeros(0, dtype=torch.long),
            row_mask=torch.zeros(0, dtype=torch.bool),
        )
    probabilities = _float_tensor(values.reshape(-1))
    return BornTargets(
        probabilities=probabilities,
        outcome_batch=torch.zeros(probabilities.numel(), dtype=torch.long),
        row_mask=torch.ones(probabilities.numel(), dtype=torch.bool),
    )


def _hard_head_masks(item: Any, parameter: bool, hilbert: bool, born: bool, backend: bool, topology: bool, hardware: bool) -> tuple[Tensor, Tensor]:
    mask = torch.zeros((1, len(HEAD_ORDER), len(STREAM_ORDER)), dtype=torch.bool)
    active = torch.zeros((1, len(HEAD_ORDER)), dtype=torch.bool)
    stream_available = {
        "circuit_graph": True,
        "parameter": parameter,
        "phasor": parameter,
        "hilbert": hilbert,
        "born": born,
        "backend": backend,
        "topology": topology,
    }
    stream_position = {name: index for index, name in enumerate(STREAM_ORDER)}
    head_position = {name: index for index, name in enumerate(HEAD_ORDER)}

    if item.task in {"joint_multitask", "hardware_masked"}:
        prefix = "joint" if item.task == "joint_multitask" else "hardware"
        names = [str(value) for value in item.arrays[f"{prefix}_head_names"].tolist()]
        group_names = [str(value) for value in item.arrays[f"{prefix}_head_input_group_names"].tolist()]
        phase12_mask = np.asarray(item.arrays[f"{prefix}_head_input_mask"], dtype=np.bool_)
        target_mask = np.asarray(item.arrays[f"{prefix}_head_target_available_mask"], dtype=np.bool_)
        if phase12_mask.shape != (len(names), len(group_names)) or target_mask.shape != (len(names),):
            raise ValueError("Phase 12 per-head mask arrays have inconsistent shapes")
        for row, phase12_name in enumerate(names):
            model_name = PHASE12_TO_MODEL_HEAD[phase12_name]
            if phase12_name == "hilbert_to_born":
                # Executed in a separate auxiliary forward pass.
                continue
            model_head = head_position[model_name]
            if bool(target_mask[row]):
                active[0, model_head] = True
            for column, group in enumerate(group_names):
                stream = INPUT_GROUP_TO_STREAM.get(group)
                if stream is not None and bool(phase12_mask[row, column]) and stream_available[stream]:
                    mask[0, model_head, stream_position[stream]] = True
        # Uncertainty is meaningful whenever at least one supervised primary head is active.
        if bool(active[0, :4].any()):
            active[0, head_position["uncertainty"]] = True
            for stream, available in stream_available.items():
                if available:
                    mask[0, head_position["uncertainty"], stream_position[stream]] = True
    else:
        model_name = PHASE12_TO_MODEL_HEAD[item.task]
        if item.task == "hilbert_to_born":
            active[0, head_position["born_prediction"]] = True
            mask[0, head_position["born_prediction"], stream_position["hilbert"]] = True
        elif item.task != "topology_audit":
            head = head_position[model_name]
            active[0, head] = True
            input_names = [str(value) for value in item.arrays["input_group_names"].tolist()]
            input_available = np.asarray(item.arrays["input_group_available_mask"], dtype=np.bool_)
            for group, available in zip(input_names, input_available.tolist(), strict=True):
                stream = INPUT_GROUP_TO_STREAM.get(group)
                if stream is not None and available and stream_available[stream]:
                    mask[0, head, stream_position[stream]] = True
            active[0, head_position["uncertainty"]] = True
            mask[0, head_position["uncertainty"]] = mask[0, head]
    if hardware and bool(mask[0, :, stream_position["hilbert"]].any()):
        raise ValueError("Hardware-masked items cannot enable the Hilbert stream")
    return mask, active


def _load_item(dataset: CompletedTrainingViewDataset, record: Any) -> Any:
    return load_training_view_item_artifact(
        resolve_safe_file(dataset.root, record.artifact_ref, f"Phase 12 item {record.view_item_id}"),
        dataset.config,
        expected_content_hash=record.content_hash,
    )


def build_training_data_spec(
    dataset: CompletedTrainingViewDataset,
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
) -> TrainingDataSpec:
    action_rows: list[np.ndarray] = []
    topology_rows: list[dict[str, float]] = []
    topology_names: set[str] = set()
    backend_rows: list[np.ndarray] = []
    backend_feature_names: tuple[str, ...] = ()
    seen_backend_entities: set[tuple[str, str]] = set()
    for record in sorted(dataset.item_records, key=lambda row: row.view_item_id):
        if record.split != "train" or record.task == "topology_audit":
            continue
        item = _load_item(dataset, record)
        if "action_candidate_features" in item.arrays:
            names = tuple(str(value) for value in item.arrays["action_candidate_feature_names"].tolist())
            if names != _ACTION_FEATURE_NAMES:
                raise ValueError(f"Unexpected Phase 12 action feature schema: {names}")
            action_rows.append(np.asarray(item.arrays["action_candidate_features"], dtype=np.float64))
        if "backend_features" in item.arrays and bool(np.asarray(item.arrays["backend_available_mask"], dtype=np.bool_).reshape(-1)[0]):
            names = tuple(str(value) for value in item.arrays["backend_feature_names"].tolist())
            if backend_feature_names and names != backend_feature_names:
                raise ValueError("Backend feature schema changed across training items")
            backend_feature_names = names
            backend_id_array = item.arrays.get("backend_id")
            backend_id = str(backend_id_array.reshape(-1)[0]) if backend_id_array is not None and backend_id_array.size else "unknown_backend"
            backend_entity = (record.split_group_id, backend_id)
            if backend_entity not in seen_backend_entities:
                seen_backend_entities.add(backend_entity)
                backend_rows.append(np.asarray(item.arrays["backend_features"], dtype=np.float64))
        pairs = _topology_pairs(item.arrays)
        if pairs:
            topology_rows.append(pairs)
            topology_names.update(pairs)
    if action_rows:
        stacked = np.concatenate(action_rows, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std < 1e-12] = 1.0
    else:
        mean = np.zeros(len(_ACTION_FEATURE_NAMES), dtype=np.float64)
        std = np.ones(len(_ACTION_FEATURE_NAMES), dtype=np.float64)
    ordered_topology = tuple(sorted(topology_names))
    if len(ordered_topology) > model_config.topology_input_dim:
        raise RuntimeError(
            f"Training topology vocabulary has {len(ordered_topology)} features, exceeding "
            f"model topology_input_dim={model_config.topology_input_dim}"
        )
    if ordered_topology:
        matrix = np.zeros((len(topology_rows), len(ordered_topology)), dtype=np.float64)
        positions = {name: index for index, name in enumerate(ordered_topology)}
        for row, pairs in enumerate(topology_rows):
            for name, value in pairs.items():
                matrix[row, positions[name]] = value
        topology_mean = matrix.mean(axis=0)
        topology_std = matrix.std(axis=0)
        topology_std[topology_std < 1e-12] = 1.0
    else:
        topology_mean = np.zeros(0, dtype=np.float64)
        topology_std = np.ones(0, dtype=np.float64)
    if backend_rows:
        backend_matrix = np.concatenate(backend_rows, axis=0)
        backend_mean = backend_matrix.mean(axis=0)
        backend_std = backend_matrix.std(axis=0)
        backend_std[backend_std < 1e-12] = 1.0
    else:
        backend_mean = np.zeros(16, dtype=np.float64)
        backend_std = np.ones(16, dtype=np.float64)
        backend_feature_names = tuple(f"backend_feature_{index}" for index in range(16))
    spec = TrainingDataSpec(
        training_view_dataset_id=dataset.training_view_dataset_id,
        distortion_labels=DISTORTION_LABELS,
        distortion_mapping=tuple(sorted(DISTORTION_TO_COARSE_LABEL.items())),
        action_edit_types=ACTION_EDIT_TYPES,
        action_edit_mapping=tuple(sorted(ACTION_EDIT_TYPE_MAP.items())),
        action_feature_names=_ACTION_FEATURE_NAMES,
        action_feature_mean=tuple(float(value) for value in mean),
        action_feature_std=tuple(float(value) for value in std),
        topology_feature_names=ordered_topology,
        topology_feature_mean=tuple(float(value) for value in topology_mean),
        topology_feature_std=tuple(float(value) for value in topology_std),
        backend_feature_names=backend_feature_names,
        backend_feature_mean=tuple(float(value) for value in backend_mean),
        backend_feature_std=tuple(float(value) for value in backend_std),
        topology_input_dim=model_config.topology_input_dim,
        normalize_action_features=training_config.normalize_action_features,
        normalize_topology_features=training_config.normalize_topology_features,
        normalize_backend_features=training_config.normalize_backend_features,
        adapter_version=TRAINING_ADAPTER_VERSION,
    )
    spec.validate()
    return spec


def load_training_examples(
    dataset: CompletedTrainingViewDataset,
    *,
    tasks: Sequence[str],
    split: str,
    spec: TrainingDataSpec,
    phase7_root: str | Path | None = None,
    allow_evaluation_splits: bool = False,
) -> list[TrainingExample]:
    allowed_splits = {"train", "validation"} | ({"test", "iid_test"} if allow_evaluation_splits else set())
    if split not in allowed_splits:
        raise ValueError("Phase 14 optimization loader supports only train/validation unless Phase 15 explicitly enables evaluation splits")
    phase7 = Path(phase7_root) if phase7_root is not None else None
    records = [
        record
        for task in tasks
        for record in dataset.records_by_task_split.get((task, split), ())
    ]
    records.sort(key=lambda row: (row.task, row.view_item_id))
    examples: list[TrainingExample] = []
    for record in records:
        if record.task == "topology_audit":
            raise ValueError("topology_audit records are audit-only and cannot enter gradients")
        item = _load_item(dataset, record)
        graph_item = item
        if item.task == "hilbert_to_born":
            anchor = dataset.graph_anchor_record_by_entity_id.get(item.entity_id)
            if anchor is None:
                raise ValueError(f"Hilbert-to-Born item {item.view_item_id} has no graph anchor")
            if anchor.split != item.split or anchor.split_group_id != item.split_group_id:
                raise ValueError("Hilbert-to-Born graph anchor crosses split boundaries")
            graph_item = _load_item(dataset, anchor)
        arrays = dict(graph_item.arrays)
        arrays.update({name: value for name, value in item.arrays.items() if not name.startswith("graph_")})
        graph = _graph_batch_from_arrays(arrays)
        n_qubits = graph.node_features.shape[0]
        parameter = _parameter_batch(arrays)
        born_input = _born_batch(arrays, "born_input")
        queries = _outcome_queries(arrays)
        hilbert, hilbert_state = _hilbert_batch(item, phase7, n_qubits)
        topology = _topology_batch(arrays, spec)
        backend = _backend_batch(arrays, spec)
        actions, action_targets = _action_batches(arrays, spec, n_qubits)
        diagnosis = _diagnosis_targets(arrays, n_qubits)
        born_targets = _born_targets(arrays)
        hardware = item.task == "hardware_masked"
        head_mask, head_active = _hard_head_masks(
            item,
            parameter is not None,
            hilbert is not None,
            born_input is not None,
            backend is not None,
            topology is not None,
            hardware,
        )
        batch = TriQTOBatch(
            graph=graph,
            parameter=parameter,
            born=born_input,
            hilbert=hilbert,
            backend=backend,
            topology=topology,
            actions=actions,
            born_queries=queries,
            hardware_mode_mask=torch.tensor([hardware]),
            topology_hilbert_dependent_mask=torch.tensor([False]),
            head_stream_mask=head_mask,
            head_active_mask=head_active,
        )
        target = TrainingTargets(
            diagnosis=diagnosis,
            action=action_targets,
            born_prediction=born_targets if item.task != "hilbert_to_born" else _born_targets({}),
            hilbert_to_born=born_targets if (item.task == "hilbert_to_born" or item.hilbert_available_mask) else _born_targets({}),
            geometry=GeometryTargets(
                target_distance=torch.zeros((1, 1)),
                pair_mask=torch.zeros((1, 1), dtype=torch.bool),
            ),
        )
        distribution = tuple(
            zip(
                [str(value) for value in arrays.get("born_target_outcome_bitstrings", np.asarray([], dtype="<U1")).tolist()],
                [float(value) for value in arrays.get("born_target_probabilities", np.asarray([], dtype=np.float64)).tolist()],
                strict=True,
            )
        )
        examples.append(
            TrainingExample(
                view_item_id=item.view_item_id,
                entity_id=item.entity_id,
                task=item.task,
                split=item.split,
                split_group_id=item.split_group_id,
                model_batch=batch,
                targets=target,
                n_qubits=n_qubits,
                born_distribution=distribution,
                hilbert_state=hilbert_state,
                privileged_target_available=item.privileged_target_available_mask,
                metadata=dict(item.metadata),
            )
        )
    return examples


def _cat_optional(parts: list[Any], attr: str, dim: int = 0) -> Tensor:
    tensors = [getattr(part, attr) for part in parts]
    return torch.cat(tensors, dim=dim) if tensors else torch.zeros(0)


def collate_training_examples(examples: Sequence[TrainingExample]) -> SupervisedBatch:
    if not examples:
        raise ValueError("Cannot collate an empty example sequence")
    graphs = [example.model_batch.graph for example in examples]
    node_offsets: list[int] = []
    gate_offsets: list[int] = []
    node_total = gate_total = 0
    for graph in graphs:
        node_offsets.append(node_total)
        gate_offsets.append(gate_total)
        node_total += graph.node_features.shape[0]
        gate_total += graph.gate_features.shape[0]
    edge_indices = [graph.edge_index + node_offsets[index] for index, graph in enumerate(graphs)]
    edge_events = [graph.edge_event_index + gate_offsets[index] for index, graph in enumerate(graphs)]
    gate_ptr_values = [0]
    gate_indices: list[Tensor] = []
    incidence_total = 0
    for index, graph in enumerate(graphs):
        local_counts = graph.gate_qubit_ptr[1:] - graph.gate_qubit_ptr[:-1]
        for count in local_counts.tolist():
            incidence_total += int(count)
            gate_ptr_values.append(incidence_total)
        gate_indices.append(graph.gate_qubit_indices + node_offsets[index])
    graph = GraphTensorBatch(
        node_features=torch.cat([value.node_features for value in graphs], dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat([value.edge_features for value in graphs], dim=0),
        edge_event_index=torch.cat(edge_events, dim=0),
        gate_features=torch.cat([value.gate_features for value in graphs], dim=0),
        gate_qubit_ptr=torch.tensor(gate_ptr_values, dtype=torch.long),
        gate_qubit_indices=torch.cat(gate_indices, dim=0),
        node_batch=torch.cat([
            torch.full((value.node_features.shape[0],), index, dtype=torch.long)
            for index, value in enumerate(graphs)
        ]),
        gate_batch=torch.cat([
            torch.full((value.gate_features.shape[0],), index, dtype=torch.long)
            for index, value in enumerate(graphs)
        ]),
        graph_count=len(examples),
    )

    def collate_parameter() -> ParameterTensorBatch | None:
        rows = [(index, example.model_batch.parameter) for index, example in enumerate(examples) if example.model_batch.parameter is not None]
        if not rows:
            return None
        available = torch.zeros(len(examples), dtype=torch.bool)
        for index, _ in rows:
            available[index] = True
        return ParameterTensorBatch(
            values=torch.cat([value.values for _, value in rows]),
            sin=torch.cat([value.sin for _, value in rows]),
            cos=torch.cat([value.cos for _, value in rows]),
            batch_index=torch.cat([torch.full((value.values.numel(),), index, dtype=torch.long) for index, value in rows]),
            available_mask=available,
        )

    def collate_born(attribute: str) -> BornTensorBatch | None:
        rows = [(index, getattr(example.model_batch, attribute)) for index, example in enumerate(examples) if getattr(example.model_batch, attribute) is not None]
        if not rows:
            return None
        width = max(value.outcome_bits.shape[1] for _, value in rows)
        bit_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        available = torch.zeros(len(examples), dtype=torch.bool)
        for index, value in rows:
            padding = width - value.outcome_bits.shape[1]
            bit_parts.append(torch.nn.functional.pad(value.outcome_bits, (0, padding)))
            mask_parts.append(torch.nn.functional.pad(value.outcome_bit_mask, (0, padding)))
            available[index] = True
        return BornTensorBatch(
            outcome_bits=torch.cat(bit_parts),
            outcome_bit_mask=torch.cat(mask_parts),
            probabilities=torch.cat([value.probabilities for _, value in rows]),
            batch_index=torch.cat([torch.full((value.probabilities.numel(),), index, dtype=torch.long) for index, value in rows]),
            available_mask=available,
        )

    def collate_hilbert() -> HilbertTensorBatch | None:
        rows = [(index, example.model_batch.hilbert) for index, example in enumerate(examples) if example.model_batch.hilbert is not None]
        if not rows:
            return None
        width = max(value.basis_bits.shape[1] for _, value in rows)
        available = torch.zeros(len(examples), dtype=torch.bool)
        bits: list[Tensor] = []
        masks: list[Tensor] = []
        for index, value in rows:
            padding = width - value.basis_bits.shape[1]
            bits.append(torch.nn.functional.pad(value.basis_bits, (0, padding)))
            masks.append(torch.nn.functional.pad(value.basis_bit_mask, (0, padding)))
            available[index] = True
        return HilbertTensorBatch(
            amplitudes_real_imag=torch.cat([value.amplitudes_real_imag for _, value in rows]),
            basis_bits=torch.cat(bits),
            basis_bit_mask=torch.cat(masks),
            batch_index=torch.cat([torch.full((value.amplitudes_real_imag.shape[0],), index, dtype=torch.long) for index, value in rows]),
            available_mask=available,
        )

    def collate_dense(attribute: str) -> DenseFeatureBatch | None:
        present = [getattr(example.model_batch, attribute) for example in examples]
        if not any(value is not None for value in present):
            return None
        first = next(value for value in present if value is not None)
        width = first.features.shape[1]
        features = torch.zeros((len(examples), width), dtype=torch.float32)
        available = torch.zeros(len(examples), dtype=torch.bool)
        for index, value in enumerate(present):
            if value is not None:
                features[index] = value.features[0]
                available[index] = True
        return DenseFeatureBatch(features=features, available_mask=available)

    def collate_actions() -> ActionCandidateTensorBatch | None:
        rows = [(index, example.model_batch.actions) for index, example in enumerate(examples) if example.model_batch.actions is not None]
        if not rows:
            return None
        candidate_offsets: list[int] = []
        total = 0
        for _, value in rows:
            candidate_offsets.append(total)
            total += value.candidate_features.shape[0]
        return ActionCandidateTensorBatch(
            candidate_features=torch.cat([value.candidate_features for _, value in rows]),
            candidate_batch=torch.cat([torch.full((value.candidate_features.shape[0],), index, dtype=torch.long) for index, value in rows]),
            candidate_available_mask=torch.cat([value.candidate_available_mask for _, value in rows]),
            edit_type_ids=torch.cat([value.edit_type_ids for _, value in rows]),
            edit_magnitudes=torch.cat([value.edit_magnitudes for _, value in rows]),
            edit_qubit_positions=torch.cat([value.edit_qubit_positions for _, value in rows]),
            edit_candidate_index=torch.cat([
                value.edit_candidate_index + candidate_offsets[position]
                for position, (_, value) in enumerate(rows)
            ]),
        )

    def collate_queries() -> OutcomeQueryTensorBatch | None:
        rows = [(index, example.model_batch.born_queries) for index, example in enumerate(examples) if example.model_batch.born_queries is not None]
        if not rows:
            return None
        width = max(value.outcome_bits.shape[1] for _, value in rows)
        available = torch.zeros(len(examples), dtype=torch.bool)
        bits: list[Tensor] = []
        masks: list[Tensor] = []
        for index, value in rows:
            padding = width - value.outcome_bits.shape[1]
            bits.append(torch.nn.functional.pad(value.outcome_bits, (0, padding)))
            masks.append(torch.nn.functional.pad(value.outcome_bit_mask, (0, padding)))
            available[index] = True
        return OutcomeQueryTensorBatch(
            outcome_bits=torch.cat(bits),
            outcome_bit_mask=torch.cat(masks),
            batch_index=torch.cat([torch.full((value.outcome_bits.shape[0],), index, dtype=torch.long) for index, value in rows]),
            available_mask=available,
        )

    model_batch = TriQTOBatch(
        graph=graph,
        parameter=collate_parameter(),
        born=collate_born("born"),
        hilbert=collate_hilbert(),
        backend=collate_dense("backend"),
        topology=collate_dense("topology"),
        actions=collate_actions(),
        born_queries=collate_queries(),
        hardware_mode_mask=torch.cat([example.model_batch.resolved_hardware_mask() for example in examples]),
        topology_hilbert_dependent_mask=torch.cat([example.model_batch.resolved_topology_hilbert_dependency() for example in examples]),
        head_stream_mask=torch.cat([example.model_batch.head_stream_mask for example in examples]),
        head_active_mask=torch.cat([example.model_batch.head_active_mask for example in examples]),
    )

    diagnosis = DiagnosisTargets(
        class_index=torch.cat([example.targets.diagnosis.class_index for example in examples]),
        class_mask=torch.cat([example.targets.diagnosis.class_mask for example in examples]),
        strength=torch.cat([example.targets.diagnosis.strength for example in examples]),
        strength_mask=torch.cat([example.targets.diagnosis.strength_mask for example in examples]),
        affected_qubit=torch.cat([example.targets.diagnosis.affected_qubit for example in examples]),
        affected_qubit_mask=torch.cat([example.targets.diagnosis.affected_qubit_mask for example in examples]),
    )
    candidate_offsets: list[int] = []
    candidate_total = 0
    for example in examples:
        candidate_offsets.append(candidate_total)
        candidate_total += example.targets.action.rank.numel()
    action = ActionTargets(
        rank=torch.cat([example.targets.action.rank for example in examples]),
        reward=torch.cat([example.targets.action.reward for example in examples]),
        selected_mask=torch.cat([example.targets.action.selected_mask for example in examples]),
        candidate_target_mask=torch.cat([example.targets.action.candidate_target_mask for example in examples]),
        privileged_oracle_mask=torch.cat([example.targets.action.privileged_oracle_mask for example in examples]),
        candidate_batch=torch.cat([
            torch.full((example.targets.action.rank.numel(),), index, dtype=torch.long)
            for index, example in enumerate(examples)
        ]),
    )

    def collate_born_target(attribute: str) -> BornTargets:
        probability_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        batch_parts: list[Tensor] = []
        for index, example in enumerate(examples):
            queries = example.model_batch.born_queries
            query_count = 0 if queries is None else queries.outcome_bits.shape[0]
            target = getattr(example.targets, attribute)
            if target.probabilities.numel() not in {0, query_count}:
                raise ValueError(
                    f"{attribute} target/query row mismatch for {example.view_item_id}"
                )
            if target.probabilities.numel() == 0:
                probability_parts.append(torch.zeros(query_count, dtype=torch.float32))
                mask_parts.append(torch.zeros(query_count, dtype=torch.bool))
            else:
                probability_parts.append(target.probabilities)
                mask_parts.append(target.row_mask)
            batch_parts.append(torch.full((query_count,), index, dtype=torch.long))
        return BornTargets(
            probabilities=torch.cat(probability_parts),
            outcome_batch=torch.cat(batch_parts),
            row_mask=torch.cat(mask_parts),
        )

    born_target = collate_born_target("born_prediction")
    hilbert_target = collate_born_target("hilbert_to_born")
    target_distance, pair_mask = _geometry_targets(examples)
    targets = TrainingTargets(
        diagnosis=diagnosis,
        action=action,
        born_prediction=born_target,
        hilbert_to_born=hilbert_target,
        geometry=GeometryTargets(target_distance=target_distance, pair_mask=pair_mask),
    )

    auxiliary = None
    hilbert_active = torch.tensor([example.targets.hilbert_to_born.probabilities.numel() > 0 for example in examples])
    if bool(hilbert_active.any()):
        auxiliary = _make_hilbert_auxiliary_batch(model_batch, hilbert_active)
    return SupervisedBatch(
        item_ids=tuple(example.view_item_id for example in examples),
        entity_ids=tuple(example.entity_id for example in examples),
        tasks=tuple(example.task for example in examples),
        splits=tuple(example.split for example in examples),
        split_group_ids=tuple(example.split_group_id for example in examples),
        model_batch=model_batch,
        auxiliary_hilbert_to_born_batch=auxiliary,
        targets=targets,
        graph_task_names=tuple(example.task for example in examples),
        privileged_item_mask=torch.tensor([example.privileged_target_available for example in examples]),
    )


def _make_hilbert_auxiliary_batch(batch: TriQTOBatch, active_rows: Tensor) -> TriQTOBatch:
    head_active = torch.zeros_like(batch.resolved_head_active_mask())
    head_stream = torch.zeros((batch.graph.graph_count, len(HEAD_ORDER), len(STREAM_ORDER)), dtype=torch.bool)
    head_index = HEAD_ORDER.index("born_prediction")
    stream_index = STREAM_ORDER.index("hilbert")
    head_active[:, head_index] = active_rows
    head_stream[:, head_index, stream_index] = active_rows
    return TriQTOBatch(
        graph=batch.graph,
        parameter=batch.parameter,
        born=None,
        hilbert=batch.hilbert,
        backend=None,
        topology=None,
        actions=None,
        born_queries=batch.born_queries,
        hardware_mode_mask=torch.zeros_like(batch.resolved_hardware_mask()),
        topology_hilbert_dependent_mask=torch.zeros_like(batch.resolved_topology_hilbert_dependency()),
        head_stream_mask=head_stream,
        head_active_mask=head_active,
    )


def _geometry_targets(examples: Sequence[TrainingExample]) -> tuple[Tensor, Tensor]:
    count = len(examples)
    distances = torch.zeros((count, count), dtype=torch.float32)
    mask = torch.zeros((count, count), dtype=torch.bool)
    for left in range(count):
        for right in range(left + 1, count):
            if examples[left].n_qubits != examples[right].n_qubits:
                continue
            born_left = dict(examples[left].born_distribution)
            born_right = dict(examples[right].born_distribution)
            if not born_left or not born_right:
                continue
            support = sorted(set(born_left) | set(born_right))
            p = np.asarray([born_left.get(key, 0.0) for key in support], dtype=np.float64)
            q = np.asarray([born_right.get(key, 0.0) for key in support], dtype=np.float64)
            hellinger = float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))
            components = [hellinger]
            if examples[left].hilbert_state is not None and examples[right].hilbert_state is not None:
                a = torch.view_as_complex(examples[left].hilbert_state)
                b = torch.view_as_complex(examples[right].hilbert_state)
                overlap = torch.abs(torch.vdot(a, b)).clamp(0.0, 1.0)
                components.append(float(torch.acos(overlap) / (math.pi / 2.0)))
            value = float(sum(components) / len(components))
            distances[left, right] = distances[right, left] = value
            mask[left, right] = mask[right, left] = True
    return distances, mask


def deterministic_budget_batches(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    *,
    epoch_seed: int,
    shuffle: bool,
) -> list[list[TrainingExample]]:
    """Pack all examples exactly once; ceilings fail instead of truncating."""
    ordered = list(examples)
    if shuffle:
        generator = torch.Generator().manual_seed(epoch_seed)
        permutation = torch.randperm(len(ordered), generator=generator).tolist()
        ordered = [ordered[index] for index in permutation]
    batches: list[list[TrainingExample]] = []
    current: list[TrainingExample] = []
    totals = {"nodes": 0, "edges": 0, "gates": 0, "candidates": 0, "outcomes": 0, "hilbert": 0}

    def size(example: TrainingExample) -> dict[str, int]:
        batch = example.model_batch
        return {
            "nodes": batch.graph.node_features.shape[0],
            "edges": batch.graph.edge_features.shape[0],
            "gates": batch.graph.gate_features.shape[0],
            "candidates": 0 if batch.actions is None else batch.actions.candidate_features.shape[0],
            "outcomes": 0 if batch.born_queries is None else batch.born_queries.outcome_bits.shape[0],
            "hilbert": 0 if batch.hilbert is None else batch.hilbert.amplitudes_real_imag.shape[0],
        }

    limits = {
        "nodes": config.max_nodes_per_batch,
        "edges": config.max_edges_per_batch,
        "gates": config.max_gates_per_batch,
        "candidates": config.max_candidates_per_batch,
        "outcomes": config.max_outcomes_per_batch,
        "hilbert": config.max_hilbert_amplitudes_per_batch,
    }
    for example in ordered:
        item_size = size(example)
        exceeded_alone = [name for name, value in item_size.items() if value > limits[name]]
        if exceeded_alone:
            raise RuntimeError(
                f"Training item {example.view_item_id} exceeds batch guardrails: {exceeded_alone}"
            )
        would_exceed = len(current) >= config.batch_size or any(
            totals[name] + item_size[name] > limits[name] for name in totals
        )
        if current and would_exceed:
            batches.append(current)
            current = []
            totals = {name: 0 for name in totals}
        current.append(example)
        for name in totals:
            totals[name] += item_size[name]
    if current:
        batches.append(current)
    if sum(len(batch) for batch in batches) != len(examples):
        raise RuntimeError("Deterministic batching lost or duplicated examples")
    return batches


__all__ = [
    "build_training_data_spec",
    "collate_training_examples",
    "deterministic_budget_batches",
    "load_training_examples",
]
