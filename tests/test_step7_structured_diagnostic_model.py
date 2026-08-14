from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from triqto.graph.constants import EDGE_FEATURE_NAMES, GATE_FEATURE_NAMES, NODE_FEATURE_NAMES
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7.model import ALL_VARIANTS, Step7DiagnosticModel


def example(*, shots: int = 512, basis_codes=(0, 1, 2), effect=True, mechanism=0, mask=True):
    local_canonical = np.asarray(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], dtype=np.float64
    )
    pair_canonical = np.asarray([[0.07], [0.08], [0.09]], dtype=np.float64)
    parity_canonical = np.asarray([0.10, 0.11, 0.12], dtype=np.float64)
    canonical_codes = [0, 1, 2]
    source_order = [canonical_codes.index(int(code)) for code in basis_codes]
    return {
        "x__graph_gate_names": np.asarray(["h", "cx", "rz"]),
        "x__graph_gate_qubit_ptr": np.asarray([0, 1, 3, 4], dtype=np.int32),
        "x__graph_gate_qubit_indices": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "x__graph_gate_parameter_ptr": np.asarray([0, 0, 0, 1], dtype=np.int32),
        "x__graph_gate_parameter_sin": np.asarray([math.sin(0.3)], dtype=np.float64),
        "x__graph_gate_parameter_cos": np.asarray([math.cos(0.3)], dtype=np.float64),
        "x__layout_logical_to_physical": np.asarray([0, 1], dtype=np.int16),
        "x__diagnostic_basis_codes": np.asarray(basis_codes, dtype=np.int8),
        "x__delta_local_expectations": local_canonical[source_order],
        "x__delta_pairwise_correlations": pair_canonical[source_order],
        "x__delta_global_parity": parity_canonical[source_order],
        "x__pair_indices": np.asarray([[0, 1]], dtype=np.int16),
        "x__observed_shots": np.full(3, shots, dtype=np.int32)[source_order],
        "x__reference_shots": np.full(3, shots, dtype=np.int32)[source_order],
        "x__reference_available_mask": np.ones(3, dtype=bool)[source_order],
        "x__reference_kind_code": np.asarray([0], dtype=np.int8),
        "y__effect_present_target": np.asarray([effect], dtype=bool),
        "y__mechanism_target": np.asarray([mechanism], dtype=np.int8),
        "y__mechanism_loss_mask": np.asarray([mask], dtype=bool),
    }


def test_adapter_builds_phase13_graph_and_targets():
    batch, targets = batch_from_step5_examples([example()])
    assert batch.graph.node_features.shape == (2, len(NODE_FEATURE_NAMES))
    assert batch.graph.edge_features.shape[1] == len(EDGE_FEATURE_NAMES)
    assert batch.graph.gate_features.shape == (3, len(GATE_FEATURE_NAMES))
    assert batch.graph.graph_count == 1
    assert batch.diagnostic.local_values.shape == (2, 3)
    assert batch.diagnostic.pair_values.shape == (1, 3)
    assert batch.diagnostic.pair_index.tolist() == [[0], [1]]
    assert targets.effect_present.tolist() == [1.0]
    assert targets.mechanism.tolist() == [0]
    assert targets.mechanism_loss_mask.tolist() == [True]


def test_basis_permutation_is_canonicalized_to_zxy_codes_012():
    batch, _ = batch_from_step5_examples([example(basis_codes=(2, 0, 1))])
    assert batch.diagnostic.basis_codes.tolist() == [[0, 1, 2]]
    assert torch.allclose(
        batch.diagnostic.local_values,
        torch.tensor([[0.01, 0.03, 0.05], [0.02, 0.04, 0.06]], dtype=torch.float32),
    )
    assert torch.allclose(
        batch.diagnostic.pair_values,
        torch.tensor([[0.07, 0.08, 0.09]], dtype=torch.float32),
    )


def test_pair_endpoints_are_offset_across_multiple_examples():
    batch, _ = batch_from_step5_examples([example(), example(mechanism=1)])
    assert batch.diagnostic.pair_index.tolist() == [[0, 2], [1, 3]]
    assert batch.diagnostic.pair_batch.tolist() == [0, 1]
    assert batch.graph.node_batch.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("variant", sorted(ALL_VARIANTS))
def test_all_frozen_variants_forward_with_finite_outputs(variant: str):
    batch, targets = batch_from_step5_examples(
        [example(), example(effect=False, mechanism=-1, mask=False)]
    )
    model = Step7DiagnosticModel(variant=variant, hidden_dim=32, graph_message_passing_layers=1, dropout=0.0)
    output = model(batch)
    assert output.effect_logit.shape == (2,)
    assert output.mechanism_logits.shape == (2, 3)
    assert output.representation.shape == (2, 32)
    assert output.magnitude_features.shape == (2, 8)
    assert torch.isfinite(output.effect_logit).all()
    assert torch.isfinite(output.mechanism_logits).all()
    targets.validate(2, batch.graph.node_features.device)


def test_snr_magnitude_feature_scales_with_sqrt_shots_for_same_diagnostic():
    batch, _ = batch_from_step5_examples([example(shots=512), example(shots=2048)])
    model = Step7DiagnosticModel(variant="structured_interaction", hidden_dim=32, graph_message_passing_layers=1, dropout=0.0)
    features = model.magnitude_features(batch)
    assert torch.allclose(features[0, :4], features[1, :4], atol=1e-7)
    assert torch.allclose(features[1, 6] / features[0, 6], torch.tensor(2.0), atol=1e-6)
    assert torch.allclose(features[1, 5] / features[0, 5], torch.tensor(0.5), atol=1e-6)


def test_masked_mechanism_target_may_be_minus_one_but_supervised_may_not():
    batch, targets = batch_from_step5_examples([example(effect=False, mechanism=-1, mask=False)])
    targets.validate(1, batch.graph.node_features.device)
    bad_batch, bad_targets = batch_from_step5_examples([example(effect=True, mechanism=-1, mask=True)])
    with pytest.raises(ValueError, match="supervised mechanism"):
        bad_targets.validate(1, bad_batch.graph.node_features.device)
