"""CPU regression: Step-14 masked MIL inference cannot depend materially
on neighboring examples, inferred microbatch boundaries, or padding width.

This is a controlled synthetic contract test. It does not certify the historical
Phase-A sample, whose blind data and model checkpoints live on the volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

V02 = Path(__file__).resolve().parents[1] / "scripts" / "v0_2"
if str(V02) not in sys.path:
    sys.path.insert(0, str(V02))

import analyze_step14_latent_frame_inference as latent
import step14_equivalence_aware_common as equiv


def test_step14_mil_batch_256_and_padding_width_are_nearly_invariant():
    rng = np.random.default_rng(20260913)
    torch.manual_seed(6201)
    model = latent.LatentFrameMIL(24, 64, 0.1)
    model.eval()
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    narrow = rng.standard_normal((12, 39, 24)).astype(np.float32)
    mask = np.zeros((12, 39), dtype=np.bool_)
    lengths = [2, 5, 9, 13, 17, 21, 25, 28, 32, 35, 37, 39]
    for i, n in enumerate(lengths):
        mask[i, :n] = True
        narrow[i, n:] = 0

    expanded = np.zeros((256, 64, 24), dtype=np.float32)
    expanded_mask = np.zeros((256, 64), dtype=np.bool_)
    expanded[:12, :39] = narrow
    expanded_mask[:12, :39] = mask
    # Fill every neighboring row with a valid bag; do not compare them.
    expanded[12:, :5] = narrow[1, :5]
    expanded_mask[12:, :5] = True

    cfg = {"latent_mil_probe": {"shared_hidden_dim": 64, "dropout": 0.1, "batch_size": 256}}
    mean = np.zeros(24, dtype=np.float32)
    std = np.ones(24, dtype=np.float32)
    _, small_out = equiv.infer_candidate_network(
        narrow, mask, state_dict=weights, mean=mean, std=std,
        latent_cfg=cfg, device=torch.device("cpu"),
    )
    _, expanded_out = equiv.infer_candidate_network(
        expanded, expanded_mask, state_dict=weights, mean=mean, std=std,
        latent_cfg=cfg, device=torch.device("cpu"),
    )
    for i, n in enumerate(lengths):
        np.testing.assert_allclose(
            small_out[i, :n], expanded_out[i, :n], atol=1e-5, rtol=0
        )

    # The exact frozen pooling function sees only left-packed valid candidates.
    profiles = rng.normal(size=(12, 39, 3)).astype(np.float64)
    distance = np.ones((12, 39, 39), dtype=np.float64)
    gates = np.zeros((12, 39, 39), dtype=np.bool_)
    for i in range(12):
        np.fill_diagonal(distance[i], 0)
        np.fill_diagonal(gates[i], True)
    padded_profiles = np.full((256, 64, 3), -np.inf, dtype=np.float64)
    padded_distance = np.ones((256, 64, 64), dtype=np.float64)
    padded_gate = np.zeros((256, 64, 64), dtype=np.bool_)
    padded_profiles[:12, :39] = profiles
    padded_distance[:12, :39, :39] = distance
    padded_gate[:12, :39, :39] = gates
    for i in range(12, 256):
        padded_profiles[i, :5] = profiles[1, :5]
        np.fill_diagonal(padded_distance[i, :5, :5], 0)
        np.fill_diagonal(padded_gate[i, :5, :5], True)

    a = equiv.equivalence_aware_pool_batch(
        small_out, profiles, distance, gates, mask, tau=0.02, frame_temperature=2.0
    )
    b = equiv.equivalence_aware_pool_batch(
        expanded_out, padded_profiles, padded_distance, padded_gate,
        expanded_mask, tau=0.02, frame_temperature=2.0
    )
    np.testing.assert_allclose(a, b[:12], atol=1e-5, rtol=0)
    np.testing.assert_array_equal(a.argmax(axis=1), b[:12].argmax(axis=1))
