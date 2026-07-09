#!/usr/bin/env python3
"""Tiny in-memory TriQTO Phase 4 smoke test."""
from __future__ import annotations

from triqto.circuits.ghz import make_ghz_circuit
from triqto.simulation import simulate_ideal_shots, simulate_ideal_statevector


def _top_items(mapping: dict[str, float] | dict[str, int], limit: int = 4) -> dict[str, float] | dict[str, int]:
    return dict(sorted(mapping.items(), key=lambda item: (-item[1], item[0]))[:limit])


if __name__ == "__main__":
    generated = make_ghz_circuit(4, measure=True)
    state_result = simulate_ideal_statevector(generated)
    shot_result = simulate_ideal_shots(generated, shots=1024, seed=1234)

    print("TriQTO Phase 4 smoke test complete.")
    print(f"Circuit family: {generated.family}")
    print(f"Qubits: {generated.n_qubits}")
    print(f"Statevector mode: {state_result.simulation_mode}")
    print(f"Shot mode: {shot_result.simulation_mode}")
    print(f"Top ideal probabilities: {_top_items(state_result.probabilities)}")
    print(f"Sampled counts: {_top_items(shot_result.counts)}")
    print(f"Ideal probability support: {len(state_result.probabilities)}")
    print(f"Shot count total: {sum(shot_result.counts.values())}")
