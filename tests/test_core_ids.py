"""Tests for deterministic TriQTO ID helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from triqto.core.ids import canonical_json, make_circuit_id, make_deterministic_id, make_run_id


@dataclass
class Payload:
    family: str
    n_qubits: int


def test_canonical_json_sorts_mapping_keys() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_deterministic_ids_are_stable_for_equivalent_payloads() -> None:
    left = make_circuit_id({"family": "ghz", "n_qubits": 4})
    right = make_circuit_id({"n_qubits": 4, "family": "ghz"})
    assert left == right
    assert left.startswith("circuit_")


def test_deterministic_ids_support_dataclasses() -> None:
    assert make_run_id(Payload("ghz", 4)) == make_run_id({"family": "ghz", "n_qubits": 4})


def test_invalid_prefix_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_deterministic_id("bad-prefix", {"x": 1})
