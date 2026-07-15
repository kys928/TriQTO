from __future__ import annotations

import math

from triqto.storage import ManifestReader, ManifestWriter


def test_dynamic_numeric_maps_round_trip_without_boolean_coercion(tmp_path):
    records = [
        {
            "sample_id": "non_parametric",
            "parameter_bindings": {},
            "metadata": {"parameter_bindings": {}},
        },
        {
            "sample_id": "qaoa",
            "parameter_bindings": {
                "beta_0": 0.25,
                "beta_1": -0.5,
                "gamma_0": 1.25,
                "gamma_1": -2.0,
            },
            "metadata": {
                "parameter_bindings": {
                    "beta_0": 0.25,
                    "beta_1": -0.5,
                    "gamma_0": 1.25,
                    "gamma_1": -2.0,
                }
            },
        },
    ]

    writer = ManifestWriter(tmp_path)
    writer.write_records("mixed_parameter_bindings", records)

    restored = ManifestReader(tmp_path).read_records("mixed_parameter_bindings")

    assert restored == records
    qaoa_bindings = restored[1]["parameter_bindings"]
    nested_bindings = restored[1]["metadata"]["parameter_bindings"]
    for bindings in (qaoa_bindings, nested_bindings):
        assert set(bindings) == {"beta_0", "beta_1", "gamma_0", "gamma_1"}
        assert all(isinstance(value, float) for value in bindings.values())
        assert all(not isinstance(value, bool) for value in bindings.values())
        assert all(math.isfinite(value) for value in bindings.values())
