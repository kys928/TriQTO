#!/usr/bin/env python
"""Safe CLI for tiny Phase 7 dataset generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    load_generation_config,
    write_dataset,
)


def demo_config() -> DatasetGenerationConfig:
    """Return a deliberately tiny safe demo configuration."""
    return DatasetGenerationConfig(
        dataset_name="triqto_phase7_demo",
        base_seed=7,
        circuit_specs=[
            CircuitGenerationSpec(
                family="hardware_efficient_ansatz",
                n_qubits=2,
                generator_kwargs={"layers": 1, "entanglement": "none", "measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(name="rx_overrotation", kwargs={"strength": 0.25, "qubits": [0]}),
            DistortionSpec(name="readout_bitflip_marker", kwargs={"probability": 0.1, "qubits": [0]}),
        ],
        ideal_shots=16,
        max_samples=4,
    )


def main() -> None:
    """Parse CLI arguments, generate a tiny dataset, and print a concise summary."""
    parser = argparse.ArgumentParser(description="Generate a deterministic TriQTO Phase 7 raw dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to a strict JSON generation config.")
    source.add_argument("--demo", action="store_true", help="Use a tiny built-in demo config.")
    parser.add_argument("--output", required=True, help="Output directory; never defaults inside the repository.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace known dataset files.")
    args = parser.parse_args()

    config = demo_config() if args.demo else load_generation_config(args.config)
    result = generate_dataset(config)
    write_result = write_dataset(result, Path(args.output), overwrite=args.overwrite)
    print(
        f"dataset={result.dataset_name} samples={result.summary['sample_count']} "
        f"manifests={len(write_result.manifest_paths)} output={write_result.output_root}"
    )
    print(
        f"scientific_generation_id={result.scientific_generation_id} "
        f"config_id={result.config_id}"
    )
    print(
        f"born_visible={result.summary['born_visible_sample_count']} "
        f"marker_only={result.summary['marker_only_sample_count']}"
    )


if __name__ == "__main__":
    main()
