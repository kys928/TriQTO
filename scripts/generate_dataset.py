#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from triqto.data_generation import CircuitGenerationSpec, DatasetGenerationConfig, DistortionSpec, generate_dataset, load_generation_config, write_dataset

def demo_config():
    return DatasetGenerationConfig(dataset_name="triqto_phase7_demo", base_seed=7, circuit_specs=[CircuitGenerationSpec("hardware_efficient_ansatz",2,{"layers":1,"entanglement":"none","measure":True},1)], distortion_specs=[DistortionSpec("rx_overrotation",{"strength":0.25,"qubits":[0]}), DistortionSpec("readout_bitflip_marker",{"probability":0.1,"qubits":[0]})], ideal_shots=16, max_samples=4)

def main():
    ap=argparse.ArgumentParser(description="Generate a tiny deterministic TriQTO Phase 7 raw dataset.")
    g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--config"); g.add_argument("--demo", action="store_true")
    ap.add_argument("--output", required=True); ap.add_argument("--overwrite", action="store_true")
    ns=ap.parse_args()
    cfg=demo_config() if ns.demo else load_generation_config(ns.config)
    res=generate_dataset(cfg); wr=write_dataset(res, Path(ns.output), overwrite=ns.overwrite)
    print(f"dataset={res.dataset_name} samples={res.summary['sample_count']} manifests={len(wr.manifest_paths)} output={wr.output_root}")
    print(f"born_visible={res.summary['born_visible_sample_count']} marker_only={res.summary['marker_only_sample_count']}")
if __name__ == "__main__": main()
