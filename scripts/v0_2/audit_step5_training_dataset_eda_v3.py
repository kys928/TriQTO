#!/usr/bin/env python3
"""Step 5 v3 promotion EDA with acquisition/context alias gates."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
V3_GEN_PATH = Path(__file__).resolve().parent / "generate_step5_matched_diagnostic_training_dataset_v3.py"
EDA_V2_PATH = Path(__file__).resolve().parent / "audit_step5_training_dataset_eda.py"

def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

V3=load_module("triqto_step5_v3_for_eda",V3_GEN_PATH)
EDA=load_module("triqto_step5_v2_eda_base",EDA_V2_PATH)
DEFAULT_POINTER=Path("/workspace/triqto-data/step5_matched_diagnostic_training_v3/current_product.json")
DEFAULT_OUTPUT=Path("/workspace/triqto-data/step5_matched_diagnostic_training_v3_eda")
DEFAULT_OVERLAY=ROOT/"configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json"
DEFAULT_V2_CONFIG=ROOT/"configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--product-dir",type=Path); p.add_argument("--product-pointer",type=Path,default=DEFAULT_POINTER); p.add_argument("--overlay-config",type=Path,default=DEFAULT_OVERLAY); p.add_argument("--base-v2-config",type=Path,default=DEFAULT_V2_CONFIG); p.add_argument("--output-parent",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--progress-every",type=int,default=250); return p.parse_args()

def read_csv(path: Path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def as_bool(v: Any)->bool:
    if isinstance(v,bool): return v
    if str(v).lower()=="true": return True
    if str(v).lower()=="false": return False
    raise ValueError(v)

def resolve(args):
    if args.product_dir: return args.product_dir.expanduser().resolve()
    pointer=json.loads(args.product_pointer.read_text()); return Path(pointer["product_dir"]).expanduser().resolve()

def main():
    args=parse_args(); product=resolve(args); overlay=json.loads(args.overlay_config.read_text()); completion=json.loads((product/"dataset_complete.json").read_text())
    if completion.get("schema") != V3.SCHEMA: raise RuntimeError("v3 EDA requires a Step 5 v3 product")
    roots=read_csv(product/"manifests/clean_circuit_manifest.csv"); examples=read_csv(product/"manifests/example_manifest.csv")
    injected=[r for r in examples if not as_bool(r["clean_control"])]; clean=[r for r in examples if as_bool(r["clean_control"])]
    gates=overlay["association_gates"]
    metrics={
      "shot_strength_cramers_v":EDA.cramers_v(injected,"shots","strength"),
      "shot_depth_cramers_v":EDA.cramers_v(injected,"shots","insertion_depth_bin"),
      "shot_family_cramers_v":EDA.cramers_v(injected,"shots","family"),
      "shot_split_cramers_v":EDA.cramers_v(injected,"shots","split"),
      "clean_shot_family_cramers_v":EDA.cramers_v(clean,"shots","family"),
      "clean_shot_split_cramers_v":EDA.cramers_v(clean,"shots","split"),
    }
    checks=[
      ("shot/strength",metrics["shot_strength_cramers_v"],"maximum_shot_strength_cramers_v"),("shot/depth",metrics["shot_depth_cramers_v"],"maximum_shot_depth_cramers_v"),("shot/family",metrics["shot_family_cramers_v"],"maximum_shot_family_cramers_v"),("shot/split",metrics["shot_split_cramers_v"],"maximum_shot_split_cramers_v"),("clean-shot/family",metrics["clean_shot_family_cramers_v"],"maximum_clean_shot_family_cramers_v"),("clean-shot/split",metrics["clean_shot_split_cramers_v"],"maximum_clean_shot_split_cramers_v")]
    failures=[]
    for name,value,key in checks:
        if value>float(gates[key]): failures.append(f"{name}={value:.6f}>{float(gates[key]):.6f}")
    per_n={}
    for nq in sorted({int(r["n_qubits"]) for r in injected}):
        sub=[r for r in injected if int(r["n_qubits"])==nq]
        vals={"shot_affected":EDA.cramers_v(sub,"shots","affected_qubit"),"strength_affected":EDA.cramers_v(sub,"strength","affected_qubit"),"depth_affected":EDA.cramers_v(sub,"insertion_depth_bin","affected_qubit")}; per_n[nq]=vals
        if vals["shot_affected"]>float(gates["maximum_per_qubit_count_shot_affected_cramers_v"]): failures.append(f"{nq}q shot/affected={vals['shot_affected']:.6f}")
        if vals["strength_affected"]>float(gates["maximum_per_qubit_count_strength_affected_cramers_v"]): failures.append(f"{nq}q strength/affected={vals['strength_affected']:.6f}")
        if vals["depth_affected"]>float(gates["maximum_per_qubit_count_depth_affected_cramers_v"]): failures.append(f"{nq}q depth/affected={vals['depth_affected']:.6f}")
    print("TRIQTO STEP 5 V3 SCHEDULING-ALIAS PRECHECK")
    for key,value in metrics.items(): print(f"{key}: {value:.6f}")
    for nq,vals in per_n.items(): print(f"{nq}q: shot/affected={vals['shot_affected']:.6f} strength/affected={vals['strength_affected']:.6f} depth/affected={vals['depth_affected']:.6f}")
    if failures:
        print("Decision: BLOCKED"); print("Failures:"); [print(f"- {x}") for x in failures]; raise SystemExit(2)
    print("Scheduling-alias gates: PASS\nRunning full artifact/hash/target/noise EDA...\n")
    EDA.PRODUCT_SCHEMA=V3.SCHEMA
    old=sys.argv
    sys.argv=[str(EDA_V2_PATH),"--product-dir",str(product),"--config",str(args.base_v2_config),"--output-parent",str(args.output_parent),"--progress-every",str(args.progress_every)]
    try: EDA.main()
    finally: sys.argv=old

if __name__=="__main__": main()
