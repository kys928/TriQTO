#!/usr/bin/env python3
"""Generate the frozen fresh legacy-retention outer cohort for Step 14.

Requires the frozen Step-14 selection marker. Reuses the established legacy
original/bridge generators in a new namespace and proves non-overlap with
historical Step-10 and Step-10C outer graphs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3
import generate_step10c_fresh_outer_cohort as step10c_outer

BASE = step5v3.BASE
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "triqto.v0_2.step14_fresh_legacy_retention_outer.v1"
DEFAULT_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
DEFAULT_STEP10C_OUTER_CONFIG = ROOT / "configs/v0_2/step10c_fresh_outer_cohort.json"
DEFAULT_V2_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
DEFAULT_V3_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json"
DEFAULT_STEP10_CONFIG = ROOT / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = ROOT / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_STEP10_PRODUCT = Path("/workspace/triqto-data/step10_training_mixture/product_0f7112597501f7ea5fbe123b")
DEFAULT_STEP10C_OUTER = Path("/workspace/triqto-data/step10c_fresh_outer_cohort/product_57ee407d62ea794bfc9ff169")
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step14_fresh_legacy_retention_outer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--selection-freeze", type=Path, required=True)
    p.add_argument("--step10c-outer-config", type=Path, default=DEFAULT_STEP10C_OUTER_CONFIG)
    p.add_argument("--base-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    p.add_argument("--v3-config", type=Path, default=DEFAULT_V3_CONFIG)
    p.add_argument("--step10-config", type=Path, default=DEFAULT_STEP10_CONFIG)
    p.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    p.add_argument("--step10-product-dir", type=Path, default=DEFAULT_STEP10_PRODUCT)
    p.add_argument("--step10c-outer-dir", type=Path, default=DEFAULT_STEP10C_OUTER)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--eda-progress-every", type=int, default=2000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_selection_freeze(path: Path, config_path: Path) -> dict[str, Any]:
    freeze = read_json(path.expanduser().resolve())
    if freeze.get("schema") != "triqto.v0_2.step14_selection_freeze.v1" or freeze.get("status") != "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION":
        raise RuntimeError("Step-14 selection is not frozen before legacy outer materialization")
    if freeze.get("protocol_config_sha256") != baseline.sha256_file(config_path):
        raise RuntimeError("Step-14 selection freeze/config mismatch")
    if not bool(freeze.get("all_three_seed_checkpoints_frozen")):
        raise RuntimeError("Step-14 selection freeze incomplete")
    return freeze


def historical_step10c_hashes(product: Path) -> set[str]:
    complete = read_json(product / "dataset_complete.json")
    if complete.get("status") != "COMPLETE_FROZEN_OUTER_VALIDATION":
        raise RuntimeError("Step-10C historical outer is not frozen/complete")
    out: set[str] = set()
    for name in ("original_root_manifest.csv", "bridge_root_manifest.csv"):
        out.update(str(r["graph_sha256"]) for r in baseline.read_csv(product / "manifests" / name))
    return out


def derived_generation_cfg(source: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(source)); spec = protocol["fresh_legacy_retention_outer"]
    o = cfg["original_domain"]; o["global_root_index_start"] = 6000; o["clean_root_count"] = int(spec["original_clean_roots"]); o["expected_examples"] = int(spec["original_expected_examples"])
    b = cfg["bridge_domain"]; b["base_seed"] = int(spec["base_seed"]); b["parent_groups"] = int(spec["bridge_parent_groups"]); b["variants_per_parent"] = int(spec["bridge_variants_per_parent"]); b["expected_clean_roots"] = int(spec["bridge_expected_roots"]); b["expected_examples"] = int(spec["bridge_expected_examples"])
    return cfg


def main() -> None:
    args = parse_args(); config_path = args.config.expanduser().resolve(); protocol = read_json(config_path)
    if protocol.get("schema") != "triqto.v0_2.step14_cross_motif_generalization_training.v1" or protocol.get("status") != "FROZEN_BEFORE_STEP14_DATASET_GENERATION":
        raise RuntimeError("unexpected Step-14 protocol schema/status")
    verify_selection_freeze(args.selection_freeze, config_path)
    source_cfg = read_json(args.step10c_outer_config.expanduser().resolve()); cfg = derived_generation_cfg(source_cfg, protocol)
    v2cfg = read_json(args.base_v2_config.expanduser().resolve()); v3cfg = read_json(args.v3_config.expanduser().resolve())
    step10cfg = read_json(args.step10_config.expanduser().resolve()); step7cfg = read_json(args.step7_config.expanduser().resolve())
    step10_product = args.step10_product_dir.expanduser().resolve()
    old_original, old_bridge, historical_identity = step10c_outer._verify_historical_products(step10_product, step10cfg, step7cfg)
    spent_step10c = historical_step10c_hashes(args.step10c_outer_dir.expanduser().resolve())
    old_original = set(old_original) | spent_step10c; old_bridge = set(old_bridge) | spent_step10c
    identity = {"schema": SCHEMA, "protocol_config_sha256": baseline.sha256_file(config_path),
                "selection_freeze_sha256": baseline.sha256_file(args.selection_freeze.expanduser().resolve()),
                "step10c_historical_outer_dataset_complete_sha256": baseline.sha256_file(args.step10c_outer_dir.expanduser().resolve()/"dataset_complete.json"),
                "derived_original_root_start": 6000, "base_seed": int(protocol["fresh_legacy_retention_outer"]["base_seed"]),
                "selects_nothing": True, "qpu_access": False}
    product_id = "legacy_outer_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    parent = args.output_parent.expanduser().resolve(); product = parent/product_id
    if product.exists():
        if read_json(product/"dataset_complete.json").get("identity") != identity: raise RuntimeError("existing Step-14 legacy outer identity mismatch")
        print("Step-14 legacy-retention outer already complete:", product); return
    staging = parent/f".{product_id}.staging-{uuid.uuid4().hex}"; staging.mkdir(parents=True, exist_ok=False)
    try:
        original_roots, original_examples = step10c_outer._generate_original(staging=staging, cfg=cfg, v2cfg=v2cfg, v3cfg=v3cfg,
            old_original_hashes=old_original, old_bridge_hashes=old_bridge, diagnostic_bound=2.000001, progress_every=args.progress_every)
        fresh_original = {str(r["graph_sha256"]) for r in original_roots}
        bridge_roots, bridge_examples = step10c_outer._generate_bridge(staging=staging, cfg=cfg, v2cfg=v2cfg,
            old_original_hashes=old_original | fresh_original, old_bridge_hashes=old_bridge,
            diagnostic_bound=2.000001, progress_every=args.progress_every)
        freshness = step10c_outer._validate_structure(cfg=cfg, original_roots=original_roots, original_examples=original_examples,
            bridge_roots=bridge_roots, bridge_examples=bridge_examples, old_original_hashes=old_original, old_bridge_hashes=old_bridge)
        all_roots = list(original_roots) + list(bridge_roots); all_examples = list(original_examples) + list(bridge_examples)
        overlap = {str(r["graph_sha256"]) for r in all_roots} & spent_step10c
        if overlap: raise RuntimeError("Step-14 legacy outer overlaps spent Step-10C outer")
        scan = step10c_outer._scan_artifacts(staging, all_examples, 2.000001, args.eda_progress_every)
        manifests = staging/"manifests"; BASE.write_csv(manifests/"original_root_manifest.csv", original_roots); BASE.write_csv(manifests/"original_example_manifest.csv", original_examples); BASE.write_csv(manifests/"bridge_root_manifest.csv", bridge_roots); BASE.write_csv(manifests/"bridge_example_manifest.csv", bridge_examples)
        eda = {"schema":"triqto.v0_2.step14_fresh_legacy_retention_eda.v1", "status":"PASS", "model_evaluated":False,
               "qpu_executed":False, "freshness":freshness, "artifact_scan":scan, "historical_step10c_graph_overlap_count":0}
        BASE.atomic_json(staging/"eda.json", eda); names = ["original_root_manifest.csv","original_example_manifest.csv","bridge_root_manifest.csv","bridge_example_manifest.csv"]
        complete = {"schema":SCHEMA, "status":"COMPLETE_FROZEN_OUTER_VALIDATION", "product_id":product_id, "identity":identity,
                    "original_clean_root_count":len(original_roots), "original_example_count":len(original_examples),
                    "bridge_parent_group_count":int(cfg["bridge_domain"]["parent_groups"]), "bridge_clean_root_count":len(bridge_roots), "bridge_example_count":len(bridge_examples),
                    "model_evaluated_before_freeze":False, "selects_nothing":True, "qpu_executed":False,
                    "manifest_hashes":{name:BASE.sha256_file(manifests/name) for name in names}, "eda_sha256":BASE.sha256_file(staging/"eda.json"),
                    "historical_identity":historical_identity}
        BASE.atomic_json(staging/"dataset_complete.json", complete); os.replace(staging, product)
        BASE.atomic_json(parent/"current_product.json", {"schema":"triqto.v0_2.step14_fresh_legacy_retention_current.v1", "product_id":product_id,
                                                         "product_dir":str(product), "dataset_complete_sha256":BASE.sha256_file(product/"dataset_complete.json")})
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print("\nTRIQTO STEP 14 FRESH LEGACY-RETENTION OUTER COMPLETE"); print("Original:", len(original_roots), "roots /", len(original_examples), "examples"); print("Bridge:", len(bridge_roots), "roots /", len(bridge_examples), "examples"); print("Historical graph overlap: 0"); print("Model evaluated: NO"); print("QPU executed: NO"); print("Output:", product)


if __name__ == "__main__":
    main()
