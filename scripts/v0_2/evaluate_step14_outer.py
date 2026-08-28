#!/usr/bin/env python3
"""One-shot frozen Step-14 outer evaluation.

Evaluates the frozen Step-14 ensemble and frozen Step-10C warm ensemble on the
same newly materialized cross-motif and legacy-retention outer cohorts. Outer
results select nothing and no QPU access occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
from triqto.step7.model import Step7DiagnosticModel

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "triqto.v0_2.step14_outer_evaluation.v1"
DEFAULT_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step14_outer_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--selection-freeze", type=Path, required=True)
    p.add_argument("--cross-motif-outer-dir", type=Path, required=True)
    p.add_argument("--legacy-retention-outer-dir", type=Path, required=True)
    p.add_argument("--step10c-benchmark-dir", type=Path)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_selection(path: Path, config_path: Path) -> dict[str, Any]:
    freeze = read_json(path)
    if freeze.get("schema") != "triqto.v0_2.step14_selection_freeze.v1" or freeze.get("status") != "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION":
        raise RuntimeError("invalid Step-14 selection freeze")
    if freeze.get("protocol_config_sha256") != baseline.sha256_file(config_path): raise RuntimeError("selection freeze/config mismatch")
    return freeze


def verify_cross_outer(product: Path) -> list[dict[str, str]]:
    c = read_json(product/"dataset_complete.json")
    if c.get("schema") != "triqto.v0_2.step14_cross_motif_dataset.v1" or c.get("status") != "COMPLETE_FROZEN_SIMULATOR_OUTER": raise RuntimeError("cross-motif outer not frozen/complete")
    if bool(c.get("model_evaluated_before_freeze", True)) or bool(c.get("future_hardware_reserve_materialized", True)): raise RuntimeError("cross-motif outer boundary failed")
    for name,wanted in c["manifest_hashes"].items():
        if baseline.sha256_file(product/"manifests"/name) != wanted: raise RuntimeError(f"cross outer manifest hash mismatch: {name}")
    rows = baseline.read_csv(product/"manifests"/"example_manifest.csv")
    if any(r["step14_partition"] != "simulator_outer" for r in rows): raise RuntimeError("cross outer contains non-outer rows")
    return rows


def verify_legacy_outer(product: Path):
    c = read_json(product/"dataset_complete.json")
    if c.get("schema") != "triqto.v0_2.step14_fresh_legacy_retention_outer.v1" or c.get("status") != "COMPLETE_FROZEN_OUTER_VALIDATION": raise RuntimeError("legacy outer not frozen/complete")
    if bool(c.get("model_evaluated_before_freeze", True)) or not bool(c.get("selects_nothing", False)): raise RuntimeError("legacy outer boundary failed")
    for name,wanted in c["manifest_hashes"].items():
        if baseline.sha256_file(product/"manifests"/name) != wanted: raise RuntimeError(f"legacy manifest hash mismatch: {name}")
    return baseline.read_csv(product/"manifests"/"original_example_manifest.csv"), baseline.read_csv(product/"manifests"/"bridge_example_manifest.csv")


def by_root(rows: Sequence[dict[str,str]]) -> dict[int,list[int]]:
    out = {}
    for i,row in enumerate(rows): out.setdefault(int(row["root_index"]), []).append(i)
    if any(len(v) != 13 for v in out.values()): raise RuntimeError("outer root does not contain 13 examples")
    return out


def materialize(product: Path, rows, label: str, batch_size: int, progress: int):
    grouped = by_root(rows)
    return step7.materialize_blocks(product=product, rows=rows, by_root=grouped, roots=sorted(grouped), root_batch_size=batch_size, label=label, progress_every=progress)


def load_candidate_models(freeze: Mapping[str,Any], freeze_path: Path, device: torch.device):
    models=[]; run_dir=freeze_path.parent
    for seed in (1701,1702,1703):
        rec=freeze["selected_seed_records"][str(seed)]; path=run_dir/rec["checkpoint"]
        if baseline.sha256_file(path) != rec["checkpoint_sha256"]: raise RuntimeError(f"candidate checkpoint hash mismatch seed={seed}")
        payload=torch.load(path,map_location="cpu",weights_only=False); model=Step7DiagnosticModel(variant="late_concat",initialization_seed=seed); model.load_state_dict(payload["state_dict"],strict=True); model.to(device); model.eval(); models.append(model)
    return models


def load_step10c(cfg: Mapping[str,Any], benchmark: Path, device: torch.device):
    if baseline.sha256_file(benchmark/"model_selection.json") != str(cfg["warm_start"]["model_selection_sha256"]): raise RuntimeError("Step-10C model-selection hash mismatch")
    threshold=float(read_json(benchmark/"model_selection.json")["ensemble_effect_thresholds"]["warm_start"]); models=[]
    for seed in (1701,1702,1703):
        filename=str(cfg["warm_start"]["checkpoint_names"][str(seed)]); path=benchmark/filename
        if baseline.sha256_file(path) != str(cfg["warm_start"]["checkpoint_sha256"][filename]): raise RuntimeError(f"Step-10C checkpoint hash mismatch: {filename}")
        payload=torch.load(path,map_location="cpu",weights_only=False); model=Step7DiagnosticModel(variant="late_concat",initialization_seed=seed); model.load_state_dict(payload["state_dict"],strict=True); model.to(device); model.eval(); models.append(model)
    return models,threshold


def ensemble(models, blocks, device):
    preds=[step7.predict_blocks(m,blocks,device) for m in models]; ref=preds[0]
    for p in preds[1:]: step7.assert_prediction_alignment(ref,p,"Step-14 outer ensemble")
    return step7.PredictionSet(source_indices=ref.source_indices,root_indices=ref.root_indices,effect_truth=ref.effect_truth,mechanism_truth_all=ref.mechanism_truth_all,mechanism_mask=ref.mechanism_mask,effect_logits=np.mean(np.stack([p.effect_logits for p in preds]),axis=0),mechanism_logits=np.mean(np.stack([p.mechanism_logits for p in preds]),axis=0))


def ba(truth, guess, classes: int) -> float:
    return float(baseline.metrics_from_cm(baseline.confusion_matrix(np.asarray(truth,dtype=np.int64),np.asarray(guess,dtype=np.int64),classes))["balanced_accuracy"])


def point(pred: step7.PredictionSet, threshold: float) -> dict[str,Any]:
    effect=ba(pred.effect_truth,(pred.effect_logits>=threshold).astype(np.int8),2); mask=pred.mechanism_mask; truth=pred.mechanism_truth_all[mask]; guess=np.argmax(pred.mechanism_logits[mask],axis=1); cm=baseline.confusion_matrix(truth,guess,3); mech=float(baseline.metrics_from_cm(cm)["balanced_accuracy"]); recalls=[float(cm[i,i]/np.sum(cm[i])) if np.sum(cm[i]) else 0.0 for i in range(3)]
    return {"effect_balanced_accuracy":effect,"mechanism_balanced_accuracy":mech,"minimum_mechanism_recall":min(recalls),"mechanism_recall":recalls}


def group_codes(rows, pred: step7.PredictionSet, kind: str) -> np.ndarray:
    values=[]
    for source in pred.source_indices.tolist():
        r=rows[int(source)]
        values.append(str(r["family_index"] if kind=="family" else r["root_index"] if kind=="root" else r["parent_group_index"]))
    lookup={v:i for i,v in enumerate(sorted(set(values)))}; return np.asarray([lookup[v] for v in values],dtype=np.int64)


def bootstrap(pred: step7.PredictionSet, threshold: float, groups: np.ndarray, reps: int, seed: int):
    rng=np.random.default_rng(seed); unique=np.unique(groups); mech=[]; effect=[]
    for _ in range(reps):
        sampled=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        effect.append(ba(pred.effect_truth[idx],(pred.effect_logits[idx]>=threshold).astype(np.int8),2)); m=pred.mechanism_mask[idx]; mech.append(ba(pred.mechanism_truth_all[idx][m],np.argmax(pred.mechanism_logits[idx][m],axis=1),3))
    return {"mechanism_ba_ci":[float(np.quantile(mech,.025)),float(np.quantile(mech,.975))],"effect_ba_ci":[float(np.quantile(effect,.025)),float(np.quantile(effect,.975))]}


def paired_delta(candidate, base, groups, reps: int, seed: int):
    step7.assert_prediction_alignment(candidate,base,"Step-14 paired outer"); rng=np.random.default_rng(seed); unique=np.unique(groups); values=[]
    for _ in range(reps):
        sampled=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([np.flatnonzero(groups==g) for g in sampled]); m=candidate.mechanism_mask[idx]; truth=candidate.mechanism_truth_all[idx][m]
        values.append(ba(truth,np.argmax(candidate.mechanism_logits[idx][m],axis=1),3)-ba(truth,np.argmax(base.mechanism_logits[idx][m],axis=1),3))
    return {"mean":float(np.mean(values)),"ci":[float(np.quantile(values,.025)),float(np.quantile(values,.975))]}


def evaluate_support_gate(cfg: Mapping[str,Any], cross, original, bridge, paired):
    g=cfg["outer_evaluation"]["support_gate"]
    checks={
        "cross_mechanism_ba":cross["candidate"]["mechanism_balanced_accuracy"]>=g["cross_motif_mechanism_balanced_accuracy_minimum"],
        "cross_mechanism_ci_lower":cross["candidate_bootstrap"]["mechanism_ba_ci"][0]>=g["cross_motif_mechanism_bootstrap_ci_lower_minimum"],
        "cross_minimum_recall":cross["candidate"]["minimum_mechanism_recall"]>=g["cross_motif_minimum_class_recall"],
        "cross_effect_ba":cross["candidate"]["effect_balanced_accuracy"]>=g["cross_motif_effect_balanced_accuracy_minimum"],
        "paired_improvement":cross["candidate"]["mechanism_balanced_accuracy"]-cross["step10c"]["mechanism_balanced_accuracy"]>=g["candidate_minus_step10c_cross_motif_mechanism_ba_minimum"] and paired["ci"][0]>g["paired_bootstrap_ci_lower_for_candidate_minus_step10c_mechanism_ba_must_exceed"],
        "legacy_original_mechanism_retention":original["candidate"]["mechanism_balanced_accuracy"]>=original["step10c"]["mechanism_balanced_accuracy"]-g["legacy_original_mechanism_ba_max_drop_vs_step10c"],
        "legacy_original_effect_retention":original["candidate"]["effect_balanced_accuracy"]>=original["step10c"]["effect_balanced_accuracy"]-g["legacy_original_effect_ba_max_drop_vs_step10c"],
        "legacy_bridge_mechanism_retention":bridge["candidate"]["mechanism_balanced_accuracy"]>=bridge["step10c"]["mechanism_balanced_accuracy"]-g["legacy_bridge_mechanism_ba_max_drop_vs_step10c"],
        "legacy_bridge_effect_retention":bridge["candidate"]["effect_balanced_accuracy"]>=bridge["step10c"]["effect_balanced_accuracy"]-g["legacy_bridge_effect_ba_max_drop_vs_step10c"]}
    passed=all(checks.values()); return {"passed":passed,"checks":checks,"interpretation":cfg["outer_evaluation"]["gate_interpretation"]["all_criteria_met" if passed else "otherwise"]}


def main() -> None:
    args=parse_args(); config_path=args.config.expanduser().resolve(); cfg=read_json(config_path)
    if cfg.get("schema")!="triqto.v0_2.step14_cross_motif_generalization_training.v1" or cfg.get("status")!="FROZEN_BEFORE_STEP14_DATASET_GENERATION": raise RuntimeError("unexpected Step-14 protocol")
    freeze_path=args.selection_freeze.expanduser().resolve(); freeze=verify_selection(freeze_path,config_path); cross_product=args.cross_motif_outer_dir.expanduser().resolve(); legacy_product=args.legacy_retention_outer_dir.expanduser().resolve(); cross_rows=verify_cross_outer(cross_product); original_rows,bridge_rows=verify_legacy_outer(legacy_product)
    device=step7.resolve_device(args.device); batch_size=int(cfg["training"]["root_batch_size"]); cross_blocks=materialize(cross_product,cross_rows,"step14-cross-outer",batch_size,args.progress_every); original_blocks=materialize(legacy_product,original_rows,"step14-original-outer",batch_size,args.progress_every); bridge_blocks=materialize(legacy_product,bridge_rows,"step14-bridge-outer",batch_size,args.progress_every)
    candidate_models=load_candidate_models(freeze,freeze_path,device); benchmark=(args.step10c_benchmark_dir or Path(cfg["warm_start"]["default_benchmark_dir"])).expanduser().resolve(); base_models,base_threshold=load_step10c(cfg,benchmark,device); candidate_threshold=float(freeze["ensemble_effect_threshold"])
    domains={"cross_motif":(cross_rows,cross_blocks,"family"),"legacy_original":(original_rows,original_blocks,"root"),"legacy_bridge":(bridge_rows,bridge_blocks,"parent")}; reps=int(cfg["outer_evaluation"]["bootstrap_replicates"]); seed=int(cfg["outer_evaluation"]["bootstrap_seed"]); results={}; stored={}
    for offset,(name,(rows,blocks,kind)) in enumerate(domains.items()):
        cand=ensemble(candidate_models,blocks,device); base=ensemble(base_models,blocks,device); groups=group_codes(rows,cand,kind); stored[name]=(cand,base,groups); results[name]={"candidate":point(cand,candidate_threshold),"step10c":point(base,base_threshold),"candidate_bootstrap":bootstrap(cand,candidate_threshold,groups,reps,seed+100*offset+1),"step10c_bootstrap":bootstrap(base,base_threshold,groups,reps,seed+100*offset+2),"bootstrap_unit":kind}
    cand,base,groups=stored["cross_motif"]; paired=paired_delta(cand,base,groups,reps,seed+999); gate=evaluate_support_gate(cfg,results["cross_motif"],results["legacy_original"],results["legacy_bridge"],paired)
    identity={"schema":SCHEMA,"protocol_config_sha256":baseline.sha256_file(config_path),"selection_freeze_sha256":baseline.sha256_file(freeze_path),"cross_outer_dataset_complete_sha256":baseline.sha256_file(cross_product/"dataset_complete.json"),"legacy_outer_dataset_complete_sha256":baseline.sha256_file(legacy_product/"dataset_complete.json"),"step10c_model_selection_sha256":baseline.sha256_file(benchmark/"model_selection.json")}
    evaluation_id="evaluation_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]; output=args.output_parent.expanduser().resolve()/evaluation_id; output.mkdir(parents=True,exist_ok=True)
    payload={"schema":SCHEMA,"status":"COMPLETE_OUTER_SPENT","evaluation_id":evaluation_id,"identity":identity,"candidate_threshold":candidate_threshold,"step10c_threshold":base_threshold,"results":results,"paired_cross_motif_mechanism_ba_delta":paired,"support_gate":gate,"outer_selects_nothing":True,"qpu_executed":False,"future_hardware_reserve_accessed":False}
    atomic_json(output/"outer_result.json",payload); atomic_json(output/"evaluation_complete.json",{"schema":SCHEMA,"status":"COMPLETE_OUTER_SPENT","evaluation_id":evaluation_id,"outer_result_sha256":baseline.sha256_file(output/"outer_result.json"),"support_gate_passed":bool(gate["passed"]),"interpretation":gate["interpretation"],"qpu_executed":False,"future_hardware_reserve_accessed":False})
    print("\nTRIQTO STEP 14 OUTER EVALUATION COMPLETE"); print("Cross-motif mechanism BA candidate / Step10C:",results["cross_motif"]["candidate"]["mechanism_balanced_accuracy"],"/",results["cross_motif"]["step10c"]["mechanism_balanced_accuracy"]); print("Paired delta CI:",paired["ci"]); print("Support gate passed:","YES" if gate["passed"] else "NO"); print("Interpretation:",gate["interpretation"]); print("QPU executed: NO"); print("Output:",output)


if __name__=="__main__":
    main()
