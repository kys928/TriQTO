#!/usr/bin/env python3
"""Read-only frozen Step-14 Phase-A replay isolation.

Compares original blind Phase-A evidence extraction against a separately
reconstructed posthoc *x-only* path for a fixed set of mismatch and control
rows. Uses the frozen networks unchanged on CPU and CUDA and compares all
intermediate stages with SHA256 provenance. No sealed truth, fit, selection,
hyperparameter changes, or outer/QPU access are permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import benchmark_step6_cheap_baselines as baseline
import predict_step14_equivalence_aware_fresh_holdout_blind as blind
import score_step14_equivalence_aware_fresh_holdout_confirmatory as confirm
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import step14_equivalence_aware_common as equiv

ROOT = Path(__file__).resolve().parents[2]
CONFIRM_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json"
LATENT_CONFIG = ROOT / "configs/v0_2/step14_latent_frame_inference.json"
HOLDOUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
METHOD_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
PRED_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout_confirmation")
OUTPUT_PARENT = Path("/workspace/triqto-data/step15_replay_diagnostics")
KNOWN_DISAGREEMENTS = (302,311,330,541,558,1980,1992,2003,2010,3135,3138,3150,3159,3445,3494,4189)

def digest_file(path: Path) -> str:
    return baseline.sha256_file(path)

def digest_array(value: np.ndarray) -> dict:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(a.tobytes()).hexdigest()
    return {"sha256":"sha256:"+h,"dtype":str(a.dtype),"shape":list(a.shape)}

def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    a=np.asarray(a)
    b=np.asarray(b)
    if a.shape != b.shape:
        return {"shape_match":False,"a":digest_array(a),"b":digest_array(b)}
    eq=np.array_equal(a,b)
    if np.issubdtype(a.dtype,np.number):
        # Both extraction paths have identical padded infinities.
        finite=np.isfinite(a)&np.isfinite(b)
        both_inf=(~np.isfinite(a))&(~np.isfinite(b))&(np.signbit(a)==np.signbit(b))
        finite_bad=finite & (a!=b)
        nonfinite_bad=~(finite | both_inf)
        good=finite|both_inf
        maxabs=float(np.max(np.abs(a[finite]-b[finite]))) if np.any(finite) else 0.
        bad=int(np.count_nonzero(finite_bad | nonfinite_bad))
    else:
        maxabs=None
        bad=int(np.count_nonzero(a!=b))
    return {"shape_match":True,"exact_equal":bool(eq),"mismatch_elements":bad,
            "max_finite_absolute_difference":maxabs,
            "a":digest_array(a),"b":digest_array(b)}

def logits_compare(reference, replay):
    a=np.asarray(reference,dtype=np.float64)
    b=np.asarray(replay,dtype=np.float64)
    if a.shape != b.shape: raise RuntimeError("logit comparison shape drift")
    error=np.abs(a-b)
    disagreements=np.flatnonzero(np.argmax(a,axis=1)!=np.argmax(b,axis=1))
    return {"max_abs_diff":float(error.max()),
            "mean_abs_diff":float(error.mean()),
            "rows_above_1e_minus_4":int(np.any(error>1e-4,axis=1).sum()),
            "argmax_mismatch_positions":[int(x) for x in disagreements.tolist()],
            "argmax_mismatch_count":int(len(disagreements))}

def posthoc_x_only_subset(product, pred_rows, latent_cfg):
    """Mirror only the deployable x-only portion of the posthoc extractor."""
    manifest={str(r["example_id"]):r for r in baseline.read_csv(product/"manifests"/"example_manifest.csv")}
    cached={}
    features=[]
    profiles=[]
    distances=[]
    gates=[]
    for row in pred_rows:
        example_id=str(row["example_id"])
        rec=manifest[example_id]
        if str(rec.get("evaluation_role"))!="target":
            raise RuntimeError("non-target row in replay diagnostic")
        root=int(row["root_index"])
        if int(rec["root_index"])!=root:
            raise RuntimeError("root mismatch")
        loaded=confirm.load_x_only(product/str(rec["artifact_path"]),str(rec["artifact_sha256"]),example_id)
        delta,weights,pairs=frame.measured_delta_and_weights(loaded)
        if root not in cached:
            clean=frame.circuit_from_serialized(loaded)
            candidates=latent.plausible_candidates(clean)
            finite_jacs=latent.finite_shot_candidate_jacobians(clean,pairs,candidates,root,latent_cfg)
            distance,gate=equiv.finite_frame_pair_geometry(finite_jacs)
            cached[root]=(finite_jacs,distance,gate)
        finite_jacs,distance,gate=cached[root]
        feats=np.stack([frame.canonicalize_evidence(delta,jac,weights)[0]
                        for jac in finite_jacs]).astype(np.float32)
        feature_profile=latent.profile_log_likelihoods(delta,weights,finite_jacs).astype(np.float64)
        features.append(feats)
        profiles.append(feature_profile)
        distances.append(distance)
        gates.append(gate)
    x,mask=latent.pad_candidate_features(features)
    count=int(x.shape[1])
    return {"x":x,"mask":mask,"profiles":blind.pad_profiles(profiles,count),
            "distance":blind.pad_pair_float(distances,count),
            "gate":blind.pad_pair_bool(gates,count)}

def write_json(path, payload):
    temp=path.with_name("."+path.name+".tmp-"+uuid.uuid4().hex)
    with temp.open("w",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=("cuda",),default="cuda")
    parser.add_argument("--max-controls",type=int,default=20)
    parser.add_argument("--output-parent",type=Path,default=OUTPUT_PARENT)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CPU/CUDA comparison requires an available GPU")
    if not (1<=args.max_controls<=40):
        raise ValueError("max-controls out of range")
    oracle_compat.apply_frozen_support_bound()
    cfg=json.loads(CONFIRM_CONFIG.read_text())
    method=cfg["method_identity"]
    hold=cfg["holdout_identity"]
    pointer=json.loads((PRED_PARENT/"current_oracle_free_predictions.json").read_text())
    pred_rows,frozen,complete=confirm.prediction_rows_and_logits(
        pointer,str(cfg["source_identity"]["oracle_free_predictions_complete_sha256"])
    ) if "source_identity" in cfg else (None,None,None)
    if pred_rows is None:
        # The confirmatory JSON pins method/holdout; the frozen prediction
        # completion hash lives in the posthoc source identity.
        posthoc_cfg=json.loads(
            (ROOT/"configs/v0_2/step14_fresh_holdout_residual_bottleneck_posthoc.json").read_text())
        expected=posthoc_cfg["source_identity"]["oracle_free_predictions_complete_sha256"]
        pred_rows,frozen,complete=confirm.prediction_rows_and_logits(pointer,str(expected))
    product_ptr=json.loads((HOLDOUT_PARENT/"current_fresh_holdout.json").read_text())
    if product_ptr["product_id"]!=hold["product_id"]:
        raise RuntimeError("holdout product identity drift")
    product=Path(product_ptr["product_dir"]).resolve()
    if digest_file(product/"dataset_complete.json")!=hold["dataset_complete_sha256"]:
        raise RuntimeError("holdout product hash drift")
    if len(pred_rows)!=7200 or frozen.shape!=(7200,3):
        raise RuntimeError("unexpected Phase-A frozen prediction shape")
    controls=np.linspace(0,7199,num=args.max_controls,dtype=np.int64)
    positions=sorted(set(KNOWN_DISAGREEMENTS)|{int(i) for i in controls})
    selected=[pred_rows[i] for i in positions]
    frozen_subset=frozen[positions]
    blind_rows={str(r["example_id"]):r for r in baseline.read_csv(product/"manifests"/"example_manifest.csv")}
    root_rows=baseline.read_csv(product/"manifests"/"root_manifest.csv")
    safe_roots={int(r["root_index"]):{"operation_signature":str(r["operation_signature"]),
                   "variant_index":str(r["variant_index"])} for r in root_rows}
    for row in selected:
        if str(row["example_id"]) not in blind_rows:
            raise RuntimeError("Phase-A example not present in blind manifest")
    latent_cfg=latent.load_frozen_config()
    phasea=blind.extract_blind_table(
        product,[blind_rows[str(r["example_id"])] for r in selected],safe_roots,
        latent_cfg,0)
    posthoc=posthoc_x_only_subset(product,selected,latent_cfg)
    if phasea["example_id"]!=[str(r["example_id"]) for r in selected]:
        raise RuntimeError("Phase-A row order drift")
    if any(int(row["candidate_count"])!=int(sum(phasea["mask"][i])) for i,row in enumerate(selected)):
        raise RuntimeError("candidate count differs from frozen Phase-A")
    stages={name:compare_arrays(phasea[name],posthoc[name])
            for name in ("x","mask","profiles","distance","gate")}
    source_dir=Path(json.loads((METHOD_PARENT/"current_equivalence_aware_fit.json").read_text())["method_dir"])
    pointer14=json.loads((METHOD_PARENT/"current_equivalence_aware_fit.json").read_text())
    if pointer14["method_id"]!=method["method_id"]:
        raise RuntimeError("method identity drift")
    if digest_file(source_dir/"method_bundle.json")!=method["method_bundle_sha256"]:
        raise RuntimeError("frozen method bundle drift")
    checkpoints={}
    outputs={}
    for name,table in (("phasea",phasea),("posthoc",posthoc)):
        outputs[name]={}
        for device_name in ("cpu","cuda"):
            per_seed={}
            for seed in [int(v) for v in method["candidate_network_seeds"]]:
                file=f"candidate_network_seed{seed}.pt"
                path=source_dir/file
                expect=method["checkpoint_sha256"][file]
                actual=digest_file(path)
                if actual!=expect: raise RuntimeError("checkpoint hash mismatch: "+file)
                payload=torch.load(path,map_location="cpu",weights_only=False)
                if payload.get("method_id")!=method["method_id"] or int(payload.get("seed"))!=seed:
                    raise RuntimeError("checkpoint internal identity drift")
                if (float(payload["selected_tau"])!=float(method["tau"]) or
                    float(payload["selected_frame_temperature"])!=float(method["frame_temperature"])):
                    raise RuntimeError("checkpoint hyperparameter identity drift")
                checkpoints[file]={"sha256":actual,"method_id":payload["method_id"],
                     "seed":int(payload["seed"]),"normalization_mean":digest_array(
                     np.asarray(payload["normalization_mean"])),
                     "normalization_std":digest_array(np.asarray(payload["normalization_std"]))}
                _,candidate=equiv.infer_candidate_network(
                    table["x"],table["mask"],state_dict=payload["state_dict"],
                    mean=np.asarray(payload["normalization_mean"],dtype=np.float32),
                    std=np.asarray(payload["normalization_std"],dtype=np.float32),
                    latent_cfg=latent_cfg,device=torch.device(device_name))
                per_seed[file]=candidate
                print(f"replay={name} device={device_name} seed={seed} complete",flush=True)
            ensemble=np.mean(np.stack(list(per_seed.values())),axis=0)
            pooled=equiv.equivalence_aware_pool_batch(
                ensemble,table["profiles"],table["distance"],table["gate"],table["mask"],
                tau=float(method["tau"]),frame_temperature=float(method["frame_temperature"]))
            outputs[name][device_name]={"pooled":pooled,"per_seed":per_seed}
    comparisons={}
    for path,devices in outputs.items():
        comparisons[path]={}
        for device_name,d in devices.items():
            comparisons[path][device_name]={
                "pooled_vs_frozen":logits_compare(frozen_subset,d["pooled"]),
                "pooled_digest":digest_array(d["pooled"]),
                "per_seed_digests":{k:digest_array(v) for k,v in d["per_seed"].items()},
            }
        comparisons[path]["cpu_vs_cuda"]=logits_compare(devices["cpu"]["pooled"],devices["cuda"]["pooled"])
        comparisons[path]["per_seed_cpu_vs_cuda"]={
            k:compare_arrays(devices["cpu"]["per_seed"][k],devices["cuda"]["per_seed"][k])
            for k in devices["cpu"]["per_seed"]
        }
    comparisons["phasea_vs_posthoc"]={
        dev:logits_compare(outputs["phasea"][dev]["pooled"],outputs["posthoc"][dev]["pooled"])
        for dev in ("cpu","cuda")
    }
    per_example=[]
    for i,pos in enumerate(positions):
        per_example.append({
            "frozen_csv_position":pos,"example_id":selected[i]["example_id"],
            "root_index":int(selected[i]["root_index"]),
            "known_prior_mismatch":pos in KNOWN_DISAGREEMENTS,
            "frozen_logits":[float(v) for v in frozen_subset[i]],
            "phasea_cpu_logits":[float(v) for v in outputs["phasea"]["cpu"]["pooled"][i]],
            "phasea_cuda_logits":[float(v) for v in outputs["phasea"]["cuda"]["pooled"][i]],
            "posthoc_cpu_logits":[float(v) for v in outputs["posthoc"]["cpu"]["pooled"][i]],
            "posthoc_cuda_logits":[float(v) for v in outputs["posthoc"]["cuda"]["pooled"][i]],
        })
    result={
      "schema":"triqto.v0_2.step14_frozen_phasea_cpu_cuda_replay_diagnostic.v1",
      "status":"COMPLETE_READ_ONLY_REPLAY_DIAGNOSTIC_NO_HOLDOUT_RESCORING",
      "fixed_positions":list(KNOWN_DISAGREEMENTS),
      "positions":positions,"example_count":len(positions),
      "scientific_boundary":{"sealed_truth_accessed":False,
          "step15_adapter_modified":False,"checkpoint_modified":False,
          "hyperparameters_selected":False,"holdout_rescored":False,
          "qpu_executed":False},
      "provenance":{
        "python":sys.version,"platform":platform.platform(),
        "torch":torch.__version__,"numpy":np.__version__,
        "cuda_version":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
        "torch_tf32_matmul_enabled":torch.backends.cuda.matmul.allow_tf32,
        "torch_tf32_cudnn_enabled":torch.backends.cudnn.allow_tf32,
        "phasea_prediction_csv_sha256":complete["oracle_free_predictions_sha256"],
        "phasea_completion_sha256":digest_file(Path(pointer["prediction_dir"])/"oracle_free_predictions_complete.json"),
        "holdout_dataset_complete_sha256":digest_file(product/"dataset_complete.json"),
        "blind_example_manifest_sha256":digest_file(product/"manifests"/"example_manifest.csv"),
        "blind_root_manifest_sha256":digest_file(product/"manifests"/"root_manifest.csv"),
        "latent_config_sha256":digest_file(LATENT_CONFIG),
        "confirm_config_sha256":digest_file(CONFIRM_CONFIG),
        "method_bundle_sha256":digest_file(source_dir/"method_bundle.json"),
        "checkpoint_provenance":checkpoints,
      },
      "stage_array_parity":stages,"replay_comparisons":comparisons,
      "example_diagnostics":per_example,
    }
    identity=hashlib.sha256((str(result["provenance"]["phasea_prediction_csv_sha256"])
                              +json.dumps(positions)+str(result["provenance"]["method_bundle_sha256"])
                              +str(result["provenance"]["gpu"])).encode()).hexdigest()[:20]
    parent=args.output_parent.resolve()
    parent.mkdir(parents=True,exist_ok=True)
    out=parent/("replay_"+identity)
    if out.exists(): raise RuntimeError("refusing to overwrite immutable replay diagnostic: "+str(out))
    out.mkdir(parents=True,exist_ok=False)
    result_path=out/"diagnostic_result.json"
    write_json(result_path,result)
    complete_doc={"schema":"triqto.v0_2.step14_cpu_cuda_replay_diagnostic_complete.v1",
                  "diagnostic_dir":str(out),"status":result["status"],
                  "result_sha256":digest_file(result_path),"example_count":len(positions)}
    write_json(out/"diagnostic_complete.json",complete_doc)
    write_json(parent/"current_step15_replay_diagnostic.json",{
        "diagnostic_dir":str(out),"result_sha256":complete_doc["result_sha256"],
        "complete_sha256":digest_file(out/"diagnostic_complete.json"),
        "status":result["status"],
    })
    print(json.dumps({"status":result["status"],"diagnostic_dir":str(out),
           "result_sha256":complete_doc["result_sha256"],
           "stage_array_parity":{k:v.get("exact_equal") for k,v in stages.items()},
           "replay_comparisons":comparisons,
           "scientific_boundary":result["scientific_boundary"]},sort_keys=True),flush=True)

if __name__ == "__main__":
    main()
