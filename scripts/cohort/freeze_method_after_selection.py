#!/usr/bin/env python3
"""Construct a post-SELECTION / pre-TEST method freeze only from complete evidence.

Must be committed to Git before acquiring any real replacement TEST observations.
Cannot be called to bless incomplete or exploratory FIT-only score reports.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def digest(b):return "sha256:"+hashlib.sha256(b).hexdigest()
def file_sha(path):return digest(path.read_bytes())
def build(selection,prelock,repo_root):
    result=json.loads(selection.read_text())
    lock=json.loads(prelock.read_text())
    if result.get("schema")!="triqto.stabilization_method_selection_decision.v1":
        raise ValueError("Not the final method-selection verdict schema")
    if result.get("status")!="COMPLETE_SELECTION_ONLY_NO_TEST_ACCESS":
        raise ValueError("The complete frozen selection is not finished")
    if result.get("families")!=60 or result.get("records")!=2880 or result.get("draws")!=16:
        raise ValueError("Full 60x4x12x16 family design missing")
    if result.get("bootstrap_replicates")!=2000 or result.get("NI_margin_absolute")!=-0.02:
        raise ValueError("Predeclared selection decision rule drift")
    if result.get("multiplicity_adjustment")!="two comparisons, 97.5% Bonferroni intervals":
        raise ValueError("Selected under unapproved multiplicity rule")
    if not result.get("no_confirmatory_TEST_access"):
        raise ValueError("TEST contamination")
    if result.get("pre_selection_lock_sha256")!=file_sha(prelock):
        raise ValueError("Selection outcome not bound to original preselection lock")
    if lock.get("status")!="FROZEN_CANDIDATES_BEFORE_SELECTION":
        raise ValueError("Invalid preselection lock")
    if lock.get("selection_inspected_before_lock",True):
        raise ValueError("Already inspected selection before locking methods")
    candidate=result.get("selected_remedy")
    if candidate not in ("R0","R1","R2"):
        raise ValueError("Invalid selected method")
    qualified=any(result["contrasts"][k]["joint_pass"] for k in ("R1","R2"))
    if candidate=="R0" and qualified:
        raise ValueError("R0 chosen despite evidence of qualifying alternative")
    if candidate!="R0" and not result["contrasts"][candidate]["joint_pass"]:
        raise ValueError("Nonqualifying remedy marked winner")
    protocol=repo_root/"docs/experiments/STABILIZATION_COMPARISON_RULES_FROZEN.md"
    code=repo_root/"scripts/cohort/remedy_feature_transforms.py"
    model=lock["candidate_network_seed_sha256"]
    estimator=digest(json.dumps(lock["fit_covariance_calibration_N"],
                    sort_keys=True,separators=(",",":")).encode())
    lineage=repo_root/"docs/evidence/finite_shot_cohort/FROZEN_MODEL_DATA_LINEAGE_20261025.md"
    if not protocol.exists() or not code.exists() or not lineage.exists():
        raise FileNotFoundError("Missing prospective protocol/implementation/lineage")
    return {
      "schema":"triqto.stabilization_final_method_freeze.v1",
      "status":"FROZEN_BEFORE_ANY_REPLACEMENT_TEST_ACQUISITION",
      "selected_remedy":candidate,
      "selected_alpha":0.0 if candidate=="R0" else lock["candidate_alpha"][candidate],
      "selected_code_sha256":file_sha(code),
      "estimator_coefficients_sha256":estimator,
      "comparison_protocol_sha256":file_sha(protocol),
      "candidate_checkpoint_sha256":digest(json.dumps(model,sort_keys=True,separators=(",",":")).encode()),
      "candidate_checkpoint_individual_sha256":model,
      "source_lineage_report_sha256":file_sha(lineage),
      "pre_selection_lock_sha256":file_sha(prelock),
      "selection_verdict_sha256":file_sha(selection),
      "source_cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
      "replacement_candidate_identity":"sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1",
      "selection_families":60,"selection_injected_examples":2880,
      "accuracy_noninferiority_gate_applied":True,
      "selection_completed_before_freeze":True,
      "confirmation_claim_explicitly_limited_to_checked_training_lineage":True,
      "external_source_exhaustive_independent_attestation":False,
      "full_step15_rank_adapter_evaluated":False,
      "scope":"Step14 candidate-head score-path correction only; original pooling fixed. N shot selection primary; 2N/4N not yet assessed.",
      "test_data_acquired":False,"test_truth_accessed":False,
      "test_measurement_policy":"Must not acquire until this exact freeze has been committed and independent recipient public key custody is verified.",
      "statistics":{
        "R0":result["baseline"]["R0"],
        "R1":result["contrasts"]["R1"],
        "R2":result["contrasts"]["R2"]},
      "scientific_claim":"Frozen local mechanism-head score-path remedy decision using permitted SELECTION; historical source-exhaustiveness and QPU generalization not claimed."
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selection-verdict",type=Path,required=True)
    p.add_argument("--preselection-lock",type=Path,required=True)
    p.add_argument("--repo",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    freeze=build(a.selection_verdict,a.preselection_lock,a.repo)
    a.output.write_text(json.dumps(freeze,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":freeze["status"],"winner":freeze["selected_remedy"],
                      "test_access":False,"full_adapter_claim":False}))
