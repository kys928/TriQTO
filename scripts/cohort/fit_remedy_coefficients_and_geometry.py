#!/usr/bin/env python3
"""FIT-only calibration coefficients and R0/R1/R2 geometry sanity pilot.

This is a diagnostic of generated candidate feature transformations, NOT a
network/model correctness study. It never reads SELECTION/TEST/truth or any
protected source; the checkpoint and pooling remain untouched.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from crossfit_nonlinear_covariance_fit import records,BUDGETS,trace,LOW,HIGH
from remedy_feature_transforms import apply,calibrated_covariance,ALPHA_R1,ALPHA_R2

def aggregate(r):
    by_family={}
    for row in r:
        f=row["fid"];d=by_family.setdefault(f,{"coordinates":[],"matched":[]})
        for group in ("coordinates","matched"):
            d[group].append(trace(row["emp"],group)/max(trace(row["estimated"],group),1e-30))
    return {name:float(np.clip(np.median([np.median(x[name]) for x in by_family.values()]),LOW,HIGH))
            for name in ("coordinates","matched")}

def feature6_to24(z):
    c=z[:3];m=z[3:]
    return np.r_[c,np.abs(c),c/max(float(np.linalg.norm(c)),1e-12),
                 m,np.abs(m),np.zeros(9,dtype=float)]

def run(corpus):
    all_rows=records(corpus)
    output={"schema":"triqto.fit_only_observed_count_remedy_calibration.v1",
            "source":"immutable 240-family cohort FIT partition only; 120 families; first root/candidate/observation",
            "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
            "covariance_procedure":"192 resamples of same N-shot observed counts; medians across 3 anchors per family then 120 families",
            "scale_limits":[LOW,HIGH],"alpha_R1":ALPHA_R1,"alpha_R2":ALPHA_R2,
            "source_sample_reused_for_coefficients":True,
            "labels_read":False,"model_network_scored":False,"selection_or_test_opened":False,
            "by_budget":{}}
    for shot in BUDGETS:
        rows=[x for x in all_rows if x["shots"]==shot]
        factors=aggregate(rows)
        stats={v:{"rms_feature_shift":[],"changed_fraction":0,"consistency_failures":0}
               for v in ("R0","R1","R2")}
        for row in rows:
            feature=feature6_to24(row["z_ref"])
            cov=calibrated_covariance(row["estimated"],factors["coordinates"],factors["matched"])
            for key,details in stats.items():
                result=apply(feature,cov,key)
                details["rms_feature_shift"].append(float(np.linalg.norm(result[:15]-feature[:15])/np.sqrt(15)))
                details["changed_fraction"]+=int(np.linalg.norm(result[:15]-feature[:15])>1e-10)
                if not np.isfinite(result).all():details["consistency_failures"]+=1
        output["by_budget"][str(shot)]={"factors":factors,
            "n_reference_predictions":len(rows),
            "feature_intervention":{key:{
              "median_rms_feature_shift":float(np.median(val["rms_feature_shift"])),
              "p90_rms_feature_shift":float(np.quantile(val["rms_feature_shift"],.9)),
              "changed_reference_count":val["changed_fraction"],
              "nonfinite_count":val["consistency_failures"]} for key,val in stats.items()}}
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();result=run(a.corpus)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["by_budget"],indent=2))
