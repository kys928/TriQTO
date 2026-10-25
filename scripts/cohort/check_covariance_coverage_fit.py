#!/usr/bin/env python3
"""FIT-only covariance predictive coverage diagnostic, no protected or TEST input."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from crossfit_nonlinear_covariance_fit import (
    records,fold,trace,FOLDS,BUDGETS,LOW,HIGH,
)
CUT3=7.814727903
CUT6=12.591587244

def factors(rows,held):
    train=[r for r in rows if r["fold"]!=held]
    fs={}
    for name in ("coordinates","matched"):
        byfamily={}
        for r in train:
            q=trace(r["emp"],name)/max(trace(r["estimated"],name),1e-30)
            byfamily.setdefault(r["fid"],[]).append(q)
        fs[name]=min(HIGH,max(LOW,float(np.median([np.median(x) for x in byfamily.values()]))))
    return fs

def quadratic(z,sigma):
    eig=np.linalg.eigvalsh(sigma)
    tol=max(float(eig[-1])*1e-10,1e-14)
    rank=int(np.sum(eig>tol))
    if rank !=len(eig):
        return None,rank
    return float(z@np.linalg.pinv(sigma,rcond=1e-10)@z),rank

def ci_family(rows,field,seed):
    # Equal-weight family-cluster mean of one-or-three-reference indicators.
    byfam={}
    for v in rows:byfam.setdefault(v["fid"],[]).append(v[field])
    means=np.array([np.mean(x) for x in byfam.values()],float)
    g=np.random.default_rng(seed)
    bs=np.array([g.choice(means,len(means),replace=True).mean() for _ in range(2000)])
    return {"mean":float(means.mean()),"low95":float(np.quantile(bs,.025)),
            "high95":float(np.quantile(bs,.975)),
            "families":len(means)}

def run(corpus):
    all_rows=records(corpus)
    final={"schema":"triqto.fit_covariance_coverage_exploration.v1",
          "family_count":120,"FOLDS":FOLDS,
          "test_or_selection_records_accessed":False,
          "prior_truth_or_oracle_accessed":False,
          "reference_draws":[0,7,15],"evaluation_draws":15,
          "nominal_chisq_thresholds":{"three_d":CUT3,"six_d":CUT6},
          "caveat":"Gaussian/chi-square approximation, not guaranteed for nonlinear normalized geometry",
          "by_budget":{}}
    for bi,shots in enumerate(BUDGETS):
        these=[r for r in all_rows if r["shots"]==shots]
        metrics=[]
        for held in range(FOLDS):
            train_factors=factors(these,held)
            sc=np.diag([np.sqrt(train_factors["coordinates"])]*3+[np.sqrt(train_factors["matched"])]*3)
            for r in these:
                if r["fold"]!=held:continue
                z=r["z_ref"]-r["z_other_mean"]
                for method,est in (
                    ("uncalibrated",r["estimated"]),
                    ("family_crossfit",sc@r["estimated"]@sc)
                ):
                    emp=r["emp"]
                    fr=float(np.linalg.norm(emp-est,"fro")/max(np.linalg.norm(emp,"fro"),1e-12))
                    stats={"fid":r["fid"],"ref":r["ref"],"method":method,"frobenius":fr}
                    for group,sl,cut in (("coords",slice(0,3),CUT3),
                                         ("matched",slice(3,6),CUT3),
                                         ("full",slice(0,6),CUT6)):
                        quad,rank=quadratic(z[sl],(1+1/15)*est[sl,sl])
                        stats[group+"_rank_deficient"]=int(quad is None)
                        stats[group+"_covered"]=(None if quad is None else int(quad<=cut))
                    metrics.append(stats)
        aggregates={}
        for method in ("uncalibrated","family_crossfit"):
            selected=[m for m in metrics if m["method"]==method]
            agg={"matrix_relative_frobenius_median":float(np.median([a["frobenius"] for a in selected]))}
            for g in ("coords","matched","full"):
                valid=[a for a in selected if a[g+"_covered"] is not None]
                agg[g]={"evaluated":len(valid),"rank_deficient":len(selected)-len(valid),
                        "coverage":ci_family(valid,g+"_covered",20261025+bi*13) if valid else None}
            aggregates[method]=agg
        final["by_budget"][str(shots)]=aggregates
    return final

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=run(args.corpus)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({b:{m:{"full_cov":a["full"]["coverage"]["mean"] if a["full"]["coverage"] else None,
       "coords_cov":a["coords"]["coverage"]["mean"] if a["coords"]["coverage"] else None,
       "matched_cov":a["matched"]["coverage"]["mean"] if a["matched"]["coverage"] else None,
       "matrix_err":a["matrix_relative_frobenius_median"]}
       for m,a in x.items()} for b,x in result["by_budget"].items()},indent=2))
