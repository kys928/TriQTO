#!/usr/bin/env python3
"""Independent, read-only adversarial audit of the frozen Step-14 development cohort.

The audit reads the RunPod Network Volume through its S3-compatible endpoint.
It never writes to the volume and never materializes Step-14 outer/reserve data.
A machine-readable verdict and human-readable report are written only to the
local GitHub Actions workspace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
from botocore.config import Config
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRODUCT_ID = "development_b087edfd6629ac250299391d"
DEFAULT_PREFIX = f"triqto-data/step14_cross_motif_dataset/{PRODUCT_ID}/"
POINTER_KEY = "triqto-data/step14_cross_motif_dataset/current_development_product.json"
EXPECTED_COUNTS = {"families": 750, "roots": 3000, "examples": 39000, "fit_families": 600, "selection_families": 150}
EXPECTED_MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
FORBIDDEN_PRODUCT_TOKENS = ("statevector", "prediction", "logit", "embedding", "checkpoint", "simulator_outer", "future_hardware_reserve")
FORBIDDEN_INPUT_TOKENS = ("mechanism", "effect", "phenotype", "affected", "boundary", "strength", "target", "label", "split", "family_id", "root_index", "example_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--output-dir", type=Path, default=Path("audit_artifacts/step14_deep"))
    p.add_argument("--fit-sample-roots", type=int, default=48)
    p.add_argument("--selection-sample-roots", type=int, default=24)
    return p.parse_args()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def s3_client():
    dc = required_env("RUNPOD_DATACENTER_ID")
    return boto3.client(
        "s3",
        endpoint_url=required_env("RUNPOD_S3_ENDPOINT"),
        region_name=dc.lower(),
        aws_access_key_id=required_env("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=30, read_timeout=120),
    )


def bucket() -> str:
    return required_env("RUNPOD_NETWORK_VOLUME_ID")


def get_bytes(client, key: str) -> bytes:
    return client.get_object(Bucket=bucket(), Key=key)["Body"].read()


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_json_bytes(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def load_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    return [dict(row) for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))]


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}: return True
    if text in {"0", "false", "no"}: return False
    raise ValueError(f"cannot parse boolean {value!r}")


def finite_number(value: Any) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


def js_divergence(a: Counter[str], b: Counter[str]) -> float:
    keys = sorted(set(a) | set(b)); sa, sb = sum(a.values()), sum(b.values())
    if not keys or not sa or not sb: return float("nan")
    pa = np.asarray([a[k] / sa for k in keys], dtype=float); pb = np.asarray([b[k] / sb for k in keys], dtype=float)
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def canonical_row(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in row.items()))


def stratified_roots(roots: list[dict[str, str]], partition: str, count: int) -> list[int]:
    candidates = sorted(int(r["root_index"]) for r in roots if r["step14_partition"] == partition)
    if count >= len(candidates): return candidates
    idx = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[i] for i in idx.tolist()]


def report_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any, severity: str = "hard") -> None:
    checks.append({"name": name, "passed": bool(passed), "severity": severity, "detail": detail})
    icon = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
    print(f"[{icon}] {name}: {detail}", flush=True)


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    client = s3_client(); prefix = args.prefix.rstrip("/") + "/"
    checks: list[dict[str, Any]] = []; warnings: list[str] = []; hard_failures: list[str] = []

    print("TRIQTO STEP-14 DEEP DEVELOPMENT COHORT AUDIT — READ ONLY", flush=True)
    print("Product:", prefix, flush=True)

    # Full object inventory, not the earlier MaxKeys=1000 listing.
    objects: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for item in page.get("Contents", []):
            objects.append({"key": item["Key"], "size": int(item.get("Size", 0)), "etag": str(item.get("ETag", ""))})
    keys = [o["key"] for o in objects]
    suspicious_names = [k for k in keys if any(tok in k.lower() for tok in FORBIDDEN_PRODUCT_TOKENS)]
    report_check(checks, "full_product_inventory", len(objects) > 39000, {"objects": len(objects), "bytes": sum(o["size"] for o in objects)})
    report_check(checks, "forbidden_product_artifact_names", not suspicious_names, suspicious_names[:20])

    required = {
        "pointer": POINTER_KEY,
        "complete": prefix + "dataset_complete.json",
        "eda": prefix + "eda.json",
        "families": prefix + "manifests/family_manifest.csv",
        "roots": prefix + "manifests/root_manifest.csv",
        "examples": prefix + "manifests/example_manifest.csv",
    }
    raw = {name: get_bytes(client, key) for name, key in required.items()}
    pointer = load_json_bytes(raw["pointer"]); complete = load_json_bytes(raw["complete"]); eda = load_json_bytes(raw["eda"])
    families = load_csv_bytes(raw["families"]); roots = load_csv_bytes(raw["roots"]); examples = load_csv_bytes(raw["examples"])

    report_check(checks, "pointer_product_id", pointer.get("product_id") == PRODUCT_ID, pointer.get("product_id"))
    report_check(checks, "pointer_completion_hash", pointer.get("dataset_complete_sha256") == sha256_bytes(raw["complete"]), sha256_bytes(raw["complete"]))
    expected_hashes = complete.get("manifest_hashes", {})
    manifest_hashes = {"family_manifest.csv": sha256_bytes(raw["families"]), "root_manifest.csv": sha256_bytes(raw["roots"]), "example_manifest.csv": sha256_bytes(raw["examples"])}
    report_check(checks, "manifest_hash_chain", manifest_hashes == expected_hashes, manifest_hashes)
    report_check(checks, "eda_hash_chain", complete.get("eda_sha256") == sha256_bytes(raw["eda"]), sha256_bytes(raw["eda"]))
    report_check(checks, "frozen_boundary_flags", complete.get("status") == "COMPLETE_FROZEN_DEVELOPMENT" and not bool(complete.get("model_evaluated_before_freeze")) and not bool(complete.get("qpu_executed")) and not bool(complete.get("future_hardware_reserve_materialized")), {k: complete.get(k) for k in ("status","model_evaluated_before_freeze","qpu_executed","future_hardware_reserve_materialized")})

    report_check(checks, "manifest_counts", (len(families), len(roots), len(examples)) == (750,3000,39000), {"families":len(families),"roots":len(roots),"examples":len(examples)})
    for name, rows in (("family",families),("root",roots),("example",examples)):
        report_check(checks, f"{name}_exact_row_duplicates", len({canonical_row(r) for r in rows}) == len(rows), len(rows)-len({canonical_row(r) for r in rows}))

    # ID uniqueness and referential integrity.
    family_ids = [r["family_id"] for r in families]; family_indices = [int(r["family_index"]) for r in families]
    root_indices = [int(r["root_index"]) for r in roots]; example_ids = [r["example_id"] for r in examples]
    report_check(checks, "family_id_uniqueness", len(set(family_ids)) == len(family_ids), len(set(family_ids)))
    report_check(checks, "family_index_uniqueness", len(set(family_indices)) == len(family_indices), len(set(family_indices)))
    report_check(checks, "root_index_uniqueness", len(set(root_indices)) == len(root_indices), len(set(root_indices)))
    report_check(checks, "example_id_uniqueness", len(set(example_ids)) == len(example_ids), len(set(example_ids)))
    famset, rootset = set(family_ids), set(root_indices)
    report_check(checks, "root_to_family_references", all(r["family_id"] in famset for r in roots), sum(r["family_id"] not in famset for r in roots))
    report_check(checks, "example_to_family_references", all(r["family_id"] in famset for r in examples), sum(r["family_id"] not in famset for r in examples))
    report_check(checks, "example_to_root_references", all(int(r["root_index"]) in rootset for r in examples), sum(int(r["root_index"]) not in rootset for r in examples))

    fam_by_id = {r["family_id"]: r for r in families}; root_by_idx = {int(r["root_index"]):r for r in roots}
    roots_by_family: dict[str,list[dict[str,str]]] = defaultdict(list)
    for r in roots: roots_by_family[r["family_id"]].append(r)
    by_root: dict[int,list[dict[str,str]]] = defaultdict(list)
    for r in examples: by_root[int(r["root_index"])].append(r)
    report_check(checks, "four_roots_per_family", all(len(v)==4 for v in roots_by_family.values()) and len(roots_by_family)==750, Counter(len(v) for v in roots_by_family.values()))
    report_check(checks, "thirteen_examples_per_root", all(len(v)==13 for v in by_root.values()) and len(by_root)==3000, Counter(len(v) for v in by_root.values()))

    # Split isolation at every available lineage level.
    fp = Counter(r["step14_partition"] for r in families); rp = Counter(r["step14_partition"] for r in roots); ep = Counter(r["step14_partition"] for r in examples)
    report_check(checks, "partition_counts", fp==Counter({"fit":600,"selection":150}) and rp==Counter({"fit":2400,"selection":600}) and ep==Counter({"fit":31200,"selection":7800}), {"families":dict(fp),"roots":dict(rp),"examples":dict(ep)})
    allowed = {"fit","selection"}
    report_check(checks, "no_outer_or_reserve_rows", set(fp)|set(rp)|set(ep) <= allowed, sorted(set(fp)|set(rp)|set(ep)))
    fit_fam = {r["family_id"] for r in families if r["step14_partition"]=="fit"}; sel_fam = {r["family_id"] for r in families if r["step14_partition"]=="selection"}
    fit_sig = {r["family_signature_sha256"] for r in families if r["step14_partition"]=="fit"}; sel_sig = {r["family_signature_sha256"] for r in families if r["step14_partition"]=="selection"}
    fit_graph = {r["graph_sha256"] for r in roots if r["step14_partition"]=="fit"}; sel_graph = {r["graph_sha256"] for r in roots if r["step14_partition"]=="selection"}
    report_check(checks, "family_split_disjoint", not (fit_fam & sel_fam), len(fit_fam & sel_fam))
    report_check(checks, "family_signature_split_disjoint", not (fit_sig & sel_sig), len(fit_sig & sel_sig))
    report_check(checks, "root_graph_split_disjoint", not (fit_graph & sel_graph), len(fit_graph & sel_graph))
    report_check(checks, "global_family_signature_uniqueness", len({r["family_signature_sha256"] for r in families})==750, len({r["family_signature_sha256"] for r in families}))
    report_check(checks, "global_root_graph_uniqueness", len({r["graph_sha256"] for r in roots})==3000, len({r["graph_sha256"] for r in roots}))

    # Near-duplicate structure audit: same operation signature is allowed but documented.
    fit_ops = {r.get("operation_signature","") for r in roots if r["step14_partition"]=="fit"}; sel_ops = {r.get("operation_signature","") for r in roots if r["step14_partition"]=="selection"}
    op_overlap = fit_ops & sel_ops
    report_check(checks, "operation_signature_cross_split_overlap", not op_overlap, {"overlap_count":len(op_overlap)}, severity="warning")
    if op_overlap: warnings.append(f"{len(op_overlap)} parameter-free operation signatures occur in both fit and selection; graph hashes and family signatures remain disjoint.")

    # Within-root crossing proves bookkeeping metadata cannot identify mechanism.
    crossing_bad = 0; metadata_bad = 0
    direct_target_cols = {"mechanism","effect_present","mechanism_loss_mask","phenomenology","clean_control","example_id","artifact_path","artifact_sha256"}
    for root, rows in by_root.items():
        clean = [r for r in rows if r["mechanism"]=="clean_control"]; distorted = [r for r in rows if r["mechanism"]!="clean_control"]
        counts = Counter(r["mechanism"] for r in distorted)
        if len(clean)!=1 or any(counts[m]!=4 for m in EXPECTED_MECHANISMS): crossing_bad += 1
        groups: dict[tuple[tuple[str,str],...],set[str]] = defaultdict(set)
        for row in distorted:
            sig = tuple(sorted((k,str(v)) for k,v in row.items() if k not in direct_target_cols))
            groups[sig].add(row["mechanism"])
        if len(groups)!=4 or any(v!=set(EXPECTED_MECHANISMS) for v in groups.values()): metadata_bad += 1
    report_check(checks, "mechanism_strength_crossing_every_root", crossing_bad==0, crossing_bad)
    report_check(checks, "full_bookkeeping_signature_mechanism_independence", metadata_bad==0, metadata_bad)

    # Values, labels, ranges and acquisition schedules.
    missing = sum(any(v is None or str(v).strip()=="" for v in r.values()) for r in families+roots+examples)
    report_check(checks, "no_missing_manifest_cells", missing==0, missing)
    numeric_cols = {"family_index","candidate_seed_offset","n_qubits","identifiability_min_delta_norm","identifiability_min_pairwise_distance"}
    nonfinite = sum(not finite_number(r[c]) for r in families for c in numeric_cols if c in r)
    nonfinite += sum(not finite_number(r[c]) for r in roots for c in ("root_index","n_qubits","affected_qubit","injection_boundary_rank") if c in r)
    nonfinite += sum(not finite_number(r[c]) for r in examples for c in ("root_index","n_qubits","affected_qubit","insertion_boundary_rank","strength","shots") if c in r)
    report_check(checks, "finite_numeric_manifest_values", nonfinite==0, nonfinite)
    report_check(checks, "identifiability_admission_all_pass", all(r.get("identifiability_status")=="PASS" for r in families), Counter(r.get("identifiability_status") for r in families))
    bad_clean = sum(not (as_bool(r["clean_control"]) and float(r["strength"])==0.0 and r["mechanism"]=="clean_control" and not as_bool(r["effect_present"]) and not as_bool(r["mechanism_loss_mask"])) for r in examples if r["mechanism"]=="clean_control")
    bad_injected = sum(float(r["strength"])<=0.0 or as_bool(r["clean_control"]) for r in examples if r["mechanism"]!="clean_control")
    report_check(checks, "clean_control_semantics", bad_clean==0, bad_clean)
    report_check(checks, "injected_row_semantics", bad_injected==0, bad_injected)

    # Coverage / support and fit-selection comparability.
    coarse = lambda r: (r["n_qubits"], r["topology_class"], r["injection_context_class"])
    fit_strata = {coarse(r) for r in families if r["step14_partition"]=="fit"}; sel_strata = {coarse(r) for r in families if r["step14_partition"]=="selection"}
    unsupported = sorted(sel_strata-fit_strata)
    report_check(checks, "selection_coarse_strata_supported_by_fit", not unsupported, unsupported)
    js = {}
    for col in ("n_qubits","topology_class","injection_context_class"):
        fa = Counter(r[col] for r in families if r["step14_partition"]=="fit"); sb = Counter(r[col] for r in families if r["step14_partition"]=="selection")
        js[col] = js_divergence(fa,sb)
    report_check(checks, "fit_selection_marginal_js_divergence", all(v<=0.10 for v in js.values()), js, severity="warning")
    if any(v>0.10 for v in js.values()): warnings.append(f"Fit/selection marginal JSD exceeds 0.10: {js}")

    # Static feature-pipeline audit: actual model batch is assembled only from x__ arrays;
    # y__ arrays become Step7Targets, and manifest metadata is not passed into the batch.
    repo = Path(__file__).resolve().parents[2]
    adapter_text = (repo/"src/triqto/step7/graph_adapter.py").read_text(encoding="utf-8")
    training_text = (repo/"scripts/v0_2/run_step14_cross_motif_training.py").read_text(encoding="utf-8")
    x_keys = sorted(set(re.findall(r'"(x__[A-Za-z0-9_]+)"', adapter_text)))
    y_keys = sorted(set(re.findall(r'"(y__[A-Za-z0-9_]+)"', adapter_text)))
    suspicious_inputs = [k for k in x_keys if any(tok in k.lower() for tok in FORBIDDEN_INPUT_TOKENS)]
    pipeline_contract = ("batch_from_step5_examples(examples" in training_text and "smoke_runner.load_example(product, row)" in training_text and {"y__effect_present_target","y__mechanism_target","y__mechanism_loss_mask"} <= set(y_keys))
    report_check(checks, "static_model_input_key_audit", not suspicious_inputs and pipeline_contract, {"x_keys":x_keys,"y_keys":y_keys,"suspicious_x_keys":suspicious_inputs})

    # Metadata-only shortcut probe, trained on fit and tested on selection.
    distorted = [r for r in examples if r["mechanism"] in EXPECTED_MECHANISMS]
    feat_cols = ["n_qubits","affected_qubit","insertion_boundary_rank","insertion_depth_bin","strength","shots","reference_kind","backend_identity","physical_layout_identity","variant_index","topology_class","injection_context_class"]
    available = [c for c in feat_cols if c in distorted[0]]
    categorical = [c for c in available if c not in {"n_qubits","affected_qubit","insertion_boundary_rank","strength","shots","variant_index"}]
    numerical = [c for c in available if c not in categorical]
    def X(rows): return [[r[c] if c in categorical else float(r[c]) for c in available] for r in rows]
    fit_rows = [r for r in distorted if r["step14_partition"]=="fit"]; sel_rows = [r for r in distorted if r["step14_partition"]=="selection"]
    yfit = np.asarray([EXPECTED_MECHANISMS.index(r["mechanism"]) for r in fit_rows]); ysel=np.asarray([EXPECTED_MECHANISMS.index(r["mechanism"]) for r in sel_rows])
    cat_idx=[available.index(c) for c in categorical]; num_idx=[available.index(c) for c in numerical]
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat_idx),("num",StandardScaler(),num_idx)])
    clf=Pipeline([("prep",prep),("clf",LogisticRegression(max_iter=400,solver="lbfgs"))]); clf.fit(X(fit_rows),yfit)
    meta_ba=float(balanced_accuracy_score(ysel,clf.predict(X(sel_rows))))
    rng=np.random.default_rng(140014); yperm=yfit.copy(); rng.shuffle(yperm)
    pclf=Pipeline([("prep",prep),("clf",LogisticRegression(max_iter=400,solver="lbfgs"))]); pclf.fit(X(fit_rows),yperm)
    shuffled_meta_ba=float(balanced_accuracy_score(ysel,pclf.predict(X(sel_rows))))
    report_check(checks, "metadata_only_mechanism_probe", meta_ba<=0.40, {"balanced_accuracy":meta_ba,"chance":1/3,"columns":available})
    report_check(checks, "metadata_shuffled_label_negative_control", shuffled_meta_ba<=0.40, {"balanced_accuracy":shuffled_meta_ba,"chance":1/3})

    # Deterministic stratified raw-artifact sample: hash, key namespace, finiteness and target consistency.
    selected_root_ids = set(stratified_roots(roots,"fit",args.fit_sample_roots)+stratified_roots(roots,"selection",args.selection_sample_roots))
    sample_rows=[r for r in examples if int(r["root_index"]) in selected_root_ids]
    print(f"Auditing {len(sample_rows)} raw NPZ artifacts across {len(selected_root_ids)} roots...",flush=True)
    def inspect(row: dict[str,str]) -> dict[str,Any]:
        key=prefix+row["artifact_path"]; blob=get_bytes(s3_client(),key); result={"row":row,"hash_ok":sha256_bytes(blob)==row["artifact_sha256"],"bad_finite":[],"forbidden_keys":[],"keys":[],"feature":None}
        with np.load(io.BytesIO(blob),allow_pickle=False) as z:
            ks=list(z.files); result["keys"]=ks
            result["forbidden_keys"]=[k for k in ks if k.startswith("x__") and any(tok in k.lower() for tok in FORBIDDEN_INPUT_TOKENS)] + [k for k in ks if any(tok in k.lower() for tok in ("statevector","prediction","logit","embedding"))]
            for k in ks:
                a=np.asarray(z[k])
                if k.startswith("x__") and np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all(): result["bad_finite"].append(k)
            effect=bool(np.asarray(z["y__effect_present_target"]).reshape(-1)[0]); mask=bool(np.asarray(z["y__mechanism_loss_mask"]).reshape(-1)[0]); mech=int(np.asarray(z["y__mechanism_target"]).reshape(-1)[0])
            result["target"]=(effect,mask,mech)
            feats=[]
            for k in ("x__delta_local_expectations","x__delta_pairwise_correlations","x__delta_global_parity"):
                a=np.asarray(z[k],dtype=float).reshape(-1); feats.extend([float(a.mean()),float(a.std()),float(np.max(np.abs(a)))])
            result["feature"]=feats
        return result
    inspected=[]
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs=[pool.submit(inspect,r) for r in sample_rows]
        for i,f in enumerate(as_completed(futs),1):
            inspected.append(f.result())
            if i%200==0: print(f"  inspected {i}/{len(futs)} artifacts",flush=True)
    hash_bad=sum(not r["hash_ok"] for r in inspected); finite_bad=sum(bool(r["bad_finite"]) for r in inspected); forbidden_bad=sum(bool(r["forbidden_keys"]) for r in inspected)
    report_check(checks,"sampled_artifact_hashes",hash_bad==0,{"sampled":len(inspected),"mismatches":hash_bad})
    report_check(checks,"sampled_artifact_input_finiteness",finite_bad==0,finite_bad)
    report_check(checks,"sampled_artifact_forbidden_model_input_keys",forbidden_bad==0,forbidden_bad)
    target_bad=0; mechanism_code_map: dict[str,set[int]]=defaultdict(set); clean_codes=set()
    for item in inspected:
        row=item["row"]; effect,mask,code=item["target"]
        if effect!=as_bool(row["effect_present"]) or mask!=as_bool(row["mechanism_loss_mask"]): target_bad+=1
        if row["mechanism"]=="clean_control": clean_codes.add(code)
        else: mechanism_code_map[row["mechanism"]].add(code)
    code_values={next(iter(v)) for v in mechanism_code_map.values() if len(v)==1}
    code_ok=all(len(mechanism_code_map[m])==1 for m in EXPECTED_MECHANISMS) and len(code_values)==3 and clean_codes=={-1}
    report_check(checks,"sampled_manifest_target_consistency",target_bad==0,target_bad)
    report_check(checks,"sampled_mechanism_code_bijection",code_ok,{k:sorted(v) for k,v in mechanism_code_map.items()}|{"clean":sorted(clean_codes)})

    # Legitimate diagnostic-feature shuffled-label negative control on the raw artifact sample.
    sample_dist=[i for i in inspected if i["row"]["mechanism"] in EXPECTED_MECHANISMS]
    xf=np.asarray([i["feature"] for i in sample_dist if i["row"]["step14_partition"]=="fit"],dtype=float); yf=np.asarray([EXPECTED_MECHANISMS.index(i["row"]["mechanism"]) for i in sample_dist if i["row"]["step14_partition"]=="fit"])
    xs=np.asarray([i["feature"] for i in sample_dist if i["row"]["step14_partition"]=="selection"],dtype=float); ys=np.asarray([EXPECTED_MECHANISMS.index(i["row"]["mechanism"]) for i in sample_dist if i["row"]["step14_partition"]=="selection"])
    shuffled_scores=[]; rng=np.random.default_rng(140015)
    for _ in range(10):
        yp=yf.copy(); rng.shuffle(yp); c=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=300))]); c.fit(xf,yp); shuffled_scores.append(float(balanced_accuracy_score(ys,c.predict(xs))))
    report_check(checks,"diagnostic_feature_shuffled_label_negative_control",float(np.mean(shuffled_scores))<=0.40,{"mean_balanced_accuracy":float(np.mean(shuffled_scores)),"max":float(np.max(shuffled_scores)),"repeats":10,"chance":1/3})

    # Consolidate verdict. Warnings do not erase a clean hard-gate result.
    for c in checks:
        if not c["passed"]:
            if c["severity"]=="hard": hard_failures.append(c["name"])
            elif c["name"] not in warnings: warnings.append(c["name"])
    verdict="FAIL" if hard_failures else ("PASS WITH DOCUMENTED LIMITATIONS" if warnings else "PASS")
    result={"schema":"triqto.v0_2.step14_development_deep_audit.v1","product_id":PRODUCT_ID,"product_prefix":prefix,"verdict":verdict,"training_safe":not hard_failures,"hard_failures":hard_failures,"warnings":warnings,"checks":checks,"inventory":{"object_count":len(objects),"total_bytes":sum(o["size"] for o in objects)},"distribution":{"family_partitions":dict(fp),"root_partitions":dict(rp),"example_partitions":dict(ep),"fit_selection_js":js,"unsupported_selection_strata":unsupported},"shortcut_probes":{"metadata_mechanism_ba":meta_ba,"metadata_shuffled_ba":shuffled_meta_ba,"diagnostic_shuffled_scores":shuffled_scores},"artifact_sample":{"roots":len(selected_root_ids),"examples":len(inspected),"fit_roots":args.fit_sample_roots,"selection_roots":args.selection_sample_roots},"manifest_hashes":manifest_hashes,"dataset_complete_sha256":sha256_bytes(raw["complete"]),"eda_sha256":sha256_bytes(raw["eda"])}
    (args.output_dir/"step14_deep_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# TriQTO Step-14 Deep Development Cohort Audit","",f"**Verdict: {verdict}**",f"**Training safe: {not hard_failures}**","",f"Product: `{PRODUCT_ID}`","",f"Objects inventoried: {len(objects):,}",f"Raw artifacts sampled: {len(inspected):,} across {len(selected_root_ids)} roots","", "## Checks",""]
    for c in checks: lines.append(f"- {'PASS' if c['passed'] else ('WARN' if c['severity']=='warning' else 'FAIL')}: **{c['name']}** — `{c['detail']}`")
    if warnings: lines += ["","## Documented limitations",""]+[f"- {w}" for w in warnings]
    if hard_failures: lines += ["","## Hard failures",""]+[f"- {w}" for w in hard_failures]
    (args.output_dir/"step14_deep_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\nFINAL VERDICT:",verdict,flush=True); print("TRAINING SAFE:",not hard_failures,flush=True)


if __name__ == "__main__":
    main()
