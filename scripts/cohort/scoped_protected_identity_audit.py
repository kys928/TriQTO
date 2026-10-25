#!/usr/bin/env python3
"""S3 read-only protected-source identity manifest audit.

New user-authorized identity-only metadata scope; never touch sealed truth,
labels, predictions, raw counts or observational data. Unknown manifests are
explicitly unverified, never assumed absent/disjoint. No S3 writes.

Uses previously saved ID-only candidate exports for old and new cohorts.
"""
from __future__ import annotations
import csv,hashlib,io,json,os,re,argparse
from pathlib import Path

IDENTITY_FIELDS=("family_signature_sha256","context_free_signature_sha256",
 "graph_sha256","operation_signature","wire_signature_sha256")
CANDIDATE_KEYS=("sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
 "sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1")
SAFE_MANIFEST_NAMES=("family_manifest.csv","root_manifest.csv",
                     "clean_circuit_manifest.csv","bridge_root_manifest.csv",
                     "circuit_identity_manifest.csv","identity_manifest.csv")
NEVER_READ_FRAGMENTS=("sealed","truth","label","counts","measurement","prediction","logit",
    "encrypted","secret","key","recipe","observations","raw","diagnostic","calibration","checkpoint")
LABEL_COLUMNS=("mechanism","truth","label","effect","strength","affected","boundary","target",
    "observed","outcome","prediction","logit","count","amplitude","state","angle","salt","seed")
MAX_CSV=10_000_000
MAX_PRODUCTS=300
MAX_NAMESPACES=250
MAX_MANIFESTS_PER_PRODUCT=6

def sha(b):return "sha256:"+hashlib.sha256(b).hexdigest()

def candidate_set(path):
    obj=json.loads(Path(path).read_text())
    if obj.get("cohort_identity") not in CANDIDATE_KEYS:raise ValueError("Unexpected candidate identity")
    if not isinstance(obj.get("candidates"),list) or not obj["candidates"]:raise ValueError("Missing candidates")
    return obj["cohort_identity"],{
       col:{str(row[col]) for row in obj["candidates"] if row.get(col)}
       for col in IDENTITY_FIELDS}

def list_folders(client,bucket,prefix):
    names=[]
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket,Prefix=prefix,Delimiter="/"):
        names.extend(x["Prefix"] for x in page.get("CommonPrefixes",[]))
    return sorted(set(names))

def manifest_key_allowed(key):
    parts=key.split("/")
    return (
        len(parts)>=4 and key.startswith("triqto-data/")
        and parts[-2]=="manifests" and parts[-1] in SAFE_MANIFEST_NAMES
        and not any(any(word in component.lower() for word in NEVER_READ_FRAGMENTS)
                    for component in parts[1:-2])
    )

def classify_header(client,bucket,key):
    # Exactly 256 bytes. If a row starts in this range, it is never parsed,
    # stored, logged, or returned. No data-bearing columns may be read in full.
    chunk=client.get_object(Bucket=bucket,Key=key,Range="bytes=0-255")["Body"].read()
    if b"\n" not in chunk:return {"status":"HEADER_LONG_OR_UNPARSEABLE"},None
    first=chunk.split(b"\n",1)[0]
    try:headers=next(csv.reader([first.decode("utf-8-sig")]))
    except Exception:return {"status":"INVALID_HEADER"},None
    if not headers or len(set(headers))!=len(headers):
        return {"status":"INVALID_OR_DUPLICATE_COLUMNS"},None
    if any(any(token in column.lower() for token in LABEL_COLUMNS) for column in headers):
        return {"status":"REFUSED_LABEL_BEARING_HEADER"},None
    allowed_metadata={"root_id","family_id","clean_circuit_group_id","variant_index",
                      "root_index","variant","n_qubits","topology_class","reference_layers",
                      "partition","dataset_partition","reference_graph_sha256",
                      "family_index","parent_group_id","source_hash"}
    if any(h not in IDENTITY_FIELDS and h not in allowed_metadata for h in headers):
        return {"status":"UNKNOWN_HEADER_COLUMNS_REFUSED"},None
    if not any(x in headers for x in IDENTITY_FIELDS):
        return {"status":"NO_IDENTITY_COLUMNS"},None
    return {"status":"STRICT_IDENTITY_HEADER","identity_fields_present":[x for x in IDENTITY_FIELDS if x in headers]},headers

def safe_compare(client,bucket,key,blob_size,targets):
    if not manifest_key_allowed(key):return {"status":"DENIED_OBJECT_NAME"}
    if blob_size>MAX_CSV:return {"status":"TOO_LARGE"}
    schema,headers=classify_header(client,bucket,key)
    if headers is None:return schema
    raw=client.get_object(Bucket=bucket,Key=key)["Body"].read()
    if len(raw)!=blob_size:raise RuntimeError("Manifest size changed")
    rows=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if list(rows.fieldnames)!=headers:raise RuntimeError("TOCTOU header changed")
    observed={f:set() for f in IDENTITY_FIELDS}
    count=0
    for row in rows:
        count+=1
        for field in IDENTITY_FIELDS:
            if field in headers and row[field]:observed[field].add(str(row[field]).strip())
    return {"status":"IDENTITY_ONLY_COMPARED","sha256":sha(raw),"rows":count,
       "fields_present":[f for f in IDENTITY_FIELDS if f in headers],
       "identity_counts":{f:len(observed[f]) for f in IDENTITY_FIELDS if f in headers},
       "intersections":{tid:{f:len(observed[f] & target[f]) for f in IDENTITY_FIELDS if f in headers}
                         for tid,target in targets.items()}}

def audit(s3,bucket,targets):
    names=list_folders(s3,bucket,"triqto-data/")
    if len(names)>MAX_NAMESPACES:raise RuntimeError("Namespace listing too large")
    out={"schema":"triqto.scoped_protected_identity_only_audit.v1",
      "target_identity_commitments":list(targets),
      "top_level_namespaces_seen":len(names),"reviewed_namespaces":[],
      "candidate_overlap_totals":{k:{f:0 for f in IDENTITY_FIELDS} for k in targets},
      "protected_truth_read":False,"sealed_test_decrypted":False,
      "s3_writes":False,"full_source_coverage_certified":False,
      "externally_attested":False,"reserve_nonmaterialization_proven":False}
    for base in names:
        name=base[len("triqto-data/"):].rstrip("/")
        # Scope focus: only namespaces skipped in prior general survey:
        # possible reserve/outer/future/sealed, plus still-unclassified ones.
        is_protected=any(x in name.lower() for x in ("reserve","future","outer","sealed"))
        is_step=name.lower().startswith("step")
        if not is_protected and is_step:continue
        # Guarded name only, then immediate product prefixes.
        dirs=list_folders(s3,bucket,base)
        if len(dirs)>MAX_PRODUCTS:raise RuntimeError("Too many product dirs")
        parents=[base]+dirs
        entry={"source_namespace":name,"protected_name":is_protected,
               "immediate_product_count":len(dirs),"manifest_statuses":[],
               "source_coverage_complete":False}
        # Need to support root-level manifests and immediate product-level manifests.
        for parent in parents:
            if any(k in parent.lower() for k in ("sealed_truth","secret","labels","predictions",
                                                "model_outputs","raw","observations")):
                entry["manifest_statuses"].append({"source":"HIDDEN_SUBTREE","status":"DENIED_SUBTREE"})
                continue
            for fname in SAFE_MANIFEST_NAMES:
                key=parent+"manifests/"+fname
                if not manifest_key_allowed(key):continue
                try:meta=s3.head_object(Bucket=bucket,Key=key)
                except Exception as ex:
                    status=getattr(ex,"response",{}).get("Error",{}).get("Code","")
                    if status in ("404","NoSuchKey","NotFound"):continue
                    entry["manifest_statuses"].append({"file":fname,"status":"HEAD_FAILED","code":str(status)[:60]})
                    continue
                result=safe_compare(s3,bucket,key,int(meta["ContentLength"]),targets)
                entry["manifest_statuses"].append({"file":fname,"product_index":parents.index(parent),**result})
                for tid,counts in result.get("intersections",{}).items():
                    for field,v in counts.items():out["candidate_overlap_totals"][tid][field]+=v
        out["reviewed_namespaces"].append(entry)
    out["limitations"]=[
      "Source registry incomplete: names outside triqto-data are not surveyed.",
      "Only root/immediate-product manifests of six allowlisted basenames inspected.",
      "Unknown/label-bearing headers refused; no proof of disjointness for missing identities.",
      "Namespace listings and HEAD are metadata only: no proof of no later reserve materialization.",
      "Same research GitHub/S3 connection; not independent source-complete certification.",
      "Any old TEST with public deterministic label mapping remains invalid for blind confirmation."]
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",action="append",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    targets=dict(candidate_set(x) for x in a.candidate)
    if set(targets)!=set(CANDIDATE_KEYS):raise ValueError("Must audit both exact candidate sets")
    import boto3
    from botocore.config import Config
    e=os.environ
    conn=boto3.client("s3",endpoint_url=e["RUNPOD_S3_ENDPOINT"],region_name=e["RUNPOD_DATACENTER_ID"].lower(),
      aws_access_key_id=e["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=e["RUNPOD_S3_SECRET_ACCESS_KEY"],
      config=Config(signature_version="s3v4",connect_timeout=10,read_timeout=20,retries={"max_attempts":2}))
    result=audit(conn,e["RUNPOD_NETWORK_VOLUME_ID"],targets)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    status={}
    for entry in result["reviewed_namespaces"]:
        for x in entry["manifest_statuses"]:status[x["status"]]=status.get(x["status"],0)+1
    print(json.dumps({"namespaces_seen":result["top_level_namespaces_seen"],
        "newly_inspected_namespace_count":len(result["reviewed_namespaces"]),
        "protected_namespace_count":sum(x["protected_name"] for x in result["reviewed_namespaces"]),
        "manifest_status_counts":status,"overlap_totals":result["candidate_overlap_totals"],
        "externally_attested":False,"source_coverage_complete":False},indent=2))
if __name__=="__main__":main()
