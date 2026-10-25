#!/usr/bin/env python3
"""Read-only inventory of S3 *identity metadata*, with protected-source stop gates.

- S3 LIST of top-level triqto-data namespaces. Never descend inside any
  namespace containing 'reserve', 'outer', 'sealed', 'future', or 'hardware'.
- For safe historical TRAIN/development namespaces, LIST immediate product
  prefixes and examine only allowlisted identity manifest CSVs.
- Never load candidate/label/mechanism predictions or protected data content.
- No S3 writes, no model inference, no QPU calls.
- Zero local intersections are NOT an independent signed attestation.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,re
from pathlib import Path

IDENTITIES=("family_signature_sha256","context_free_signature_sha256",
            "graph_sha256","operation_signature","wire_signature_sha256")
IDENTITY_FILES=("family_manifest.csv","root_manifest.csv",
                "clean_circuit_manifest.csv","bridge_root_manifest.csv")
DENIED=("reserve","simulator_outer","sealed","future_hardware","future_","_outer","outer_",
        "test_truth","test_labels")
MAX_SOURCE_FAMILIES=200
MAX_PRODUCTS_PER_FAMILY=200
MAX_MANIFEST_BYTES=10_000_000
LABEL_COLUMNS=("mechanism","effect_label","truth","strength","affected_qubit",
               "injection_boundary","prediction","logit","target_unitary",
               "statevector","density_matrix","observed_counts","count_histogram")

def digest(raw):
    return "sha256:"+hashlib.sha256(raw).hexdigest()

def denied(path):
    return any(part in path.lower() for part in DENIED)

def check_safe(path):
    if denied(path):raise PermissionError("Refuse protected or outer namespace")
    if not path.startswith("triqto-data/"):raise ValueError("Unexpected object prefix")

def subfolders(s3,bucket,prefix,protected=False):
    if not protected:check_safe(prefix)
    names=[]
    paginator=s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket,Prefix=prefix,Delimiter="/"):
        names.extend(p["Prefix"] for p in page.get("CommonPrefixes",()))
    if len(names)!=len(set(names)):raise RuntimeError("Duplicate S3 prefixes")
    return sorted(names)

def safe_header(s3,bucket,key):
    check_safe(key)
    # One ranged read for CSV HEADER only. No hidden rows or truth are fetched
    # if disallowed columns are present.
    raw=s3.get_object(Bucket=bucket,Key=key,Range="bytes=0-4095")["Body"].read()
    if not raw or b"\n" not in raw:raise ValueError("Unparseable or oversized header")
    names=next(csv.reader([raw.split(b"\n",1)[0].decode("utf-8-sig")]))
    if any(any(bad in col.lower() for bad in LABEL_COLUMNS) for col in names):
        return names,False
    return names,True

def permitted_manifest(s3,bucket,key,filesize):
    if filesize>MAX_MANIFEST_BYTES:return {"state":"TOO_LARGE_FOR_READ_ONLY_AUDIT","bytes":filesize},None
    try:
        head,allowed=safe_header(s3,bucket,key)
    except Exception as ex:
        return {"state":"HEADER_UNVERIFIED","error_type":type(ex).__name__},None
    metadata={"columns":head,"bytes":filesize}
    if not allowed:
        metadata["state"]="LABEL_BEARING_HEADER_REFUSED"
        return metadata,None
    if not any(field in head for field in IDENTITIES):
        metadata["state"]="NO_COMPARABLE_IDENTITY_COLUMNS"
        return metadata,None
    raw=s3.get_object(Bucket=bucket,Key=key)["Body"].read()
    if len(raw)!=filesize:raise RuntimeError("S3 manifest length changed")
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if list(reader.fieldnames)!=head:raise RuntimeError("Manifest header changed")
    ident={field:set() for field in IDENTITIES}
    count=0
    for row in reader:
        count+=1
        for field in IDENTITIES:
            value=(row.get(field) or "").strip()
            if value:ident[field].add(value)
    metadata.update({"state":"SAFE_IDENTITY_ONLY_READ","rows":count,
                     "sha256":digest(raw),
                     "available_identity_fields":[k for k,v in ident.items() if v],
                     "identity_counts":{k:len(v) for k,v in ident.items()}})
    return metadata,ident

def load_candidate(path):
    obj=json.loads(path.read_text())
    if not obj.get("candidates"):raise ValueError("Missing candidate identities")
    ids={f:set() for f in IDENTITIES}
    for row in obj["candidates"]:
        for f in IDENTITIES:
            if row.get(f):ids[f].add(str(row[f]))
    return {"identity":obj["cohort_identity"],"manifest_sha256":digest(path.read_bytes()),
            "rows":len(obj["candidates"])},ids

def audit(s3,bucket,candidate_paths):
    proposals=[load_candidate(p) for p in candidate_paths]
    root=subfolders(s3,bucket,"triqto-data/")
    if len(root)>MAX_SOURCE_FAMILIES:raise RuntimeError("Too many S3 namespaces for bounded listing")
    summary={"schema":"triqto.readonly_s3_identity_namespace_audit.v1",
             "scope":"triqto-data namespace LIST + explicitly safe identity CSV GET",
             "attempted_protected_object_reads":False,"s3_writes":False,
             "all_historical_sources_certified":False,
             "nonmaterialization_certified":False,
             "prior_test_blinding_valid":False,
             "candidate_sets":[{"identity":p[0]["identity"],"candidate_rows":p[0]["rows"],
                                "manifest_sha256":p[0]["manifest_sha256"]} for p in proposals],
             "top_level_namespace_count":len(root),"protected_namespace_present":[],
             "safe_source_namespaces":[],"limited_sources":[],
             "comparison_counts":{p[0]["identity"]:{f:0 for f in IDENTITIES} for p in proposals},
             "coverage_limits":[]}
    for parent in root:
        family=parent[len("triqto-data/"):-1]
        if denied(parent):
            # We do not descend into these names; presence is not evidence
            # that any truth/labels/results inside exist or are accessible.
            summary["protected_namespace_present"].append(family)
            continue
        # Only list namespaces evidently from dataset/training products.
        if not re.match(r"^(step\d+[a-z]?[_-]|step\d+$)",family,re.IGNORECASE):
            summary["limited_sources"].append({"namespace":family,"reason":"NOT_A_STEP_PRODUCT_NAMESPACE"})
            continue
        products=subfolders(s3,bucket,parent)
        if len(products)>MAX_PRODUCTS_PER_FAMILY:
            raise RuntimeError("Too many products within "+family)
        if not products:
            summary["limited_sources"].append({"namespace":family,"reason":"NO_IMMEDIATE_SUBPREFIXES"})
            continue
        source_meta={"namespace":family,"product_prefix_count":len(products),"products":[]}
        for base in products:
            if denied(base):
                summary["protected_namespace_present"].append(base[:-1].split("/")[-1])
                continue
            candidate_keys=[base+"manifests/"+name for name in IDENTITY_FILES]
            prod={"relative_product":base[len(parent):-1],"manifests":[]}
            for key in candidate_keys:
                try:
                    metadata=s3.head_object(Bucket=bucket,Key=key)
                except Exception as ex:
                    code=getattr(ex,"response",{}).get("Error",{}).get("Code","")
                    if code in ("404","NoSuchKey","NotFound"):continue
                    prod["manifests"].append({"manifest_name":key.split("/")[-1],
                                               "state":"HEAD_UNVERIFIED",
                                               "error_type":type(ex).__name__})
                    continue
                result,ident=permitted_manifest(s3,bucket,key,int(metadata["ContentLength"]))
                result["manifest_name"]=key.split("/")[-1]
                if ident is not None:
                    result["overlap_counts"]={}
                    for prop, ids in proposals:
                        hits={f:len(ident[f] & ids[f]) for f in IDENTITIES}
                        result["overlap_counts"][prop["identity"]]=hits
                        for field,n in hits.items():
                            summary["comparison_counts"][prop["identity"]][field]+=n
                prod["manifests"].append(result)
            source_meta["products"].append(prod)
        summary["safe_source_namespaces"].append(source_meta)
    summary["coverage_limits"]=[
      "LIST only triqto-data/; datasets in other buckets or prefixes are not excluded",
      "Protected/reserve/outer namespaces not descended or read",
      "Only manifest CSVs at one product directory depth with approved filenames were scanned",
      "Missing structural fields, absent manifests, unknown identity canonicalizations remain unverified",
      "An absent top-level reserve namespace does not prove reserve nonmaterialization elsewhere",
      "Overlap counts against enumerated hashes do not establish arbitrary circuit/unitary equivalence",
      "A signed independently reviewed source registry is still needed for scientific certification"]
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",action="append",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    import boto3
    from botocore.config import Config
    env=os.environ
    s3=boto3.client("s3",endpoint_url=env["RUNPOD_S3_ENDPOINT"],
         region_name=env["RUNPOD_DATACENTER_ID"].lower(),
         aws_access_key_id=env["RUNPOD_S3_ACCESS_KEY_ID"],
         aws_secret_access_key=env["RUNPOD_S3_SECRET_ACCESS_KEY"],
         config=Config(signature_version="s3v4",connect_timeout=15,read_timeout=25,
                       retries={"max_attempts":2}))
    result=audit(s3,env["RUNPOD_NETWORK_VOLUME_ID"],args.candidate)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    summary={"namespace_count":result["top_level_namespace_count"],
             "scanned_safe_namespaces":len(result["safe_source_namespaces"]),
             "protected_namespace_count":len(result["protected_namespace_present"]),
             "comparisons":result["comparison_counts"],
             "identity_registry_independent_attested":False,
             "reserve_nonmaterialization_certified":False}
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
