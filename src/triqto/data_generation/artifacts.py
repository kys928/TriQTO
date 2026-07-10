"""Artifact writer for Phase 7 generated datasets."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
from qiskit import qpy
from triqto.storage import ManifestWriter
from .records import DatasetGenerationResult, DatasetWriteResult
from .specs import config_to_dict

KNOWN=["generation_config.json","dataset_summary.json","manifests/sample_manifest.parquet","manifests/circuit_manifest.parquet","manifests/simulation_manifest.parquet","manifests/distortion_manifest.parquet","manifests/metric_manifest.parquet"]

def _write_json(path: Path, obj: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite: raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n")

def _ref(path: str) -> str:
    if Path(path).is_absolute(): raise ValueError("artifact reference must be relative")
    return Path(path).as_posix()

def _with_refs(result: DatasetGenerationResult):
    crecs=[]; srecs=[]
    for r in result.circuit_records:
        md=dict(r.metadata); md["artifact_ref"]=_ref(f"artifacts/circuits/{r.circuit_id}.qpy"); crecs.append(replace(r, metadata=md))
    for r in result.simulation_records:
        if r.simulation_mode=="ideal_statevector":
            sv=r.metadata.get("statevector_ref"); pr=r.metadata.get("probabilities_ref")
            srecs.append(replace(r, statevector_ref=sv, probabilities_ref=pr, metadata={k:v for k,v in r.metadata.items() if k not in {"statevector_ref","probabilities_ref"}}))
        elif r.simulation_mode=="ideal_shot":
            cr=r.metadata.get("counts_ref")
            srecs.append(replace(r, counts_ref=cr, metadata={k:v for k,v in r.metadata.items() if k!="counts_ref"}))
        else: srecs.append(r)
    return crecs, srecs

def write_dataset(result: DatasetGenerationResult, output_root: str | Path, *, overwrite: bool=False) -> DatasetWriteResult:
    root=Path(output_root)
    planned=[root/p for p in KNOWN]
    planned += [root/"artifacts/circuits"/f"{r.circuit_id}.qpy" for r in result.circuit_records]
    planned += [root/"artifacts/probabilities"/f"{s.clean_run_id}.json" for s in result.samples]
    planned += [root/"artifacts/probabilities"/f"{s.distorted_run_id}.json" for s in result.samples]
    if result.config.store_statevectors:
        planned += [root/"artifacts/statevectors"/f"{s.clean_run_id}.npy" for s in result.samples]
        planned += [root/"artifacts/statevectors"/f"{s.distorted_run_id}.npy" for s in result.samples]
    if any(s.clean_shot_result for s in result.samples):
        for rec in result.simulation_records:
            if rec.simulation_mode=="ideal_shot": planned.append(root/"artifacts/counts"/f"{rec.run_id}.json")
    for p in set(planned):
        if p.exists() and not overwrite: raise FileExistsError(str(p))
    written=[]
    _write_json(root/"generation_config.json", config_to_dict(result.config), overwrite); written.append(root/"generation_config.json")
    _write_json(root/"dataset_summary.json", result.summary, overwrite); written.append(root/"dataset_summary.json")
    # circuits
    by_circuit={s.clean_circuit_id:s.clean_circuit for s in result.samples}; by_circuit.update({s.distorted_circuit_id:s.distorted_circuit for s in result.samples})
    for cid,circ in sorted(by_circuit.items()):
        p=root/"artifacts/circuits"/f"{cid}.qpy"; p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not overwrite: raise FileExistsError(str(p))
        with p.open("wb") as fh: qpy.dump(circ, fh)
        written.append(p)
    for s in result.samples:
        for rid, probs in [(s.clean_run_id,s.clean_result.probabilities),(s.distorted_run_id,s.distorted_result.probabilities)]:
            p=root/"artifacts/probabilities"/f"{rid}.json"; 
            if not p.exists() or overwrite: _write_json(p, probs, True); written.append(p)
        if result.config.store_statevectors:
            for rid, sv in [(s.clean_run_id,s.clean_result.statevector),(s.distorted_run_id,s.distorted_result.statevector)]:
                p=root/"artifacts/statevectors"/f"{rid}.npy"; p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists() or overwrite: np.save(p, np.asarray(sv.data)); written.append(p)
    for rec in result.simulation_records:
        if rec.simulation_mode=="ideal_shot":
            sample=next(x for x in result.samples if x.clean_circuit_id==rec.circuit_id or x.distorted_circuit_id==rec.circuit_id)
            counts = sample.clean_shot_result.counts if rec.circuit_id==sample.clean_circuit_id else sample.distorted_shot_result.counts
            p=root/"artifacts/counts"/f"{rec.run_id}.json"; _write_json(p, counts, overwrite); written.append(p)
    crecs,srecs=_with_refs(result)
    mw=ManifestWriter(root/"manifests")
    manifests={
      "sample_manifest": mw.write_records("sample_manifest", result.sample_records, overwrite=overwrite),
      "circuit_manifest": mw.write_records("circuit_manifest", crecs, overwrite=overwrite),
      "simulation_manifest": mw.write_records("simulation_manifest", srecs, overwrite=overwrite),
      "distortion_manifest": mw.write_records("distortion_manifest", result.distortion_records, overwrite=overwrite),
      "metric_manifest": mw.write_records("metric_manifest", result.metric_records, overwrite=overwrite),
    }
    written.extend(manifests.values())
    for rec in crecs:
        assert (root/rec.metadata["artifact_ref"]).exists()
    for rec in srecs:
        for ref in [rec.statevector_ref, rec.probabilities_ref, rec.counts_ref]:
            if ref: assert not Path(ref).is_absolute() and (root/ref).exists()
    return DatasetWriteResult(root, sorted(set(written)), manifests, {}, root/"dataset_summary.json", root/"generation_config.json")
