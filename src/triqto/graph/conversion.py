"""Phase 8 deterministic, framework-neutral circuit graph conversion."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib, json, math, os, shutil, uuid
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.data_generation import config_from_dict, scientific_generation_id, validate_dataset_joins, verify_dataset_references
from triqto.storage import CircuitRecord, DatasetSampleRecord, DistortionRecord, ManifestReader, ManifestWriter, MetricRecord, SimulationRecord
from triqto.storage.schema import GraphPairRecord, GraphRecord

GRAPH_SCHEMA_VERSION = "triqto.graph.phase8.v1"
GATE_VOCAB_VERSION = "triqto.graph.gate_vocab.v1"
ANGLE_SLOT_MAPPING_VERSION = "triqto.graph.angle_slots.v1"
EDGE_REPRESENTATION_VERSION = "triqto.graph.directed_multiedge.v1"
PROBABILITY_ATOL = 1e-9
NODE_FEATURE_NAMES = ("measured_flag","measurement_count","reset_count","single_qubit_gate_incidence_count","two_qubit_gate_incidence_count","total_gate_incidence_count","angular_parameter_incidence_count","sum_sin_angle","sum_cos_angle","unique_interaction_neighbor_count","first_active_layer_normalized","last_active_layer_normalized","active_layer_span_normalized")
EDGE_FEATURE_NAMES = ("normalized_gate_order","normalized_layer","forward_flag","source_operand_position","destination_operand_position","known_control_source_flag","known_target_destination_flag","symmetric_gate_flag","parameter_count","angular_parameter_count")
GATE_FEATURE_NAMES = ("vocabulary_id","arity","circuit_order_index","normalized_order","layer_index","normalized_layer","measurement_flag","reset_flag","barrier_flag","one_qubit_flag","two_qubit_flag","multi_qubit_flag","parameter_count","angular_parameter_count","known_control_semantics_mask","symmetric_interaction_mask")
GLOBAL_FEATURE_NAMES: tuple[str,...] = ()
GATE_VOCAB = {name:i for i,name in enumerate(["UNK","barrier","ccx","cp","crx","cry","crz","cswap","cx","cz","h","id","measure","p","reset","rx","ry","rz","s","sdg","swap","sx","t","tdg","u","u1","u2","u3","x","y","z"])}
CONTROL_GATES = {"cx","cz","cp","crx","cry","crz","ccx","cswap"}
SYMMETRIC_GATES = {"swap","cz"}
ANGULAR_SLOTS = {"rx":{0},"ry":{0},"rz":{0},"p":{0},"cp":{0},"crx":{0},"cry":{0},"crz":{0},"u":{0,1,2},"u1":{0},"u2":{0,1},"u3":{0,1,2}}


def _nonblank(v: Any, name: str) -> str:
    if not isinstance(v,str) or not v.strip(): raise ValueError(f"{name} must contain non-whitespace text")
    return v.strip()
def _pos_int(v: Any, name: str) -> int:
    if isinstance(v,bool) or not isinstance(v,int): raise TypeError(f"{name} must be a positive integer and not bool")
    if v <= 0: raise ValueError(f"{name} must be positive")
    return v
def _bool(v: Any, name: str) -> bool:
    if not isinstance(v,bool): raise TypeError(f"{name} must be exactly bool")
    return v
def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, sort_keys=True, allow_nan=False))

def _safe_ref(root: Path, ref: str) -> Path:
    if not isinstance(ref,str) or not ref or ref == ".": raise ValueError(f"unsafe empty path {ref!r}")
    pp = PurePosixPath(ref)
    if pp.is_absolute() or any(part == ".." for part in pp.parts): raise ValueError(f"unsafe relative path {ref!r}")
    path = (root / Path(*pp.parts)).resolve(); base = root.resolve()
    if path != base and base not in path.parents: raise ValueError(f"path escapes root: {ref}")
    return path

@dataclass(frozen=True, slots=True)
class GraphConversionConfig:
    schema_version: str = GRAPH_SCHEMA_VERSION
    max_gate_events: int = 100_000
    max_probability_outcomes: int = 1_000_000
    include_supplemental_counts: bool = True
    reject_conditioned_operations: bool = True
    def __post_init__(self):
        object.__setattr__(self,"schema_version",_nonblank(self.schema_version,"schema_version"))
        object.__setattr__(self,"max_gate_events",_pos_int(self.max_gate_events,"max_gate_events"))
        object.__setattr__(self,"max_probability_outcomes",_pos_int(self.max_probability_outcomes,"max_probability_outcomes"))
        object.__setattr__(self,"include_supplemental_counts",_bool(self.include_supplemental_counts,"include_supplemental_counts"))
        object.__setattr__(self,"reject_conditioned_operations",_bool(self.reject_conditioned_operations,"reject_conditioned_operations"))
        _jsonable(asdict(self))

def graph_config_to_dict(c: GraphConversionConfig)->dict[str,Any]: return asdict(c)
def graph_config_from_dict(p: Mapping[str,Any])->GraphConversionConfig:
    if not isinstance(p,Mapping): raise TypeError("graph config must be a mapping")
    extra=set(p)-set(GraphConversionConfig.__dataclass_fields__) # type: ignore
    if extra: raise ValueError(f"Unknown graph config fields: {sorted(extra)}")
    return GraphConversionConfig(**dict(p))
def load_graph_config(path: str|Path)->GraphConversionConfig:
    return graph_config_from_dict(json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid non-finite JSON constant: {x}"))))
def save_graph_config(c: GraphConversionConfig,path: str|Path)->None:
    Path(path).write_text(json.dumps(graph_config_to_dict(c),sort_keys=True,indent=2,allow_nan=False)+"\n")

def graph_schema_id()->str:
    return make_deterministic_id("graphschema", {"schema_version":GRAPH_SCHEMA_VERSION,"node":NODE_FEATURE_NAMES,"edge":EDGE_FEATURE_NAMES,"gate":GATE_FEATURE_NAMES,"global":GLOBAL_FEATURE_NAMES,"gate_vocab_version":GATE_VOCAB_VERSION,"angle_slot_mapping_version":ANGLE_SLOT_MAPPING_VERSION,"edge_representation_version":EDGE_REPRESENTATION_VERSION})
def graph_id(circuit_id:str, role:str)->str: return make_deterministic_id("graph",{"circuit_id":circuit_id,"role":role,"graph_schema_id":graph_schema_id()})
def graph_pair_id(sample_id:str, clean_graph_id:str, distorted_graph_id:str)->str: return make_deterministic_id("graphpair",{"sample_id":sample_id,"clean_graph_id":clean_graph_id,"distorted_graph_id":distorted_graph_id,"graph_schema_version":GRAPH_SCHEMA_VERSION})

@dataclass(slots=True)
class CircuitGraphData:
    graph_id:str; circuit_id:str; sample_id:str; role:str; family:str; graph_schema_version:str; n_qubits:int
    node_index:np.ndarray; node_features:np.ndarray; edge_index:np.ndarray; edge_event_index:np.ndarray; edge_features:np.ndarray; gate_names:np.ndarray; gate_features:np.ndarray; gate_qubit_ptr:np.ndarray; gate_qubit_indices:np.ndarray; gate_clbit_ptr:np.ndarray; gate_clbit_indices:np.ndarray; gate_parameter_ptr:np.ndarray; gate_parameter_values:np.ndarray; gate_parameter_sin:np.ndarray; gate_parameter_cos:np.ndarray; gate_parameter_angle_mask:np.ndarray; parameter_names:np.ndarray; parameter_values:np.ndarray; parameter_sin:np.ndarray; parameter_cos:np.ndarray; outcome_bitstrings:np.ndarray; exact_probabilities:np.ndarray; global_features:np.ndarray
    node_feature_names:tuple[str,...]=NODE_FEATURE_NAMES; edge_feature_names:tuple[str,...]=EDGE_FEATURE_NAMES; gate_feature_names:tuple[str,...]=GATE_FEATURE_NAMES; global_feature_names:tuple[str,...]=GLOBAL_FEATURE_NAMES
    exact_probability_available_mask:bool=True; supplemental_counts_available_mask:bool=False; hilbert_available_mask:bool=False; source_statevector_ref:str|None=None; source_probability_ref:str|None=None; source_counts_ref:str|None=None; metadata:dict[str,Any]=field(default_factory=dict)

@dataclass(slots=True)
class GraphSamplePair:
    graph_pair_id:str; sample_id:str; clean_graph_id:str; distorted_graph_id:str; distortion_id:str; metric_id:str; clean_graph:CircuitGraphData; distorted_graph:CircuitGraphData; born_metric_names:np.ndarray; born_metric_values:np.ndarray; born_metric_positive_infinity_mask:np.ndarray; born_zero_shift:bool; born_observable_shift_absent:bool; marker_only:bool; applicability_warning:str|None; metadata:dict[str,Any]=field(default_factory=dict)

@dataclass(slots=True)
class CompletedPhase7Dataset:
    source_root:Path; generation_config:dict[str,Any]; source_scientific_generation_id:str; samples:list[DatasetSampleRecord]; circuits:list[CircuitRecord]; simulations:list[SimulationRecord]; distortions:list[DistortionRecord]; metrics:list[MetricRecord]; circuits_by_id:dict[str,QuantumCircuit]; probabilities_by_run_id:dict[str,dict[str,float]]; counts_by_run_id:dict[str,dict[str,int]]; statevector_storage_enabled:bool; completion_marker:dict[str,Any]

@dataclass(slots=True)
class GraphConversionResult:
    source_root:Path; source_scientific_generation_id:str; graph_conversion_id:str; operational_config_id:str; graph_schema_id:str; graphs:list[CircuitGraphData]; pairs:list[GraphSamplePair]; graph_records:list[GraphRecord]; graph_pair_records:list[GraphPairRecord]; summary:dict[str,Any]

@dataclass(slots=True)
class GraphWriteResult:
    output_root:Path; graph_complete_path:Path; managed_files:list[str]; graph_count:int; pair_count:int

def _load_qpy(path:Path)->QuantumCircuit:
    from qiskit import qpy
    with path.open('rb') as h: circuits=qpy.load(h)
    if len(circuits)!=1: raise ValueError(f"QPY artifact must contain exactly one circuit: {path}")
    return circuits[0]
def _read_json(path:Path)->Any:
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid non-finite JSON constant: {x}")))

def load_completed_phase7_dataset(source_root: str|Path)->CompletedPhase7Dataset:
    root=Path(source_root); 
    if not root.exists(): raise FileNotFoundError(root)
    marker_path=root/'dataset_complete.json'
    if not marker_path.exists(): raise FileNotFoundError(marker_path)
    marker=_read_json(marker_path)
    if marker.get('complete') is not True: raise ValueError('dataset_complete.json complete must be true')
    for ref in marker.get('managed_files',[]):
        p=_safe_ref(root,ref)
        if not p.exists() or not p.is_file(): raise FileNotFoundError(ref)
    for name in ('sample_manifest','circuit_manifest','simulation_manifest','distortion_manifest','metric_manifest'):
        if not (root/'manifests'/f'{name}.parquet').exists(): raise FileNotFoundError(name)
    gen_payload=_read_json(root/'generation_config.json'); gen_config=config_from_dict(gen_payload)
    reader=ManifestReader(root/'manifests')
    samples=reader.read_typed_records('sample_manifest', DatasetSampleRecord) # type: ignore
    circuits=reader.read_typed_records('circuit_manifest', CircuitRecord) # type: ignore
    sims=reader.read_typed_records('simulation_manifest', SimulationRecord) # type: ignore
    dists=reader.read_typed_records('distortion_manifest', DistortionRecord) # type: ignore
    metrics=reader.read_typed_records('metric_manifest', MetricRecord) # type: ignore
    validate_dataset_joins(samples,circuits,sims,dists,metrics); verify_dataset_references(root,circuits,sims, require_statevectors=bool(gen_payload.get('store_statevectors')))
    circuits_by_id={}
    for rec in circuits:
        ref=rec.metadata.get('artifact_ref')
        if not isinstance(ref,str): raise ValueError(f"Circuit {rec.circuit_id} missing artifact_ref")
        circuits_by_id[rec.circuit_id]=_load_qpy(_safe_ref(root,ref))
    probs={}; counts={}
    for sim in sims:
        if sim.probabilities_ref:
            payload=_read_json(_safe_ref(root,sim.probabilities_ref));
            if not isinstance(payload,Mapping): raise TypeError('probability artifact must be mapping')
            probs[sim.run_id]={str(k):float(v) for k,v in payload.items()}
        if sim.counts_ref:
            payload=_read_json(_safe_ref(root,sim.counts_ref)); counts[sim.run_id]={str(k):int(v) for k,v in payload.items()}
    return CompletedPhase7Dataset(root, gen_payload, scientific_generation_id(gen_config), samples, circuits, sims, dists, metrics, circuits_by_id, probs, counts, bool(gen_payload.get('store_statevectors')), marker)

def _qindex(circuit:QuantumCircuit,q:Any)->int: return circuit.find_bit(q).index
def _cindex(circuit:QuantumCircuit,c:Any)->int: return circuit.find_bit(c).index
def _layers(circuit:QuantumCircuit)->list[int]:
    next_layer=[0]*circuit.num_qubits; out=[]
    for inst in circuit.data:
        qs=[_qindex(circuit,q) for q in inst.qubits]
        layer=max([next_layer[q] for q in qs], default=0); out.append(layer)
        for q in qs: next_layer[q]=layer+1
    return out

def _numeric_params(inst:Any)->tuple[list[float],list[bool]]:
    vals=[]; masks=[]; name=inst.operation.name
    for i,p in enumerate(inst.operation.params):
        try: val=float(p)
        except Exception as exc: raise ValueError(f"Unsupported/unbound nonnumeric parameter {p!r} in gate {name}") from exc
        if not math.isfinite(val): raise ValueError('gate parameter must be finite')
        vals.append(val); masks.append(i in ANGULAR_SLOTS.get(name,set()))
    return vals,masks

def _validate_probs(exact:Mapping[str,float], n:int)->tuple[np.ndarray,np.ndarray]:
    seen=set(); rows=[]
    for k,v in exact.items():
        if not isinstance(k,str) or len(k)!=n or any(ch not in '01' for ch in k): raise ValueError(f"malformed probability bitstring {k!r}")
        if k in seen: raise ValueError(f"duplicate outcome {k}")
        seen.add(k); f=float(v)
        if f < -PROBABILITY_ATOL: raise ValueError('negative probability')
        rows.append((k, max(0.0,f)))
    rows.sort(key=lambda x:x[0]); total=sum(v for _,v in rows)
    if abs(total-1.0)>PROBABILITY_ATOL: raise ValueError(f"probabilities must sum to one, got {total}")
    return np.array([k for k,_ in rows], dtype=f"<U{max([1]+[len(k) for k,_ in rows])}"), np.array([v for _,v in rows], dtype=np.float64)

def circuit_to_graph(circuit:QuantumCircuit, *, circuit_id:str, sample_id:str, role:str, family:str, parameter_bindings:Mapping[str,float], exact_probabilities:Mapping[str,float], metadata:Mapping[str,Any])->CircuitGraphData:
    if role not in {'clean','distorted'}: raise ValueError('role must be clean or distorted')
    if circuit.parameters: raise ValueError('Phase 8 requires fully bound circuits')
    circ=circuit.copy(); n=circ.num_qubits; layers=_layers(circ); max_layer=max(layers, default=0); max_order=max(len(circ.data)-1,1)
    node=np.zeros((n,len(NODE_FEATURE_NAMES)),dtype=np.float64); neighbors=[set() for _ in range(n)]; first=[None]*n; last=[None]*n
    gate_names=[]; gate_feats=[]; qptr=[0]; qidx=[]; cptr=[0]; cidx=[]; pptr=[0]; pvals=[]; psin=[]; pcos=[]; pmask=[]; edges=[]; eevent=[]; efeats=[]; multi=0
    for order, inst in enumerate(circ.data):
        op=inst.operation; name=op.name
        if getattr(op,'condition',None) is not None: raise NotImplementedError('conditioned operations are not supported in Phase 8 v1')
        qs=[_qindex(circ,q) for q in inst.qubits]; cs=[_cindex(circ,c) for c in inst.clbits]; vals,masks=_numeric_params(inst)
        layer=layers[order]; norm_o=0.0 if len(circ.data)<=1 else order/max_order; norm_l=0.0 if max_layer<=0 else layer/max_layer
        arity=len(qs); meas=name=='measure'; reset=name=='reset'; barr=name=='barrier'; two=arity==2 and not barr; one=arity==1 and not meas and not reset and not barr; multi=multi+(1 if arity>2 else 0)
        gate_names.append(name); gate_feats.append([GATE_VOCAB.get(name,0),arity,order,norm_o,layer,norm_l,meas,reset,barr,one,two,arity>2,len(vals),sum(masks),name in CONTROL_GATES,name in SYMMETRIC_GATES])
        qidx.extend(qs); qptr.append(len(qidx)); cidx.extend(cs); cptr.append(len(cidx)); pvals.extend(vals); pmask.extend(masks); psin.extend([math.sin(v) if m else 0.0 for v,m in zip(vals,masks)]); pcos.extend([math.cos(v) if m else 0.0 for v,m in zip(vals,masks)]); pptr.append(len(pvals))
        for q in qs:
            first[q]=layer if first[q] is None else min(first[q],layer); last[q]=layer if last[q] is None else max(last[q],layer); node[q,5]+=1; node[q,6]+=sum(masks); node[q,7]+=sum(math.sin(v) for v,m in zip(vals,masks) if m); node[q,8]+=sum(math.cos(v) for v,m in zip(vals,masks) if m)
            if meas: node[q,0]=1; node[q,1]+=1
            if reset: node[q,2]+=1
            if one: node[q,3]+=1
            if two: node[q,4]+=1
        if two:
            a,b=qs; neighbors[a].add(b); neighbors[b].add(a)
            for src,dst,fw,sp,dp in [(a,b,1.0,0.0,1.0),(b,a,0.0,1.0,0.0)]:
                edges.append((src,dst)); eevent.append(order); efeats.append([norm_o,norm_l,fw,sp,dp, float(name in CONTROL_GATES and sp==0), float(name in CONTROL_GATES and dp==1), float(name in SYMMETRIC_GATES), len(vals), sum(masks)])
    for q in range(n):
        node[q,9]=len(neighbors[q])
        if first[q] is not None:
            node[q,10]=0.0 if max_layer<=0 else first[q]/max_layer; node[q,11]=0.0 if max_layer<=0 else last[q]/max_layer; node[q,12]=0.0 if max_layer<=0 else (last[q]-first[q])/max_layer
    names=sorted(parameter_bindings); vals=[float(parameter_bindings[k]) for k in names]
    if any(not math.isfinite(v) for v in vals): raise ValueError('parameter binding must be finite')
    bit,prob=_validate_probs(exact_probabilities,n)
    md=_jsonable(dict(metadata)); md.setdefault('global_phase', str(circ.global_phase)); md.setdefault('global_phase_excluded_from_features', True); md['multi_qubit_event_count']=multi
    return CircuitGraphData(graph_id(circuit_id,role),circuit_id,sample_id,role,family,GRAPH_SCHEMA_VERSION,n,np.arange(n,dtype=np.int64),node,np.array(edges,dtype=np.int64).T.reshape(2,len(edges)) if edges else np.empty((2,0),dtype=np.int64),np.array(eevent,dtype=np.int64),np.array(efeats,dtype=np.float64).reshape(len(edges),len(EDGE_FEATURE_NAMES)) if edges else np.empty((0,len(EDGE_FEATURE_NAMES)),dtype=np.float64),np.array(gate_names,dtype=f"<U{max([1]+[len(x) for x in gate_names])}"),np.array(gate_feats,dtype=np.float64).reshape(len(gate_names),len(GATE_FEATURE_NAMES)) if gate_feats else np.empty((0,len(GATE_FEATURE_NAMES)),dtype=np.float64),np.array(qptr,dtype=np.int64),np.array(qidx,dtype=np.int64),np.array(cptr,dtype=np.int64),np.array(cidx,dtype=np.int64),np.array(pptr,dtype=np.int64),np.array(pvals,dtype=np.float64),np.array(psin,dtype=np.float64),np.array(pcos,dtype=np.float64),np.array(pmask,dtype=np.bool_),np.array(names,dtype=f"<U{max([1]+[len(x) for x in names])}"),np.array(vals,dtype=np.float64),np.sin(vals,dtype=np.float64),np.cos(vals,dtype=np.float64),bit,prob,np.empty((0,),dtype=np.float64),metadata=md)

def graph_arrays(g:CircuitGraphData)->dict[str,np.ndarray]:
    return {k:getattr(g,k) for k in ['node_index','node_features','edge_index','edge_event_index','edge_features','gate_names','gate_features','gate_qubit_ptr','gate_qubit_indices','gate_clbit_ptr','gate_clbit_indices','gate_parameter_ptr','gate_parameter_values','gate_parameter_sin','gate_parameter_cos','gate_parameter_angle_mask','parameter_names','parameter_values','parameter_sin','parameter_cos','outcome_bitstrings','exact_probabilities','global_features']}

def content_hash(arrays:Mapping[str,np.ndarray], metadata:Mapping[str,Any])->str:
    h=hashlib.sha256(); h.update(canonical_json(_jsonable(dict(metadata))).encode())
    for name in sorted(arrays):
        a=np.ascontiguousarray(arrays[name]); h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0'+canonical_json(list(a.shape)).encode()+b'\0'+a.tobytes())
    return 'sha256:'+h.hexdigest()

def _artifact_metadata(g:CircuitGraphData)->dict[str,Any]:
    return {"graph_id":g.graph_id,"circuit_id":g.circuit_id,"sample_id":g.sample_id,"role":g.role,"family":g.family,"graph_schema_version":g.graph_schema_version,"n_qubits":g.n_qubits,"node_feature_names":list(g.node_feature_names),"edge_feature_names":list(g.edge_feature_names),"gate_feature_names":list(g.gate_feature_names),"global_feature_names":list(g.global_feature_names),"exact_probability_available_mask":g.exact_probability_available_mask,"supplemental_counts_available_mask":g.supplemental_counts_available_mask,"hilbert_available_mask":g.hilbert_available_mask,"source_statevector_ref":g.source_statevector_ref,"source_probability_ref":g.source_probability_ref,"source_counts_ref":g.source_counts_ref,"metadata":g.metadata}

def _decode_metrics(born:Mapping[str,Any])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    names=[]; vals=[]; mask=[]
    for k in sorted(x for x in born if not x.endswith('__nonfinite')):
        marker=born.get(k+'__nonfinite')
        if born[k] is None and marker=='positive_infinity': names.append(k); vals.append(0.0); mask.append(True)
        else:
            v=float(born[k]);
            if not math.isfinite(v): raise ValueError('metric must be finite or encoded positive infinity')
            names.append(k); vals.append(v); mask.append(False)
    return np.array(names,dtype=f"<U{max([1]+[len(x) for x in names])}"),np.array(vals,dtype=np.float64),np.array(mask,dtype=np.bool_)

def convert_completed_dataset_to_graphs(source_root:str|Path, config:GraphConversionConfig|None=None)->GraphConversionResult:
    cfg=config or GraphConversionConfig(); ds=load_completed_phase7_dataset(source_root)
    if len(ds.samples)>cfg.max_gate_events*10: pass
    circ_rec={r.circuit_id:r for r in ds.circuits}; sim_rec={r.run_id:r for r in ds.simulations}; metric_rec={r.metric_id:r for r in ds.metrics}; dist_rec={r.distortion_id:r for r in ds.distortions}
    cache={}; graphs=[]; pairs=[]
    for s in sorted(ds.samples,key=lambda r:r.sample_id):
        made=[]
        for cid,rid,role in [(s.clean_circuit_id,s.clean_run_id,'clean'),(s.distorted_circuit_id,s.distorted_run_id,'distorted')]:
            key=(cid,role)
            if key not in cache:
                sim=sim_rec[rid]; meta={"source_run_id":rid,"source_probability_ref":sim.probabilities_ref,"source_statevector_ref":sim.statevector_ref,"source_counts_ref":sim.counts_ref}
                g=circuit_to_graph(ds.circuits_by_id[cid],circuit_id=cid,sample_id=s.sample_id,role=role,family=s.family,parameter_bindings=s.parameter_bindings,exact_probabilities=ds.probabilities_by_run_id[rid],metadata=meta)
                g.source_probability_ref=sim.probabilities_ref; g.source_statevector_ref=sim.statevector_ref; g.source_counts_ref=sim.counts_ref if cfg.include_supplemental_counts else None; g.supplemental_counts_available_mask=bool(sim.counts_ref and cfg.include_supplemental_counts)
                cache[key]=g; graphs.append(g)
            made.append(cache[key])
        m=metric_rec[s.metric_id]; names,vals,infmask=_decode_metrics(m.born_metrics); d=dist_rec[s.distortion_id]
        pair=GraphSamplePair(graph_pair_id(s.sample_id,made[0].graph_id,made[1].graph_id),s.sample_id,made[0].graph_id,made[1].graph_id,s.distortion_id,s.metric_id,made[0],made[1],names,vals,infmask,bool(s.metadata.get('born_zero_shift',False)),bool(s.metadata.get('born_observable_shift_absent',False)),bool(d.metadata.get('marker_only',False)),d.metadata.get('applicability_warning'),{"distortion_type":d.distortion_type})
        pairs.append(pair)
    records=[]
    for g in sorted(graphs,key=lambda x:x.graph_id):
        ch=content_hash(graph_arrays(g), _artifact_metadata(g)); records.append(GraphRecord(g.graph_id,g.sample_id,g.circuit_id,g.role,g.family,g.graph_schema_version,f"artifacts/graphs/{g.graph_id}.npz",ch,g.n_qubits,g.edge_index.shape[1],g.gate_features.shape[0],g.node_features.shape[1],g.edge_features.shape[1],g.gate_features.shape[1],{"logical_content_hash_basis":"sorted arrays plus canonical metadata","phase":"8"}))
    pair_records=[GraphPairRecord(p.graph_pair_id,p.sample_id,p.clean_graph_id,p.distorted_graph_id,p.distortion_id,p.metric_id,f"artifacts/pairs/{p.graph_pair_id}.npz",{"phase":"8"}) for p in sorted(pairs,key=lambda x:x.graph_pair_id)]
    fam=Counter(s.family for s in ds.samples); dis=Counter(dist_rec[s.distortion_id].distortion_type for s in ds.samples); qs=Counter(str(s.n_qubits) for s in ds.samples)
    summary={"source_sample_count":len(ds.samples),"graph_count":len(graphs),"pair_count":len(pairs),"clean_graph_count":sum(g.role=='clean' for g in graphs),"distorted_graph_count":sum(g.role=='distorted' for g in graphs),"family_counts":dict(sorted(fam.items())),"distortion_counts":dict(sorted(dis.items())),"marker_only_pair_count":sum(p.marker_only for p in pairs),"born_zero_shift_pair_count":sum(p.born_zero_shift for p in pairs),"variable_qubit_count_distribution":dict(sorted(qs.items())),"total_nodes":sum(g.n_qubits for g in graphs),"total_directed_edges":sum(g.edge_index.shape[1] for g in graphs),"total_gate_events":sum(g.gate_features.shape[0] for g in graphs),"multi_qubit_event_count":sum(g.metadata.get('multi_qubit_event_count',0) for g in graphs),"supplemental_count_availability_count":sum(g.supplemental_counts_available_mask for g in graphs),"schema_versions":{"graph":GRAPH_SCHEMA_VERSION,"gate_vocab":GATE_VOCAB_VERSION,"angle_slots":ANGLE_SLOT_MAPPING_VERSION,"edge_representation":EDGE_REPRESENTATION_VERSION}}
    schema=graph_schema_id(); conv=make_deterministic_id('graphconv',{"source_scientific_generation_id":ds.source_scientific_generation_id,"graph_schema_id":schema,"schema_version":GRAPH_SCHEMA_VERSION}); op=make_deterministic_id('graphconfig',graph_config_to_dict(cfg))
    return GraphConversionResult(ds.source_root,ds.source_scientific_generation_id,conv,op,schema,sorted(graphs,key=lambda x:x.graph_id),pairs,records,pair_records,summary)

def _npz_meta_array(meta:Mapping[str,Any])->np.ndarray: return np.array(json.dumps(_jsonable(dict(meta)),sort_keys=True,allow_nan=False), dtype='<U65535')
def _read_meta_array(a:np.ndarray)->dict[str,Any]: return json.loads(str(a.reshape(())))

def save_graph_artifact(g:CircuitGraphData,path:Path)->None:
    arrays=graph_arrays(g); arrays['__metadata__']=_npz_meta_array(_artifact_metadata(g)); np.savez_compressed(path, **arrays)
def load_graph_artifact(path:str|Path, expected_content_hash:str|None=None)->CircuitGraphData:
    with np.load(path, allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files if k!='__metadata__'}; meta=_read_meta_array(data['__metadata__'])
    for a in arrays.values():
        if a.dtype.hasobject: raise ValueError('object dtype is forbidden')
    g=CircuitGraphData(**{k:meta[k] for k in ['graph_id','circuit_id','sample_id','role','family','graph_schema_version','n_qubits']}, **arrays, node_feature_names=tuple(meta['node_feature_names']), edge_feature_names=tuple(meta['edge_feature_names']), gate_feature_names=tuple(meta['gate_feature_names']), global_feature_names=tuple(meta['global_feature_names']), exact_probability_available_mask=bool(meta['exact_probability_available_mask']), supplemental_counts_available_mask=bool(meta['supplemental_counts_available_mask']), hilbert_available_mask=bool(meta['hilbert_available_mask']), source_statevector_ref=meta.get('source_statevector_ref'), source_probability_ref=meta.get('source_probability_ref'), source_counts_ref=meta.get('source_counts_ref'), metadata=meta.get('metadata',{}))
    validate_graph_data(g); ch=content_hash(graph_arrays(g), _artifact_metadata(g))
    if expected_content_hash and ch!=expected_content_hash: raise ValueError('content_hash mismatch')
    return g

def validate_graph_data(g:CircuitGraphData)->None:
    if g.node_index.dtype!=np.int64 or g.edge_index.dtype!=np.int64: raise TypeError('indices must be int64')
    if g.node_features.shape!=(g.n_qubits,len(g.node_feature_names)): raise ValueError('node feature shape mismatch')
    if g.edge_index.shape[0]!=2: raise ValueError('edge_index must have shape (2,E)')
    E=g.edge_index.shape[1]; G=g.gate_features.shape[0]
    if g.edge_event_index.shape!=(E,): raise ValueError('edge_event_index shape mismatch')
    if E and (g.edge_index.min()<0 or g.edge_index.max()>=g.n_qubits): raise ValueError('invalid edge index')
    if g.edge_event_index.size and (g.edge_event_index.min()<0 or g.edge_event_index.max()>=G): raise ValueError('invalid event index')
    for ptr,flat in [(g.gate_qubit_ptr,g.gate_qubit_indices),(g.gate_clbit_ptr,g.gate_clbit_indices),(g.gate_parameter_ptr,g.gate_parameter_values)]:
        if ptr.dtype!=np.int64 or ptr[0]!=0 or np.any(np.diff(ptr)<0) or ptr[-1]!=len(flat): raise ValueError('malformed CSR pointer')
    if len(g.gate_qubit_ptr)!=G+1 or len(g.gate_clbit_ptr)!=G+1 or len(g.gate_parameter_ptr)!=G+1: raise ValueError('pointer length mismatch')
    for a in [g.node_features,g.edge_features,g.gate_features,g.gate_parameter_values,g.gate_parameter_sin,g.gate_parameter_cos,g.parameter_values,g.parameter_sin,g.parameter_cos,g.exact_probabilities,g.global_features]:
        if not np.all(np.isfinite(a)): raise ValueError('nonfinite feature/probability')
    _validate_probs(dict(zip([str(x) for x in g.outcome_bitstrings.tolist()], [float(x) for x in g.exact_probabilities])), g.n_qubits)

def save_pair_artifact(p:GraphSamplePair,path:Path)->None:
    np.savez_compressed(path, metadata=_npz_meta_array({"graph_pair_id":p.graph_pair_id,"sample_id":p.sample_id,"clean_graph_id":p.clean_graph_id,"distorted_graph_id":p.distorted_graph_id,"distortion_id":p.distortion_id,"metric_id":p.metric_id,"born_zero_shift":p.born_zero_shift,"born_observable_shift_absent":p.born_observable_shift_absent,"marker_only":p.marker_only,"applicability_warning":p.applicability_warning,"metadata":p.metadata}), born_metric_names=p.born_metric_names, born_metric_values=p.born_metric_values, born_metric_positive_infinity_mask=p.born_metric_positive_infinity_mask)
def load_pair_artifact(path:str|Path)->dict[str,Any]:
    with np.load(path,allow_pickle=False) as d:
        for k in d.files:
            if d[k].dtype.hasobject: raise ValueError('object dtype is forbidden')
        return {"metadata":_read_meta_array(d['metadata']),"born_metric_names":d['born_metric_names'],"born_metric_values":d['born_metric_values'],"born_metric_positive_infinity_mask":d['born_metric_positive_infinity_mask']}

def write_graph_dataset(result:GraphConversionResult, output_root:str|Path)->GraphWriteResult:
    out=Path(output_root)
    if out.exists(): raise FileExistsError(out)
    parent=out.parent; parent.mkdir(parents=True,exist_ok=True); staging=parent/(out.name+f".staging-{uuid.uuid4().hex}")
    managed=[]
    try:
        (staging/'manifests').mkdir(parents=True); (staging/'artifacts'/'graphs').mkdir(parents=True); (staging/'artifacts'/'pairs').mkdir(parents=True)
        save_graph_config(GraphConversionConfig(), staging/'graph_config.json'); managed.append('graph_config.json')
        (staging/'graph_summary.json').write_text(json.dumps(result.summary,sort_keys=True,indent=2,allow_nan=False)+"\n"); managed.append('graph_summary.json')
        for g in result.graphs:
            p=staging/'artifacts'/'graphs'/f'{g.graph_id}.npz'; save_graph_artifact(g,p); managed.append(f'artifacts/graphs/{g.graph_id}.npz')
        for p0 in result.pairs:
            p=staging/'artifacts'/'pairs'/f'{p0.graph_pair_id}.npz'; save_pair_artifact(p0,p); managed.append(f'artifacts/pairs/{p0.graph_pair_id}.npz')
        ManifestWriter(staging/'manifests').write_records('graph_manifest', result.graph_records, overwrite=True); managed.append('manifests/graph_manifest.parquet')
        ManifestWriter(staging/'manifests').write_records('graph_pair_manifest', result.graph_pair_records, overwrite=True); managed.append('manifests/graph_pair_manifest.parquet')
        for r in result.graph_records: load_graph_artifact(staging/r.graph_ref, r.content_hash)
        for r in result.graph_pair_records: load_pair_artifact(staging/r.pair_ref)
        managed=sorted(managed+['graph_complete.json'])
        complete={"complete":True,"source_scientific_generation_id":result.source_scientific_generation_id,"graph_conversion_id":result.graph_conversion_id,"graph_schema_id":result.graph_schema_id,"graph_count":len(result.graphs),"pair_count":len(result.pairs),"managed_files":managed}
        (staging/'graph_complete.json').write_text(json.dumps(complete,sort_keys=True,indent=2,allow_nan=False)+"\n")
        for ref in managed:
            if not _safe_ref(staging,ref).exists(): raise FileNotFoundError(ref)
        os.replace(staging,out)
        return GraphWriteResult(out,out/'graph_complete.json',managed,len(result.graphs),len(result.pairs))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
