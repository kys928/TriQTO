"""Phase 7 deterministic raw data generation pipeline."""
from __future__ import annotations

import inspect, json, math, random
from collections import Counter
from typing import Any


from triqto.circuits.circuit_metadata import GeneratedCircuit, count_two_qubit_gates
from triqto.circuits.families import generate_circuit_family, get_circuit_family
from triqto.core.ids import make_circuit_id, make_deterministic_id, make_run_id, make_sample_id
from triqto.distortions import apply_distortion
from triqto.metrics import compare_born_distributions
from triqto.metrics.results import BornMetricBundle
from triqto.simulation import simulate_ideal_shots, simulate_ideal_statevector
from triqto.storage import CircuitRecord, DistortionRecord, MetricRecord, SimulationRecord
from triqto.storage.schema import DatasetSampleRecord
from .records import DatasetGenerationResult, GeneratedDatasetSample
from .seeding import derive_child_seed
from .specs import DatasetGenerationConfig, config_id, config_to_dict


def _jsoncopy(x: Any) -> Any:
    return json.loads(json.dumps(x, sort_keys=True, allow_nan=False))

def _generator_kwargs(family: str, kwargs: dict[str, Any], seed: int) -> dict[str, Any]:
    out = _jsoncopy(kwargs)
    sig = inspect.signature(get_circuit_family(family))
    if "seed" in sig.parameters and "seed" not in out:
        out["seed"] = seed
    return out

def _bind_parameters(generated: GeneratedCircuit, seed: int, low: float, high: float):
    circ = generated.circuit.copy()
    params = sorted(circ.parameters, key=lambda p: p.name)
    rng = random.Random(seed)
    bindings = {p.name: float(rng.uniform(low, high)) for p in params}
    assignment = {p: bindings[p.name] for p in params}
    bound = circ.assign_parameters(assignment, inplace=False) if assignment else circ
    enc = {k: [math.sin(v), math.cos(v)] for k, v in bindings.items()}
    return bound, bindings, enc

def _circuit_record(circuit_id: str, circuit, family: str, metadata: dict[str, Any]) -> CircuitRecord:
    return CircuitRecord(circuit_id, family, circuit.num_qubits, circuit.num_clbits, circuit.depth(), count_two_qubit_gates(circuit), len(circuit.parameters), _jsoncopy(metadata))

def _sim_record(run_id: str, circuit_id: str, mode: str, shots: int | None, metadata: dict[str, Any]) -> SimulationRecord:
    return SimulationRecord(run_id, circuit_id, mode, "qiskit.quantum_info.Statevector", shots, metadata=_jsoncopy(metadata))

def _metric_values(bundle: BornMetricBundle) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, result in sorted(bundle.metrics.items()):
        v = float(result.value)
        if math.isfinite(v): out[name] = v
        elif v > 0: out[name] = None; out[f"{name}__nonfinite"] = "positive_infinity"
        else: out[name] = None; out[f"{name}__nonfinite"] = "negative_infinity"
    return out

def _add_unique(store: dict[str, Any], rec: Any, key: str) -> None:
    old = store.get(key)
    if old is None: store[key] = rec
    elif old.to_dict() != rec.to_dict(): raise ValueError(f"Conflicting deterministic ID detected: {key}")

def generate_dataset(config: DatasetGenerationConfig) -> DatasetGenerationResult:
    cid = config_id(config)
    circuits: dict[str, CircuitRecord] = {}; sims: dict[str, SimulationRecord] = {}; dists: dict[str, DistortionRecord] = {}; metrics: dict[str, MetricRecord] = {}; sample_ids: set[str] = set()
    samples=[]; sample_records=[]
    for spec_index, spec in enumerate(config.circuit_specs):
      for rep in range(spec.repetitions):
        base_payload={"config_id":cid,"spec_index":spec_index,"spec":config_to_dict(config)["circuit_specs"][spec_index],"repetition_index":rep}
        gseed=derive_child_seed(config.base_seed,"circuit_generation",base_payload); pseed=derive_child_seed(config.base_seed,"parameter_binding",base_payload)
        gkwargs=_generator_kwargs(spec.family, spec.generator_kwargs, gseed)
        generated=generate_circuit_family(spec.family, spec.n_qubits, **gkwargs)
        bound, bindings, enc = _bind_parameters(generated,pseed,config.parameter_low,config.parameter_high)
        clean_id=make_circuit_id({"family":spec.family,"n_qubits":spec.n_qubits,"generator_kwargs":gkwargs,"repetition_index":rep,"generation_seed":gseed,"parameter_bindings":bindings,"schema_version":config.schema_version})
        clean_run=make_run_id({"circuit_id":clean_id,"simulation_mode":"ideal_statevector","parameter_bindings":bindings,"schema_version":config.schema_version})
        clean_res=simulate_ideal_statevector(bound)
        clean_shot=None
        _add_unique(circuits,_circuit_record(clean_id,bound,spec.family,{"role":"clean","family":spec.family,"parameter_bindings":bindings,"parameter_sin_cos":enc,"generator_kwargs":gkwargs}),clean_id)
        _add_unique(sims,_sim_record(clean_run,clean_id,"ideal_statevector",None,{"role":"clean","probabilities_ref":f"artifacts/probabilities/{clean_run}.json","statevector_ref":f"artifacts/statevectors/{clean_run}.npy" if config.store_statevectors else None}),clean_run)
        if config.ideal_shots:
            clean_shot_seed=derive_child_seed(config.base_seed,"clean_shots",base_payload)
            clean_shot=simulate_ideal_shots(bound,shots=config.ideal_shots,seed=clean_shot_seed)
            srid=make_run_id({"circuit_id":clean_id,"simulation_mode":"ideal_shot","shots":config.ideal_shots,"seed":clean_shot_seed,"schema_version":config.schema_version})
            _add_unique(sims,_sim_record(srid,clean_id,"ideal_shot",config.ideal_shots,{"source_run_id":clean_run,"counts_ref":f"artifacts/counts/{srid}.json","seed":clean_shot_seed}),srid)
        for dist_index, dspec in enumerate(config.distortion_specs):
            distortion=apply_distortion(dspec.name,bound,**_jsoncopy(dspec.kwargs))
            did=make_deterministic_id("distortion",{"clean_circuit_id":clean_id,"name":dspec.name,"kwargs":dspec.kwargs,"metadata":distortion.metadata,"schema_version":config.schema_version})
            distorted_id=make_circuit_id({"clean_circuit_id":clean_id,"distortion_id":did,"schema_version":config.schema_version})
            distorted_run=make_run_id({"distorted_circuit_id":distorted_id,"simulation_mode":"ideal_statevector","distortion_id":did,"schema_version":config.schema_version})
            distorted_res=simulate_ideal_statevector(distortion.distorted_circuit)
            context=dict(distortion.metadata); context.update({"distortion_family":distortion.distortion_family})
            bundle=compare_born_distributions(clean_res,distorted_res,context_metadata=context)
            mid=make_deterministic_id("metric",{"clean_run_id":clean_run,"distorted_run_id":distorted_run,"metric_family":"born","metric_names":sorted(bundle.metrics),"schema_version":config.schema_version})
            sid=make_sample_id({"config_id":cid,"clean_circuit_id":clean_id,"distortion_id":did,"metric_id":mid})
            if sid in sample_ids: raise ValueError(f"Duplicate sample_id detected: {sid}")
            sample_ids.add(sid)
            marker=bool(distortion.metadata.get("marker_only")); tv=abs(bundle.metrics["total_variation"].value); zero=tv==0.0
            md={"distortion_name":dspec.name,"distortion_kwargs":dspec.kwargs,"parameter_sin_cos":enc,"marker_only":marker,"born_zero_shift":zero,"born_observable_shift_absent":zero}
            _add_unique(circuits,_circuit_record(distorted_id,distortion.distorted_circuit,spec.family,{"role":"distorted","source_clean_circuit_id":clean_id,"distortion_id":did,"parameter_bindings":bindings}),distorted_id)
            _add_unique(sims,_sim_record(distorted_run,distorted_id,"ideal_statevector",None,{"role":"distorted","distortion_id":did,"probabilities_ref":f"artifacts/probabilities/{distorted_run}.json","statevector_ref":f"artifacts/statevectors/{distorted_run}.npy" if config.store_statevectors else None}),distorted_run)
            dshot=None
            if config.ideal_shots:
                dseed=derive_child_seed(config.base_seed,"distorted_shots",{**base_payload,"distortion_index":dist_index,"distortion_id":did})
                dshot=simulate_ideal_shots(distortion.distorted_circuit,shots=config.ideal_shots,seed=dseed)
                drid=make_run_id({"circuit_id":distorted_id,"simulation_mode":"ideal_shot","shots":config.ideal_shots,"seed":dseed,"distortion_id":did,"schema_version":config.schema_version})
                _add_unique(sims,_sim_record(drid,distorted_id,"ideal_shot",config.ideal_shots,{"source_run_id":distorted_run,"counts_ref":f"artifacts/counts/{drid}.json","seed":dseed}),drid)
            _add_unique(dists,DistortionRecord(did,clean_id,distortion.distortion_type,distortion.strength,distortion.affected_qubits,distortion.affected_gates,metadata={**_jsoncopy(distortion.metadata),"distorted_circuit_id":distorted_id}),did)
            mrec=MetricRecord(mid,distorted_run,distorted_id,did,born_metrics=_metric_values(bundle),hilbert_metrics={"computed": False},parameter_metrics={"computed": False},topology_metrics={"computed": False},hilbert_available_mask=False,metadata={"clean_run_id":clean_run,"distorted_run_id":distorted_run,"sample_id":sid,"metric_family":"born","metric_schema_version":"triqto.born.phase6","support_size":len(bundle.support),"nonfinite_encoding":"positive infinity encoded as null plus metric__nonfinite", "applicability_warning":bundle.metadata.get("applicability_warning")})
            _add_unique(metrics,mrec,mid)
            srec=DatasetSampleRecord(sid,config.dataset_name,config.schema_version,clean_id,distorted_id,clean_run,distorted_run,did,mid,spec.family,spec.n_qubits,rep,_jsoncopy(bindings),config.base_seed,md)
            srec.validate(); sample_records.append(srec)
            samples.append(GeneratedDatasetSample(sid,clean_id,distorted_id,clean_run,distorted_run,did,mid,spec.family,spec.n_qubits,rep,bindings,gseed,pseed,bound,distortion.distorted_circuit,clean_res,distorted_res,distortion,bundle,clean_shot,dshot,md))
    fam=Counter(s.family for s in samples); distc=Counter(s.metadata["distortion_name"] for s in samples)
    summary={"sample_count":len(samples),"unique_clean_circuit_count":len({s.clean_circuit_id for s in samples}),"unique_distorted_circuit_count":len({s.distorted_circuit_id for s in samples}),"simulation_record_count":len(sims),"distortion_record_count":len(dists),"metric_record_count":len(metrics),"family_counts":dict(sorted(fam.items())),"distortion_counts":dict(sorted(distc.items())),"marker_only_sample_count":sum(s.metadata["marker_only"] for s in samples),"born_visible_sample_count":sum(not s.metadata["born_zero_shift"] for s in samples),"born_zero_shift_sample_count":sum(s.metadata["born_zero_shift"] for s in samples),"base_seed":config.base_seed,"schema_version":config.schema_version,"scientific_scope":"synthetic simulator-derived raw data; no hardware, training, correction actions, topology, or quantum-advantage claim"}
    return DatasetGenerationResult(config.dataset_name,config.schema_version,cid,config,samples,list(circuits.values()),list(sims.values()),list(dists.values()),list(metrics.values()),sample_records,summary)
