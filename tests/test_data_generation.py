import json, math, sys
from pathlib import Path
import pytest

from triqto.data_generation import *
from triqto.metrics import BornMetricBundle
from triqto.storage import ManifestReader
from triqto.storage.schema import DatasetSampleRecord


def cfg(seed=11, shots=None):
    return DatasetGenerationConfig(
        dataset_name="tiny",
        base_seed=seed,
        circuit_specs=[CircuitGenerationSpec("hardware_efficient_ansatz", 2, {"layers": 1, "entanglement": "none", "measure": True}, 1)],
        distortion_specs=[DistortionSpec("rx_overrotation", {"strength": 0.3, "qubits": [0]}), DistortionSpec("readout_bitflip_marker", {"probability": 0.1, "qubits": [0]})],
        ideal_shots=shots,
        max_samples=4,
    )


def test_config_validation_and_roundtrip():
    with pytest.raises(ValueError): DatasetGenerationConfig("", 1, [CircuitGenerationSpec("bell",2,{})], [DistortionSpec("rx_overrotation",{"strength":.1})])
    with pytest.raises(ValueError): DatasetGenerationConfig("x", 1, [], [DistortionSpec("rx_overrotation",{"strength":.1})])
    with pytest.raises(ValueError): DatasetGenerationConfig("x", 1, [CircuitGenerationSpec("bell",2,{})], [])
    with pytest.raises(ValueError): CircuitGenerationSpec("bell", 2, {}, 0)
    with pytest.raises(ValueError): DatasetGenerationConfig("x",1,[CircuitGenerationSpec("bell",2,{})],[DistortionSpec("rx_overrotation",{"strength":.1})],max_samples=0)
    with pytest.raises(ValueError): DatasetGenerationConfig("x",1,[CircuitGenerationSpec("bell",2,{},2)],[DistortionSpec("rx_overrotation",{"strength":.1})],max_samples=1)
    c=cfg(); assert config_from_dict(config_to_dict(c)) == c
    with pytest.raises(ValueError): config_from_dict({**config_to_dict(c), "split":"nope"})


def test_seed_derivation_deterministic_and_namespaced():
    a=derive_child_seed(1,"parameter_binding",{"x":1})
    assert a == derive_child_seed(1,"parameter_binding",{"x":1})
    assert a != derive_child_seed(1,"clean_shots",{"x":1})


def test_generation_ids_parameters_counts_and_marker_honesty():
    r1=generate_dataset(cfg(11)); r2=generate_dataset(cfg(11)); r3=generate_dataset(cfg(12))
    assert [s.sample_id for s in r1.samples] == [s.sample_id for s in r2.samples]
    assert [s.parameter_bindings for s in r1.samples] == [s.parameter_bindings for s in r2.samples]
    assert [s.parameter_bindings for s in r1.samples] != [s.parameter_bindings for s in r3.samples]
    assert len(r1.samples)==2
    assert r1.summary["unique_clean_circuit_count"] == 1
    assert len([rec for rec in r1.simulation_records if rec.simulation_mode=="ideal_statevector" and rec.circuit_id==r1.samples[0].clean_circuit_id]) == 1
    for s in r1.samples:
        assert not s.clean_circuit.parameters and not s.distorted_circuit.parameters
        assert s.sample_id and s.clean_circuit_id and s.distorted_circuit_id and s.clean_run_id and s.distorted_run_id and s.distortion_id and s.metric_id
        assert s.clean_result.probabilities and s.distorted_result.probabilities
        assert isinstance(s.born_metrics, BornMetricBundle)
    visible=[s for s in r1.samples if s.metadata["distortion_name"]=="rx_overrotation"][0]
    assert visible.born_metrics.metrics["total_variation"].value > 0
    marker=[s for s in r1.samples if s.metadata["marker_only"]][0]
    assert marker.clean_result.probabilities == marker.distorted_result.probabilities
    assert marker.born_metrics.metadata.get("applicability_warning")
    assert marker.metadata["born_zero_shift"] is True


def test_phase_rz_zero_shift_label():
    c=DatasetGenerationConfig("phase",4,[CircuitGenerationSpec("ghz",2,{"measure":True})],[DistortionSpec("phase_rz_drift",{"strength":0.4,"qubits":[0]})])
    r=generate_dataset(c)
    assert r.samples[0].metadata["born_zero_shift"] is True
    assert "undistorted" not in json.dumps(r.summary).lower()


def test_dataset_sample_record_roundtrip():
    rec=DatasetSampleRecord("s","d","v","c","dc","r","dr","dist","m","bell",2,0,{"x":1.0},5,{})
    rec.validate()
    assert DatasetSampleRecord.from_dict(rec.to_dict()) == rec
    with pytest.raises(ValueError): DatasetSampleRecord("","d","v","c","dc","r","dr","dist","m","bell",0,0,{},0,{}).validate()


def test_duplicate_conflicting_ids_detected():
    from triqto.data_generation.pipeline import _add_unique
    from triqto.storage import CircuitRecord
    a = CircuitRecord("same", "bell", 2, 2, 1, 0, 0, {"role": "clean"})
    b = CircuitRecord("same", "ghz", 2, 2, 1, 0, 0, {"role": "clean"})
    store = {}
    _add_unique(store, a, "same")
    with pytest.raises(ValueError): _add_unique(store, b, "same")


def test_write_dataset_manifests_artifacts_readback_and_overwrite(tmp_path):
    pyarrow=pytest.importorskip("pyarrow")
    r=generate_dataset(cfg(shots=8))
    w=write_dataset(r,tmp_path/"a")
    expected={"sample_manifest","circuit_manifest","simulation_manifest","distortion_manifest","metric_manifest"}
    assert set(w.manifest_paths)==expected
    assert (tmp_path/"a"/"generation_config.json").exists() and (tmp_path/"a"/"dataset_summary.json").exists()
    reader=ManifestReader(tmp_path/"a"/"manifests")
    rows=reader.read_records("sample_manifest")
    assert len(rows)==2
    for manifest in expected: assert reader.read_records(manifest)
    sim_rows=reader.read_records("simulation_manifest")
    circuit_rows=reader.read_records("circuit_manifest")
    refs=[]
    for row in circuit_rows: refs.append(row["metadata"]["artifact_ref"])
    for row in sim_rows:
        for k in ("statevector_ref","probabilities_ref","counts_ref"):
            if row.get(k) is not None and row.get(k) == row.get(k): refs.append(row[k])
    for ref in refs:
        assert not Path(ref).is_absolute()
        assert (tmp_path/"a"/ref).exists()
    with pytest.raises(FileExistsError): write_dataset(r,tmp_path/"a")
    w2=write_dataset(r,tmp_path/"b")
    assert [x["sample_id"] for x in reader.read_records("sample_manifest")] == [x["sample_id"] for x in ManifestReader(tmp_path/"b"/"manifests").read_records("sample_manifest")]


def test_kl_infinity_encoded_and_no_aer_import():
    from triqto.data_generation.pipeline import _metric_values
    from triqto.metrics import compare_born_distributions
    vals=_metric_values(compare_born_distributions({"0":1.0},{"1":1.0}))
    assert vals["kl_clean_to_distorted"] is None
    assert vals["kl_clean_to_distorted__nonfinite"] == "positive_infinity"
    json.dumps(vals, allow_nan=False)
    assert "qiskit_aer" not in sys.modules
