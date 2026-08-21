from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "v0_2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_step10c_frozen_scientific_delta_is_only_horizon():
    step10b = _json(ROOT / "configs/v0_2/step10_warmstart_vs_scratch.json")
    step10c = _json(ROOT / "configs/v0_2/step10c_crashsafe_long_horizon.json")
    assert step10c["status"] == "FROZEN_BEFORE_STEP10C_TRAINING_OUTCOME"
    assert step10c["architecture"]["variant"] == step10b["architecture"]["variant"] == "late_concat"
    assert step10c["architecture"]["expected_trainable_parameter_count"] == 453829
    for key in (
        "root_batch_size",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "effect_loss_weight",
        "mechanism_loss_weight",
        "domain_schedule",
    ):
        assert step10c["training"][key] == step10b["training"][key]
    assert step10b["training"]["max_epochs"] == 20
    assert step10c["training"]["max_epochs"] == 40
    assert step10c["selection"]["eligible_checkpoint_order"] == step10b["selection"]["eligible_checkpoint_order"]
    assert step10c["evaluation"]["bridge_mechanism_gate"] == step10b["evaluation"]["bridge_mechanism_gate"]
    assert step10c["evaluation"]["original_retention_gate"]["mechanism_balanced_accuracy_max_drop"] == 0.02
    assert step10c["evaluation"]["original_retention_gate"]["effect_balanced_accuracy_max_drop"] == 0.02


def test_step10c_outer_is_fresh_outer_only_contract():
    cfg = _json(ROOT / "configs/v0_2/step10c_fresh_outer_cohort.json")
    assert cfg["status"] == "FROZEN_BEFORE_STEP10C_OUTER_OUTCOME"
    assert cfg["source_training_mixture"]["step10b_outer_is_spent"] is True
    assert cfg["original_domain"]["global_root_index_start"] == 5000
    assert cfg["original_domain"]["clean_root_count"] == 1000
    assert cfg["original_domain"]["reuse_previous_outer_rows"] is False
    assert cfg["bridge_domain"]["base_seed"] == 2026082101
    assert cfg["bridge_domain"]["parent_groups"] == 60
    assert cfg["bridge_domain"]["variants_per_parent"] == 8
    assert cfg["bridge_domain"]["expected_clean_roots"] == 480
    assert cfg["bridge_domain"]["exact_step9d_pilot_graph_forbidden"] is True


def test_step10c_architecture_parameter_count_unchanged():
    step7 = _load("step10c_test_step7", "run_step7_full_development_benchmark.py")
    experiment = _json(ROOT / "configs/v0_2/step7_structured_diagnostic_model.json")
    model = step7.instantiate_model("late_concat", 1701, experiment, torch.device("cpu"))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 453829


def test_atomic_checkpoint_helpers_roundtrip(tmp_path: Path):
    runner = _load("step10c_test_runner", "run_step10c_crashsafe_long_horizon.py")
    json_path = tmp_path / "progress.json"
    runner.atomic_json_fsync(json_path, {"status": "ok", "epoch": 7})
    assert _json(json_path) == {"epoch": 7, "status": "ok"}

    checkpoint = tmp_path / "resume.pt"
    runner.atomic_torch_save(checkpoint, {"schema": runner.SCHEMA, "tensor": torch.arange(4)})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == runner.SCHEMA
    assert torch.equal(payload["tensor"], torch.arange(4))
    assert not list(tmp_path.glob("*.tmp-*"))


def test_resume_identity_is_exact_not_partial():
    runner = _load("step10c_test_identity", "run_step10c_crashsafe_long_horizon.py")
    identity = {
        "schema": runner.SCHEMA,
        "training_config_sha256": "sha256:a",
        "runner_sha256": "sha256:b",
        "mixture_product_id": "mix",
        "fresh_outer_product_id": "outer",
        "step9a_bundle_id": "bundle",
        "initialization": "warm_start",
        "seed": 1701,
        "architecture": "late_concat",
        "trainable_parameter_count": 453829,
        "execution_device": "cpu",
        "runtime_fingerprint": {
            "python_version": "3.12.x",
            "torch_version": "2.8.0+cpu",
            "numpy_version": "2.3.2",
            "device_type": "cpu",
            "torch_num_threads": 8,
            "cuda_version_if_any": None,
            "cuda_device_name_if_any": None,
        },
    }
    assert runner._identity_equal(identity, dict(identity))
    changed = dict(identity)
    changed["seed"] = 1702
    assert not runner._identity_equal(identity, changed)


def test_resume_identity_rejects_runtime_change():
    runner = _load("step10c_test_runtime_identity", "run_step10c_crashsafe_long_horizon.py")
    base = {
        "schema": runner.SCHEMA,
        "training_config_sha256": "sha256:a",
        "runner_sha256": "sha256:b",
        "mixture_product_id": "mix",
        "fresh_outer_product_id": "outer",
        "step9a_bundle_id": "bundle",
        "initialization": "warm_start",
        "seed": 1701,
        "architecture": "late_concat",
        "trainable_parameter_count": 453829,
        "execution_device": "cpu",
        "runtime_fingerprint": {"python_version": "3.12.3", "torch_version": "2.8.0+cpu", "numpy_version": "2.3.2", "device_type": "cpu", "torch_num_threads": 8, "cuda_version_if_any": None, "cuda_device_name_if_any": None},
    }
    changed = json.loads(json.dumps(base))
    changed["runtime_fingerprint"]["torch_num_threads"] = 16
    assert not runner._identity_equal(base, changed)
