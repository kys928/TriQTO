from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

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


def test_outer_boundary_requires_all_six_selections(tmp_path: Path):
    strict = _load(
        "step10c_strict_boundary_incomplete",
        "run_step10c_crashsafe_long_horizon_strict.py",
    )
    records = {
        "warm_start__seed1701": {"selected_epoch": 10},
    }
    with pytest.raises(RuntimeError, match="all six Step-10C selections"):
        strict._write_outer_boundary_marker(
            tmp_path / "outer_evaluation_started.json",
            benchmark_id="benchmark_test",
            outer_product_id="outer_test",
            selected_records=records,
        )


def test_outer_boundary_marker_roundtrip_after_all_six(tmp_path: Path):
    strict = _load(
        "step10c_strict_boundary_complete",
        "run_step10c_crashsafe_long_horizon_strict.py",
    )
    records = {
        f"{initialization}__seed{seed}": {"selected_epoch": 17 + (seed % 3)}
        for initialization in ("warm_start", "scratch")
        for seed in (1701, 1702, 1703)
    }
    path = tmp_path / "outer_evaluation_started.json"
    marker = strict._write_outer_boundary_marker(
        path,
        benchmark_id="benchmark_test",
        outer_product_id="outer_test",
        selected_records=records,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert marker["all_six_seed_selections_complete"] is True
    assert loaded["selected_run_count_before_outer_materialization"] == 6
    assert loaded["fresh_outer_materialized_during_training"] is False
    assert loaded["fresh_outer_used_for_epoch_selection"] is False
    assert loaded["fresh_outer_used_for_threshold_selection"] is False
    assert set(loaded["selected_epochs"]) == set(records)


def test_authoritative_main_defers_fresh_outer_materialization():
    strict = _load(
        "step10c_strict_boundary_source",
        "run_step10c_crashsafe_long_horizon_strict.py",
    )
    source = inspect.getsource(strict.main)
    boundary = source.index("boundary_marker = _write_outer_boundary_marker")
    original_outer = source.index("fresh_original_outer = step7.materialize_blocks")
    bridge_outer = source.index("fresh_bridge_outer = step7.materialize_blocks")
    outer_prediction = source.index("outer_predictions: dict")

    # The executable sequencing is the contract. Do not rely on a comment or
    # prose sentinel: neither fresh outer artifact set may be materialized, nor
    # may the outer-prediction phase begin, before the six-selection boundary
    # marker is successfully written.
    assert boundary < original_outer < outer_prediction
    assert boundary < bridge_outer < outer_prediction

    before_boundary = source[:boundary]
    assert "fresh_original_outer = step7.materialize_blocks" not in before_boundary
    assert "fresh_bridge_outer = step7.materialize_blocks" not in before_boundary
    assert "outer_predictions: dict" not in before_boundary
    assert "step7.predict_blocks(model, fresh_original_outer" not in before_boundary
    assert "step7.predict_blocks(model, fresh_bridge_outer" not in before_boundary
