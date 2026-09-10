from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v0_2"))

import generate_step14_equivalence_aware_fresh_holdout_redacted as redacted  # noqa: E402
import runpod_step14_training_worker as worker  # noqa: E402


def test_x_only_redaction_preserves_raw_bytes_and_x_values(tmp_path: Path) -> None:
    raw_rel = Path("cross_motif_artifacts/outer_validation/example.npz")
    raw_path = tmp_path / raw_rel
    raw_path.parent.mkdir(parents=True)
    x_numeric = np.asarray([[1.0, -2.5], [3.25, 0.0]], dtype=np.float64)
    x_names = np.asarray(["h", "cx", "rz"], dtype="<U2")
    np.savez_compressed(
        raw_path,
        x__numeric=x_numeric,
        x__names=x_names,
        y__mechanism_target=np.asarray([2], dtype=np.int8),
        audit__affected_qubit=np.asarray([1], dtype=np.int16),
        meta__example_id=np.asarray(["example"], dtype="<U7"),
    )
    raw_sha = redacted.baseline.sha256_file(raw_path)
    full = [{
        "example_id": "sha256:abc",
        "artifact_path": raw_rel.as_posix(),
        "artifact_sha256": raw_sha,
    }]
    projected = [{
        "root_index": 1,
        "family_id": "family",
        "step14_partition": "fresh_equivalence_holdout",
        "example_id": "sha256:abc",
        "artifact_path": raw_rel.as_posix(),
        "artifact_sha256": raw_sha,
        "evaluation_role": "target",
    }]

    blind_rows, audit = redacted.redact_artifacts(
        tmp_path, full, projected, {"x__numeric", "x__names"}
    )

    sealed = tmp_path / full[0]["artifact_path"]
    blind = tmp_path / blind_rows[0]["artifact_path"]
    assert redacted.baseline.sha256_file(sealed) == raw_sha
    with np.load(sealed, allow_pickle=False) as z:
        assert "y__mechanism_target" in z.files
        assert "audit__affected_qubit" in z.files
        assert "meta__example_id" in z.files
    with np.load(blind, allow_pickle=False) as z:
        assert set(z.files) == {"x__numeric", "x__names"}
        np.testing.assert_array_equal(z["x__numeric"], x_numeric)
        np.testing.assert_array_equal(z["x__names"], x_names)
    assert audit["raw_bytes_preserved"] is True
    assert audit["x_array_names_dtypes_shapes_values_preserved"] is True
    assert audit["numeric_measurements_transformed"] is False
    assert audit["forbidden_prefix_array_count_removed"] == {"audit__": 1, "meta__": 1, "y__": 1}


def test_fresh_holdout_worker_routes_to_redacted_generator() -> None:
    job = {
        "task": {
            "runner": "step14_training_selection",
            "command": "generate_equivalence_aware_fresh_holdout",
            "workspace": "/workspace/triqto-data/step14_cross_motif_training",
            "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
            "progress_every": 10,
            "expected_training_run_id": "training_18e0b4ed6e685af30b6c4a35",
            "expected_selection_freeze_sha256": "sha256:" + "a" * 64,
            "expected_final_method_freeze_payload_sha256": "sha256:" + "b" * 64,
        }
    }
    command = worker.build_command(job)
    joined = " ".join(command)
    assert "generate_step14_equivalence_aware_fresh_holdout_redacted.py" in joined
    assert "--final-method-freeze-payload-sha256" in joined
    assert "--device" not in joined
