from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0_2/audit_step5_v3_stage_nesting.py"
SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v3"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_product(root: Path, root_count: int, mutate: bool = False) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "dataset_complete.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "COMPLETE",
                "product_id": f"product-{root_count}",
                "clean_circuit_root_count": root_count,
            }
        ),
        encoding="utf-8",
    )
    roots = []
    examples = []
    for index in range(root_count):
        roots.append(
            {
                "root_index": index,
                "clean_circuit_group_id": f"group-{index}",
                "split": "validation" if index % 5 == 0 else "train",
                "family": "toy",
                "n_qubits": 3,
                "graph_sha256": f"graph-{index}",
            }
        )
        examples.append(
            {
                "example_id": f"example-{index}",
                "root_index": index,
                "clean_circuit_group_id": f"group-{index}",
                "split": "validation" if index % 5 == 0 else "train",
                "artifact_path": f"artifacts/example-{index}.npz",
                "artifact_sha256": f"artifact-{index}",
            }
        )
    if mutate:
        examples[0]["artifact_sha256"] = "changed"
    write_csv(root / "manifests/clean_circuit_manifest.csv", roots)
    write_csv(root / "manifests/example_manifest.csv", examples)


def run(previous: Path, current: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--previous-product-dir",
            str(previous),
            "--current-product-dir",
            str(current),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_exact_nested_product(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    output = tmp_path / "nesting.json"
    make_product(previous, 2)
    make_product(current, 4)
    result = run(previous, current, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Decision: NESTING_VALID" in result.stdout
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["decision"] == "NESTING_VALID"
    assert evidence["previous_product_id"] == "product-2"
    assert evidence["current_product_id"] == "product-4"
    assert evidence["artifact_hash_mismatch_count"] == 0


def test_rejects_changed_prior_artifact(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    output = tmp_path / "nesting.json"
    make_product(previous, 2)
    make_product(current, 4, mutate=True)
    result = run(previous, current, output)
    assert result.returncode != 0
    assert "Decision: BLOCKED" in result.stdout
    assert "Artifact-hash mismatches: 1" in result.stdout
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["decision"] == "BLOCKED"
    assert evidence["artifact_hash_mismatch_count"] == 1
