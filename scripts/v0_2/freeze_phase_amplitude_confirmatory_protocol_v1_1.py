#!/usr/bin/env python3
"""Freeze the fully implemented TriQTO confirmatory protocol v1.1.

Version 1.1 supersedes the untouched v1 freeze by binding the holdout generator
and one-shot evaluator into the immutable marker before any holdout generation.
It does not generate data, access confirmatory labels, fit models, or read the
historical v0.1 test split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG = Path("configs/v0_2/phase_amplitude_confirmatory_v1_1.json")
REQUIRED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "qiskit",
    "qiskit-aer",
    "ripser",
    "gudhi",
    "torch",
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != (
        "triqto.v0_2.phase_amplitude_confirmatory_protocol.v1_1"
    ):
        raise ValueError("Unexpected confirmatory protocol schema")
    if protocol.get("study_role") != "one_shot_confirmatory":
        raise ValueError("Protocol is not marked one-shot confirmatory")
    if protocol.get("historical_v0_1_test_access_allowed") is not False:
        raise ValueError("Historical v0.1 test access must be forbidden")
    if protocol.get("development_validation_reuse_allowed") is not False:
        raise ValueError("Development validation reuse must be forbidden")
    if protocol.get(
        "development_validation_feature_artifact_access_allowed"
    ) is not False:
        raise ValueError("Development validation feature access must be forbidden")

    implementation = protocol.get("implementation", {})
    required_implementation = {
        "freeze_script",
        "holdout_generator",
        "holdout_generator_payload",
        "one_shot_evaluator",
        "one_shot_evaluator_payload",
        "base_evaluator",
        "relational_feature_runner",
    }
    if set(implementation) != required_implementation:
        raise ValueError("Implementation path set is incomplete or unexpected")

    holdout = protocol["holdout"]
    total = int(holdout["total_entities"])
    if total != 160:
        raise ValueError("Confirmatory holdout must contain 160 entities")
    if sum(int(item["count"]) for item in holdout["strata"]) != total:
        raise ValueError("Frozen stratum counts do not sum to 160")
    if sum(
        int(item["count"])
        for item in holdout["raw_label_strength_counts"]
    ) != total:
        raise ValueError("Frozen label-strength counts do not sum to 160")
    raw_counts: dict[str, int] = {}
    coarse_counts: dict[str, int] = {}
    for item in holdout["raw_label_strength_counts"]:
        raw = str(item["raw_label"])
        coarse = str(item["coarse_label"])
        count = int(item["count"])
        raw_counts[raw] = raw_counts.get(raw, 0) + count
        coarse_counts[coarse] = coarse_counts.get(coarse, 0) + count
    if raw_counts != {
        key: int(value) for key, value in holdout["raw_label_counts"].items()
    }:
        raise ValueError("Raw-label totals disagree with label-strength rows")
    if coarse_counts != {"phase_like": 80, "amplitude_like": 80}:
        raise ValueError("Coarse labels are not balanced 80/80")
    if holdout.get("new_circuit_instances_required") is not True:
        raise ValueError("New circuit instances must be required")
    if holdout.get("entity_id_overlap_with_development_forbidden") is not True:
        raise ValueError("Development entity overlap must be forbidden")
    if holdout.get("split_group_overlap_with_development_forbidden") is not True:
        raise ValueError("Development group overlap must be forbidden")

    regimes = protocol["regimes"]
    if tuple(regimes) != ("B_absolute", "B_delta", "C_summary"):
        raise ValueError("Only B_absolute/B_delta/C_summary are allowed")
    if any(
        regimes[name].get("raw_statevector_components_exposed") is not False
        for name in regimes
    ):
        raise ValueError("Raw statevector components must not be exposed")

    schema = protocol["feature_schema"]
    if int(schema["max_state_dimension"]) != 256:
        raise ValueError("Frozen state dimension must be 256")
    if int(schema["max_qubits"]) != 8:
        raise ValueError("Frozen max-qubit count must be 8")
    if tuple(schema["hilbert_summary_names"]) != (
        "aligned_state_l2_delta",
        "fidelity",
        "fubini_study_distance",
        "magnitude_l1_delta",
        "magnitude_l2_delta",
    ):
        raise ValueError("Unexpected Hilbert summary schema")

    if protocol["tuning"].get("holdout_used_for_tuning") is not False:
        raise ValueError("Holdout tuning is forbidden")
    if protocol["calibration"].get("holdout_used_for_calibration") is not False:
        raise ValueError("Holdout calibration is forbidden")
    rules = protocol["one_shot_rules"]
    if rules.get("freeze_before_holdout_generation") is not True:
        raise ValueError("Freeze-before-generation must be enabled")
    if rules.get("generate_all_blinded_predictions_before_label_access") is not True:
        raise ValueError("Blinded predictions must precede label access")


def package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required package is missing: {package}") from exc
    return output


def verify_development_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    product_root = Path(str(protocol["development_product_dir"])).resolve()
    generation_complete = product_root / "generation_complete.json"
    if not generation_complete.is_file():
        raise FileNotFoundError(generation_complete)
    generation = read_json(generation_complete)
    if generation.get("product_id") != protocol["development_product_id"]:
        raise RuntimeError("Development product ID does not match protocol")
    if generation.get("test_split_accessed") is not False:
        raise RuntimeError("Development product lacks historical-test isolation")

    result_root = Path(str(protocol["development_result_dir"])).resolve()
    complete_path = result_root / "complete.json"
    results_path = result_root / "results.json"
    if not complete_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"Development result is incomplete: {result_root}")
    complete = read_json(complete_path)
    if complete.get("confirmatory") is not False:
        raise RuntimeError("Development result must be non-confirmatory")
    if complete.get("historical_v0_1_test_accessed") is not False:
        raise RuntimeError("Development result lacks historical-test isolation")
    return {
        "product_generation_complete_sha256": sha256_file(generation_complete),
        "development_result_complete_sha256": sha256_file(complete_path),
        "development_results_sha256": sha256_file(results_path),
    }


def verify_superseded_freeze(protocol: Mapping[str, Any]) -> dict[str, Any]:
    supersedes = protocol["supersedes"]
    old_study_id = str(supersedes["study_id"])
    old_root = (
        Path("/workspace/triqto-data/phase15_6_confirmatory_v1").resolve()
        / old_study_id
    )
    old_freeze = old_root / "protocol_frozen.json"
    if not old_freeze.is_file():
        raise FileNotFoundError(f"Superseded freeze is missing: {old_freeze}")
    old = read_json(old_freeze)
    if old.get("study_id") != old_study_id or old.get("status") != "FROZEN":
        raise RuntimeError("Superseded study marker is invalid")
    forbidden = [
        name
        for name in (
            "holdout_generated.json",
            "evaluation_started.json",
            "labels_accessed.json",
            "evaluation_complete.json",
        )
        if (old_root / name).exists()
    ]
    if forbidden:
        raise RuntimeError(
            "Cannot supersede a study that progressed beyond freeze: "
            + ", ".join(forbidden)
        )
    return {
        "study_id": old_study_id,
        "protocol_frozen_sha256": sha256_file(old_freeze),
        "holdout_generated": False,
        "confirmatory_metrics_accessed": False,
        "reason": str(supersedes["reason"]),
    }


def implementation_hashes(
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in protocol["implementation"].values():
        path = repository_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Implementation file is missing: {path}")
        output[str(relative)] = sha256_file(path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    protocol = read_json(config_path)
    validate_protocol(protocol)

    protocol_sha256 = sha256_text(canonical(protocol))
    study_id = f"confirm_{protocol_sha256.removeprefix('sha256:')[:20]}"
    configured_root = Path(str(protocol["confirmatory_root"])).resolve()
    output_root = (
        arguments.output_root.expanduser().resolve()
        if arguments.output_root is not None
        else configured_root
    )
    study_root = output_root / study_id
    marker_path = study_root / "protocol_frozen.json"

    repository_root = Path(git_output("rev-parse", "--show-toplevel")).resolve()
    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--porcelain")
    if git_status:
        raise RuntimeError(
            "Refusing to freeze from a dirty working tree:\n" + git_status
        )

    development_binding = verify_development_binding(protocol)
    supersession_binding = verify_superseded_freeze(protocol)
    file_hashes = implementation_hashes(repository_root, protocol)
    frozen = {
        "schema": "triqto.v0_2.phase_amplitude_confirmatory_freeze.v1_1",
        "status": "FROZEN",
        "study_id": study_id,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "source": {
            "config_path_relative_to_repo": str(
                config_path.relative_to(repository_root)
            ),
            "config_file_sha256": sha256_file(config_path),
            "freeze_script_path_relative_to_repo": str(
                Path(__file__).resolve().relative_to(repository_root)
            ),
            "freeze_script_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit,
            "git_worktree_clean": True,
        },
        "implementation_binding": {
            "files": file_hashes,
            "all_implementation_files_frozen_before_holdout_generation": True,
        },
        "development_binding": development_binding,
        "supersession_binding": supersession_binding,
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_holdout_generated": False,
        "confirmatory_metrics_accessed": False,
        "historical_v0_1_test_accessed": False,
        "development_validation_feature_artifacts_accessed": False,
    }

    if marker_path.exists():
        existing = read_json(marker_path)
        immutable = (
            "study_id",
            "protocol_sha256",
            "protocol",
            "source",
            "implementation_binding",
            "development_binding",
            "supersession_binding",
        )
        if any(existing.get(name) != frozen.get(name) for name in immutable):
            raise RuntimeError(
                f"Existing protocol freeze differs from invocation: {marker_path}"
            )
        print("Confirmatory protocol v1.1 was already frozen identically.")
    else:
        if study_root.exists() and any(study_root.iterdir()):
            raise RuntimeError(
                f"Refusing to freeze into non-empty directory: {study_root}"
            )
        atomic_json(marker_path, frozen)

    pointer = output_root / "current_protocol.json"
    pointer_value = {
        "study_id": study_id,
        "study_root": str(study_root),
        "protocol_frozen": str(marker_path),
        "protocol_sha256": protocol_sha256,
    }
    if pointer.exists() and read_json(pointer) != pointer_value:
        raise RuntimeError(
            f"Confirmatory root already points to another protocol: {pointer}"
        )
    if not pointer.exists():
        atomic_json(pointer, pointer_value)

    print()
    print("=" * 78)
    print("TRIQTO CONFIRMATORY PROTOCOL V1.1 FROZEN")
    print("=" * 78)
    print(f"study_id: {study_id}")
    print(f"protocol_sha256: {protocol_sha256}")
    print(f"git_commit: {git_commit}")
    print(f"study_root: {study_root}")
    print(f"supersedes: {supersession_binding['study_id']}")
    print("implementation scripts bound: YES")
    print("holdout generated: NO")
    print("confirmatory metrics accessed: NO")
    print("historical v0.1 test accessed: NO")


if __name__ == "__main__":
    main()
