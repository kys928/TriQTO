#!/usr/bin/env python3
"""Freeze the one-shot TriQTO phase/amplitude confirmatory protocol.

This command must run before confirmatory holdout generation. It binds the
predeclared hypotheses, feature regimes, grids, thresholds, seeds, calibration,
uncertainty procedure, development product, repository commit, and software
environment into an immutable protocol marker.

It does not generate examples, access confirmatory labels, fit models, or read
historical v0.1 test data.
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

DEFAULT_CONFIG = Path(
    "configs/v0_2/phase_amplitude_confirmatory_v1.json"
)
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
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
        "triqto.v0_2.phase_amplitude_confirmatory_protocol.v1"
    ):
        raise ValueError("Unexpected confirmatory protocol schema")
    if protocol.get("study_role") != "one_shot_confirmatory":
        raise ValueError("Protocol is not marked one-shot confirmatory")
    if protocol.get("historical_v0_1_test_access_allowed") is not False:
        raise ValueError("Historical v0.1 test access must be forbidden")
    if protocol.get("development_validation_reuse_allowed") is not False:
        raise ValueError("Development validation reuse must be forbidden")

    holdout = protocol["holdout"]
    raw_counts = holdout["raw_label_counts"]
    if int(holdout["total_entities"]) != 160:
        raise ValueError("Confirmatory holdout must contain 160 entities")
    if sum(int(value) for value in raw_counts.values()) != 160:
        raise ValueError("Raw-label counts do not sum to 160")
    if int(holdout["phase_like_entities"]) != int(raw_counts["phase_rz"]):
        raise ValueError("Phase count disagrees with phase_rz count")
    amplitude_count = int(raw_counts["amplitude_rx"]) + int(
        raw_counts["amplitude_ry"]
    )
    if int(holdout["amplitude_like_entities"]) != amplitude_count:
        raise ValueError("Amplitude count disagrees with RX/RY counts")
    if holdout.get("new_circuit_instances_required") is not True:
        raise ValueError("New circuit instances must be required")
    if holdout.get("entity_id_overlap_with_development_forbidden") is not True:
        raise ValueError("Development entity overlap must be forbidden")
    if holdout.get("split_group_overlap_with_development_forbidden") is not True:
        raise ValueError("Development group overlap must be forbidden")

    if tuple(protocol["regimes"]) != (
        "B_absolute",
        "B_delta",
        "C_summary",
    ):
        raise ValueError("Only frozen B_absolute/B_delta/C_summary are allowed")
    if protocol["regimes"]["B_delta"][
        "raw_statevector_components_exposed"
    ] is not False:
        raise ValueError("B_delta must not expose statevector components")
    if protocol["regimes"]["C_summary"][
        "raw_statevector_components_exposed"
    ] is not False:
        raise ValueError("C_summary must not expose statevector components")

    if protocol["tuning"].get("holdout_used_for_tuning") is not False:
        raise ValueError("Holdout tuning is forbidden")
    if protocol["calibration"].get("holdout_used_for_calibration") is not False:
        raise ValueError("Holdout calibration is forbidden")
    if protocol["one_shot_rules"].get(
        "freeze_before_holdout_generation"
    ) is not True:
        raise ValueError("Freeze-before-generation must be enabled")


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
        raise FileNotFoundError(
            f"Development product is unavailable: {generation_complete}"
        )
    generation = read_json(generation_complete)
    if str(generation.get("product_id")) != str(
        protocol["development_product_id"]
    ):
        raise RuntimeError("Development product ID does not match protocol")
    if generation.get("test_split_accessed") is not False:
        raise RuntimeError("Development product lacks historical-test isolation")

    result_root = Path(str(protocol["development_result_dir"])).resolve()
    complete_path = result_root / "complete.json"
    results_path = result_root / "results.json"
    if not complete_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"Development relational-observable result is incomplete: {result_root}"
        )
    complete = read_json(complete_path)
    if complete.get("confirmatory") is not False:
        raise RuntimeError("Development report must be marked non-confirmatory")
    if complete.get("historical_v0_1_test_accessed") is not False:
        raise RuntimeError("Development report lacks historical-test isolation")

    return {
        "product_generation_complete_sha256": sha256_file(
            generation_complete
        ),
        "development_result_complete_sha256": sha256_file(complete_path),
        "development_results_sha256": sha256_file(results_path),
    }


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

    config_sha256 = sha256_text(canonical(protocol))
    study_id = f"confirm_{config_sha256.removeprefix('sha256:')[:20]}"
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

    binding = verify_development_binding(protocol)
    frozen = {
        "schema": "triqto.v0_2.phase_amplitude_confirmatory_freeze.v1",
        "status": "FROZEN",
        "study_id": study_id,
        "protocol_sha256": config_sha256,
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
        "development_binding": binding,
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_holdout_generated": False,
        "confirmatory_metrics_accessed": False,
        "historical_v0_1_test_accessed": False,
    }

    if marker_path.exists():
        existing = read_json(marker_path)
        immutable_fields = (
            "study_id",
            "protocol_sha256",
            "protocol",
            "source",
            "development_binding",
        )
        if any(existing.get(name) != frozen.get(name) for name in immutable_fields):
            raise RuntimeError(
                f"Existing protocol freeze differs from this invocation: {marker_path}"
            )
        print("Confirmatory protocol was already frozen identically.")
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
        "protocol_sha256": config_sha256,
    }
    if pointer.exists() and read_json(pointer) != pointer_value:
        raise RuntimeError(
            f"Confirmatory root already points to another protocol: {pointer}"
        )
    if not pointer.exists():
        atomic_json(pointer, pointer_value)

    print()
    print("=" * 78)
    print("TRIQTO PHASE/AMPLITUDE CONFIRMATORY PROTOCOL FROZEN")
    print("=" * 78)
    print(f"study_id: {study_id}")
    print(f"protocol_sha256: {config_sha256}")
    print(f"git_commit: {git_commit}")
    print(f"study_root: {study_root}")
    print("holdout generated: NO")
    print("confirmatory metrics accessed: NO")
    print("historical v0.1 test accessed: NO")


if __name__ == "__main__":
    main()
