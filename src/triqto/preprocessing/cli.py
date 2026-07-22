"""Command-line interface for immutable TriQTO dataset preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .config import load_preprocessing_config
from .pipeline import preprocess_phase7_dataset
from .splits import verify_saved_split_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triqto-preprocess",
        description=(
            "Validate and preprocess a completed immutable TriQTO Phase 7 dataset "
            "into a fresh auditable preprocessing run."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the complete preprocessing DAG")
    _add_run_arguments(run)
    validate = subparsers.add_parser(
        "validate", help="Run ingestion, schema, numerical and physical validation only"
    )
    _add_run_arguments(validate, validation_only_default=True)
    dry = subparsers.add_parser(
        "dry-run", help="Resolve and validate the run plan without writing outputs"
    )
    _add_run_arguments(dry, dry_run_default=True)
    verify = subparsers.add_parser(
        "verify-splits", help="Independently verify split assignments and leakage invariants"
    )
    verify.add_argument("--preprocessing-root", required=True, type=Path)
    verify.add_argument("--split-name", action="append", default=[])
    return parser


def _add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    validation_only_default: bool = False,
    dry_run_default: bool = False,
) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, dest="input_root")
    parser.add_argument("--output", required=True, type=Path, dest="output_root")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validation-only", action="store_true", default=validation_only_default)
    parser.add_argument("--dry-run", action="store_true", default=dry_run_default)


def _progress(payload: dict[str, Any]) -> None:
    stage = payload.get("stage", "preprocessing")
    completed = payload.get("completed")
    total = payload.get("total")
    message = str(payload.get("message", ""))
    if completed is not None and total is not None:
        print(f"[{stage}] {completed}/{total} {message}".rstrip(), flush=True)
    else:
        print(f"[{stage}] {message}".rstrip(), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-splits":
            result = verify_saved_split_directory(
                args.preprocessing_root,
                split_names=tuple(args.split_name) or None,
            )
        else:
            config = load_preprocessing_config(args.config)
            result = preprocess_phase7_dataset(
                phase7_root=args.input_root,
                output_root=args.output_root,
                config=config,
                run_id=args.run_id,
                validation_only=bool(args.validation_only),
                dry_run=bool(args.dry_run),
                progress_callback=_progress,
            )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        status = str(result.get("status", ""))
        return 0 if status in {"complete", "dry_run", "valid"} else 2
    except KeyboardInterrupt:
        print("Preprocessing interrupted; no incomplete run was published.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"TriQTO preprocessing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
