"""Atomic, hashable output artifacts for preprocessing runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
import uuid


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_value(item) for item in value), key=repr)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_value(value.tolist())
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("complex JSON value contains NaN or infinity")
        return {"real": value.real, "imag": value.imag}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON value contains NaN or infinity")
        return value
    raise TypeError(f"Unsupported strict JSON value type: {type(value)!r}")


def strict_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
    )


def atomic_write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, strict_json_dumps(payload) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    lines = [strict_json_dumps(dict(row), indent=None) for row in rows]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _parquet_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return strict_json_dumps(value, indent=None)
    return value


def write_parquet(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = [
        {str(key): _parquet_scalar(value) for key, value in dict(row).items()}
        for row in rows
    ]
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas and pyarrow are required for preprocessing Parquet output") from exc
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    frame = (
        pd.DataFrame(materialized)
        if materialized
        else pd.DataFrame({"__triqto_empty__": pd.Series(dtype="bool")})
    )
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return target


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def output_inventory(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in base.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(base).as_posix(),
    ):
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(base).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def create_staging_directory(output_root: str | Path, run_id: str) -> tuple[Path, Path]:
    base = Path(output_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    final = base / run_id
    if final.exists():
        raise FileExistsError(f"preprocessing run already exists: {final}")
    staging = base / f".{run_id}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    return staging, final


def publish_staging_directory(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite preprocessing run: {final}")
    os.replace(staging, final)


def discard_staging_directory(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
