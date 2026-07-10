"""Strict JSON and path helpers shared by Phase 8 modules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON constant: {value}")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-finite constants."""
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object_pairs,
    )


def strict_json_load(path: str | Path) -> Any:
    return strict_json_loads(Path(path).read_text())


def strict_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":") if indent is None else None,
    )


def write_strict_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(strict_json_dumps(payload, indent=2) + "\n")
    return target


def json_copy(payload: Any) -> Any:
    return strict_json_loads(strict_json_dumps(payload, indent=None))


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def require_nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-whitespace text")
    return value.strip()


def require_exact_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def normalize_relative_posix_ref(reference: Any, name: str) -> str:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    if "\\" in reference:
        raise ValueError(f"{name} must use POSIX separators")
    parsed = PurePosixPath(reference)
    if parsed.is_absolute() or parsed == PurePosixPath("."):
        raise ValueError(f"{name} must be a relative file path")
    if any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{name} is not normalized or contains traversal")
    normalized = parsed.as_posix()
    if normalized != reference:
        raise ValueError(f"{name} must be normalized POSIX path: {reference!r}")
    return normalized


def resolve_safe_file(root: str | Path, reference: Any, name: str) -> Path:
    ref = normalize_relative_posix_ref(reference, name)
    base = Path(root).resolve()
    candidate = (base / Path(*PurePosixPath(ref).parts)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{name} escapes root: {ref}") from exc
    if not candidate.exists():
        raise FileNotFoundError(f"{name} references missing file: {ref}")
    if not candidate.is_file():
        raise ValueError(f"{name} does not reference a file: {ref}")
    return candidate


def ensure_sorted_unique_strings(values: Sequence[Any], name: str) -> tuple[str, ...]:
    normalized = tuple(require_nonblank(value, f"{name} entry") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates")
    if tuple(sorted(normalized)) != normalized:
        raise ValueError(f"{name} must be sorted")
    return normalized


__all__ = [
    "ensure_sorted_unique_strings",
    "json_copy",
    "normalize_relative_posix_ref",
    "require_exact_bool",
    "require_mapping",
    "require_nonblank",
    "require_positive_int",
    "resolve_safe_file",
    "strict_json_dumps",
    "strict_json_load",
    "strict_json_loads",
    "write_strict_json",
]
