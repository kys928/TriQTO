"""Manifest reader and writer for TriQTO's manifest-centered data lake.

The implementation supports small-to-medium Parquet manifests backed by pandas and
pyarrow. Manifests hold metadata rows and references to large external artifacts; the
writer deliberately does not serialize large tensors inline.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from triqto.storage.schema import ManifestRecordMixin

_EMPTY_MAP_SENTINEL = "__triqto_parquet_empty_map_v1__"
_NUMERIC_MAP_SENTINEL = "__triqto_parquet_numeric_map_v1__"
_RESERVED_MAP_KEYS = frozenset({_EMPTY_MAP_SENTINEL, _NUMERIC_MAP_SENTINEL})


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a dataclass record or mapping into a plain manifest row."""
    if isinstance(record, ManifestRecordMixin):
        record.validate()
        return record.to_dict()
    if is_dataclass(record):
        from dataclasses import asdict

        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"Unsupported manifest record type: {type(record)!r}")


def _is_numeric_map(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping can use the stable numeric-map Parquet encoding."""
    return bool(value) and all(
        isinstance(key, str)
        and isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for key, item in value.items()
    )


def _encode_parquet_value(value: Any, path: str) -> Any:
    """Encode mappings into Parquet-stable recursive representations.

    PyArrow cannot persist a struct with no child field, so empty mappings use a
    reserved one-field sentinel. Numeric mappings with dynamic keys (for example
    circuit parameter bindings) are encoded as a sorted list of name/value records.
    Encoding those maps as ordinary structs is unsafe when rows have different keys:
    Arrow can otherwise infer or materialize an incorrect scalar type during a mixed
    manifest round trip.
    """
    if isinstance(value, Mapping):
        reserved = _RESERVED_MAP_KEYS.intersection(value)
        if reserved:
            raise ValueError(
                f"{path} uses reserved manifest key {sorted(reserved)[0]!r}"
            )
        if not value:
            return {_EMPTY_MAP_SENTINEL: True}
        if _is_numeric_map(value):
            return {
                _NUMERIC_MAP_SENTINEL: [
                    {"name": key, "value": float(value[key])}
                    for key in sorted(value)
                ]
            }
        return {
            key: _encode_parquet_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _encode_parquet_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return [
            _encode_parquet_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _decode_numeric_map(value: Any, path: str) -> dict[str, float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} has malformed numeric-map sentinel value")
    decoded: dict[str, float] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}[{index}] must be a name/value mapping")
        name = entry.get("name")
        raw_value = entry.get("value")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}[{index}].name must be a nonblank string")
        if name in decoded:
            raise ValueError(f"{path} contains duplicate numeric-map key {name!r}")
        if (
            not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
            or not math.isfinite(float(raw_value))
        ):
            raise ValueError(f"{path}[{index}].value must be finite numeric data")
        decoded[name] = float(raw_value)
    return decoded


def _decode_parquet_value(value: Any, path: str) -> Any:
    """Restore recursively encoded mappings after Parquet readback."""
    if (
        not isinstance(value, (str, bytes, Mapping, list, tuple))
        and hasattr(value, "tolist")
    ):
        return _decode_parquet_value(value.tolist(), path)
    if isinstance(value, Mapping):
        empty_sentinel = value.get(_EMPTY_MAP_SENTINEL)
        numeric_sentinel = value.get(_NUMERIC_MAP_SENTINEL)
        other_items = {
            key: item for key, item in value.items() if key not in _RESERVED_MAP_KEYS
        }
        if empty_sentinel is True:
            if numeric_sentinel is not None and not _is_missing(numeric_sentinel):
                raise ValueError(
                    f"{path} has both empty-map and numeric-map sentinels"
                )
            if any(not _is_missing(item) for item in other_items.values()):
                raise ValueError(
                    f"{path} has an empty-map sentinel alongside real values"
                )
            return {}
        if empty_sentinel is not None and not _is_missing(empty_sentinel):
            raise ValueError(f"{path} has malformed empty-map sentinel value")
        if numeric_sentinel is not None and not _is_missing(numeric_sentinel):
            if any(not _is_missing(item) for item in other_items.values()):
                raise ValueError(
                    f"{path} has a numeric-map sentinel alongside real values"
                )
            return _decode_numeric_map(numeric_sentinel, path)
        return {
            key: _decode_parquet_value(item, f"{path}.{key}")
            for key, item in other_items.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _decode_parquet_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


class ManifestWriter:
    """Write TriQTO manifest rows to Parquet files under a manifest root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def manifest_path(self, manifest_name: str) -> Path:
        """Return a normalized `.parquet` path for a manifest name."""
        name = (
            manifest_name
            if manifest_name.endswith(".parquet")
            else f"{manifest_name}.parquet"
        )
        return self.root / name

    def write_records(
        self,
        manifest_name: str,
        records: Iterable[Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write records to a Parquet manifest and return the output path."""
        rows = [
            _encode_parquet_value(
                _record_to_dict(record),
                f"{manifest_name}[{index}]",
            )
            for index, record in enumerate(records)
        ]
        path = self.manifest_path(manifest_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pandas and pyarrow are required for Parquet manifest writing"
            ) from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path


class ManifestReader:
    """Read TriQTO Parquet manifests as row dictionaries or schema records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def manifest_path(self, manifest_name: str) -> Path:
        """Return a normalized `.parquet` path for a manifest name."""
        name = (
            manifest_name
            if manifest_name.endswith(".parquet")
            else f"{manifest_name}.parquet"
        )
        return self.root / name

    def read_records(self, manifest_name: str) -> list[dict[str, Any]]:
        """Read a Parquet manifest and restore encoded mappings."""
        path = self.manifest_path(manifest_name)
        if not path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pandas and pyarrow are required for Parquet manifest reading"
            ) from exc
        raw_rows = pd.read_parquet(path).to_dict(orient="records")
        decoded: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_rows):
            value = _decode_parquet_value(raw, f"{manifest_name}[{index}]")
            if not isinstance(value, dict):
                raise TypeError("Decoded manifest row must be a dictionary")
            decoded.append(value)
        return decoded

    def read_typed_records(
        self,
        manifest_name: str,
        record_type: type[ManifestRecordMixin],
    ) -> list[ManifestRecordMixin]:
        """Read a manifest and instantiate validated schema records."""
        records = [
            record_type.from_dict(row)
            for row in self.read_records(manifest_name)
        ]
        for record in records:
            record.validate()
        return records
