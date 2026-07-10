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


def _encode_parquet_value(value: Any, path: str) -> Any:
    """Replace empty mappings with a reserved one-field struct for Parquet.

    PyArrow cannot persist a struct with no child field. The sentinel is decoded on every
    read, so application-level records still receive exact empty dictionaries at any
    nesting depth. Legitimate data may not use the reserved sentinel key.
    """
    if isinstance(value, Mapping):
        if _EMPTY_MAP_SENTINEL in value:
            raise ValueError(
                f"{path} uses reserved manifest key {_EMPTY_MAP_SENTINEL!r}"
            )
        if not value:
            return {_EMPTY_MAP_SENTINEL: True}
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


def _decode_parquet_value(value: Any, path: str) -> Any:
    """Restore recursively encoded empty mappings after Parquet readback."""
    if (
        not isinstance(value, (str, bytes, Mapping, list, tuple))
        and hasattr(value, "tolist")
    ):
        return _decode_parquet_value(value.tolist(), path)
    if isinstance(value, Mapping):
        sentinel = value.get(_EMPTY_MAP_SENTINEL)
        other_items = {
            key: item for key, item in value.items() if key != _EMPTY_MAP_SENTINEL
        }
        if sentinel is True:
            if any(not _is_missing(item) for item in other_items.values()):
                raise ValueError(
                    f"{path} has an empty-map sentinel alongside real values"
                )
            return {}
        if sentinel is not None and not _is_missing(sentinel):
            raise ValueError(f"{path} has malformed empty-map sentinel value")
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
        """Read a Parquet manifest and restore encoded empty mappings."""
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
