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

EMPTY_PARAMETER_BINDINGS_ENCODING = "parquet_null_normalized_to_empty_dict"
EMPTY_PARAMETER_SIN_COS_ENCODING = "parquet_null_normalized_to_empty_dict"


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


def _encode_empty_sample_maps(rows: list[dict[str, Any]]) -> None:
    """Encode empty sample maps so all-empty nested Parquet structs stay writable."""
    for row in rows:
        if "parameter_bindings" not in row:
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "Rows with parameter_bindings require mapping metadata for "
                "empty-map Parquet encoding"
            )
        encoded_metadata = dict(metadata)
        changed = False
        if row["parameter_bindings"] == {}:
            encoded_metadata["empty_parameter_bindings_storage_encoding"] = (
                EMPTY_PARAMETER_BINDINGS_ENCODING
            )
            row["parameter_bindings"] = None
            changed = True
        if encoded_metadata.get("parameter_sin_cos") == {}:
            encoded_metadata["empty_parameter_sin_cos_storage_encoding"] = (
                EMPTY_PARAMETER_SIN_COS_ENCODING
            )
            encoded_metadata["parameter_sin_cos"] = None
            changed = True
        if changed:
            row["metadata"] = encoded_metadata


def _missing_manifest_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _decode_empty_sample_maps(
    rows: list[dict[str, Any]],
    record_type: type[ManifestRecordMixin],
) -> list[dict[str, Any]]:
    if record_type.__name__ != "DatasetSampleRecord":
        return rows
    decoded: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "DatasetSampleRecord.metadata must be a mapping during typed readback"
            )
        decoded_metadata = dict(metadata)
        if _missing_manifest_value(row.get("parameter_bindings")):
            if (
                decoded_metadata.get("empty_parameter_bindings_storage_encoding")
                != EMPTY_PARAMETER_BINDINGS_ENCODING
            ):
                raise ValueError(
                    "DatasetSampleRecord.parameter_bindings is null without the "
                    "documented empty-map storage encoding"
                )
            row["parameter_bindings"] = {}
        if _missing_manifest_value(decoded_metadata.get("parameter_sin_cos")):
            if (
                decoded_metadata.get("empty_parameter_sin_cos_storage_encoding")
                != EMPTY_PARAMETER_SIN_COS_ENCODING
            ):
                raise ValueError(
                    "DatasetSampleRecord.metadata.parameter_sin_cos is null without "
                    "the documented empty-map storage encoding"
                )
            decoded_metadata["parameter_sin_cos"] = {}
            row["metadata"] = decoded_metadata
        decoded.append(row)
    return decoded


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
        rows = [_record_to_dict(record) for record in records]
        _encode_empty_sample_maps(rows)
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
        """Read a Parquet manifest into a list of dictionaries."""
        path = self.manifest_path(manifest_name)
        if not path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pandas and pyarrow are required for Parquet manifest reading"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")

    def read_typed_records(
        self,
        manifest_name: str,
        record_type: type[ManifestRecordMixin],
    ) -> list[ManifestRecordMixin]:
        """Read a manifest and instantiate validated schema records."""
        rows = _decode_empty_sample_maps(
            self.read_records(manifest_name),
            record_type,
        )
        records = [record_type.from_dict(row) for row in rows]
        for record in records:
            record.validate()
        return records
