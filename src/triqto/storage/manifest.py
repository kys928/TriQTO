"""Manifest reader and writer for TriQTO's manifest-centered data lake.

The Phase 2 implementation supports small-to-medium Parquet manifests backed by
pandas/pyarrow.  Manifests hold metadata rows and references to large external
artifacts; the writer deliberately does not serialize large tensors inline.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Iterable

from triqto.storage.schema import ManifestRecordMixin


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a dataclass record or mapping into a plain manifest row."""
    if isinstance(record, ManifestRecordMixin):
        record.validate()
        return record.to_dict()
    if is_dataclass(record):
        # Support future dataclass records that do not inherit the mixin yet.
        from dataclasses import asdict

        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"Unsupported manifest record type: {type(record)!r}")


class ManifestWriter:
    """Write TriQTO manifest rows to Parquet files under a manifest root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def manifest_path(self, manifest_name: str) -> Path:
        """Return a normalized `.parquet` path for a manifest name."""
        name = manifest_name if manifest_name.endswith(".parquet") else f"{manifest_name}.parquet"
        return self.root / name

    def write_records(self, manifest_name: str, records: Iterable[Any], *, overwrite: bool = False) -> Path:
        """Write records to a Parquet manifest and return the output path."""
        rows = [_record_to_dict(record) for record in records]
        path = self.manifest_path(manifest_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("pandas and pyarrow are required for Parquet manifest writing") from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path


class ManifestReader:
    """Read TriQTO Parquet manifests as row dictionaries or schema records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def manifest_path(self, manifest_name: str) -> Path:
        """Return a normalized `.parquet` path for a manifest name."""
        name = manifest_name if manifest_name.endswith(".parquet") else f"{manifest_name}.parquet"
        return self.root / name

    def read_records(self, manifest_name: str) -> list[dict[str, Any]]:
        """Read a Parquet manifest into a list of dictionaries."""
        path = self.manifest_path(manifest_name)
        if not path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("pandas and pyarrow are required for Parquet manifest reading") from exc
        return pd.read_parquet(path).to_dict(orient="records")

    def read_typed_records(self, manifest_name: str, record_type: type[ManifestRecordMixin]) -> list[ManifestRecordMixin]:
        """Read a manifest and instantiate rows as a provided record dataclass."""
        return [record_type.from_dict(row) for row in self.read_records(manifest_name)]
