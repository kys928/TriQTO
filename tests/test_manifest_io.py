"""Tests for Phase 2 Parquet manifest IO."""
from __future__ import annotations

import pytest

from triqto.storage.manifest import ManifestReader, ManifestWriter
from triqto.storage.schema import CircuitRecord


def test_manifest_writer_reader_round_trip(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    record = CircuitRecord("circuit_1", "ghz", 4, 4, 3, 3, 0, {"tag": "roundtrip"})
    writer = ManifestWriter(tmp_path)
    path = writer.write_records("circuit_manifest", [record])
    assert path.name == "circuit_manifest.parquet"

    reader = ManifestReader(tmp_path)
    rows = reader.read_records("circuit_manifest")
    assert rows[0]["circuit_id"] == "circuit_1"
    typed = reader.read_typed_records("circuit_manifest", CircuitRecord)
    assert typed[0].family == "ghz"


def test_manifest_writer_refuses_overwrite_by_default(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    writer = ManifestWriter(tmp_path)
    writer.write_records("manifest", [{"a": 1}])
    with pytest.raises(FileExistsError):
        writer.write_records("manifest", [{"a": 2}])
