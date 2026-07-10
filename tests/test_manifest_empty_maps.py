from __future__ import annotations

import pytest

from triqto.storage import ManifestReader, ManifestWriter


def test_nested_empty_maps_roundtrip_losslessly(tmp_path):
    rows = [
        {
            "row_id": "row_1",
            "top_level_empty": {},
            "metadata": {
                "nested_empty": {},
                "deeper": {"empty": {}},
                "values": [1, 2, 3],
            },
        }
    ]
    writer = ManifestWriter(tmp_path)
    writer.write_records("nested_empty_maps", rows)
    assert ManifestReader(tmp_path).read_records("nested_empty_maps") == rows


def test_reserved_empty_map_sentinel_is_rejected(tmp_path):
    rows = [
        {
            "row_id": "row_1",
            "metadata": {"__triqto_parquet_empty_map_v1__": True},
        }
    ]
    with pytest.raises(ValueError, match="reserved manifest key"):
        ManifestWriter(tmp_path).write_records("reserved_key", rows)
