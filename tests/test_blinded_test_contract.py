"""Privacy-contract pilot with synthetic task labels only: no TEST acquisition."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/cohort'))
import blinded_test_contract as b

def test_keyed_schedule_not_recoverable_from_public_identifiers():
    keyA=bytes(range(32));keyB=bytes(reversed(range(32)))
    root="independent-opaque-root-1"
    orderA=b.private_order(keyA,root)
    orderB=b.private_order(keyB,root)
    assert sorted(orderA)==list(range(13))
    assert sorted(orderB)==list(range(13))
    assert orderA != orderB
    assert orderA==b.private_order(keyA,root)

def test_public_schema_contains_no_schedule_or_labels():
    shots=[2048]+[512]*3+[1024]*3+[2048]*3+[4096]*3
    public,private=b.blinded_observation_metadata(bytes(range(32)),"synthetic-root",shots)
    assert b.validate_public_metadata(public)
    assert len(public)==len(private)==13
    assert len({x["observation_id"] for x in public})==13
    assert all(set(x)=={"observation_id","shots"} for x in public)
    assert sorted(x["private_schedule_index"] for x in private)==list(range(13))
    assert all("private_schedule_index" not in x for x in public)

def test_secret_required_for_schedule():
    import pytest
    with pytest.raises(ValueError):b.private_order(b"too short","root")
    with pytest.raises(ValueError):b.private_order(bytes(range(32)),"")
