import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import select_stabilization_remedy as s

def few_rows():
    rows=[]
    for k in range(60):
        label=k%3
        rows.append({"family_id":str(k),"root_id":f"root-{k}",
          "observation_id":f"obs-{k}","truth_mechanism":label,
          "predictions":{"R0":[label]*16,"R1":[0]*16,"R2":[label]*16}})
    return rows

def test_noninferiority_blocks_stable_constant_guess(monkeypatch):
    monkeypatch.setattr(s,"REPEATS",40)
    a=s.boot_once(few_rows())
    assert a["selected_remedy"]=="R0"
    assert not a["contrasts"]["R1"]["noninferiority_pass"]

def test_rejects_incomplete_selection():
    with pytest.raises(ValueError,match="60 full"):
        s.validate(few_rows())

def test_pairwise_disagreement_endpoints():
    assert s.pair_disagreement([1]*16)==0.
    assert s.pair_disagreement([0]*8+[1]*8)>0.5

def test_rejects_prediction_shape_drift():
    rows=few_rows()
    rows[0]["predictions"]["R2"]=[1]*15
    with pytest.raises(ValueError):
        s.validate(rows)
