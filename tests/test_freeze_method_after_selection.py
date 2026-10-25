"""Post-SELECTION freeze must not certify incomplete or manipulated evidence."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import freeze_method_after_selection as f

def payload():
    return {"schema":"triqto.stabilization_method_selection_decision.v1",
      "status":"COMPLETE_SELECTION_ONLY_NO_TEST_ACCESS",
      "families":60,"records":2880,"draws":16,"bootstrap_replicates":2000,
      "NI_margin_absolute":-0.02,
      "multiplicity_adjustment":"two comparisons, 97.5% Bonferroni intervals",
      "no_confirmatory_TEST_access":True,
      "selected_remedy":"R0",
      "contrasts":{"R1":{"joint_pass":False},"R2":{"joint_pass":False}}}

def test_refuse_incomplete_evidence(tmp_path):
    p=tmp_path/"incomplete.json";p.write_text(json.dumps({}))
    lk=tmp_path/"lock.json";lk.write_text("{}")
    with pytest.raises(ValueError):
        f.build(p,lk,tmp_path)

def test_refuse_wrong_family_count(tmp_path):
    p=payload();p["families"]=59
    t=tmp_path/"verdict.json";t.write_text(json.dumps(p))
    l=tmp_path/"lock.json";l.write_text("{}")
    with pytest.raises(ValueError):
        f.build(t,l,tmp_path)

def test_refuse_r0_when_remedy_passed(tmp_path):
    p=payload();p["contrasts"]["R1"]["joint_pass"]=True
    t=tmp_path/"verdict.json";t.write_text(json.dumps(p))
    l=tmp_path/"lock.json";l.write_text("{}")
    p["pre_selection_lock_sha256"]=f.file_sha(l)
    t.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        f.build(t,l,tmp_path)
