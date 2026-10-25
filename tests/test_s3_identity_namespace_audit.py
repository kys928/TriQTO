"""Proof that source inventory never GETs protected or label-bearing data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import readonly_s3_identity_namespace_audit as a

class Body:
    def __init__(self,payload):self.payload=payload
    def read(self):return self.payload

class S3:
    def __init__(self):
        self.contents={
          "triqto-data/step5_matched_diagnostic_training/product_a/manifests/root_manifest.csv":
             b"graph_sha256,operation_signature,foo\nsha256:abc,ry:q0,x\n",
          "triqto-data/step5_matched_diagnostic_training/product_a/manifests/family_manifest.csv":
             b"graph_sha256,mechanism_label\nsha256:def,RZ\n"
        }
        self.reads=[]
    def get_object(self,**kw):
        key=kw["Key"]
        self.reads.append((key,kw.get("Range")))
        if "reserve" in key:raise AssertionError("READ OF PROTECTED OBJECT")
        data=self.contents[key]
        if "Range" in kw:return {"Body":Body(data[:4096])}
        return {"Body":Body(data)}
    def head_object(self,**kw):
        return {"ContentLength":len(self.contents[kw["Key"]])}

def test_protected_objects_refused_before_get():
    s=S3()
    import pytest
    with pytest.raises(PermissionError):
        a.safe_header(s,"bucket","triqto-data/step14_future_hardware_reserve/sealed_truth.csv")
    assert not s.reads

def test_header_label_refused_without_full_get():
    s=S3()
    key="triqto-data/step5_matched_diagnostic_training/product_a/manifests/family_manifest.csv"
    result,ident=a.permitted_manifest(s,"bucket",key,len(s.contents[key]))
    assert ident is None
    assert result["state"]=="LABEL_BEARING_HEADER_REFUSED"
    assert s.reads==[(key,"bytes=0-4095")]

def test_identity_csv_only_reads_identity_fields():
    s=S3()
    key="triqto-data/step5_matched_diagnostic_training/product_a/manifests/root_manifest.csv"
    info,ids=a.permitted_manifest(s,"bucket",key,len(s.contents[key]))
    assert ids["graph_sha256"]=={"sha256:abc"}
    assert ids["operation_signature"]=={"ry:q0"}
    assert info["state"]=="SAFE_IDENTITY_ONLY_READ"

def test_denied_prefixes():
    assert a.denied("triqto-data/step14_simulator_outer/") is True
    assert a.denied("triqto-data/step14_future_hardware_reserve/") is True
    assert a.denied("triqto-data/step10_training_mixture/") is False
