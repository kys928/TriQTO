"""No protected label/truth data may be read or compared by identity checker."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import scoped_protected_identity_audit as s

class Obj:
    def __init__(self,raw):self.raw=raw
    def read(self):return self.raw
class Client:
    def __init__(self,data):
        self.data=data;self.requests=[]
    def get_object(self,**kw):
        key=kw["Key"];self.requests.append((key,kw.get("Range")))
        if "truth" in key or "secret" in key:raise AssertionError("PROTECTED TRUTH READ")
        raw=self.data[key]
        if "Range" in kw:raw=raw[:256]
        return {"Body":Obj(raw)}
class Dummy:
    pass

def test_truth_paths_refused():
    for key in ("triqto-data/step14_reserve/sealed/manifests/identity_manifest.csv",
                "triqto-data/step14_reserve/secret/manifests/family_manifest.csv",
                "triqto-data/step14_reserve/manifests/truth.csv",
                "triqto-data/step14_reserve/manifests/label.csv"):
        assert not s.manifest_key_allowed(key)

def test_header_rejection_never_fetches_entire_file():
    key="triqto-data/step14_future_hardware_reserve/manifests/identity_manifest.csv"
    c=Client({key:b"family_signature_sha256,mechanism_label\nfoo,RZ\n"})
    z=s.safe_compare(c,"bucket",key,len(c.data[key]),{})
    assert z["status"]=="REFUSED_LABEL_BEARING_HEADER"
    assert len(c.requests)==1 and c.requests[0][1]=="bytes=0-255"

def test_identity_manifest_returns_counts_only():
    key="triqto-data/step14_outer/manifests/identity_manifest.csv"
    raw=b"graph_sha256,family_signature_sha256\nsha256:abc,sha256:family\n"
    c=Client({key:raw})
    fields={f:set() for f in s.IDENTITY_FIELDS}
    fields["graph_sha256"].add("sha256:abc")
    m=s.safe_compare(c,"bucket",key,len(raw),{"sample":fields})
    assert m["status"]=="IDENTITY_ONLY_COMPARED"
    assert m["intersections"]["sample"]["graph_sha256"]==1
    assert b"family" not in str(m).encode()

def test_unknown_header_fails_closed():
    key="triqto-data/step14_outer/manifests/family_manifest.csv"
    raw=b"graph_sha256,execution_input\nabc,secret\n"
    c=Client({key:raw})
    r=s.safe_compare(c,"bucket",key,len(raw),{})
    assert r["status"]=="UNKNOWN_HEADER_COLUMNS_REFUSED"
    assert len(c.requests)==1
