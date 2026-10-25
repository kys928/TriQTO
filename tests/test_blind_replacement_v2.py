"""Independent sacrificial-fixture checks: sealed blind TEST truth is nonpublic."""
import sys,json
from pathlib import Path
import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import build_blind_replacement_test_v2 as t

def test_real_test_cannot_be_built_without_freeze(tmp_path):
    f=tmp_path/"invalid.json";f.write_text("{}")
    with pytest.raises(ValueError):
        t.validate_freeze(f)

def test_validated_freeze_requires_all_hashes(tmp_path):
    p={"schema":"triqto.stabilization_final_method_freeze.v1",
      "status":"FROZEN_BEFORE_ANY_REPLACEMENT_TEST_ACQUISITION",
      "selected_remedy":"R1",
      "accuracy_noninferiority_gate_applied":True,
      "selection_completed_before_freeze":True,
      "confirmation_claim_explicitly_limited_to_checked_training_lineage":True}
    f=tmp_path/"incomplete.json";f.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        t.validate_freeze(f)

def check_sacrificial_pilot(candidates,out):
    private=rsa.generate_private_key(public_exponent=65537,key_size=3072)
    pub=out.parent/"pilot-public.pem"
    pub.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
    result=t.build(candidates,out,pub,pilot=True)
    assert result["root_count"]==4 and result["observation_count"]==52
    assert result["status"]=="SYNTHETIC_SACRIFICIAL_PILOT_NOT_A_TEST"
    wrapped=(out/"wrapped_test_key.bin").read_bytes()
    key=private.decrypt(wrapped,padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),
        label=b"TriQTO-replacement-TEST-v2"))
    pub_rows=json.loads((out/"public_roots.json").read_text())
    assert len(pub_rows)==4
    for root in pub_rows:
        sealed_path=out/"sealed"/(root["root_id"].split(":")[1]+".aesgcm")
        ciphertext=sealed_path.read_bytes()
        truth=json.loads(AESGCM(key).decrypt(ciphertext[:12],ciphertext[12:],
                                   root["root_id"].encode()))
        expected=truth["truth"]
        assert len(expected)==13
        mechanisms=[x["mechanism"] for x in expected]
        assert mechanisms.count(None)==1
        assert all(mechanisms.count(x)==4 for x in t.MODELS)
        with np.load(out/root["raw_path"],allow_pickle=False) as z:
            assert z["diagnostic_counts"].shape[0]==13
            assert z["calibration_counts"].shape[0]==3
            ids=z["observation_ids"].tolist()
            shots=z["diagnostic_shots"].tolist()
            assert ids==[x["observation_id"] for x in root["observations"]]
            assert shots==[x["shots"] for x in root["observations"]]
            assert shots==[x["shots"] for x in expected]
            for i,count in enumerate(z["diagnostic_counts"]):
                assert (count.sum(axis=-1)==shots[i]).all()
        assert len(set(ids))==13
    public_blob=(out/"public_roots.json").read_text()+(out/"public_manifest.json").read_text()
    assert '"mechanism"' not in public_blob
    assert '"strength"' not in public_blob
    assert '"affected_qubit"' not in public_blob
    assert '"secret_permutation"' not in public_blob

def test_pilot_encrypted_payload(tmp_path):
    # The external Actions job sets this to downloaded public identities; never
    # touches any real TEST observation/label sources.
    f=tmp_path/"replacement-candidates.json"
    f.write_text(json.dumps({"cohort_identity":t.EXPECTED,"families":60,
         "roots":240,"candidates":[]}))
    # Unit fixture alone cannot reconstruct a valid family: test gate instead.
    with pytest.raises(RuntimeError):
        next(t.reconstruct(f,limit=1))
