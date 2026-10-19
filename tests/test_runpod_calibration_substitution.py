"""The campaign must release paid resources and verify donor evidence first."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/runpod_calibration_substitution.py'
SPEC = importlib.util.spec_from_file_location('calibration_campaign_test', PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
IMAGE = 'ghcr.io/kys928/triqto-runpod@sha256:' + 'a' * 64


def setup_control(monkeypatch, tmp_path, *, failed=False, corrupt=False):
    monkeypatch.chdir(tmp_path)
    events = []
    state = {}
    control = types.ModuleType('runpod_control_v2')
    common = types.ModuleType('runpod_worker')
    common.safe_job_id = lambda value: value
    control.required_env = lambda key: {'RUNPOD_ALLOWED_GPU_TYPES':
        'NVIDIA GeForce RTX 5090,NVIDIA GeForce RTX 4090',
        'RUNPOD_DATACENTER_ID': 'EU-CZ-1', 'RUNPOD_NETWORK_VOLUME_ID': 'volume'}[key]
    control.utc_now = lambda: 'fixed-time'
    control.active_key = lambda value: 'active/' + value
    control.status_key = lambda name, run: 'status/' + name
    control.internal_put_json = lambda *a: None
    control.archive_active = lambda *a, **k: None
    control.bucket = lambda: 'volume'
    def request(method, path, payload):
        job = json.loads(base64.b64decode(payload['env']['TRIQTO_JOB_B64']))
        state.update(job=job)
        events.append(('launch', job['role'], payload['imageName'], job['donor']))
        return {'id': 'pod-' + job['role']}
    control.runpod_request = request
    control.delete_pod = lambda pod: events.append(('delete', pod))
    donor = {'directory': '/workspace/donor', 'manifest_sha256': 'sha256:' + 'b' * 64}
    def result():
        return {'donor': donor, 'numeric_comparisons': {}, 'intervention': {}}
    def read_status(key):
        raw = json.dumps(result()).encode()
        return {'state': 'failed' if failed else 'completed', 'result_path': '/workspace/result.json',
                'result_sha256': 'sha256:' + ('0' * 64 if corrupt else hashlib.sha256(raw).hexdigest()),
                'donor': donor}
    control.read_json_object = read_status
    control.s3_client = lambda: types.SimpleNamespace(get_object=lambda **k: {'Body': io.BytesIO(json.dumps(result()).encode())})
    monkeypatch.setitem(sys.modules, 'runpod_control_v2', control)
    monkeypatch.setitem(sys.modules, 'runpod_worker', common)
    return events


def test_campaign_uses_one_digest_and_verified_donor_then_releases_both_pods(monkeypatch, tmp_path):
    events = setup_control(monkeypatch, tmp_path)
    campaign.controller('fixed-test', IMAGE)
    assert [event[0] for event in events] == ['launch', 'delete', 'launch', 'delete']
    assert events[0][2] == events[2][2] == IMAGE
    assert events[0][3] is None
    assert events[2][3]['directory'] == '/workspace/donor'
    assert not Path('calibration_active_pod.json').exists()


@pytest.mark.parametrize('failure', ['worker', 'hash'])
def test_campaign_releases_failed_donor_and_never_launches_recipient(monkeypatch, tmp_path, failure):
    events = setup_control(monkeypatch, tmp_path, failed=failure == 'worker', corrupt=failure == 'hash')
    with pytest.raises(RuntimeError):
        campaign.controller('fixed-test', IMAGE)
    assert [event[0] for event in events] == ['launch', 'delete']
    assert not Path('calibration_active_pod.json').exists()
