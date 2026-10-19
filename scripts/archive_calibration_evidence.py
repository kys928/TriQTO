#!/usr/bin/env python3
"""Read only the fixed calibration experiment from S3; export a verified closure.

No pod operations, S3 writes, sealed truth, or unselected blind payload reads.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent / 'v0_2'))
from portable_calibration import CalibrationBundle, canonical, sha

EXPERIMENT = 'counts-substitution-20261016-r1'
ROLE_HASHES = {
    'capture': 'sha256:f57461e7dba7201664f290440fff37726bcc97a2ad7cb9a3334576c229e609b5',
    'probe': 'sha256:420b48cd97953af76c552892ed4fa22ffe608d8367292a6aa1c6a9636b776650'}


def key_for(path):
    path = PurePosixPath(path)
    if '..' in path.parts or not str(path).startswith('/workspace/triqto-data/') or 'sealed_truth' in path.parts:
        raise ValueError('refused path outside blind experimental evidence')
    return str(path).removeprefix('/workspace/')


def export(output):
    import runpod_control_v2 as control
    output.mkdir(parents=True, exist_ok=False)
    client, bucket = control.s3_client(), control.bucket()
    inventory, reads = {}, []

    def read(path, expected=None):
        key = key_for(path)
        raw = client.get_object(Bucket=bucket, Key=key)['Body'].read()
        if expected and sha(raw) != expected:
            raise ValueError('source hash mismatch: ' + path)
        reads.append({'key': key, 'sha256': sha(raw), 'bytes': len(raw)})
        return raw

    def save(rel, raw):
        path = output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        inventory[rel] = sha(raw)

    reports = {}
    bundles = {}
    for role, expected in ROLE_HASHES.items():
        raw = read(f'/workspace/triqto-data/calibration_substitution/{EXPERIMENT}/{role}/experiment_result.json', expected)
        save(f'reports/{role}.json', raw)
        result = json.loads(raw)
        reports[role] = result
        for profile, report in result['reports'].items():
            if profile == 'substituted':
                continue
            rec = report['result']['portable_calibration']
            rel = f'bundles/{role}_{profile}'
            manifest = read(rec['directory'] + '/manifest.json', rec['manifest_sha256'])
            doc = json.loads(manifest)
            arrays = read(rec['directory'] + '/arrays.npz', doc['arrays_sha256'])
            save(rel + '/manifest.json', manifest)
            save(rel + '/arrays.npz', arrays)
            CalibrationBundle(output / rel, rec['manifest_sha256'])
            bundles[role + '_' + profile] = {'directory': rel, 'manifest_sha256': rec['manifest_sha256']}
    donor = reports['capture']['reports']['native']['result']
    provenance = donor['provenance']
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json').read_text())
    product = '/workspace/triqto-data/step14_equivalence_aware_fresh_holdout/' + config['holdout_identity']['product_id']
    # Only blinded manifest fields are retained. No truth manifest is fetched.
    manifest = read(product + '/manifests/example_manifest.csv', provenance['blind_example_manifest_sha256'])
    by_id = {r['example_id']: r for r in csv.DictReader(io.StringIO(manifest.decode()))}
    roots_raw = read(product + '/manifests/root_manifest.csv', provenance['blind_root_manifest_sha256'])
    roots = {int(r['root_index']): {k: r[k] for k in ('operation_signature', 'variant_index')}
             for r in csv.DictReader(io.StringIO(roots_raw.decode()))}
    selected = []
    for example in donor['example_diagnostics']:
        row = by_id[example['example_id']]
        if row['evaluation_role'] != 'target' or int(row['root_index']) != example['root_index']:
            raise ValueError('selected blind cohort drift')
        rel = PurePosixPath(row['artifact_path'])
        if not rel.parts or rel.parts[0] != 'blind_artifacts' or rel.is_absolute() or '..' in rel.parts:
            raise ValueError('refused nonblind artifact')
        raw = read(product + '/' + str(rel), row['artifact_sha256'])
        import numpy as np
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if not data.files or any(not k.startswith('x__') for k in data.files):
                raise ValueError('refused non-x payload')
        save('product/' + str(rel), raw)
        selected.append({k: row[k] for k in ('example_id', 'root_index', 'family_id', 'evaluation_role', 'artifact_path', 'artifact_sha256')})
    method = '/workspace/triqto-data/step14_equivalence_aware_development/' + config['method_identity']['method_id']
    save('checkpoints/method_bundle.json', read(method + '/method_bundle.json', provenance['method_bundle_sha256']))
    for name, rec in provenance['checkpoint_provenance'].items():
        save('checkpoints/' + name, read(method + '/' + name, rec['sha256']))
    fixture = {'schema': 'triqto.calibration_evidence_archive.v1', 'experiment': EXPERIMENT,
               'bundles': bundles, 'selected_rows': selected,
               'safe_roots': {str(r): roots[r] for r in sorted({e['root_index'] for e in donor['example_diagnostics']})},
               'donor_examples': donor['example_diagnostics'], 'scientific_provenance': provenance,
               'source_reads': reads, 'files': inventory,
               'boundaries': {'sealed_truth_accessed': False, 'unselected_payloads_read': False, 's3_writes': False}}
    (output / 'archive_manifest.json').write_bytes(canonical(fixture))
    digest = sha((output / 'archive_manifest.json').read_bytes())
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
    report = {'archive': archive.name, 'archive_sha256': sha(archive.read_bytes()),
              'manifest_sha256': digest, 'bundle_count': len(bundles),
              'examples': len(selected), 'files': len(inventory), 'bytes': archive.stat().st_size}
    archive.with_suffix('.sha256.json').write_bytes(canonical(report))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    export(p.parse_args().output)
