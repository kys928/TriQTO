#!/usr/bin/env python3
"""Bounded 5090 donor -> 4090 count intervention, with CPU kernel ablations.

Uses the existing private GitHub Actions control plane; credentials never enter
experiment artifacts. Writes only new experiment and controller records.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('/workspace/triqto-data/calibration_substitution')
HISTORIC = Path('/workspace/triqto-data/step15_replay_diagnostics/replay_ca7600e6576b71f5ced2/diagnostic_result.json')
HISTORIC_SHA = 'ba51512c5ecf7d49340823587eb39c446b4853cfcb9ae9cf3e3e50dbdda8b6c5'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def checked_result(parent):
    pointer = json.loads((parent / 'current_step15_replay_diagnostic.json').read_text())
    directory = Path(pointer['diagnostic_dir'])
    raw = (directory / 'diagnostic_result.json').read_bytes()
    if 'sha256:' + hashlib.sha256(raw).hexdigest() != pointer['result_sha256']:
        raise RuntimeError('diagnostic result hash mismatch')
    return json.loads(raw), pointer


def worker():
    import runpod_worker as common
    job = common.load_job()
    name, run_id = common.safe_job_id(job['id']), common.safe_job_id(job['_control_run_id'])
    experiment = common.safe_job_id(job['experiment'])
    role = job['role']
    if role not in {'capture', 'probe'}:
        raise ValueError('invalid experiment role')
    directory = PARENT / experiment / role
    directory.mkdir(parents=True, exist_ok=False)  # fail closed after a restart
    status_path = common.CONTROL_ROOT / name / run_id / 'status.json'
    started = {'job_id': name, 'control_run_id': run_id, 'state': 'running',
               'started_at': common.utc_now(), 'role': role,
               'image': job['image'], 'experiment_directory': str(directory)}
    write(status_path, started)
    try:
        # Verify frozen Step-15 artifacts before and after all read-only diagnostics.
        verifier = [sys.executable, str(ROOT / 'scripts/v0_2/verify_step15_frame_ranker_freeze.py')]
        subprocess.run(verifier, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_portable_calibration.py',
                        'tests/test_step14_phasea_calibration_trace.py', '-q'], cwd=ROOT, check=True)
        reports = {}
        cpu_env = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
        profiles = [('native', {})]
        if role == 'capture':
            profiles += [('single_thread', cpu_env),
                         ('haswell_single_thread', {**cpu_env, 'OPENBLAS_CORETYPE': 'Haswell'}),
                         ('numpy_no_avx512_single_thread', {**cpu_env,
                          'NPY_DISABLE_CPU_FEATURES': 'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'})]
        else:
            profiles += [('substituted', {})]
        donor = job.get('donor')
        for profile, overrides in profiles:
            output = directory / profile
            command = [sys.executable, str(ROOT / 'scripts/v0_2/diagnose_step14_phasea_replay.py'),
                       '--output-parent', str(output), '--max-controls', '20']
            if profile == 'substituted':
                command += ['--substitute-calibration', donor['directory'],
                            '--calibration-manifest-sha256', donor['manifest_sha256']]
            else:
                command += ['--capture-calibration', str(directory / 'bundles' / profile)]
            env = {**os.environ, **overrides}
            with (directory / (profile + '.log')).open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result, pointer = checked_result(output)
            expected_gpu = 'RTX 5090' if role == 'capture' else 'RTX 4090'
            if expected_gpu not in result['execution_environment']['gpu_name']:
                raise RuntimeError('unexpected experiment GPU')
            reports[profile] = {'result': result, 'pointer': pointer, 'environment_overrides': overrides}
            # Fail immediately if donor cannot reproduce the independently verified run.
            if role == 'capture' and profile == 'native':
                raw = HISTORIC.read_bytes()
                if hashlib.sha256(raw).hexdigest() != HISTORIC_SHA:
                    raise RuntimeError('historic 5090 diagnostic hash drift')
                historical = json.loads(raw)
                if result['positions'] != historical['positions']:
                    raise RuntimeError('donor cohort drift')
                for device in ('cpu', 'cuda'):
                    if result['replay_comparisons']['phasea'][device]['pooled_vs_frozen']['argmax_mismatch_count']:
                        raise RuntimeError('donor does not reproduce frozen predictions')
                a, b = result['sampled_calibration_provenance']['per_root'], historical['sampled_calibration_provenance']['per_root']
                if set(a) != set(b) or any(a[r]['sample_hashes']['sample_counts'] != b[r]['sample_hashes']['sample_counts'] or
                        a[r]['finite_shot_jacobians'] != b[r]['finite_shot_jacobians'] for r in a):
                    raise RuntimeError('donor does not reproduce historic sampled counts/Jacobians')
            write(directory / 'progress.json', {'completed_profiles': list(reports)})
        sys.path.insert(0, str(ROOT / 'scripts/v0_2'))
        from portable_calibration import compare_bundles
        reference = (reports['native']['result']['portable_calibration'] if role == 'capture' else donor)
        if role == 'capture':
            reference = {**reference, **reports['native']['pointer']}
        comparisons = {}
        for profile, report in reports.items():
            if profile == 'substituted' or (role == 'capture' and profile == 'native'):
                continue
            current = report['result']['portable_calibration']
            comparisons[profile] = compare_bundles(reference['directory'], reference['manifest_sha256'],
                                                   current['directory'], current['manifest_sha256'])
        subprocess.run(verifier, check=True)
        result = {'schema': 'triqto.calibration_substitution_experiment.v1',
                  'role': role, 'reports': reports, 'numeric_comparisons': comparisons,
                  'donor': reference, 'scientific_boundary': {
                      'sealed_truth_accessed': False, 'weights_modified': False,
                      'frozen_predictions_modified': False, 'holdout_rescored': False}}
        if role == 'probe':
            donor_raw = (Path(donor['diagnostic_dir']) / 'diagnostic_result.json').read_bytes()
            if 'sha256:' + hashlib.sha256(donor_raw).hexdigest() != donor['result_sha256']:
                raise RuntimeError('donor diagnostic result hash mismatch')
            donor_result = json.loads(donor_raw)
            substituted_result = reports['substituted']['result']
            donor_roots = donor_result['sampled_calibration_provenance']['per_root']
            substituted_roots = substituted_result['sampled_calibration_provenance']['per_root']
            if set(donor_roots) != set(substituted_roots) or any(
                    donor_roots[r]['sample_hashes']['sample_counts'] != substituted_roots[r]['sample_hashes']['sample_counts']
                    for r in donor_roots):
                raise RuntimeError('intervention did not preserve exact donor counts')
            jacobian_different_roots = [r for r in donor_roots
                if donor_roots[r]['finite_shot_jacobians'] != substituted_roots[r]['finite_shot_jacobians']]
            result['intervention'] = {
                'exact_donor_count_parity': True,
                'jacobian_different_roots': jacobian_different_roots,
                'native': reports['native']['result']['replay_comparisons']['phasea'],
                'substituted': reports['substituted']['result']['replay_comparisons']['phasea'],
                'description': 'Only integer calibration counts substituted; probabilities still simulated on recipient.'}
        result_path = directory / 'experiment_result.json'
        write(result_path, result)
        digest = 'sha256:' + hashlib.sha256(result_path.read_bytes()).hexdigest()
        write(status_path, {**started, 'state': 'completed', 'result_path': str(result_path),
                            'result_sha256': digest, 'donor': reference, 'completed_at': common.utc_now()})
    except BaseException:
        write(status_path, {**started, 'state': 'failed', 'traceback': traceback.format_exc(),
                            'completed_at': common.utc_now()})
        raise


def controller(experiment, image):
    import runpod_control_v2 as control
    import runpod_worker as common
    experiment = common.safe_job_id(experiment)
    if not image.startswith('ghcr.io/kys928/triqto-runpod@sha256:') or len(image.rsplit(':', 1)[1]) != 64:
        raise ValueError('experiment image must be pinned by digest')
    donor = None
    results = {}
    active_pod = None
    try:
        for role, gpu in [('capture', 'NVIDIA GeForce RTX 5090'), ('probe', 'NVIDIA GeForce RTX 4090')]:
            allowed = control.required_env('RUNPOD_ALLOWED_GPU_TYPES').split(',')
            if gpu not in [v.strip() for v in allowed]:
                raise ValueError('GPU not authorized by repository configuration: ' + gpu)
            name = common.safe_job_id(experiment + '-' + role)
            run_id = role + '-' + str(int(time.time()))
            job = {'id': name, '_control_run_id': run_id, 'experiment': experiment,
                   'role': role, 'image': image, 'donor': donor}
            payload = {'name': 'triqto-' + name, 'computeType': 'GPU', 'cloudType': 'SECURE',
                       'gpuTypeIds': [gpu], 'gpuCount': 1,
                       'dataCenterIds': [control.required_env('RUNPOD_DATACENTER_ID')],
                       'networkVolumeId': control.required_env('RUNPOD_NETWORK_VOLUME_ID'),
                       'volumeMountPath': '/workspace', 'containerDiskInGb': 50,
                       'dockerEntrypoint': ['/bin/bash', '-lc'],
                       'dockerStartCmd': ['cd /opt/TriQTO && python scripts/runpod_calibration_substitution.py --worker'],
                       'ports': [], 'interruptible': False, 'imageName': image,
                       'env': {'TRIQTO_JOB_B64': base64.b64encode(json.dumps(job).encode()).decode(),
                               'TRIQTO_EXPECTED_IMAGE_TAG': image,
                               'TRIQTO_EXPECTED_IMAGE_DIGEST': image.rsplit('@', 1)[1], 'PYTHONUNBUFFERED': '1'}}
            auth = os.environ.get('RUNPOD_CONTAINER_REGISTRY_AUTH_ID', '').strip()
            if auth:
                payload['containerRegistryAuthId'] = auth
            # A failed placement creates no pod. Retry availability only, bounded.
            for attempt in range(20):
                try:
                    pod = control.runpod_request('POST', '/pods', payload)
                    break
                except RuntimeError as exc:
                    if 'There are no instances currently available' not in str(exc) or attempt == 19:
                        raise
                    print('Waiting for ' + gpu + ' capacity', flush=True)
                    time.sleep(30)
            active_pod = str(pod['id'])
            write(Path('calibration_active_pod.json'), {'pod_id': active_pod})
            active_record = {'job_id': name, 'control_run_id': run_id, 'pod_id': active_pod,
                             'status_key': control.status_key(name, run_id),
                             'launched_at': control.utc_now(), 'lifecycle_mode': 'detached',
                             'runner': 'calibration_substitution', 'image_name': image}
            control.internal_put_json(control.active_key(run_id), active_record)
            print(json.dumps({'role': role, 'pod_id': active_pod, 'status_key': active_record['status_key']}), flush=True)
            deadline = time.monotonic() + 3600
            last_progress = None
            while time.monotonic() < deadline:
                status = control.read_json_object(active_record['status_key'])
                if status and status.get('state') in {'completed', 'failed'}:
                    break
                progress_key = str((PARENT / experiment / role / 'progress.json')).removeprefix('/workspace/')
                progress = control.read_json_object(progress_key)
                if progress != last_progress:
                    print(json.dumps({'role': role, 'progress': progress}), flush=True)
                    last_progress = progress
                time.sleep(20)
            else:
                raise TimeoutError('experiment host exceeded 60-minute execution budget')
            if status['state'] != 'completed':
                raise RuntimeError(json.dumps(status))
            key = status['result_path'].removeprefix('/workspace/')
            raw = control.s3_client().get_object(Bucket=control.bucket(), Key=key)['Body'].read()
            if 'sha256:' + hashlib.sha256(raw).hexdigest() != status['result_sha256']:
                raise RuntimeError('experiment result verification failed')
            results[role] = json.loads(raw)
            write(Path('calibration_' + role + '_result.json'), results[role])
            donor = status['donor']
            control.delete_pod(active_pod)
            active_pod = None
            Path('calibration_active_pod.json').unlink(missing_ok=True)
            control.archive_active(control.active_key(run_id), active_record, outcome='completed', detail={'status': status})
        summary = {'experiment': experiment, 'image': image,
                   'donor': donor, 'intervention': results['probe']['intervention'],
                   'numeric_summary': {role: {profile: {k: v for k, v in rec.items() if k != 'per_root'}
                       for profile, rec in result['numeric_comparisons'].items()} for role, result in results.items()}}
        write(Path('calibration_experiment_summary.json'), summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if active_pod:
            control.delete_pod(active_pod)
            Path('calibration_active_pod.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--experiment')
    parser.add_argument('--image')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        if not args.experiment or not args.image:
            parser.error('controller requires experiment and image')
        controller(args.experiment, args.image)


if __name__ == '__main__':
    main()
