"""Additional identity-only historical audit; no protected reserves or labels."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    import boto3
    from botocore.config import Config
    env=os.environ
    s3=boto3.client('s3',endpoint_url=env['RUNPOD_S3_ENDPOINT'],region_name=env['RUNPOD_DATACENTER_ID'].lower(),
        aws_access_key_id=env['RUNPOD_S3_ACCESS_KEY_ID'],aws_secret_access_key=env['RUNPOD_S3_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4',connect_timeout=15,read_timeout=30))
    bucket=env['RUNPOD_NETWORK_VOLUME_ID'];sources=[];missing=[];identities={k:set() for k in ['family_signature_sha256','graph_sha256','operation_signature']}
    def get(key):
        if any(x in key.lower() for x in ['sealed_truth','future_hardware','reserve','simulator_outer']):raise RuntimeError('Protected path refused')
        return s3.get_object(Bucket=bucket,Key=key)['Body'].read()
    def consume(base,name,expected):
        key=base+'manifests/'+name
        try:raw=get(key)
        except s3.exceptions.NoSuchKey:missing.append(key);return []
        digest='sha256:'+hashlib.sha256(raw).hexdigest()
        if expected is not None and digest!=expected:raise RuntimeError('Historical metadata checksum mismatch: '+key)
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        for row in rows:
            for field in identities:
                if row.get(field):identities[field].add(row[field])
        sources.append({'key':key,'sha256':digest,'expected_verified':expected is not None,'rows':len(rows)})
        return rows
    # These prefixes are prior TRAIN/validation products, not outer or reserves.
    prefixes=['step5_matched_diagnostic_training','step5_matched_diagnostic_training_v2',
              'step5_matched_diagnostic_training_v3','step10_training_mixture']
    for prefix in prefixes:
        parent='triqto-data/'+prefix+'/'
        page=s3.list_objects_v2(Bucket=bucket,Prefix=parent,Delimiter='/',MaxKeys=1000)
        if page.get('IsTruncated'):raise RuntimeError('Historical product listing truncated')
        products=[x['Prefix'] for x in page.get('CommonPrefixes',[]) if '/product_' in x['Prefix']]
        if not products:missing.append(parent+'[no product directories]')
        for base in products:
            try:complete=json.loads(get(base+'dataset_complete.json'))
            except s3.exceptions.NoSuchKey:missing.append(base+'dataset_complete.json');continue
            for name in ['family_manifest.csv','root_manifest.csv']:
                consume(base,name,complete.get('manifest_hashes',{}).get(name))
    # Published, already-spent fresh holdout: BLIND manifests only, never sealed_truth.
    fresh='triqto-data/step14_equivalence_aware_fresh_holdout/fresh_holdout_f805dd871059cb342dafbd59/'
    completion_raw=get(fresh+'dataset_complete.json')
    if 'sha256:'+hashlib.sha256(completion_raw).hexdigest()!='sha256:515cc17016894506e040d79bfef2dd7bf1c605578deccfb4fbef2ba865794675':
        raise RuntimeError('Frozen spent-holdout completion checksum mismatch')
    complete=json.loads(completion_raw);hm=complete['blind_manifest_sha256']
    consume(fresh,'family_manifest.csv',hm['family_manifest.csv'])
    fresh_roots=consume(fresh,'root_manifest.csv',hm['root_manifest.csv'])
    # Diagnostic identity metadata only; do not extract/checkpoint or read any labels.
    subprocess.run(['gh','release','download','calibration-evidence-20261019','--repo','kys928/TriQTO',
                    '-p','calibration-evidence.tar.gz','-D','diagnostic-meta'],check=True,capture_output=True)
    with tarfile.open('diagnostic-meta/calibration-evidence.tar.gz') as archive:
        m=next(x for x in archive.getmembers() if x.name.endswith('archive_manifest.json'))
        raw=archive.extractfile(m).read();d=json.loads(raw)
    diag_rows=d['selected_rows'];mapped=0
    index={(str(r.get('family_id')),str(r.get('root_index'))):r for r in fresh_roots}
    for row in diag_rows:
        for field in identities:
            if row.get(field):identities[field].add(str(row[field]))
        target=index.get((str(row.get('family_id')),str(row.get('root_index'))))
        if target:
            mapped+=1
            for field in identities:
                if target.get(field):identities[field].add(target[field])
    sources.append({'key':'GitHub release calibration-evidence-20261019/archive_manifest.json',
                    'sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),'rows':len(diag_rows),'diagnostic_rows_mapped_to_blind_fresh_roots':mapped})
    out={'schema':'triqto.cohort.additional_exclusions.v1','sources':sources,'missing':missing,
         'identities':{k:sorted(v) for k,v in identities.items()},'complete_historical_coverage':False,
         'unverified':['inaccessible sealed/future reserves','any historical products not enumerated in this explicit source inventory'],
         'sealed_data_accessed':False,'model_outputs_accessed':False,'s3_mutations':False}
    Path('additional-exclusions.json').write_text(json.dumps(out,sort_keys=True,indent=2)+chr(10))
    print(json.dumps({'sources':sources,'missing':missing,'identity_counts':{k:len(v) for k,v in identities.items()}}))

if __name__=='__main__':main()
