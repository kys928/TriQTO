"""Read only accessible historical training identity metadata; never truth/reserves."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path


def main():
    import boto3
    from botocore.config import Config
    env = os.environ
    s3 = boto3.client('s3', endpoint_url=env['RUNPOD_S3_ENDPOINT'],
        region_name=env['RUNPOD_DATACENTER_ID'].lower(),
        aws_access_key_id=env['RUNPOD_S3_ACCESS_KEY_ID'],
        aws_secret_access_key=env['RUNPOD_S3_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4', connect_timeout=15, read_timeout=30))
    bucket = env['RUNPOD_NETWORK_VOLUME_ID']
    sources = []
    identities = {x: set() for x in ['family_signature_sha256','graph_sha256','operation_signature']}
    # Explicit approved historical development product, not any protected reserve.
    base = 'triqto-data/step14_cross_motif_dataset/development_b087edfd6629ac250299391d/'
    completion_raw = s3.get_object(Bucket=bucket,Key=base+'dataset_complete.json')['Body'].read()
    completion = json.loads(completion_raw)
    for name in ['family_manifest.csv', 'root_manifest.csv']:
        key=base+'manifests/'+name
        raw=s3.get_object(Bucket=bucket,Key=key)['Body'].read()
        digest='sha256:'+hashlib.sha256(raw).hexdigest()
        if digest != completion['manifest_hashes'][name]:
            raise RuntimeError('Historical manifest hash mismatch: '+name)
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        for row in rows:
            for field in identities:
                if row.get(field): identities[field].add(row[field])
        sources.append({'key':key,'sha256':digest,'rows':len(rows)})
    out={'schema':'triqto.cohort.exclusions.v1',
         'sources':sources,'identities':{k:sorted(v) for k,v in identities.items()},
         'complete_historical_coverage':False,
         'unverified':['earlier non-Step14 training products','spent fresh-holdout blind graph inventory',
                       'selected 36 diagnostic graph inventory','inaccessible sealed/future reserves'],
         'sealed_data_accessed':False,'s3_mutations':False}
    Path('historical-exclusions.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'sources':sources,'identity_counts':{k:len(v) for k,v in identities.items()},
                      'coverage_complete':False}))

if __name__=='__main__': main()
