# Reproduction and custody

## Frozen execution source

Construction, acquisition-time audit and primary EDA were committed **before** scientific materialization at `e62d3f6128a3ee4bf6aa5e1a68f6d4ad103d28bd`, above PR #107 (`1a2d25defa7cb21fc495c9a634431f5ac88c1f73`).

Use Python 3.11 and `requirements-cohort.txt`. The cohort environment deliberately has no torch, model training or QPU dependencies. Selected frozen function definitions are loaded from source without importing training entry points; their complete source-file SHA256 hashes are asserted before acquisition. The original source files are unchanged.

```bash
git clone https://github.com/kys928/TriQTO.git
cd TriQTO
git checkout e62d3f6128a3ee4bf6aa5e1a68f6d4ad103d28bd
python3.11 -m venv .venv-cohort
source .venv-cohort/bin/activate
pip install -r requirements-cohort.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q tests/test_independent_cohort.py
```

## Exact evidence replay (no TEST key required)

Download the versioned raw archive and its `SHA256SUMS` from the linked release. Verify before extraction. The archive contains blind raw counts/circuits for all partitions, development truth/acquisition targets in separate directories, compact derived samples and reconstruction hashes, and **encrypted** TEST labels/recipes. The private recovery key is not part of the archive or repository. Do not decrypt TEST for construction/EDA.

```bash
sha256sum -c SHA256SUMS
tar -xzf TriQTO_Finite_Shot_Cohort_20261025.tar.gz
python scripts/cohort/audit.py --product corpus --output cohort-audit-reproduced.json
python scripts/cohort/eda.py --product corpus --output reproduced-report
```

`audit.py` uses a separate implementation for candidate legality, outcome conventions, count means, Jacobians, ridge features and simulator/RNG replay; it does not import `core.py` or invoke generator validation functions. All raw files get hash/shape/count-mass checks. A fixed content-selected sample gets independent numerical and simulation reconstruction. This is independent code, **not an independent human replication**.

`eda.py` excludes TEST files at the top of its data loop. It reconstructs every development Jacobian, 24-D canonical feature and profile array and checks the original array hashes. The 64-replicate observed-count bootstrap uses one first variant/candidate/opaque observation per development family at N, with diagnostic evidence fixed. It is a conditional plug-in estimate, not a physical new measurement or guarantee of coverage near singularities.

Pair distances/gates can be reconstructed with the unchanged, source-bound `finite_frame_pair_geometry` function. Candidate network scores and pooled decisions are intentionally **not** produced in this task. They require the frozen Step-15 bundle and a separately authorized, gated comparison.

## New generation versus exact reproduction

With authorized read-only S3 credentials, `s3_exclusion_inventory.py` resolves the historical Step-14 development identity inventory. `build.py --pilot` creates the separate engineering pilot. A full fresh generation uses the public sealing key:

```bash
python scripts/cohort/s3_exclusion_inventory.py
python scripts/cohort/build.py --pilot --exclusions historical-exclusions.json --output pilot
python scripts/cohort/build.py --exclusions historical-exclusions.json \
  --public-key docs/evidence/finite_shot_cohort/test_sealing_public_key.pem --output new-corpus
```

Do not confuse this with reproducing the exact archived cohort: fresh TEST generation uses a new private salt and randomized encryption. The original TEST generation recipes/seeds are encrypted. Only a later authorized custodian can recover them; blind count replay needs no key. FIT/SELECTION generation and calibration seed schedules are deterministic. The materialization identity additionally binds timing metadata and encryption envelopes; a separate statistical-evidence identity binds the original blind root/circuit/count content.

## Additional exclusion and diagnostic audit

The identity-only extension reads verified early training identity manifests, the already-spent fresh holdout's **blind** manifests, and identity fields from the published diagnostic archive. It never opens `sealed_truth`, future reserves or model outcomes. Some historical products do not advertise every desired signature field; absent data remain an exclusion gap.

The later `post_materialization_audit.py` adds structural partition tables, historical identity comparisons and independent reference/observation diagnostic replay on permitted development examples. This is a supplementary validation revision; it changes no corpus bytes or selection rule. Its exact commit and output hash must be recorded with the final report.

```bash
python scripts/cohort/post_materialization_audit.py --product corpus \
  --additional-exclusions additional-exclusions.json --output supplemental-report
```

No actual TEST label decryption is part of any command above. Generation-time code alone, rather than an independent post-sealing truth audit, supports TEST label scheduling. This limitation is disclosed rather than claimed as an independently checked outcome.
