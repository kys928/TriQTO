# TriQTO — independent read-only S3 identity namespace inventory (25 October 2026)

**Scientific status: INTERNAL_IDENTITY_COMPARISONS_PASS_BUT_EXTERNAL_EXCLUSION_INCOMPLETE.** This is additional evidence, **not** an independently signed source-complete protected reserve attestation and not authorization to acquire replacement TEST data or evaluate R0/R1/R2.

## Provenance

- Script: `scripts/cohort/readonly_s3_identity_namespace_audit.py`.
- Four synthetic denylist and label-header checks passed.
- Successful main run: [GitHub Actions 40216188215](https://github.com/kys928/TriQTO/actions/runs/40216188215), research branch `research/read-only-identity-inventory-20261025`.
- Preserved run artifact `read-only-s3-identity-preflight`, id **12607268886**, 30-day retention, including per-source SHA256 of read manifest CSVs and identity-field count metadata.
- Offline artifact coverage digest [GitHub Actions 40216354888](https://github.com/kys928/TriQTO/actions/runs/40216354888), artifact `identity-coverage-digest`, id **12607382513**.
- Compared two published identity-only candidate sets: original 240-family cohort `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`; prospective 60-family replacement TEST identity `sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1`.

## Read-only coverage and results

| Inventory / audit measure | Result |
|---|---:|
| Top-level namespaces listed under `triqto-data/` | 53 |
| Non-protected Step-related namespace subtrees inspected at one product depth | 36 |
| Safe Step-related namespace subtrees containing one of the target identity manifests | 7 |
| Non-protected identity manifests read and hashed | 11 |
| Total rows in 11 readable identity manifests | 15,400 |
| Manifests detected with label-bearing headers and refused for full-file parsing | 1 |
| Protected namespace names detected without descending into their objects | 4 |
| Unclassified top-level namespace names not treated as Step products | 14 |
| Five-field identity intersections for original cohort against scanned sources | **0** |
| Five-field identity intersections for prospective 60-family candidate set | **0** |
| Independent source-exhaustive signed attestation issued | **No** |

All reported zero overlap counts are limited to identity fields **actually present and comparable** in the 11 permitted manifests. An absent `context_free_signature_sha256`, family/operation signature or wire normalization is **unknown**, not zero proof of equality-avoidance under those equivalence classes. The earlier explicitly enumerated Step-14 development, Step-5/10, and spent-holdout inventories substantially overlap the set this safe scanner can parse; this second audit is an independent read-only scan and cross-comparison, **not evidence of 15,400 entirely new circuit records**.

## What was and was not accessed

The scanner uses S3 LIST under `triqto-data/`, with one level of non-protected product-directory descent. It may GET only four known identity-manifest CSV filenames outside any namespace containing `reserve`, `outer`, `sealed` or `future_hardware`. It compares hashes/signatures and outputs **aggregate intersection counts**, manifest content hashes and field availability; individual source identity values are not printed. A ranged header probe on one permitted historical manifest revealed a column name indicating labels, and it skipped full content parsing; a ranged GET can technically transfer some bytes following the header, but those data rows were not interpreted and the object was not a protected namespace. **No protected reserve or sealed truth object GET, no S3 write, no QPU, model prediction or frozen checkpoint change.**

The four protected namespace names were **not opened or classified by type**; their existence does not prove a materialized future-hardware *circuit reserve* and their absence from the checked path would not prove nonmaterialization elsewhere. The original frozen Step-14 config sets `future_hardware_reserve_materialized_in_step14=false` and the Step-14 code's materialization modes are `development` and `simulator_outer` only. That documents **intent and generation behavior in Step 14**, not the later storage state. No protected directory metadata or label contents were inspected to resolve this.

## Why there is still an independence blocker

1. **Incomplete source discovery:** only `triqto-data/` was enumerated, one safe subtree depth. Other storage prefixes or buckets, unknown dataset naming and 14 unclassified namespaces might hold relevant source products.
2. **Incomplete source identity schema:** 11 readable manifests do not necessarily include the five comparable structural identities needed for equivalence-aware exclusion; early Step-5/10 products in particular exposed mostly graph hashes in earlier audit.
3. **Protected source metadata not inspected:** four denied namespace names; no confirmed complete identity-only source inventory for relevant reserves/spent outer products.
4. **No independent custodian provenance:** the workflow used existing research GitHub/RunPod credentials; it cannot sign a truly independent, source-exhaustive attestation.
5. **The original TEST remains invalid:** published deterministic label-order metadata gives exact mechanism labels; structural independence, even if later certified, cannot repair its blinding.

## Scientific next step

The identity-only exclusion scripts and signature verification path in PR #110 remain valid infrastructure. To *close* the gate without opening labels, an independently authorized custodian must either (a) attest protected identity-only manifests and confirm source-exhaustive coverage across all scientifically relevant storage, or (b) provide independently verifiable evidence that a claimed reserve was never materialized. Both existing FIT/SELECTION and the **new 60-family TEST candidate identity** require separate attestation. Do not self-sign with the same GitHub runner and call it independent.

**Decision:** keep PR #112 draft, original 240-family archive immutable, no R0/R1/R2 selection, no replacement TEST acquisition, no claims of reserve nonmaterialization after Step 14.
