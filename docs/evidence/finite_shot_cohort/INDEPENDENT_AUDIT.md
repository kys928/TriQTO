# Independent verification and scientific adjudication

**Verdict:** `INTERNALLY_VERIFIED_QUARANTINED_EXTERNAL_INDEPENDENCE_UNVERIFIED`.

The corpus is constructed, archived, and internally verified. It is **not** marked `VERIFIED_AND_FROZEN_FOR_STABILIZATION_DEVELOPMENT`; stabilization selection remains prohibited. This is a scientific exclusion-coverage blocker, not a compute failure or a failed model-performance test.

## Preregistered execution

The [primary CPU workflow](https://github.com/kys928/TriQTO/actions/runs/40188027634) ran from commit `e62d3f6128a3ee4bf6aa5e1a68f6d4ad103d28bd`, committed before scientific materialization. It passed all 15 regression checks, an excluded four-family engineering pilot, scientific generation, separate audit, development-only EDA and archive publication. Materialization took 385.74 seconds on the runner, including screening and file construction; later audit, EDA and archival time are additional.

| Check | Result and scope |
|---|---|
| Cohort completeness | 240 families / 960 roots / 12,480 observations |
| Family split and sampling strata | Exact 120/60/60 and 192/48; all four variants remain with their family |
| Immutable inventory | All 3,366 listed files checked; no missing or extra corpus files |
| Raw counts | Every bundle checked for permitted fields, integer/nonnegative counts, shape, basis/outcome convention and exact shot mass |
| Candidate construction | Independently reconstructed from the known circuit for all 960 roots |
| Independent calibration simulation/RNG replay | 24 complete first-draw candidate count bundles reproduced exactly, including blind TEST examples; no hidden labels needed |
| Independent canonical reconstruction | 384 sampled canonical vectors, maximum absolute error 9.95e-13 |
| Exhaustive development derived recipes | All 720 roots' full Jacobian, 24-D canonical feature and profile arrays reconstructed and checked against original byte hashes by primary EDA |
| Observed-count uncertainty mechanics | Correlated within-basis covariance, independent signs/bases, joint coordinate/matched bootstrap and PSD/numerical edge checks |
| TEST sealing | 240 authenticated ciphertexts; AES key wrapped with public RSA key; private key absent from runner/repo/archive |
| TEST EDA boundary | Zero TEST array files opened by EDA; no TEST labels decrypted |
| Frozen model boundary | No model loaded/evaluated, no retraining, no remedy parameter selected |
| Resource boundary | No QPU executions, new RunPod pod or S3 mutation |

The regression suite intentionally catches one missing shot, a label injected into a blind bundle, a swapped outcome convention, axis/sign order mistakes, RNG stream collisions, loss of within-basis/joint covariance, structural wire-relabel duplicates, null-frame failures and disagreement with frozen canonical/profile algebra.

## Additional adversarial checks

An independently implemented [post-materialization audit](SUPPLEMENTAL_AUDIT.md) did not modify any corpus bytes. It checked all 720 FIT/SELECTION truth schedules and independently replayed reference and observed diagnostics for 156 development observations. No replay failed. The archive's full SHA256 was also verified after an independent download.

This audit compared the new corpus with the identity-only [expanded historical inventory](additional-exclusions.json), collected by [this successful read-only workflow](https://github.com/kys928/TriQTO/actions/runs/40188502697). It uses checksummed early Step-5 training products, the Step-10 bridge training product, the spent holdout's blind manifests, and only identity metadata from the published diagnostic archive. All 36 diagnostic rows mapped to their blind spent-holdout roots.

| Historical comparison | Result |
|---|---|
| Original 750-family/3,000-root Step-14 development | Zero family, graph, ordered-operation or wire-relabel skeleton matches |
| Additional early training + spent holdout | 8,171 unique parameterized graph hashes and 150 family/operation signatures available; zero matches |
| 150-family/600-root spent fresh holdout | Blind family/graph/operation/wire identities checked; zero matches |
| Selected 36 diagnostic observations | All identities mapped within that blind 600-root inventory; zero matches |
| Protected sealed/future reserves | **Unverified; not opened** |
| Complete historical registry and missing legacy structural signatures | **Unverified** |

The extra inventory has ten source entries and no missing requested files. That does **not** make it exhaustive: legacy clean-root manifests often expose graph hashes without family or ordered-operation signatures, and unenumerated historical products are not certified. The 8,171 graph identities are unique across the extra sources and should not be confused with a count of independent families.

## Independence of the verification

`audit.py` does not import the generator or `core.py`. It separately implements outcome statistics, candidate legality, signed finite differences, weighted ridge canonicalization and statevector/count replay. The post-materialization script uses that separate audit path and permitted development truth to check diagnostic generation. The generator's validation is not the sole evidence.

Both paths were authored and run by the same investigator and use Qiskit for simulation. They are **independent code paths, not independent human replication or hardware validation**. Numerical reconstruction is sampled, while byte integrity, shapes and count totals are exhaustive. TEST label schedule correctness is supported by generation code, not independent inspection after sealing. Exact ordered-skeleton checks up to wire relabeling do not certify arbitrary algebraic/unitary/measurement equivalence.

## Scientific properties retained rather than repaired

- The original first-RY structural proxy identifies the intended affected qubit in the known clean circuit. No explicit true-location field enters the extraction API, but no claim of uninformative circuit structure is warranted.
- Original admission changes the realized grammar distribution and removes many non-identifiable screening families; 376 development attempts failed it. The screening log preserves those attempts. No difficult admitted family was removed after EDA.
- 72.9% of representative development families satisfy the stress rule; stress enrichment is limited. This does not invalidate the representative sample but prevents an easy/hard interpretation of the 80/20 sampling labels.
- Nonlinear coordinate bootstrap variability is not yet well calibrated. The analytic Jacobian count covariance check passing does not validate nonlinear reliability estimates or stabilization remedies.
- Diagnostics are held fixed across calibration draws; stability under repeated diagnostic acquisition is not measured.

## Remaining gate

Obtain an authorized **blind exclusion inventory or custodian overlap attestation** for protected reserves and complete the historical source/signature coverage. Do not open protected labels to resolve this. Any later overlap must fail the cohort gate and be handled as a documented integrity repair under the original overlap rule, not as outcome-driven filtering.

The current acceptance state and evidence identity are in [COHORT_ACCEPTANCE.json](COHORT_ACCEPTANCE.json). Original acquisition-time status/coverage fields remain unchanged inside the frozen [DATASET_MANIFEST.json](DATASET_MANIFEST.json); the supplemental records update what has subsequently been verified without rewriting the source archive.
