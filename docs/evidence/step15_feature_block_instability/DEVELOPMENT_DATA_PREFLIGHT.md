# TriQTO — read-only development-data feasibility preflight

**Date:** 2026-10-23. **Status:** Metadata preflight completed; **independent remedy-development cohort not yet admitted**.  
**Execution:** [GitHub Actions run 40000082799](https://github.com/kys928/TriQTO/actions/runs/40000082799), success. Artifact `development-manifest-preflight` (12521078563), short retention.

## Verified observations

Using the RunPod network volume's S3-compatible **read-only** API on a GitHub runner (no RunPod pod), the workflow fetched only the historical Step-14 development product's `dataset_complete.json`, `family_manifest.csv` and `root_manifest.csv`. The two manifest SHA256 hashes matched the dataset completion manifest.

- Existing product ID: `development_b087edfd6629ac250299391d`
- 750 historical families; 3,000 roots.
- Partitions: 600 historical `fit` families, 150 historical `selection` families.
- When compared with the 36 selected calibration diagnostic rows from the published `calibration-evidence-20261019` release, **zero identical family IDs** were found.
- The diagnostic archive rows did **not** expose family signature hashes, so **signature-level non-overlap was not verified**. A family ID comparison alone is insufficient proof of graph or circuit equivalence exclusion.
- No example payload, target mechanism label, true location, simulator outer evaluation, sealed truth or future hardware reserve was accessed.

The development source was used in earlier Step-14/Step-15 model-development work. Its family count and lack of identical diagnostic IDs do **not** make it a new independent method-selection or confirmatory cohort. The feasibility result is **a source inventory only**.

## Research decision

**Do not tune or select matched-response stabilization on the diagnostic 36 examples or represent this historical development product as untouched validation.**

Before the planned R0/R1/R2 study, create a new family-disjoint development cohort with complete identity/signature/graph-hash overlap checks and split it into development-FIT, development-SELECTION and reserved remedy-TEST families. Existing historical data may be used only for non-selection engineering/sanity if appropriately labeled, and no sealed or held-out benchmark is to be opened.

An initial preflight run failed because `gh release download` was invoked without an explicit repository in a workflow without checkout. That invocation was repaired; the documented successful run independently verified the manifests. The failure involved no data mutation.

**Scientific boundary:** `no independent remedy cohort -> no remedy-selection claims`.
