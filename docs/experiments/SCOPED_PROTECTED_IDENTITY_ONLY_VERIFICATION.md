# Narrow identity-only protected-source verification authorization (2026-10-25)

**Purpose:** User requested the remaining protected historical / reserve identity-only overlap check. This is a *new research audit* with a narrower data-access permission than the original Step-14 frozen no-inspection rule, not a retroactive claim that the old Step-14 reserve was opened or used during training. Original TEST label confidentiality and Step-15 frozen model remain unchanged.

## Allowed
- LIST and HEAD metadata of `triqto-data/` immediate namespaces and product folders, including previously excluded `reserve`, `outer`, `sealed`, `future` **folder names** and candidate manifest file names only; logs contain only redacted summary/manifest-type counts.
- Object GET **only** a recognized identity-specific metadata CSV from `manifests/`, explicitly allowlisted file basename, with non-label header checked before rows are processed. Return only up to five cryptographic or ordered-operation identity fields, per-source SHA-256, record count, coverage flags, aggregate overlap counts for two frozen candidate sets. If any column suggests mechanism, strength, truth, prediction, label, measurement, hidden recipe or secret, refuse the manifest for full-file reading; no blind attempt to reconstruct missing fields from truth.
- If a specific source has no safe usable manifest, report it as `NO_VERIFIABLE_IDENTITY_MANIFEST`, not as disjoint.

## Forbidden
- Access to any `truth`, `sealed`, `label`, `predictions`, `model_outputs`, measurement counts, raw observations, hidden salts/keys or archived protected data products, even for verification.
- Inspect candidate mechanism inference, run test scoring, change model weights, reveal hidden identities/labels, write anything to S3 or change any pre-existing immutable evidence archive.
- Invent a source-registry exhaustive claim, sign as an independent custodian with an author-controlled key, or promote a reserve into confirmatory data based solely on partial S3 metadata.

## Gate
This investigation is **author-permitted identity-only provenance work**, not an independent third-party/source-complete certification. Missing historical signatures and unknown source prefixes remain limitations. The old public TEST continues to be metadata-leaky even after identity checks. The prospective replacement TEST must receive its own source-overlap attestation after the exact identity manifest is frozen. If source absence is suggested by LIST, say *not found in scope*, never *never materialized* without independent storage-lifecycle evidence.

**Revision applies prospectively to this scoped metadata audit only.** It does not change the original Step-14 training, Step-15 model or TEST scoring contracts.
