# TriQTO — identity-only historical/reserve attestation handoff

**Status: NO CUSTODIAN ATTESTATION RECEIVED.** Original cohort ID `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`. This is NOT evidence that the exclusion gate passed.

## Existing evidence (no new access to protected reserves)

The prior [internal audit](../evidence/finite_shot_cohort/INDEPENDENT_AUDIT.md) and [supplemental inventory](../evidence/finite_shot_cohort/SUPPLEMENTAL_AUDIT.md) checked the 750-family Step-14 development product, the published 36 diagnostics mapped to a spent 150-family holdout, and an additional **8,171 unique parameterized graph hashes** from earlier training/spent data. All checked intersections were zero. Crucially, the additional 8,171 identities did **not** supply full family/operation/wire equivalence metadata, and inaccessible or unenumerated sources were not certified.

Known audited public/approved historical identities include the following explicit product manifests:
- Step-14 development: `triqto-data/step14_cross_motif_dataset/development_b087edfd6629ac250299391d/manifests/family_manifest.csv` and `root_manifest.csv`
- Step-5 baseline: `triqto-data/step5_matched_diagnostic_training/product_bb5cbdcd33a591f2f8c97b7c/manifests/clean_circuit_manifest.csv`
- Step-5 v2: `triqto-data/step5_matched_diagnostic_training_v2/product_025667cdd6191f70d4a77957/manifests/clean_circuit_manifest.csv`
- Step-5 v3: four products `product_502097524aa54c78800fb459`, `product_7fef7da30ee7710d189ed27d`, `product_85a26bf36d9ed1769b6a5e23`, `product_b2d78ad2309b71a55f9bb54f`, each with `manifests/clean_circuit_manifest.csv`
- Step-10 mixture: `triqto-data/step10_training_mixture/product_0f7112597501f7ea5fbe123b/manifests/bridge_root_manifest.csv`
- Spent fresh holdout: `triqto-data/step14_equivalence_aware_fresh_holdout/fresh_holdout_f805dd871059cb342dafbd59/manifests/family_manifest.csv` and `root_manifest.csv`
- Published diagnostics: `calibration-evidence-20261019` release `archive_manifest.json` (36 rows, mapped to spent holdout).

These are **an inventory of verified prior comparisons**, not an exhaustive declaration of all model-development, experimental and reserved historical products. The missing historical schema fields must be reconstructed from authorized circuit metadata or explicitly marked *not verifiable*. Do not assume that a parameterized graph-hash comparison proves no skeleton/wire equivalence.

## Custodian execution workflow (NOT run from public GitHub Actions)

1. Trusted custodian with authorized access enumerates **all relevant existing / spent model fit, model selection, experiment diagnostic, sealed holdout and reserve identity sources**. They author and sign a **frozen source registry** before reading candidate IDs. This may contain only identities, no truth columns. If a reserve was genuinely never materialized, custody must provide signed non-materialization evidence; do not silently omit it.
2. Export original cohort candidate graph/family/operation/wire identity tokens using [the safe exporter](../../scripts/cohort/export_blind_candidate_identities.py) and the original hash-verified raw release. The exporter never reads TEST raw arrays/truth.
3. In a private authorized custodian environment, normalize each source into a source-named CSV with all five fields: `family_signature_sha256`, `context_free_signature_sha256`, `graph_sha256`, `operation_signature`, `wire_signature_sha256`. The absence of any mandatory field fails coverage until reconstructed. Different original source encodings must be converted with a documented and independently tested identity canonicalization; this script does **not** silently translate hashes.
4. Run `scripts/cohort/custodian_identity_attestation.py` with the candidate export, exhaustive frozen registry and private source CSV folder. A **custodian-held external** Ed25519 signing key is required; the ordinary researcher has no access to it. The script outputs aggregate overlap counts, source coverage and SHA256 commitments but never emits source identities.
5. Researcher verifies the signature with `scripts/cohort/verify_custodian_attestation.py`, using a **pre-registered trusted public key fingerprint** and exact candidate/source-registry hashes. A valid signature proves provenance/integrity of the attestation, **not** completeness of a false custodian source registry: an independent human/procedural review must inspect registry scope.
6. Do not set cohort admission to VERIFIED until *every* signed source scope and frozen protocol gate is accepted, including the separate publicly compromised original TEST finding. The original TEST remains unusable for blind confirmation **even if identity sources match none**.

## Source registry contract

The expected JSON shape has `custodian_asserts_source_inventory_exhaustive:true` plus `required_sources`: objects with unique `name` and `required_identity_fields` listing all five identity types. An incomplete registry (`false`, missing data, absent file, empty required identity, overlap) **fails closed**. An unsigned zero-overlap check is a local *fixture*, not an admissible external attestation.

Original frozen no-inspection restrictions still apply: these scripts do not automatically access any protected reserve, and no script here decrypts historical truth or loads model outputs.

## Separate replacement TEST gate

The 60-family replacement TEST specified in [the replacement protocol](../experiments/REPLACEMENT_BLINDED_STABILIZATION_TEST_PROTOCOL.md) **has not been generated or evaluated**. Construction awaits custodian attestation and a locked final method, as prescribed. Secret-scheduled ID/order primitives and synthetic tests can be developed and inspected without generating an actual TEST. Do not promote the publicly leaked original TEST to confirmatory use.

## Escalation / evidence needed

The remaining material blocker is **a real authorized independent custodian, exhaustive identity-source registry, protected-source attestations and trusted signing key**. Neither a standard GitHub Action using the same repository credentials nor a model-created self-signature is independent custody, and neither suffices to claim that the reserve exclusion gate passed.
