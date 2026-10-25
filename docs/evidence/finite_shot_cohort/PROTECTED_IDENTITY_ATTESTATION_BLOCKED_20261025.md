# Protected-source identity-only attestation: blocked execution record

**Date:** 2026-10-25. **Status:** `IMPLEMENTED_NOT_EXECUTED_PROTECTED_SOURCES_NOT_VERIFIED`.

The user explicitly requested a protected-source **identity-only** check. The proposed procedure is described in [SCOPED_PROTECTED_IDENTITY_ONLY_VERIFICATION.md](../../experiments/SCOPED_PROTECTED_IDENTITY_ONLY_VERIFICATION.md) and `scripts/cohort/scoped_protected_identity_audit.py`. Synthetic refusal tests were authored in `tests/test_scoped_protected_identity_audit.py`.

**Execution result:** Workflows permitting the script to use the connected S3 credentials against previously excluded protected namespaces could **not** be installed or started from this environment. No new workflow executed; no protected S3 identity manifest, reserve, label, measurement, prediction or sealed truth was inspected. The new script and tests have not been run against live protected sources and do not constitute verified identity evidence.

## Evidence already available before this request

- [Prior read-only S3 internal scan 40216188215](https://github.com/kys928/TriQTO/actions/runs/40216188215): 53 top-level namespaces; 11 allowed identity-only CSVs containing 15,400 rows from known permitted products; zero comparable structural collisions for each of the two candidate cohorts, **subject to incomplete coverage**.
- [Prior coverage digest 40216354888](https://github.com/kys928/TriQTO/actions/runs/40216354888): 4 protected source namespace names intentionally not opened; 14 unclassified namespaces; one label-bearing historical manifest refused.
- Original cohort identity: `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`.
- Proposed replacement TEST candidate identity: `sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1`.

**Scientific status remains:** protected-source independence **UNVERIFIED**, exhaustive historical registry **UNVERIFIED**, no independent custodian signed attestation; original TEST still fails public mechanism-label blindness. No R0/R1/R2 inference or replacement TEST measurement acquisition authorized.

## Required approved execution

An independently authorized custodian in a segregated environment must:
1. Confirm the exhaustive source registry (including the four protected namespaces and any off-prefix historical sources) without exposing labels.
2. Run a reviewed identity-only manifest read; never open sealed truth or prediction/counts objects. If a source has no safe identity-only manifest, report that coverage is missing rather than attempting truth access.
3. Compare five structural identity types for each original and prospective TEST candidate manifest. Source cryptographic canonicalization needs independent equivalence checks where historical schemas differ.
4. Produce source-coverage statement, checksum commitments and a signed aggregate overlap-only report, with a pre-registered external trusted public-key identity; then independently review registry completeness.
5. Only after a confirmed zero-overlap/source-complete attestation may subsequent methods be selected and the actual replacement blind TEST acquired under the separate frozen protocol.

**Do not interpret this draft script, unsigned audit, or tool execution refusal as a successful attestation.**
