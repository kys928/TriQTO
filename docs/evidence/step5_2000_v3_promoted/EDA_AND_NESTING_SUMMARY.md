# Step 5 v3 — 2000-root promoted development gate

Status: **PROMOTION_READY + NESTING_VALID**

This document records the accepted 2000-independent-clean-root Step 5 v3 development cohort reported from the controlled pod run on 2026-08-11. It is a dataset-quality gate only; no classifier was trained and no historical v0.1 test or spent Phase 15.6 confirmatory cohort was accessed.

## Product

- Product ID: `product_502097524aa54c78800fb459`
- Clean roots: 2000
- Examples: 26000
- Train / validation clean roots: 1600 / 400
- Expected composition: 2000 clean controls + 8000 RZ + 8000 RX + 8000 RY
- 3q train / validation roots: 296 / 66

## Generator-stage design checks

- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- shot / strength Cramer's V: `0.022638`
- shot / depth Cramer's V: `0.018271`
- clean-shot / family Cramer's V: `0.000000`

## Scheduling-alias precheck

Global:

- shot / strength: `0.022638`
- shot / depth: `0.018271`
- shot / family: `0.000000`
- shot / split: `0.000000`
- clean-shot / family: `0.000000`
- clean-shot / split: `0.000000`

Per qubit count (`shot/affected`, `strength/affected`, `depth/affected`):

- 2q: `0.065607`, `0.022822`, `0.008802`
- 3q: `0.040425`, `0.045160`, `0.046078`
- 4q: `0.055314`, `0.048068`, `0.048936`
- 6q: `0.062228`, `0.033594`, `0.073074`
- 8q: `0.074698`, `0.062944`, `0.078587`

Scheduling-alias gates: **PASS**.

## Full-artifact EDA

Audit ID: `audit_8a20d67048326db847149401`

Decision: **PROMOTION_READY**

- artifacts verified: `26000 / 26000`
- distinct NPZ schemas: `1`
- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- 3q train / validation roots: `296 / 66`
- family split total-variation distance: `0.000000`
- phenomenology train / validation TV: `0.007188`
- shot-noise SNR remains **REPORT ONLY**
- historical v0.1 test accessed: **NO**
- spent confirmatory cohort accessed: **NO**
- classifier trained: **NO**

## 1000 -> 2000 nesting audit

Previous accepted product: `product_7fef7da30ee7710d189ed27d`

Current product: `product_502097524aa54c78800fb459`

- previous roots: 1000
- current roots: 2000
- previous examples checked: 13000
- root-row mismatches: 0
- missing previous examples: 0
- example-row mismatches: 0
- artifact-hash mismatches: 0

Decision: **NESTING_VALID**

The accepted 1000-root development universe is therefore preserved exactly inside the 2000-root cohort.

## Promotion decision

The 2000-root v3 cohort passes both required Step 5 gates:

1. full-artifact EDA = `PROMOTION_READY`;
2. 1000 -> 2000 nesting = `NESTING_VALID`.

Therefore the final 5000-root stage is unlocked. The 5000-root cohort must again pass the same scheduling/full-artifact EDA gates and preserve the accepted 2000-root product exactly. Only after that final gate should Step 5 be closed and Step 6 cheap baselines begin.
