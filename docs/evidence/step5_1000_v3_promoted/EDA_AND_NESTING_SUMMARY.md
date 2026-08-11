# Step 5 v3 — 1000-root promoted development gate

Status: **PROMOTION_READY + NESTING_VALID**

This document records the accepted 1000-independent-clean-root Step 5 v3 development cohort reported from the controlled pod run on 2026-08-11. It is a dataset-quality gate only; no classifier was trained and no historical v0.1 test or spent Phase 15.6 confirmatory cohort was accessed.

## Product

- Product ID: `product_7fef7da30ee7710d189ed27d`
- Clean roots: 1000
- Examples: 13000
- Train / validation clean roots: 800 / 200
- Expected composition: 1000 clean controls + 4000 RZ + 4000 RX + 4000 RY
- 3q train / validation roots: 140 / 31

## Generator-stage design checks

- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- shot / strength Cramer's V: `0.024698`
- shot / depth Cramer's V: `0.022613`
- clean-shot / family Cramer's V: `0.004808`

## Scheduling-alias precheck

Global:

- shot / strength: `0.024698`
- shot / depth: `0.022613`
- shot / family: `0.000000`
- shot / split: `0.000000`
- clean-shot / family: `0.004808`
- clean-shot / split: `0.024002`

Per qubit count (`shot/affected`, `strength/affected`, `depth/affected`):

- 2q: `0.053808`, `0.008403`, `0.061177`
- 3q: `0.071232`, `0.079740`, `0.063341`
- 4q: `0.059758`, `0.051998`, `0.092117`
- 6q: `0.063918`, `0.057191`, `0.083920`
- 8q: `0.101584`, `0.129017`, `0.080231`

Scheduling-alias gates: **PASS**.

## Full-artifact EDA

Audit ID: `audit_9473903073b1ded46f174e4f`

Decision: **PROMOTION_READY**

- artifacts verified: `13000 / 13000`
- distinct NPZ schemas: `1`
- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- 3q train / validation roots: `140 / 31`
- family split total-variation distance: `0.000000`
- phenomenology train / validation TV: `0.006146`
- shot-noise SNR remains **REPORT ONLY**
- historical v0.1 test accessed: **NO**
- spent confirmatory cohort accessed: **NO**
- classifier trained: **NO**

## 500 -> 1000 nesting audit

Previous accepted product: `product_85a26bf36d9ed1769b6a5e23`

Current product: `product_7fef7da30ee7710d189ed27d`

- previous roots: 500
- current roots: 1000
- previous examples checked: 6500
- root-row mismatches: 0
- missing previous examples: 0
- example-row mismatches: 0
- artifact-hash mismatches: 0

Decision: **NESTING_VALID**

The accepted 500-root development universe is therefore preserved exactly inside the 1000-root cohort.

## Promotion decision

The 1000-root v3 cohort passes both required Step 5 gates:

1. full-artifact EDA = `PROMOTION_READY`;
2. 500 -> 1000 nesting = `NESTING_VALID`.

Therefore the 2000-root stage is unlocked. The 2000-root cohort must again pass the same scheduling/full-artifact EDA gates and must preserve the accepted 1000-root product exactly before 5000 roots can be generated.
