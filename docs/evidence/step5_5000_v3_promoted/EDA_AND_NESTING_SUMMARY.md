# Step 5 v3 — 5000-root final promoted development gate

Status: **PROMOTION_READY + NESTING_VALID**

This document records the accepted final 5000-independent-clean-root Step 5 v3 development cohort reported from the controlled pod run on 2026-08-13. It is a dataset-quality gate only; no classifier was trained and no historical v0.1 test or spent Phase 15.6 confirmatory cohort was accessed.

## Product

- Product ID: `product_b2d78ad2309b71a55f9bb54f`
- Clean roots: 5000
- Examples: 65000
- Train / validation clean roots: 4000 / 1000
- Expected composition: 5000 clean controls + 20000 RZ + 20000 RX + 20000 RY
- 3q train / validation roots: 743 / 178

## Generator-stage design checks

- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- shot / strength Cramer's V: `0.003868`
- shot / depth Cramer's V: `0.016423`
- clean-shot / family Cramer's V: `0.000961`

## Scheduling-alias precheck

Global:

- shot / strength: `0.003868`
- shot / depth: `0.016423`
- shot / family: `0.000000`
- shot / split: `0.000000`
- clean-shot / family: `0.000961`
- clean-shot / split: `0.004800`

Per qubit count (`shot/affected`, `strength/affected`, `depth/affected`):

- 2q: `0.025294`, `0.014551`, `0.004042`
- 3q: `0.019668`, `0.039059`, `0.028693`
- 4q: `0.029179`, `0.031266`, `0.030308`
- 6q: `0.043880`, `0.026087`, `0.046274`
- 8q: `0.043250`, `0.037748`, `0.042906`

Scheduling-alias gates: **PASS**.

## Full-artifact EDA

Audit ID: `audit_205c163da08f6d7f1f3f1618`

Decision: **PROMOTION_READY**

- artifacts verified: `65000 / 65000`
- distinct NPZ schemas: `1`
- family / split Cramer's V: `0.000000`
- depth / strength Cramer's V: `0.000000`
- 3q train / validation roots: `743 / 178`
- family split total-variation distance: `0.000000`
- phenomenology train / validation TV: `0.001188`
- shot-noise SNR remains **REPORT ONLY**
- historical v0.1 test accessed: **NO**
- spent confirmatory cohort accessed: **NO**
- classifier trained: **NO**

## 2000 -> 5000 nesting audit

Previous accepted product: `product_502097524aa54c78800fb459`

Current product: `product_b2d78ad2309b71a55f9bb54f`

- previous roots: 2000
- current roots: 5000
- previous examples checked: 26000
- root-row mismatches: 0
- missing previous examples: 0
- example-row mismatches: 0
- artifact-hash mismatches: 0

Decision: **NESTING_VALID**

The accepted 2000-root development universe is therefore preserved exactly inside the final 5000-root cohort.

## Step 5 closure

The final 5000-root v3 cohort passes both required Step 5 gates:

1. full-artifact EDA = `PROMOTION_READY`;
2. 2000 -> 5000 nesting = `NESTING_VALID`.

Therefore the staged Step 5 dataset construction is complete. The v3 generator and final 5000-root cohort are frozen for downstream Step 6 baseline evaluation. Step 6 may compare deployable evidence baselines, but must not silently alter Step 5 sampling, labels, split membership, or privileged-target masking.