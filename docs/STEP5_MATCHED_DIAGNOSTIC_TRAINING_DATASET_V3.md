# Step 5 v3 — matched diagnostic training dataset

Step 5 v3 is the accepted staged data-generation contract for the first serious TriQTO diagnostic training cohort.

The staged progression is based on **independent clean-circuit roots**, not derived perturbation examples:

`500 -> 1000 -> 2000 -> 5000`

Each larger stage must satisfy two gates before promotion:

1. the full-artifact EDA and scheduling-alias audit returns `PROMOTION_READY`;
2. the stage-nesting audit proves the entire previous accepted cohort is preserved exactly (`NESTING_VALID`).

## History

- v1 500-root candidate: rejected after full EDA found family/split aliasing and perfect depth/strength confounding.
- v2 500-root candidate: repaired v1 and added 3q, but rejected after deeper EDA found shot-count aliasing with affected-qubit context.
- v3 500-root candidate: accepted.
- v3 1000-root candidate: accepted and exactly nested over 500.
- v3 2000-root candidate: accepted and exactly nested over 1000.
- v3 5000-root candidate: final Step 5 stage; not yet accepted.

## Current accepted gates

### 500 roots

Product: `product_85a26bf36d9ed1769b6a5e23`

EDA: `audit_3ac00c9b31d1be6c54d7b852`

Decision: `PROMOTION_READY`.

### 1000 roots

Product: `product_7fef7da30ee7710d189ed27d`

EDA: `audit_9473903073b1ded46f174e4f`

Decisions: `PROMOTION_READY` + `NESTING_VALID` against the accepted 500-root product.

### 2000 roots

Product: `product_502097524aa54c78800fb459`

EDA: `audit_8a20d67048326db847149401`

Decisions: `PROMOTION_READY` + `NESTING_VALID` against the accepted 1000-root product.

The final 5000-root stage is therefore unlocked.

## Core input contract

Primary deployable inputs remain:

- intended/reference clean circuit graph;
- signed local X/Y/Z expectation deltas;
- signed same-basis pairwise XX/YY/ZZ correlation deltas;
- signed global X/Y/Z parity deltas;
- basis identity;
- observed/reference shot counts;
- reference availability/kind;
- neutral layout/backend metadata.

The hidden injected RZ/RX/RY mechanism never appears in the graph input or deployable metadata. Raw semantic reference-window IDs remain audit/meta-only. Statevectors and privileged exact simulator quantities remain target/audit-only.

## Supervision contract

Targets include:

- effect present / negligible;
- mechanism RZ/RX/RY;
- phenomenology phase-dominant / population-dominant / mixed / negligible;
- privileged continuous state-derived audit targets.

For injected-but-negligible interventions, `mechanism_loss_mask = false` so the model is not trained to hallucinate a mechanism from essentially null evidence.

## Final Step 5 closeout

The 5000-root cohort must pass:

- scheduling-alias gates;
- full-artifact EDA = `PROMOTION_READY`;
- exact 2000 -> 5000 nesting = `NESTING_VALID`.

Only then should Step 5 be closed and Step 6 cheap baselines begin.
