# Step 9G deployment-domain coverage bridge — frozen outcome

Scientific boundary: post-hoc exploratory audit only. No QPU submission and no TriQTO retraining occurred.

## Frozen returned artifacts

`deployment_domain_coverage_bridge_v1.json`

SHA-256:

`71b16f01745d18765ea4114264299210a7d437c4caa525741def9b9899ebbd3f`

`deployment_domain_coverage_bridge_v1.log`

SHA-256:

`2be7b3794623261f35cb3b321292e3587eb26db826dc528edabe4db7732a92f4`

Step-5 product:

`product_b2d78ad2309b71a55f9bb54f`

## Primary targeted result

The frozen Step-9G decision status is:

`DEPLOYMENT_DOMAIN_COVERAGE_RESCUES_CIRCUIT_ONLY_FRAME_IN_TARGETED_AUDIT`

Step5-only support fails catastrophically on the bridge domain (Euclidean BA 0.2639, lower 95% CI 0.2014, minimum recall 0.0417) and gets only 1/3 frozen Step-9D phase QPU mechanisms correct.

Bridge-only support passes strongly on held-out bridge validation:

- Euclidean BA = 0.9722;
- lower 95% CI = 0.9444;
- minimum mechanism recall = 0.9583;
- frozen Step-9D phase QPU = 3/3;
- predicted QPU local-frame median cosine to exact simulator frame ≈ 1.0000;
- QPU nearest support distance = 0.0731 versus bridge-validation maximum 1.8823.

Step5+bridge support also passes strongly on the held-out bridge/QPU target:

- Euclidean BA = 0.9653;
- lower 95% CI = 0.9375;
- minimum mechanism recall = 0.9375;
- frozen Step-9D phase QPU = 3/3;
- predicted QPU local-frame median cosine to exact simulator frame ≈ 1.0000;
- QPU nearest support distance = 0.0880 versus bridge-validation maximum 1.6841.

The exact frozen Step-9D pilot circuit was excluded from the bridge generator.

## Required qualification

The headline coverage rescue is **targeted**, not a demonstration that one global circuit-only frame estimator now works across both original Step-5 and deployment-bridge domains.

For `step5_plus_bridge`, the same estimator evaluated on the original frozen Step-5 validation contexts obtains only:

- cosine BA = 0.5632;
- Euclidean BA = 0.5862;
- Euclidean lower 95% CI = 0.4943;
- Euclidean minimum recall = 0.5172.

Therefore adding deployment-domain support rescues the targeted deployment region but does not by itself establish compatibility with the original Step-5 response-frame population under this simple global RBF estimator.

A second qualification is architectural. Step 9G's successful hand-built circuit feature map includes exact gate-level phasor information and a candidate local context (candidate qubit and insertion boundary). The current Step-7 graph adapter does not expose exactly that same information to the mechanism head: gate-level angle phasors are collapsed into aggregate qubit features, and the true candidate distortion boundary/qubit is not a deployable known input.

Accordingly, Step 9G supports `missing deployment-domain coverage` as a real failure mode, but it does **not** yet prove that dataset expansion alone, with the current Step-7 input/representation contract unchanged, is sufficient.

## Hard next action

Before retraining TriQTO, run a representation-contract ablation that separates:

1. information already available under the current Step-7 graph contract;
2. adding gate-level phasor preservation;
3. adding/using an explicit candidate-local-context mechanism.

Only after that audit should the project decide between pure dataset redesign, a minimal graph-adapter/input-projection change with checkpoint reuse, or a larger local-context architectural change.
