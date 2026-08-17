# Step 9A v1 replay failure — evidence note

Status: **FAILED CLOSED — NO DEPLOYMENT BUNDLE PRODUCED**

Date: 2026-08-17

Source branch/merge state:

- Step 9A v1 merge commit: `aa5275a68b8dc51ba9ddf498a1b2e1d2813c9925`
- development product: `product_b2d78ad2309b71a55f9bb54f`
- device: CUDA
- architecture: `late_concat`
- seed under failure: `1701`

## Observed failure

The v1 deployment-freeze runner attempted to reconstruct the archived Step-8 checkpoint selection by rerunning the original seeded Step-8 training procedure and requiring the same selected epoch.

Archived Step-8 seed-1701 selected epoch:

`8`

Observed Step-9A v1 replay selected epoch:

`19`

The runner terminated with:

`RuntimeError: seed 1701 selected epoch changed: 19 != 8`

No deployment bundle was written.

## Interpretation

The failure invalidates the v1 assumption that the exact Step-8 CUDA checkpoint weights can be reconstructed from seed + data + training code alone. Step 8 seeded Python, NumPy and PyTorch, but did not enable deterministic CUDA algorithms. The graph model also uses segmented reductions implemented with operations including `index_add_`.

This does **not** invalidate the Step-8 confirmatory result. Step 8 evaluated the actual in-memory models trained during that one-shot confirmatory run. It does mean that those exact weights are now irrecoverable because they were not serialized.

## Corrective action

Step 9A v2 changes the deployment boundary to a **post-confirmation fixed-epoch refit**:

- architecture remains `late_concat`;
- seeds remain `1701/1702/1703`;
- epoch counts are fixed to the archived Step-8 selected epochs `8/11/17`;
- the current refit cannot select new epochs;
- the deployed effect threshold remains the archived Step-8 value;
- the current refit cannot select a new deployment threshold;
- the Step-8 confirmatory cohort is not accessed;
- exact Step-8 checkpoint identity is explicitly not claimed;
- actual serialized checkpoint SHA-256 hashes define the deployment weight identity;
- only one successful deployment bundle is allowed before Step 9B consumes it.

The failed v1 attempt is retained as evidence rather than hidden or repaired post hoc.
