# Step 7 structured diagnostic model — smoke result

Status: **STEP7_SMOKE_PASS**

This is a non-scientific plumbing gate. It does not provide model-quality evidence and did not access Step-7 selection or outer-development-validation roots.

## Source identity

- Step 5 product: `product_b2d78ad2309b71a55f9bb54f`
- smoke ID: `smoke_185f69415ea6bb082dd93ef7`
- uploaded `smoke_complete.json` SHA-256: `4fe46b4c2f3c88f14c00df7d8e6c45fd5d74f06f9d76b415282e3f0a568e85c0`
- experiment config SHA-256 recorded by smoke: `2715c93c0d22e4aff6caf71a5e4c8ae872f5f17c236ab3445808bcefd1895a77`
- smoke config SHA-256: `b16c64f08cbb6006025e03cf07eb14d366a3cff68f048bd30d263d2657122c29`
- runner SHA-256: `e3201ba469bcbff68defe7a07bbdb2d707303a1ce0fa8fa630546bb2d7fe8f35`
- selected fit roots: `2, 3, 5, 6, 8, 9, 11, 12`

## Access boundary

- fit roots/examples accessed: **8 / 104**
- selection roots accessed: **0**
- outer validation roots accessed: **0**
- historical v0.1 test accessed: **NO**
- spent Phase 15.6 confirmatory cohort accessed: **NO**
- scientific metric claim: **NO**

## Plumbing result

All seven frozen variants completed forward pass, effect loss, masked mechanism loss, backward pass, finite-gradient validation, gradient clipping, optimizer step, and post-step finite-output validation.

All variants intentionally have the same trainable parameter count: **453,829**.

| Variant | Total smoke loss | Gradient norm |
|---|---:|---:|
| diagnostic_only | 2.1054 | 34.6251 |
| graph_only | 2.0722 | 7.0838 |
| late_concat | 2.0849 | 29.4289 |
| structured_interaction | 2.0019 | 36.3096 |
| structured_no_magnitude | 1.9528 | 32.4975 |
| structured_no_pairwise | 2.0132 | 32.9602 |
| structured_no_parity | 2.1231 | 41.8913 |

These losses must not be ranked or interpreted scientifically: they are one optimizer step on eight fit roots and were generated solely to validate the implementation contract.

## Consequence

The real-artifact smoke gate is satisfied. The frozen Step-7 architecture/ablation contract may proceed to the full development benchmark. The full runner must preserve the predeclared fit/selection/outer-validation split and may not alter the architecture-specific gate after observing neural outcomes.
