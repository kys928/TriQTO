# Step 7.1 — final parity-residual architecture revision

Status: **FROZEN BEFORE STEP-7.1 OUTCOME**

Evidence status: **adaptive development, not confirmatory**.

Step 7 established that the learned late-concatenation graph+diagnostic model is materially stronger than the cheap Step-6 mechanism baselines, while the bespoke local/pair structured interaction is slightly and significantly worse than the matched late-concat control. Step 7 also showed a one-seed positive global-parity ablation and no demonstrated positive contribution from the current pairwise gated interaction.

Step 7.1 therefore performs exactly one final, narrowly motivated architecture revision. It is not another architecture sweep.

## Champion

`late_concat`

This is the Step-7 winning architecture: independent graph and diagnostic encoders, nonlinear graph-level late fusion, explicit magnitude/shot-aware effect pathway, and no local/pair graph-diagnostic gated interaction.

Frozen Step-7 outer-development metrics:

- effect BA: `0.7128`
- mechanism BA: `0.5118`
- integrated BA: `0.4614`

## Candidate

`late_concat_parity_residual`

The candidate keeps the late-concat champion unchanged except for one explicit global parity interaction:

1. encode graph geometry as before;
2. encode signed local/pair/parity diagnostics as before;
3. construct the same late-concat representation as the champion;
4. form one graph-conditioned global-parity interaction using the existing Step-7 global gate;
5. add that interaction as a scalar-gated residual to the late-concat representation.

The scalar residual parameter is initialized to exactly zero. Therefore, before training, candidate and champion representations are identical. Training must earn any departure from the champion.

No local qubit-diagnostic interaction is reintroduced. No pair-endpoint gated interaction is reintroduced. Pairwise diagnostic values remain available only inside the generic diagnostic encoder, exactly as in late concat.

All Step-7.1 variants instantiate the same parameter set: **453,830 trainable parameters**.

## Diagnostic ablation

`late_concat_no_parity`, seed `1701` only.

This is not a third architecture candidate. It asks whether global parity contributes inside the late-concat family itself. Its result is diagnostic development evidence only.

## Data and training

Step 7.1 reuses the exact Step-7 development partitions and training contract:

- fit: 3,000 roots;
- internal selection: 1,000 roots;
- outer development validation: 1,000 roots;
- all 13 derivatives of one clean root remain together;
- primary seeds: `1701`, `1702`, `1703`;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- root batch size 32;
- max 20 epochs;
- patience 4;
- class-balanced effect BCE and masked mechanism CE;
- checkpoint selection by internal-selection mechanism BA first, then effect BA;
- effect threshold selected only on internal-selection roots.

The outer validation cohort has already been inspected in Steps 6 and 7. Step 7.1 is adaptive development evidence and cannot be called confirmation.

## Primary decision gate

The candidate wins only if **all** of the following hold:

1. the re-run late-concat champion reproduces the frozen Step-7 late-concat mechanism BA within absolute `0.01`;
2. paired clean-root 95% CI lower bound for
   `mechanism BA(late_concat_parity_residual) - mechanism BA(late_concat)`
   is greater than zero;
3. paired clean-root 95% CI lower bound for candidate-minus-champion effect BA is at least `-0.005`.

If all three pass, freeze `late_concat_parity_residual` as the final development architecture.

If any fail or tie, freeze `late_concat` as the final development architecture.

## Stop rule

**There is no Step 7.2 architecture search on this development validation cohort.**

After Step 7.1, architecture selection stops. The next scientific stage is to generate and freeze a genuinely new untouched confirmatory cohort for whichever final architecture Step 7.1 selects.

## Run

PR #57 must merge first. The Step-7.1 branch is currently stacked from the exact completed Step-7 head so its scientific contract could be frozen immediately without modifying Step 7.

After #57 is merged and this branch is updated/retargeted:

```bash
cd /workspace/triqto

git fetch origin
git switch agent/step7-1-parity-residual
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q \
  tests/test_step7_structured_diagnostic_model.py \
  tests/test_step7_full_development_benchmark.py \
  tests/test_step7_1_parity_residual.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step7_1_parity_residual_benchmark.py \
  --product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f \
  --step7-dir /workspace/triqto-data/step7_structured_diagnostic_benchmark/benchmark_9989fbf9ef9feeaf283fe23f
```

The runner SHA-verifies the frozen Step-5 source and all recorded Step-7 result files, reuses the exact Step-7 partitions, and materializes the frozen artifacts once.

Expected completion marker:

`STEP7_1_FINAL_REVISION_COMPLETE`

Completion means the frozen one-shot revision executed correctly. It does not imply the candidate won; the selected final architecture is recorded separately in `decision.json` and `benchmark_complete.json`.
