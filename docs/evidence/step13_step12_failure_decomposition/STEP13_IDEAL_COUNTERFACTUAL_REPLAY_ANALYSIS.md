# Step 13 ideal-diagnostic counterfactual replay analysis

## Question

Does the Step-12 mechanism failure disappear if the frozen Step-10C model is given exact ideal diagnostic tensors on the exact same Step-12 circuit graphs/layouts?

## Result

No.

The frozen Step-10C ensemble reproduces the Step-12 hardware result at 8/18 mechanism classifications correct. Replacing all acquired QPU diagnostic tensors with exact statevector-derived local, pairwise, and parity diagnostics leaves aggregate mechanism correctness unchanged at 8/18. Replacing only local diagnostics with ideal values also gives 8/18. Replacing only pairwise/parity diagnostics gives 8/18.

This replay used no QPU, no retraining, no threshold changes, and no checkpoint selection. The original hardware predictions were reproduced before the counterfactual comparison.

## Motif structure

The strongest failure is graph/motif dependent.

- `cz_echo_ramsey`: 0/6 under hardware diagnostics and 0/6 under fully ideal diagnostics.
- `dual_arm_recombination`: 3/6 under hardware diagnostics and 2/6 under fully ideal diagnostics.
- `three_qubit_phase_fanout`: 5/6 under hardware diagnostics and 6/6 under fully ideal diagnostics.

For `cz_echo_ramsey`, idealization does not alter the qualitative mechanism mapping. At both strengths, RZ is mapped to RY, RX is mapped to RZ, and RY is mapped to RZ. This is a systematic learned mapping error rather than a finite-shot perturbation around otherwise-correct decision boundaries.

## Mechanism structure

Under acquired hardware diagnostics Step-10C obtains RZ 3/6, RX 1/6, and RY 4/6. Under fully ideal diagnostics it obtains RZ 2/6, RX 2/6, and RY 4/6. There is no global ideal-data rescue and no isolated diagnostic block whose idealization restores the classifier.

## Relation to the first Step-13 decomposition

The earlier evidence-geometry decomposition found that the QPU diagnostic vectors remain strongly aligned with their ideal counterparts and that a posthoc nearest-ideal-template diagnostic can separate all 18 distorted cases. The counterfactual replay independently strengthens that interpretation: even when hardware diagnostic imperfections are removed completely, the learned model still fails on the same new circuit context.

Therefore simple hardware-evidence collapse is not supported as the primary failure explanation.

## Supported conclusion

The primary failure class is learned out-of-distribution mapping/generalization. The dependence on motif, especially the invariant 0/6 `cz_echo_ramsey` result under hardware and ideal diagnostics, makes graph-conditioned context generalization a leading candidate.

The current evidence does not yet isolate whether the failure originates specifically in the graph encoder, diagnostic encoder, graph-diagnostic fusion, mechanism head, or their jointly learned representation. It is therefore not scientifically justified to replace the architecture based only on Step 13.

## Training implication

Additional training/generalization is now a justified next intervention, but it must be distributional rather than memorization of the 21 Step-12 cases. A next-stage training corpus should broaden phase-mechanism supervision across many independently generated motifs, injection positions, circuit sizes, interaction patterns, strengths, and layouts while retaining previous-domain examples to protect against forgetting.

Step-12 cases are now post-outcome diagnostic evidence and must not become the evaluation set for a model trained from information derived from them. A future hardware validation must use a new frozen, blind circuit/backend cohort.

The default checkpoint policy remains warm-start from the frozen Step-10C model unless a later controlled ablation demonstrates that its learned representation is actively harmful.
