# Step 11 post-outcome freeze

Status: **FROZEN AFTER FIRST COMPLETED PHYSICAL-QPU PILOT**

This freeze records the first completed Step-11 physical-QPU outcome before any posthoc diagnostic experiment, reinterpretation, retraining, retuning, model replacement, or additional exploratory hardware attempt.

## Frozen identity

- plan: `qpuplan_eaef7207095be0da5e12212b`
- IBM Runtime job: `da6n7es6l22c73dmiaag`
- primary model: `step10c_warm_start`
- paired report-only baseline: `step9a_deployment_ensemble`
- backend: `ibm_kingston` version `1.0.0`
- processor: `Heron` revision `2`
- physical chain: `[10, 11, 12]`
- programs: 72
- shots/program: 4096
- total circuit shots: 294912
- QPU usage: 78 seconds
- source bundle SHA256: `62d34c456f9c362ee8d2bd55e5f6002cbd7736117282c56e12b42a6fb5a54200`

## Frozen outcome

Primary predeclared targeted phase-mechanism result: **3/3 correct**, corresponding to the predeclared phrase **strong exploratory targeted-repair transfer signal**.

Secondary descriptive result: Step10C distorted mechanism correctness **6/9** versus Step9A report-only **4/9**. Step10C-only / Step9A-only correct counts were **4 / 2**.

This outcome remains explicitly exploratory. No confirmatory claim is created by the 3/3 targeted result.

## No post-outcome reinterpretation or tuning

Before this freeze, the QPU result was not used to:

- train or fine-tune model weights;
- change the architecture;
- change the Step-10C or Step-9A checkpoint identities;
- change either model's effect threshold;
- change the shot count;
- adapt mitigation, dynamical decoupling, or twirling;
- choose the backend or physical chain from model predictions;
- replace the predeclared primary candidate;
- redefine the primary metric or its 3/3, 2/3, 0–1/3 interpretation rule;
- retroactively mark the Step-10 simulator gate as passed.

The initial failed execution attempts did not start physical QPU access and produced no IBM Runtime job ID. They are not replacements for this completed plan. The completed job above is the first Step-11 physical-QPU outcome and is immutable as the first exploratory result.

## Forward-use rule

Posthoc diagnosis may begin only after this freeze is committed. Any later hardware run is a **new exploratory attempt** with separate provenance and may not overwrite, erase, or replace this result. Any future confirmatory hardware study requires a newly frozen design with independent claim boundaries.
