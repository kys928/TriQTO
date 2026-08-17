# Step 8 untouched confirmatory evaluation — result summary

Status: **COMPLETE — CONFIRMATORY COHORT SPENT**

Evidence status: **predeclared confirmatory same-generator replication**

## Identity and integrity

- confirmatory cohort: `confirm_653c328159326c605e3f3d98`
- evaluation: `confirm_eval_e4ef740739a4390bb850c3c0`
- architecture: `late_concat`
- seeds: `1701`, `1702`, `1703`
- clean roots: 2,000
- examples: 26,000
- mechanism-supervised examples: 21,428
- effect/no-effect support: 21,428 / 4,572
- uploaded evaluation ZIP SHA-256: `sha256:9dc2941875d1d8e7f74b8476643ac2be31a7b83a7a554984b1965553897b1cf5`
- sealed cohort SHA-256 recorded by evaluator: `sha256:dec4b034c28796d68c6873f7fc2b92f4c0667957052fca75a713c0edaacda722`
- Step-8 config SHA-256: `sha256:afab2c31139748636a7bf5f975212706dc7f037168c5566c78c20de2852f4063`
- evaluator SHA-256: `sha256:bca15ca118ad6655b666034a37ad7eabea2001c45a3097d9871d149ff12646d1`
- Step-7 architecture config SHA-256: `sha256:2715c93c0d22e4aff6caf71a5e4c8ae872f5f17c236ab3445808bcefd1895a77`
- confirmatory access marker SHA-256: `sha256:fdfdfe1ce4d5059bca598dd5da51d1aba36747097c1a5ca816011359ca07baa9`

All seven evaluation payload hashes recorded in `evaluation_complete.json` were independently re-hashed from the uploaded bundle and matched exactly. The spent marker also matches the recorded `evaluation_complete.json` hash.

The access marker records that the ensemble effect threshold (`0.059394106269`) was selected before confirmatory access. The result bundle records `confirmatory_labels_selected_nothing=true`, `architecture_changed_after_confirmatory_access=false`, no historical v0.1 access, no Phase-15.6 reuse, and no hardware execution.

## Predeclared gates

| Claim | Frozen gate | Observed | Decision |
|---|---|---|---|
| Primary mechanism | BA 95% CI lower > 0.45 AND every mechanism recall >= 0.40 | BA **0.4998**, 95% CI **[0.4921, 0.5072]**; recalls RZ **0.5387**, RX **0.5246**, RY **0.4361** | **SUPPORTED** |
| Secondary effect | BA 95% CI lower > 0.65 | BA **0.7119**, 95% CI **[0.7037, 0.7200]** | **SUPPORTED** |
| Secondary integrated | BA 95% CI lower > 0.40 | BA **0.4516**, 95% CI **[0.4445, 0.4585]** | **SUPPORTED** |

Additional confirmatory metrics:

- mechanism macro-F1: **0.4959**, 95% CI **[0.4881, 0.5033]**
- mechanism macro one-vs-rest ROC AUC: **0.6955**
- effect macro-F1: **0.6343**
- effect ROC AUC: **0.8073**
- integrated macro-F1: **0.4245**

The confirmatory mechanism BA is **0.0120 lower** than the Step-7 development reference (0.5118). This is a modest generalization drop, not a failure of the frozen confirmation gate.

## Seed behavior

- seed 1701 mechanism BA: **0.4923**
- seed 1702 mechanism BA: **0.4950**
- seed 1703 mechanism BA: **0.4986**
- three-seed mean-logit ensemble mechanism BA: **0.4998**

The ensemble result is not driven by one exceptional seed.

## Confirmatory stratified diagnostics

These are post-outcome descriptive diagnostics, not additional confirmation gates.

### Shots

- 512: mechanism BA **0.4558**
- 1024: **0.4717**
- 2048: **0.5148**
- 4096: **0.5565**

Finite-shot acquisition remains a major bottleneck.

### Intervention strength

- 0.05: mechanism BA **0.4267**
- 0.15: **0.5728**

Weak interventions remain substantially harder.

### Circuit family

- Bell-like: **0.5392**
- GHZ: **0.5369**
- hardware-efficient ansatz: **0.3596**
- phase-interference: **0.3748**
- QAOA-like: **0.6860**
- QFT-like: **0.5998**
- random-shallow: **0.3875**

Family-conditioned performance is highly uneven even within the same generator population. This is an important limitation.

### Qubit count

- 2q: **0.5319**
- 3q: **0.5006**
- 4q: **0.4874**
- 6q: **0.4852**
- 8q: **0.4837**

There is a modest decline with larger qubit counts, but no collapse.

## Interpretation

Step 8 supports the predeclared claim that the frozen `late_concat` model generalizes to **new independent roots from the same frozen finite-shot simulator population** with useful mechanism, effect, and integrated diagnosis.

It does **not** establish:

- real-QPU robustness;
- calibration-drift robustness;
- out-of-distribution circuit-family generalization;
- universal mechanism identification;
- quantum advantage;
- fault tolerance.

The weakest confirmed mechanism class is RY (recall 0.4361), only modestly above the frozen 0.40 floor. The model remains strongly shot-, strength-, and family-dependent. These limitations should be carried forward rather than hidden.

## Scientific boundary

The cohort is permanently spent. It must never be reused as confirmatory evidence or used to adapt architecture, thresholds, losses, seeds, or decision rules and then re-described as untouched confirmation.
