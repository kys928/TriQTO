# Step 12 independent cross-backend phase generalization — reviewer-resistant analysis

Status: **POSTHOC ANALYSIS OF A COMPLETED PREDECLARED QPU GENERALIZATION TEST**

This analysis freezes what the completed Step-12 result does and does not support. It does not alter the pre-QPU protocol, the frozen support gate, the model, or the Step-11 result.

## 1. Integrity and provenance

The uploaded source execution bundle `step12plan_24b2631b5cd94c58af951492.zip` has SHA256 `9c9a4a726e0403a466960669917a394ef7df73c2f6fb12239dbdf799f818d7ff` and passes ZIP CRC validation. `step12_complete.json` records `status=COMPLETE`, plan `step12plan_24b2631b5cd94c58af951492`, IBM Runtime job `da6od4k6l22c73dmjscg`, backend `ibm_marrakesh` version `1.0.21`, physical chain `[134, 135, 139]`, QPU usage `136` seconds, `qpu_results_used_for_tuning=false`, and `step11_result_replaced=false`.

Every execution-file SHA256 declared in `step12_complete.json` was verified against the bytes in the uploaded source bundle and matched.

The public freeze intentionally does not publish the full IBM instance CRN. Its SHA256 is preserved in `STEP12_HARDWARE_PROVENANCE.json`; the exact local `generalization_plan.json` and `backend_snapshot.json` hashes are frozen in `STEP12_RESULT_MANIFEST.json`.

## 2. The predeclared question

Step 12 asked whether the Step-10C phase-mechanism capability seen in Step 11 generalized to a new backend and new circuit context without retraining or QPU-driven adaptation.

Before QPU execution, the support gate required all of the following:

- total Step-10C mechanism correctness >=14/18;
- each mechanism >=4/6;
- each strength >=6/9;
- distorted effect detection >=14/18;
- clean false positives <=1/3;
- Step-10C minus Step-9A mechanism-correct advantage >=3.

Observed Step-10C result: **8/18** mechanism-correct. The gate failed. The frozen interpretation is therefore exactly:

`NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_NOT_SUPPORTED`

## 3. Which criteria failed

Four of the six gate criteria failed:

- total mechanism correctness: 8/18 < 14/18;
- per-mechanism floor: RZ 3/6, RX 1/6, RY 4/6, so the floor failed;
- per-strength floor: 0.13 -> 5/9 and 0.27 -> 3/9, so both are below 6/9;
- paired advantage: Step10C 8/18 versus Step9A 9/18, giving -1 rather than >=+3.

Two criteria passed:

- distorted effect detection: 18/18;
- clean false positives: 1/3.

Thus the failure is primarily a **mechanism-discrimination generalization failure**, not a failure to detect that a distortion occurred.

## 4. Strong context dependence

The motif breakdown is:

| Motif | Step 10C | Step 9A |
| --- | ---: | ---: |
| `cz_echo_ramsey` | 0/6 | 0/6 |
| `dual_arm_recombination` | 3/6 | 3/6 |
| `three_qubit_phase_fanout` | 5/6 | 6/6 |

The result is therefore highly context-dependent. Step-10C performed well on `three_qubit_phase_fanout`, partially on `dual_arm_recombination`, and failed completely on `cz_echo_ramsey`. The Step-9A baseline shows almost the same ordering, which argues against describing Step-10C as generally superior on this new hardware context.

The mechanism confusion is also uneven: Step-10C obtained RZ 3/6, RX 1/6, and RY 4/6. RX overrotation is the weakest mechanism in this attempt.

## 5. The identifiability audit passed

The mandatory model-blind statevector audit passed before hardware execution. The minimum low-strength reference-to-distorted diagnostic delta was `0.073691` against the frozen minimum `0.04`. The minimum low-strength pairwise mechanism distance was `0.118163` against the frozen minimum `0.10`.

This means the chosen idealized diagnostic evidence was not trivially degenerate under the predeclared statevector audit. It does **not** prove that the mechanisms remained equally separable after physical-device noise, finite sampling, compilation, and the model's learned representation.

## 6. Relationship to Step 11

Step 11 remains a valid localized exploratory observation: Step-10C scored 3/3 on the targeted phase-interference anchor context. Step 12 does not retroactively invalidate those measurements.

What Step 12 changes is the scope of the conclusion. The Step-11 success cannot be promoted to a robust cross-backend/cross-motif phase-generalization claim, because the predeclared independent test failed and Step-10C did not outperform the paired older baseline.

## 7. What this result supports

The result supports the following narrow statements:

- Step-10C can detect the presence of the Step-12 controlled distortions on this acquisition very reliably (18/18 effect detections);
- its mechanism labels are not robustly invariant to the Step-12 shift in backend/circuit context;
- the earlier targeted phase repair is not sufficient evidence of broad phase-mechanism generalization;
- additional diagnosis is required before any new training or architectural intervention is justified.

## 8. What this result does not establish

Step 12 does not by itself prove:

- that the architecture is fundamentally incapable of cross-backend generalization;
- that `ibm_marrakesh` hardware noise alone caused the failure;
- that the new circuit motifs alone caused the failure;
- that stronger distortions are intrinsically harder in general;
- that one particular representation feature is responsible;
- that retraining, threshold adjustment, mitigation, or architecture changes are warranted without a separate diagnosis.

Those are hypotheses for a new stage, not conclusions from Step 12.

## 9. Frozen scientific conclusion

**The predeclared independent Step-12 test failed. The Step-11 targeted phase-mechanism success did not generalize robustly to the frozen cross-backend, cross-motif Step-12 setting. Step-10C detected distortions reliably but did not classify their mechanisms reliably, and it did not outperform the Step-9A paired baseline. Step 11 remains a localized exploratory result; Step 12 limits its generalization scope.**
