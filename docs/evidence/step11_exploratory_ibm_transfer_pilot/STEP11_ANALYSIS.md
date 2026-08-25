# Step 11 exploratory IBM transfer pilot — reviewer-resistant analysis

Status: **POSTHOC ANALYSIS OF A COMPLETED PREDECLARED EXPLORATORY QPU PILOT**

This analysis freezes what the completed Step-11 result does and does not support. It does not modify the pre-QPU protocol, the model, or the interpretation rule.

## 1. Integrity and provenance

The source execution bundle `qpuplan_eaef7207095be0da5e12212b.zip` has SHA256 `62d34c456f9c362ee8d2bd55e5f6002cbd7736117282c56e12b42a6fb5a54200` and passes ZIP CRC validation. The completed artifact identifies plan `qpuplan_eaef7207095be0da5e12212b`, IBM Runtime job `da6n7es6l22c73dmiaag`, backend `ibm_kingston` version `1.0.0`, physical chain `[10, 11, 12]`, and QPU usage `78` seconds.

`pilot_complete.json` records `status=COMPLETE`, `exploratory_only=true`, `confirmatory_claim=false`, `simulator_full_gate_previously_met=false`, and `qpu_results_used_for_tuning=false`. Every required execution-file SHA256 recorded by `pilot_complete.json` matches the bytes in the uploaded source bundle.

The public freeze intentionally does not publish the full IBM instance CRN. Its SHA256 is preserved in `STEP11_HARDWARE_PROVENANCE.json`, while the exact local `pilot_plan.json` and `backend_snapshot.json` hashes remain in `STEP11_RESULT_MANIFEST.json`.

## 2. Predeclared primary question

The primary targeted metric was fixed before QPU access: mechanism correctness of the frozen Step-10C warm ensemble on the three distorted `phase_interference` anchor cases. The interpretation rule was also fixed before execution:

- 3/3 = strong exploratory targeted-repair transfer signal;
- 2/3 = partial exploratory targeted-repair transfer signal;
- 0/3 or 1/3 = weak or absent exploratory targeted-repair transfer signal.

Observed Step-10C result: **3/3**. Therefore the correct predeclared interpretation is **strong exploratory targeted-repair transfer signal**.

The paired Step-9A baseline, run only as inference on the same acquired counts, scored **0/3** on those three phase cases.

## 3. Broader result is mixed, not universally positive

Across all nine distorted cases, Step 10C obtained **6/9** correct mechanisms (0.6667) and Step 9A obtained **4/9** (0.4444). The paired comparison was 4 Step10C-only correct, 2 Step9A-only correct, 2 both correct, and 1 both wrong.

The family breakdown prevents an overbroad claim:

| Family | Step 10C | Step 9A |
| --- | ---: | ---: |
| `phase_interference` | **3/3** | **0/3** |
| `ghz` | 2/3 | 2/3 |
| `bell_like` | 1/3 | 2/3 |

Thus the observed advantage is concentrated in the targeted phase-interference family. Step 10C was not uniformly superior: Step 9A was better on the Bell-like mechanism subset.

## 4. Effect detection did not explain the mechanism result

Both models detected an effect in **7/9** distorted cases. Step 10C had **0/3** clean effect false positives, while Step 9A had **1/3**. The main targeted Step-11 signal is therefore mechanism discrimination on phase-interference cases, not a generic increase in effect-detection rate.

The frozen code compares the ensemble mean **effect logit** with the stored effect threshold; the reported probability is a sigmoid-transformed display value. A probability column should therefore not be directly compared numerically with the logit threshold.

## 5. What this supports

The result is consistent with the hypothesis that the simulator-side coverage repair represented by the Step-10C training state addressed the specific phase-mechanism weakness that had appeared in the earlier hardware context, and that this targeted repair transferred back to physical IBM hardware.

Because the exact Step-9D anchor family had already influenced development decisions, Step 11 is not an independent confirmation. The exact anchor graph was excluded from bridge training, but prior knowledge of the hardware weakness remains part of the experimental history.

## 6. What this does not support

This single pilot does not establish:

- full TriQTO hardware validation;
- general superiority of Step 10C across circuit families, layouts, backends, strengths, or times;
- a causal proof that bridge coverage alone caused the improvement;
- a confirmatory estimate of generalization error;
- retroactive passage of the Step-10 full simulator gate;
- permission to switch models or tune using these QPU outcomes.

The targeted primary sample contains only three distorted phase cases, all on one backend and one selected physical chain at one calibration epoch. The broader 12-case matrix uses controlled synthetic perturbations of fixed strength 0.15, with physical-device noise superimposed. These limitations materially constrain external generalization.

## 7. Frozen scientific conclusion

**Step 11 provides strong exploratory evidence of targeted phase-mechanism transfer for the frozen Step-10C model on the predeclared IBM hardware anchor context. The result is localized, exploratory, and non-confirmatory. It does not justify a general hardware-superiority claim or any retroactive rewrite of Step 10 outcomes.**
