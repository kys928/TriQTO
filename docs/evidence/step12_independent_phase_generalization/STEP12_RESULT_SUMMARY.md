# Step 12 independent cross-backend phase generalization — result summary

Status: **FROZEN POSTOUTCOME HARDWARE RESULT — PREDECLARED GATE FAILED**

Plan: `step12plan_24b2631b5cd94c58af951492`  
IBM Runtime job: `da6od4k6l22c73dmjscg`  
Backend: `ibm_marrakesh` (`Heron` revision 2)  
Physical chain: `[134, 135, 139]`  
QPU usage: `136` quantum seconds

## Predeclared primary result

The frozen primary metric was Step-10C mechanism correctness across all 18 distorted Step-12 cases on a backend different from Step 11, with new phase-sensitive motifs and two frozen strengths.

**Observed Step-10C mechanism correctness: 8/18 (0.4444).**

The predeclared support gate therefore failed. The only permitted gate interpretation is:

`NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_NOT_SUPPORTED`

The paired Step-9A report-only baseline scored **9/18 (0.5000)** on the same acquired QPU counts. Step-10C minus Step-9A mechanism correctness was **-1**, rather than the predeclared required advantage of at least +3.

## Gate criteria

| Criterion | Frozen requirement | Observed | Pass |
| --- | ---: | ---: | :---: |
| Step-10C total mechanism correctness | >=14/18 | 8/18 | NO |
| Each mechanism | >=4/6 | RZ 3/6; RX 1/6; RY 4/6 | NO |
| Each strength | >=6/9 | 0.13: 5/9; 0.27: 3/9 | NO |
| Distorted effect detection | >=14/18 | 18/18 | YES |
| Clean false positives | <=1/3 | 1/3 | YES |
| Step-10C minus Step-9A mechanism correct | >=+3 | -1 | NO |

## Context breakdown

| Motif | Step 10C | Step 9A report-only |
| --- | ---: | ---: |
| `cz_echo_ramsey` | 0/6 | 0/6 |
| `dual_arm_recombination` | 3/6 | 3/6 |
| `three_qubit_phase_fanout` | 5/6 | 6/6 |

Step-10C detected an effect in **18/18** distorted cases but often assigned the wrong mechanism label. The failure therefore cannot be summarized as simple inability to notice a perturbation.

## Claim boundary

Step 12 does **not** support the planned narrow cross-backend phase-generalization statement. It also does not erase or replace the earlier Step-11 localized exploratory result. The correct combined reading is that the Step-11 phase success did not generalize robustly under the harder Step-12 change of backend, circuit motifs, strengths, and layouts.

No model weight, threshold, architecture, checkpoint, shot count, backend choice, mitigation setting, or gate interpretation was changed after seeing Step-12 QPU results.
