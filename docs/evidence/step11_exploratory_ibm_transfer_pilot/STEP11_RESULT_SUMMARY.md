# Step 11 exploratory IBM transfer pilot — result summary

Status: **FROZEN POSTOUTCOME EXPLORATORY HARDWARE RESULT**

Plan: `qpuplan_eaef7207095be0da5e12212b`  
IBM Runtime job: `da6n7es6l22c73dmiaag`  
Backend: `ibm_kingston` (`Heron` revision 2)  
Physical chain: `[10, 11, 12]`  
QPU usage: `78` quantum seconds

## Predeclared primary targeted result

The frozen primary metric was Step-10C mechanism correctness on the three distorted `phase_interference` anchor cases.

**Observed: 3/3 correct.**

By the pre-QPU rule in `PROTOCOL.md`, 3/3 maps to **strong exploratory targeted-repair transfer signal**. This wording was fixed before physical execution and is descriptive, not a confirmatory pass/fail gate.

The report-only Step-9A paired baseline scored **0/3** on the same three acquired hardware cases.

## Secondary results

| Metric | Step 10C primary | Step 9A report-only |
| --- | ---: | ---: |
| Distorted mechanism correctness | 6/9 (0.6667) | 4/9 (0.4444) |
| Distorted effect detections | 7/9 | 7/9 |
| Clean effect false positives | 0/3 | 1/3 |
| Phase-interference mechanism correctness | 3/3 | 0/3 |
| GHZ mechanism correctness | 2/3 | 2/3 |
| Bell-like mechanism correctness | 1/3 | 2/3 |

Paired distorted-case comparison: Step10C-only correct = **4**, Step9A-only correct = **2**, both correct = **2**, both wrong = **1**.

## Claim boundary

This result supports a **localized exploratory transfer statement**: the Step-10C model recovered all three targeted phase-mechanism labels on the physical IBM anchor context that motivated the repair work. It does **not** establish general hardware superiority, does not make Step 11 confirmatory, and does not retroactively satisfy the previously unmet full simulator gate.

No model weight, threshold, architecture, checkpoint, shot count, backend choice, mitigation setting, or candidate identity was changed from QPU results.
