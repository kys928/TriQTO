# Training Plan

Future training views: distortion diagnosis, action ranking, Born prediction, Hilbert-to-Born prediction, topology audit, joint multitask, and hardware-masked training.

The documented future loss is:

`L_total = L_task + λ_geo L_geo + λ_diag L_diag + λ_action L_action + λ_top L_top`

Early `λ_top = 0`. Persistent homology is computed, logged, and used diagnostically from the beginning, but topology loss becomes active only after topology signals are validated.
