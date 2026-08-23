# Step 11 pre-QPU freeze

Status: **FROZEN BEFORE PHYSICAL-QPU EXECUTION**

This file freezes the implementation identity for the first IBM hardware stage after the Step-10D simulator hard stop.

## Frozen source outcome

- Step-10D benchmark: `benchmark_1455864a09de8804a7e7958a`
- Step-10D uploaded audit ZIP SHA256: `e62eb3de2c7e5ccd00b0087d095c4f60394d79ffbd601a1c2042ac7ee926901b`
- Step-10D hardware-candidate decision: `step10c_warm_start`
- further simulator tuning before hardware: **forbidden**

## Frozen primary hardware model

- Step-10C benchmark: `benchmark_f9478da45d68795655259054`
- architecture: `late_concat`
- trainable parameters: 453,829
- seeds: 1701/1702/1703
- effect threshold: `-0.125638447701931`
- checkpoint SHA256 values are frozen in `configs/v0_2/step11_exploratory_ibm_transfer_pilot.json`.

Step-9A is a report-only paired baseline on the same counts and cannot become the primary candidate after QPU results are observed.

## Frozen Step-11 file identities

Git blob SHA values on branch `agent/step11-exploratory-ibm-transfer-pilot` before physical-QPU execution:

- config `configs/v0_2/step11_exploratory_ibm_transfer_pilot.json`: `c2a701e28c30fc7ec250a4d4bf14ae1a475bb3d3`
- protocol `docs/evidence/step11_exploratory_ibm_transfer_pilot/PROTOCOL.md`: `9fdd3ce9e4d73bad9ed4398ff99cf326dfccc3d8`
- runner `scripts/v0_2/run_step11_exploratory_ibm_transfer_pilot.py`: `d0c796b3227b13f3d6de1912c1547b3bea45d505`
- contract tests `tests/test_step11_exploratory_ibm_transfer_contract.py`: `6a3e058d1d919db0f44a10b7a80b41a16f4ae707`

Any change to these files after this freeze requires a documented amendment before QPU submission. Execution-enabling software compatibility repairs are allowed only if they leave the scientific design, model identities, shot count, circuit matrix, analysis rule, and no-tuning boundaries unchanged.

## Frozen acquisition

- explicit IBM Open Plan instance only; paid-plan execution forbidden;
- one quality-selected backend and connected three-qubit chain;
- 12 anchor cases;
- six programs per case;
- 4096 shots per program;
- 72 programs / 294,912 total circuit shots;
- SamplerV2 single-job mode;
- max QPU execution time 300 s;
- no mitigation, dynamical decoupling, or twirling;
- no model-prediction-driven backend choice;
- exact plan must be generated and inspected before explicit execution.

## Claim boundary

Step 11 is exploratory targeted-transfer evidence under an incompletely satisfied simulator gate. It is not a confirmatory test of the full TriQTO hypothesis and may not be used to retroactively declare the Step-10 simulator gate passed.

No physical QPU access is authorized by this freeze alone. The merged runner requires an explicit saved plan and confirmation token before submission.
