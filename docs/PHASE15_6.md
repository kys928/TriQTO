# Phase 15.6 — Research campaign environment and execution

Phase 15.6 turns the Phase 15.5 engineering path into a user-operated research campaign. It prepares a reproducible pod environment, a versioned experiment plan, one shared immutable data universe, multi-seed Phase 14 training, per-seed Phase 15.5 benchmarking, and a cross-seed summary.

This phase **does not launch a cloud pod, spend cloud credits, generate the full research dataset in CI, or use physical quantum hardware**. The user chooses the pod and persistent volume, then runs the provided commands.

## What is implemented

- strict `triqto.phase15_6.campaign.v1` configuration;
- source-config hashes and an immutable campaign identity;
- conservative resource estimates;
- pod preflight checks for Python, packages, CPU, RAM, disk, Torch, and CUDA;
- a shared Phase 7 → 8 → 9 → 11 → 12 data stage;
- one Phase 14 run per configured training seed;
- one Phase 15.5 noisy operational-policy benchmark per trained seed;
- cross-seed aggregation without inventing a paper-level claim;
- stage locks and completed-stage reuse;
- rejection of generated workspaces inside the Git checkout;
- enforced `physical_hardware=false` and `topology_loss_weight=0.0`.

The repository includes a first research pilot configuration with:

- 21 circuit specifications across Bell, GHZ, phase-interference, QFT-like, hardware-efficient, random-shallow, and QAOA-like families;
- 2–8 qubits;
- 10 controlled distortion specifications;
- 13,440 Phase 7 clean/distorted samples;
- X/Y/Z measurement settings;
- three independent Phase 14 seeds;
- five independently interpretable noisy-Aer profiles for Phase 15.5 evaluation.

This is a serious pilot configuration, not yet a final publication campaign.

## Pod sizing

TriQTO has two different resource bottlenecks:

1. **Phase 7/8/9/11/12 data construction is CPU, RAM, and storage heavy.** Qiskit Aer is CPU-first in the supported Phase 15.6 profile.
2. **Phase 14 neural training benefits from CUDA.** The current model is not large enough to justify a multi-GPU setup for the pilot.

Recommended combined pod for the included 13,440-sample pilot:

| Resource | Recommended |
| --- | --- |
| CPU | 24 vCPU |
| System RAM | 96 GB |
| GPU | One CUDA GPU with 24 GB VRAM |
| Persistent NVMe | 500 GB |
| Python | 3.11.x |
| OS | Linux |
| GPU driver | Compatible with the selected PyTorch CUDA wheel |

A smaller exploratory run can use roughly 8–16 vCPU, 32–64 GB RAM, 12–16 GB VRAM, and 150–250 GB disk. Expect less headroom and more risk of storage or memory pressure.

For a later campaign above roughly 100,000 Phase 7 samples or with materially larger action/topology settings, plan for 48–64 vCPU, 192–256 GB RAM, a 48 GB GPU, and 1–2 TB of persistent NVMe. These are planning recommendations, not measured runtime guarantees.

A GPU is not required to generate the data. Renting a GPU during the entire CPU-heavy data stage may waste money. A cost-efficient approach is:

1. use a high-CPU pod for the data stage;
2. keep the campaign workspace on persistent storage;
3. attach that storage to a GPU pod for training and evaluation.

## Install on a pod

Clone the repository on a Python 3.11 pod, then run:

```bash
bash docker/phase15_6/bootstrap_pod.sh
```

The bootstrap installs the pinned CPU-safe repository environment first and, by default, replaces CPU Torch with the CUDA 12.8 Torch 2.8 wheel. It intentionally keeps Qiskit Aer on the repository-supported CPU path.

For a CPU-only environment:

```bash
TRIQTO_TORCH_PROFILE=cpu bash docker/phase15_6/bootstrap_pod.sh
```

A container build is also provided:

```bash
docker build -f docker/phase15_6/Dockerfile -t triqto-phase15-6 .
```

## Prepare and inspect the campaign

Use a persistent path outside the Git checkout:

```bash
export TRIQTO_WORKSPACE=/workspace/triqto-data/phase15_6_pilot_v1

python scripts/run_phase15_6_campaign.py prepare \
  --config configs/experiments/phase15_6_research_pilot.json \
  --workspace "$TRIQTO_WORKSPACE"

python scripts/run_phase15_6_campaign.py preflight \
  --workspace "$TRIQTO_WORKSPACE"
```

Preparation does not generate data or train a model. It snapshots and hashes the selected configuration files, writes `campaign_plan.json`, records the exact sample count, and produces a resource recommendation.

## Execute the expensive stages

Run data construction once:

```bash
python scripts/run_phase15_6_campaign.py data \
  --workspace "$TRIQTO_WORKSPACE"
```

Then train one seed at a time:

```bash
python scripts/run_phase15_6_campaign.py train \
  --workspace "$TRIQTO_WORKSPACE" \
  --seed 2026

python scripts/run_phase15_6_campaign.py train \
  --workspace "$TRIQTO_WORKSPACE" \
  --seed 2027

python scripts/run_phase15_6_campaign.py train \
  --workspace "$TRIQTO_WORKSPACE" \
  --seed 2028
```

Evaluate each trained seed:

```bash
python scripts/run_phase15_6_campaign.py evaluate \
  --workspace "$TRIQTO_WORKSPACE" \
  --seed 2026
```

Omitting `--seed` runs every configured seed that is not already complete.

Aggregate after all seed reports exist:

```bash
python scripts/run_phase15_6_campaign.py aggregate \
  --workspace "$TRIQTO_WORKSPACE"
```

An `all` command exists, but separate stages are safer for a first cloud run because they make resource changes and failure diagnosis easier.

## Workspace layout

```text
campaign_plan.json
campaign_state.json
source_config_snapshots/
data/
  phase7/
  phase8/
  phase9/
  phase11/
  phase12/
  phase15_6_data_complete.json
runs/
  seed-2026/
    phase14/
    phase15_5/
    phase15_6_seed_complete.json
aggregate/
  cross_seed_summary.json
```

Completed stages are reused. Two writers are blocked from using the same workspace simultaneously. A partially existing output without its completion marker fails closed rather than being silently treated as valid.

The Phase 14 engine itself supports exact checkpoint resume. The Phase 15.6 v1 campaign wrapper deliberately does not guess which interrupted staging checkpoint should be adopted; interrupted partial stages require explicit inspection before recovery.

## Scientific interpretation

The cross-seed aggregate reports:

- trained-policy utility and regret across seeds;
- improvement over random, no-op, and family-heuristic controls;
- variation across independent training seeds;
- positive-improvement fractions;
- diagnostic success-gate booleans.

Those gates are decision aids, not publication claims. Research-quality evidence still requires reviewing split coverage, failure cases, effect sizes, computational cost, held-out axes, and whether the result survives campaign expansion.

Phase 15.6 does not establish:

- physical-hardware transfer;
- broad out-of-distribution generalization;
- calibrated uncertainty;
- universal quantum error correction;
- fault tolerance;
- topology benefit or causal topology;
- quantum advantage.

Topology remains diagnostic and its training coefficient stays exactly zero.
