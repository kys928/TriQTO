# TriQTO capability matrix

TriQTO currently provides an offline, deterministic research scaffold. It must not be described as hardware-validated, OOD-generalizing, uncertainty-calibrated, or topology-validated.

| Capability | Status | Notes |
| --- | --- | --- |
| Basis-conditioned ideal `p(y \| M)` | Implemented, offline-tested | Explicit Pauli-product `X/Y/Z` settings; exact probabilities are simulator privilege. |
| Identifiability masking and reporting | Implemented, offline-tested | Unidentifiable diagnosis/action labels are masked by default; strict rejection and explicit audited override are supported. |
| Observable readout bit-flip channel | Implemented, offline-tested | Exact independent symmetric classical readout channel; not a noisy circuit simulator. |
| Sampled ideal shots conditioned on `M` | Implemented, offline-tested | Offline only; setting-specific provenance is mandatory. |
| Noisy Aer shots / density simulation | Intentionally unsupported in active configs | Future work; configs must be marked unsupported until implemented and tested. |
| Fake-backend / transpilation evidence | Intentionally unsupported in active configs | Future work; no fabricated backend features. |
| IBM Runtime ingestion | Credential-gated placeholder only | Not run by default; no credentials in tests. |
| Topology loss | Diagnostic/audit only | `topology_loss_weight` remains `0.0` by default. |
| Phase 15 OOD/generalization claims | Unsupported | IID results must not be labeled as OOD generalization. |
| Per-example calibrated uncertainty | Unsupported | Existing uncertainty outputs are not validated calibration evidence. |

## Dependency matrix

Supported default environments: Python 3.11 and 3.12 on CPU with direct dependencies pinned by `requirements-cpu.txt` and `constraints/cpu.txt`. Transitive packages are not hash-locked, so this is not described as byte-for-byte environment reproduction.

Install default CPU profile:

```bash
python -m pip install pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
python -m pip install -r requirements.txt -c constraints/cpu.txt
python -m pip install -e .
python scripts/verify_dependency_pins.py --check-installed
```

The optional CUDA profile is independent from the CPU profile so `qiskit-aer` and `qiskit-aer-gpu` cannot collide. Default CI checks that separation statically; complete GPU resolution and runtime behavior remain unvalidated until a CUDA runner is added.

## Configuration boundary

Repository capability YAMLs are planning/claim-boundary documents. Unsupported YAMLs are rejected by the generic loader unless explicitly opened for planning inspection. Executable scientific phases continue to use their strict typed JSON/YAML loaders and real registries; the generic capability loader does not replace those schemas.

The legacy executable distortion name `readout_bitflip_marker` is no longer registered. Use `readout_bitflip`; layout markers remain audit-only and unidentifiable until real backend/layout evidence exists. See [`MEASUREMENT_IDENTIFIABILITY.md`](MEASUREMENT_IDENTIFIABILITY.md).
