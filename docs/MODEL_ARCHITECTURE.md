# Future TriQTO Model Architecture

## Input streams
- Circuit graph encoder
- Parameter encoder
- Phasor encoder
- Optional Hilbert encoder
- Born encoder
- Backend encoder
- Topology encoder

## Fusion
- Dual-mode tri-manifold fusion
- Mask-aware fusion
- Topology fusion

## Core interaction
TriQTO will use phase-coupled lattice interaction and graph/lattice message passing. Transformer Q/K/V imitation is not the central mechanism.

## Output heads
- Distortion diagnosis head
- Action ranking head
- Born prediction head
- Hilbert deformation head
- Uncertainty head
- Topology head
