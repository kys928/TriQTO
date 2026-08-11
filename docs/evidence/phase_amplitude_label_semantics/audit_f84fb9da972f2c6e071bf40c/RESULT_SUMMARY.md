# Step 2 label-semantics audit result

**Audit ID:** `audit_f84fb9da972f2c6e071bf40c`  
**Decision:** `CONTEXT_DEPENDENT`  
**Development-only:** YES  
**Classifier trained:** NO  
**Labels changed:** NO  
**Historical v0.1 test accessed:** NO  

## Primary result

The coarse phenotype names are directionally meaningful overall, but mechanism and observed phenomenology cannot be treated as equivalent across all circuit/state contexts.

### `phase_like`

- expected-sign concordance: `1.0000`
- group-bootstrap 95% CI: `[1.0000, 1.0000]`
- median dominance log-ratio: `+19.5601`
- negligible-effect fraction: `0.0143`
- resolved examples: `138 / 140`
- strong-dominance concordance among resolved examples: `1.0000`
- phenotype counts: `138 phase_dominant`, `2 negligible`, `0 mixed`, `0 population_dominant`

Within this development product, the RZ-drift cohort is therefore exceptionally consistent with a relative-phase/interference-dominant final-state effect.

### `amplitude_like`

- expected-sign concordance: `0.8031`
- group-bootstrap 95% CI: `[0.7338, 0.8672]`
- median dominance log-ratio: `-3.3807`
- negligible-effect fraction: `0.0929`
- resolved examples: `127 / 140`
- strong-dominance concordance among resolved examples: `0.7244`
- phenotype counts: `92 population_dominant`, `22 mixed`, `13 phase_dominant`, `13 negligible`

The amplitude-like mechanism cohort is population-dominant overall, but the phenotype is materially context-dependent.

## Raw-mechanism split inside `amplitude_like`

### RX overrotation

- expected-sign concordance: `0.7193`
- strong-dominance concordance: `0.6140`
- negligible fraction: `0.1857`
- median dominance log-ratio: `-1.8494`
- phenotype counts: `35 population_dominant`, `14 mixed`, `8 phase_dominant`, `13 negligible`

RX overrotation is the least semantically stable amplitude-like mechanism in this cohort.

### RY overrotation

- expected-sign concordance: `0.8714`
- strong-dominance concordance: `0.8143`
- negligible fraction: `0.0000`
- median dominance log-ratio: `-5.3146`
- phenotype counts: `57 population_dominant`, `8 mixed`, `5 phase_dominant`, `0 negligible`

RY overrotation is substantially more consistently population-dominant than RX overrotation, but is still not universally so.

## Context strata that failed the frozen stability criterion

Exactly six adequately populated context strata failed the pre-frozen stable criterion:

1. `raw_label = rx_overrotation`: expected-sign concordance `0.7193`.
2. `family = hardware_efficient_ansatz`: amplitude-like concordance `0.5926`, with only `9/27` examples population-dominant; `12/27` were mixed and `6/27` phase-dominant.
3. `family = qaoa_like`: amplitude-like concordance `0.6818`; `12/22` population-dominant and `10/22` mixed.
4. `n_qubits = 6`: amplitude-like concordance `0.7429`.
5. `n_qubits = 8`: amplitude-like concordance `0.7241`.
6. `strength = 0.05`: amplitude-like concordance `0.7969`.

A further descriptive warning is `qft_like`: its resolved amplitude-like examples were fully population-dominant, but `13/24` amplitude-like QFT examples were negligible under the frozen effect-resolution threshold. All 13 negligible QFT amplitude-like cases in this cohort came from RX overrotation; the 11 RY-overrotation QFT cases were population-dominant.

## Additional structural observation

For amplitude-like examples from families marked `phase_sensitive_family = false`, expected-sign concordance was `1.0000` (`30/30`). For amplitude-like examples from phase-sensitive families, concordance fell to `0.7423` (`97` resolved of `110` total, with `0.1182` negligible fraction).

This supports the audit decision that circuit/state context materially changes the observed phenotype of the same injected X/Y-axis mechanism.

## Scientific interpretation

Step 2 does **not** justify rewriting the historical coarse labels after the fact. It establishes a development finding:

- `phase_like` is a highly stable phenotype name for the tested RZ-drift cohort under this generator/product;
- `amplitude_like` is a useful directional shorthand overall, but it is not a physically invariant phenotype name for RX/RY overrotation;
- injected mechanism and observed final-state phenomenology must be represented separately going forward;
- future diagnostic targets should preserve mechanism identity and expose a continuous or mixed population-versus-phase effect coordinate rather than forcing every RX/RY case into a pure amplitude phenotype.

The perfect RZ result must not be generalized beyond the tested distortion-injection protocol until the insertion location and propagation policy are explicitly checked. A terminal RZ perturbation, for example, would preserve computational-basis populations by construction; an in-circuit RZ perturbation followed by non-diagonal gates need not.

## Provenance

- config SHA-256: `sha256:f0bf12c2338ec8cee168ef62d2dd73282e670630e571132eb0f88dc1acbcecd0`
- source manifest SHA-256: `sha256:d5dc1d2786d853982fb909123391818b3209995c72560d76d0ab89425bc3c4d5`
- source generation-complete SHA-256: `sha256:8ec933249923083b1b392ed3bb9f47253994362cf9836b015acb0f58f8ddd034`
- example metrics SHA-256: `sha256:82c73e41b2410b976df9b8d306026707d1c6a85bc42128e47e88519ca18c9be2`
- stratified metrics SHA-256: `sha256:bd906bbe44a4be2f3764fe6d693798b01f106ea6fc093b6366bcabbbf9ce6d4a`
- decision SHA-256: `sha256:cd90fbcb87bc4aaa8255e7e40fec58110fe9e58a98e82d9549da3ceac55afb02`
- maximum decomposition closure error: `2.220446049250313e-16`
