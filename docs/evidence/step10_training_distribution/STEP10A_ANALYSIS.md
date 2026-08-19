# Step 10A leakage-safe training-mixture result

Status: **FROZEN POST OUTCOME**

## Decision

`STEP10A_DATASET_CONTRACT_PASSES__STEP10B_TRAINING_UNLOCKED`

The generated Step-10A product is `product_0f7112597501f7ea5fbe123b`. It references the frozen Step-5 v3 product `product_b2d78ad2309b71a55f9bb54f` unchanged and adds a 2,400-root / 31,200-example deployment-domain bridge.

## Leakage and identity audit

The uploaded archive was independently checked rather than relying only on the generator summary.

- 300 parent groups were present, each with exactly 8 nearby circuit variants.
- Every parent group's 8 variants occupy exactly one of fit, selection, or outer validation; parent-group leakage violations: **0**.
- All 2,400 clean bridge roots contain exactly 13 derivatives and all derivatives remain in the same partition as the root; root-partition violations: **0**.
- Bridge partition counts are 1,440 fit, 480 selection, and 480 outer-validation roots.
- Each of the four bridge motif families contributes 360/120/120 roots to fit/selection/outer-validation respectively.
- Qubit-count balance is exact: 800 roots each at 2q, 3q, and 4q.
- Duplicate bridge graph hashes: **0**.
- The exact frozen Step-9D pilot graph is absent.
- All 31,200 manifest-referenced NPZ artifacts were found and SHA-256 verified; missing artifacts: **0**, hash mismatches: **0**.
- Mechanism examples are balanced: 9,600 each for RZ/RX/RY plus 2,400 clean controls.
- Acquisition counts are balanced across 512/1024/2048/4096 shots at 7,800 examples each.

## Frozen hashes

- uploaded `step10_training_mixture.zip`: `dc1fdea3d0f80c7ae0880417b3b9bd13efafdade9dce05a9369e3f5fb69d711d`
- `step10a_generation.log`: `15f9e0ef8bf9f1904f2aa716a1ddb214f6c95e26705d66ba8465b649bda2826f`
- `dataset_complete.json`: `fc4b529677c5165c96ac191aee33622efdbba4fade3867795f9f84778e1ca3c4`
- `stage_validation.json`: `c1437d130b048672856475646fbb1fe9dd831af4e28c54838805b1326b5c5624`
- `original_domain_reference.json`: `6938dca1cfd8479c5ea7b001f74af3d15d5803b07cc20cf97070acbdff989db9`
- `bridge_root_manifest.csv`: `96bd358640a77a4cea328588b11d3e8567e4fd8ce7b42d9e91d1575dad2c50bc`
- `bridge_example_manifest.csv`: `d5bfef900fb0264fa0b41213788be8bf7610080b1f313d27a8954eb945a5467a`

## Interpretation

Step 10A validates the data/split contract only. It does **not** show that the neural model will learn the bridge without forgetting Step 5, nor does it show that warm-start will beat scratch. Those are Step-10B questions.

The dataset is suitable for the already-frozen Step-10B benchmark because neighboring bridge variants cannot leak across model-selection boundaries, the original domain remains separately identifiable, and the bridge family is substantially broader than the three frozen Step-9D QPU pilot cases.

Step 10B may now run without changing the architecture, data contract, retention threshold, bridge success threshold, training seeds, optimizer class, or warm-start/scratch comparison after seeing Step-10A outcome.
