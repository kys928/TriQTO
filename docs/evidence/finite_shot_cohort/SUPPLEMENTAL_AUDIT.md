# Additional non-selecting post-materialization audit

This expands identity coverage and audit checks only. No data, admissions, split, measurement, labels or remedy settings changed.

## Structural differences across partitions (TEST label-blind)

n_qubits

| partition   |   2 |   3 |   4 |   5 |
|:------------|----:|----:|----:|----:|
| FIT         |  41 |  30 |  22 |  27 |
| SELECTION   |  26 |  12 |   5 |  17 |
| TEST        |  19 |  14 |  15 |  12 |

topology_class

| partition   |   directed_fan |   hub_spoke |   mixed_sparse |   path |   staggered_path |   two_branch |
|:------------|---------------:|------------:|---------------:|-------:|-----------------:|-------------:|
| FIT         |             23 |          26 |             17 |     17 |               21 |           16 |
| SELECTION   |              8 |           9 |              8 |      9 |               10 |           16 |
| TEST        |             15 |           7 |             10 |      9 |                7 |           12 |

reference_layers

| partition   |   4 |   5 |   6 |   7 |   8 |   9 |
|:------------|----:|----:|----:|----:|----:|----:|
| FIT         |   7 |  17 |  20 |  26 |  29 |  21 |
| SELECTION   |   7 |  15 |   9 |   7 |   6 |  16 |
| TEST        |  10 |  14 |  13 |   6 |   8 |   9 |

injection_context_class

| partition   |   inter_entangling |   post_entangling_recombination |   pre_entangling |
|:------------|-------------------:|--------------------------------:|-----------------:|
| FIT         |                 49 |                              53 |               18 |
| SELECTION   |                 26 |                              26 |                8 |
| TEST        |                 20 |                              26 |               14 |

Joint qubit/topology coverage by partition:

|                  |   directed_fan |   hub_spoke |   mixed_sparse |   path |   staggered_path |   two_branch |
|:-----------------|---------------:|------------:|---------------:|-------:|-----------------:|-------------:|
| ('FIT', 2)       |              8 |          10 |              5 |      5 |                7 |            6 |
| ('FIT', 3)       |              3 |           8 |              3 |      5 |                7 |            4 |
| ('FIT', 4)       |              5 |           4 |              6 |      4 |                1 |            2 |
| ('FIT', 5)       |              7 |           4 |              3 |      3 |                6 |            4 |
| ('SELECTION', 2) |              3 |           4 |              3 |      4 |                3 |            9 |
| ('SELECTION', 3) |              1 |           2 |              2 |      0 |                3 |            4 |
| ('SELECTION', 4) |              2 |           0 |              1 |      0 |                2 |            0 |
| ('SELECTION', 5) |              2 |           3 |              2 |      5 |                2 |            3 |
| ('TEST', 2)      |              3 |           0 |              6 |      4 |                2 |            4 |
| ('TEST', 3)      |              1 |           3 |              1 |      2 |                3 |            4 |
| ('TEST', 4)      |              5 |           2 |              2 |      2 |                1 |            3 |
| ('TEST', 5)      |              6 |           2 |              1 |      1 |                1 |            1 |

Zero cells are coverage gaps, not evidence that a circuit is impossible. They are retained rather than filled after observing the sample.

## Diagnostic and content audit

{
  "cohort_identity": "sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
  "statistical_evidence_identity": "sha256:9f889092113d9c65549d0191158b25ed0a845f2cce7769192d449d0564ccb6fa",
  "additional_identity_inventory_sha256": "sha256:a159d850d6744bc76b904d84795c6703b5171bb691fc811438d57cb2c8c2f1c6",
  "additional_source_count": 10,
  "overlaps": {
    "additional_family_signature": 0,
    "additional_graph_sha256": 0,
    "additional_operation_signature": 0,
    "additional_wire_relabelled_signature": 0,
    "original_wire_relabelled_signature_independent": 0
  },
  "development_roots_truth_schedule_checked": 720,
  "independent_development_diagnostic_replays": 156,
  "test_labels_accessed": false,
  "test_array_files_opened": 0,
  "errors": [],
  "status": "PASS_WITH_REMAINING_EXCLUSION_GAPS",
  "remaining_historical_gaps": [
    "inaccessible sealed/future reserves",
    "any historical products not enumerated in this explicit source inventory"
  ],
  "missing_historical_paths": [],
  "global_verified_status_permitted": false
}

Statistical evidence identity binds sorted root IDs, parameterized graph hashes and raw evidence bundles. It excludes wall-clock timing and randomized encryption envelopes; the materialization identity separately binds the full original archive inventory.

Legacy products exposing only graph hashes do not certify parameter-free skeleton or wire-relabel exclusion. The inaccessible reserve and any unenumerated products remain unverified. No global development approval is inferred.

Scope of structural equivalence: exact ordered gate/wire skeletons up to wire relabeling. Arbitrary commutation, algebraic gate simplification, unitary equivalence and measurement-equivalence classes are not certified.

Count-frequency quantiles are development only, stratifiable by qubit count/budget/stratum in the companion CSV. Diagnostic replay checks generation correctness against permitted development labels, not prediction accuracy. No TEST label correctness claim is added.
