# Portable replay manifest dependency repair

The original portable archive deliberately contains the selected 36 blind payloads
and their authenticated manifest rows in `archive_manifest.json`. It does not
contain a full product `manifests/example_manifest.csv`. Portable replay passed
exact donor-Jacobian reconstruction but then called the independent post-hoc
extractor, which still tried to open that omitted CSV. This was a packaging/API
dependency error, not a new numerical replay divergence.

`posthoc_x_only_subset` now accepts explicit blind manifest rows. Portable replay
passes the already hash-verified selected rows. The helper rejects missing,
duplicate, or extra row identities, retains target/root checks, and still uses the
original hash-checking x-only NPZ loader. Its default on-volume CSV path remains
unchanged. No archive regeneration, synthetic CSV, model, normalization, pooling,
calibration count, or scientific setting change is involved.

Validation against the original 4,445,681-byte release archive, independently
SHA256-verified after download, succeeded on CPU with sampling forbidden:

- 33/33 reconstructed Jacobian hashes exactly match the donor.
- Both extraction paths agree exactly at every compared stage.
- 0/36 disagreements against frozen predictions.
- Maximum absolute logit error: 2.3182305177726903e-6.
- 27 focused tests pass, including the missing-product-manifest regression and
  rejection of incomplete/mismatched selected-row inventories.

Archive SHA256: `8f179d98a18c695df68e1da10f31f635c8fc8a74a39112feb14ba40275f6cd8b`.
Manifest SHA256: `38a399a16ff67f880dc59d5f495457312c49c944eba16165e1f1548f3bd5f1f3`.
Corrective code commit: `d43063b4da37a0180f0d7a40dd4cb8b8e2b32374`.

## First historical sampling divergence

The independently downloaded native 5090 and 4090 arrays reproduce both original
multinomial draws in a single NumPy 2.3.2 process. The first differing call is
root 40025, ordinal 0. Its conditional-binomial loop reaches category 2 with
identical RNG state and 1,314 remaining shots. The conditional probability is
0.5 for the donor and 0.5000000000000002 for the recipient. This crosses NumPy's
`random_binomial` direct/complement branch (`p <= 0.5`). Counts change from
`[1398, 1384, 631, 683]` to `[1398, 1384, 683, 631]`.

A controlled binary draw from that same RNG state gives `[631,683]` at 0.5 and
`[683,631]` at the next representable number above 0.5. Both original
conditional-binomial reconstructions exactly match NumPy's multinomial output.
The first call's maximum probability difference is only 5.551115123125783e-17.
The numeric fixture and RNG-state hashes are recorded in `SAMPLER_REPRODUCER.json`.

This identifies the discontinuity for the first captured differing call. It does
not trace all other calls or indicate an incorrect sampling distribution. Nearly
identical distributions can map the same seed to different valid realizations.
Fresh-calibration stability is assessed separately by the preregistered pilot.

[NumPy 2.3.2 source: random_multinomial and random_binomial](https://github.com/numpy/numpy/blob/v2.3.2/numpy/random/src/distributions/distributions.c)

[Corrected workflow](https://github.com/kys928/TriQTO/actions/runs/39287947712)
[Durable original archive](https://github.com/kys928/TriQTO/releases/tag/calibration-evidence-20261019)

The corrected workflow subsequently completed successfully: all 96 resampling
draws, durable release publication, independent release-download verification,
and all 704 repository CI tests passed. The downloaded result archive and its
101 dependency hashes were independently verified again. Every resampled count
array's shot totals and all reported label-flip/pairwise metrics were checked
from the retained NPZ outputs. See `ANALYSIS.md` and `PILOT_RESULT_SUMMARY.json`.
