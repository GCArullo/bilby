# Distance investigation: sampler changes awaiting review

On14 November2026 Gregorio subsequently authorized the representative per-issue
campaign described in `../sampler_investigation/README.md`. Those explicitly
listed experiments are permitted in addition to DAG13633524. Production
adoption and merging into the paper branch remain blocked pending upstream
developer review and Gregorio's authorization. The earlier restriction below
continues to apply to other launches.

The `sg_cbc_distance_marginalization` branch contains the distance-likelihood
work and the experimental sampler changes created for that investigation:
the two code fixes from 88f321fd (zero-acceptance seed retention and donor
exclusion), the calibration-reflection launcher change from 353bf6ad, and
their regression tests. Gregorio requested this branch location on
14 November 2026. The separate `codex/sampler-fixes-upstream-review` ref
remains an archival snapshot of the same tested sampler code.

Keep these changes out of `sine_gaussians_addition` and normal paper runs
until Bilby upstream developers have checked them and Gregorio authorizes
their reintroduction. No upstream review or approval is claimed.

The sole currently authorized changed-sampler event analysis is the explicit
standard-prior NRSur7dq4 + one coherent SG impact test, CIT DAG 13633524.
Do not launch additional analyses from this experimental branch without
new authorization. Its launcher code includes experimental reflection
settings. Normal submissions use the restored paper branch and paper images.
The distance-branch image registry remains at the pre-sampler image; moving
source code here does not authorize activating the experimental runtime.

The retained impact test compares the combined sampler configuration,
including its longer walk limit; it does not isolate the two code fixes.
Their role in the historical event evidence discrepancy remains unresolved.
