# Sampler changes awaiting upstream developer review

This branch isolates the two sampler code fixes originally committed in
88f321fd (zero-acceptance seed retention and donor exclusion), the calibration
reflection setting from 353bf6ad, and their immutable runtime registration.
Its base is sg_cbc_distance_marginalization, whose validated likelihood fixes
are unchanged here.

Do not merge these sampler changes, enable them in normal paper runs, or make
their image the paper/distance branch default until Bilby upstream developers
have checked them and Gregorio authorizes their reintroduction. No upstream
review or approval is claimed by this branch.

The sole currently authorized changed-sampler event analysis is the explicit
standard-prior NRSur7dq4 + one coherent SG impact test, CIT DAG 13633524. The
main paper and distance branches retain their pre-investigation sampler
behavior. Other affected runs are replaced using paper sampler settings.

The code fixes, reflection change and walk-length changes have not yet been
shown to explain the historical event evidence discrepancy. The controlled
reflection-only evidence test did not reproduce a large discrepancy.
