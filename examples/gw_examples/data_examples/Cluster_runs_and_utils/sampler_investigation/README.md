# Representative event sampler investigation

The user authorized this diagnostic campaign on 14 November 2026. Use the
canonical `sg_cbc_distance_marginalization` branch. The paper branch and normal
production settings remain unchanged; these investigations do not constitute
upstream approval of the experimental sampler changes.

All full-event cases share the real NRSur7dq4 + coherent one-SG likelihood,
original source-frame distance prior, immutable `88f321fd` image and Dynesty
3.1.0. `sampling_variants.py` makes source-hash-guarded process-local changes;
the installed package is never overwritten. Resume through the same worker so
the case's effective source is restored before loading its checkpoint.

| Case | Difference from the common paper-behavior control |
| --- | --- |
| paper | None: original fallback, original donor pool, reflection, maxmcmc5000 |
| seed_only | Retain seed after a zero-acceptance walk |
| donor_only | Exclude the walk seed from the donor pool |
| reflection_only | Disable reflection of the40 calibration coordinates |
| walkcap_only | Increase maxmcmc to50000 |

Each full-event case uses2500 live points, naccept60, eight workers and three
independent seeds. Effective source, behavior checks, target hashes, sampled
coordinates and three likelihood replays are saved before sampling. Native
evidence explicitly uses the likelihood ratio (lnB); the result retains lnZ
and its noise offset separately. Passing preflight is not convergence evidence.

`sampling_periodic.py` varies one actual event angle at a time, holding the
other parameters fixed. Independent refined quadrature supplies the conditional
reference. Default and seam-centered coordinates compare Bilby's wrapped walk
and native Dynesty3.1/3.2 uniform/slice algorithms. The3.2 wheel is hash-pinned
and loaded under an isolated namespace. Invalid native unit coordinates are
reported as failures rather than silently repaired. These are conditional
tests, not full-event posterior or evidence measurements.

`evidence_postprocess.py` compares old/new evidence-error calculations and
post-processing on identical completed native event trajectories. It requires
an explicit lnZ/lnB convention and checks the matching HDF5 result. It never
rewrites original results. `evidence_support.py` measures constraint support,
the wrapper's normalization behavior and a bounded waveform-validity sample;
it does not apply an inferred evidence correction. `evidence_selfcheck.py`
checks the comparison machinery against a deterministic completed trajectory.

The campaign manifest and workbook record running, completed, failed and
not-applicable cases separately. Missing results remain pending. Dynamic-only
and unused-sampler issues must not be described as validated by these tests.
