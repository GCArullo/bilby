# Hyperbolic Likelihood Audit

## Goal

Compare the current branch against `sine_gaussians_addition`, focusing on the
hyperbolic likelihood implementation. Check the implementation against
`asasli/HyperWave`, report mismatches, and fix mistakes in this branch.

## Tasks

- [x] Locate existing markdown plans.
- [x] Identify branch changes relative to `sine_gaussians_addition`.
- [x] Inspect local hyperbolic likelihood code and tests.
- [x] Inspect the HyperWave reference implementation.
- [x] Compare equations and data flow across implementations.
- [x] Fix confirmed mistakes in this branch.
- [x] Run focused verification.
- [x] Summarize remaining risks and next steps.

## Comparison

- Only `bilby/gw/likelihood/hyperbolic.py` and `test/gw/likelihood_test.py`
  differ from `sine_gaussians_addition` in the tracked branch comparison.
- HyperWave computes the residual statistic as
  `4 * df * sum_ifo(|d_ifo - h_ifo|**2 / S_ifo)`. This matches the branch's
  `|r|**2 / scale2` with `scale2 = S * duration / 4`.
- For fixed network dimension `d`, HyperWave uses
  `lambda = (d + 1) / 2` and the segment term
  `lambda * log(alpha / delta) + ((1 - d) / 2) * log(2*pi)
  - log(2*alpha) - log(K_lambda(alpha*delta))
  - alpha * sqrt(delta**2 + yy)`.
- The branch is more general than HyperWave for detector masks: it forms a union
  frequency grid and lets the effective dimension vary by frequency bin.
- HyperWave does not include an explicit PSD/Jacobian determinant term in the
  hyperbolic likelihood. The branch keeps the determinant term so future PSD
  inference has the correct normalization.

## Fixes

- Corrected the hyperbolic normalisation from `log(delta / alpha)` to
  `log(alpha / delta)`.
- Restored the residual-to-whitened-residual Jacobian term
  `-sum_active_detectors(log(scale2))` in each network frequency bin.
- Updated the direct test oracle to use the HyperWave sign.
- Replaced internal Hyperbolic likelihood uses of the deprecated public
  `parameters` property with `_parameters`.
- Made the auxiliary Hyperbolic noise-only likelihood initialize defaults through
  internal state so it works when `PARAMETERS_AS_STATE` is disabled.

## Verification

- `python -m pytest test/gw/likelihood_test.py::TestHyperbolicGWTransient -q`
- Direct comparison against
  `/private/tmp/HyperWave_codex_hyp_audit_20260625/src/hyperwave/likelihoods/distributions_fd.py`
  for the whitened hyperbolic kernel gave matching values:
  `local=-164.025065434277 hyperwave=-164.025065434277`.
- `BILBY_ALLOW_PARAMETERS_AS_STATE=FALSE` construction/evaluation and mocked
  nested noise-evidence calculation pass.

## Remaining Differences

- HyperWave assumes a common frequency grid and fixed dimension
  `d = 2 * n_ifo`; the branch supports varying detector masks.
- HyperWave exposes `hyperbolic_classic` for direct `alpha, delta` parameters
  and `hyperbolic` for `alpha, ratio` where `delta = alpha * ratio`; the branch
  implements the direct `alpha, delta` parameterization.
- The network hyperbolic density couples detectors in a single residual vector,
  so there is no unique additive per-detector decomposition of the network log
  likelihood. The branch's per-detector output is a single-detector diagnostic.
- Existing tests still emit deprecation warnings when they deliberately use
  stored parameters. That is outside this hyperbolic-equation fix.
