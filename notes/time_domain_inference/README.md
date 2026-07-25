# Time-Domain Inference

## Goal

Implement gravitational-wave time-domain likelihoods in bilby for:

- Gaussian noise
- Student-t noise
- Hyperbolic noise
- detector-independent and detector-dependent non-Gaussian parameters

Use pyRing as the scientific reference when possible, but fit the implementation
to bilby's waveform, detector, sampler, and result interfaces.

## Active Plan

- [x] Inspect existing markdown plans and current GW likelihood code.
- [x] Inspect pyRing time-domain likelihood reference code.
- [x] Implement a shared bilby time-domain likelihood core.
- [x] Implement Gaussian time-domain likelihood.
- [x] Implement Student-t time-domain likelihood.
- [x] Implement Hyperbolic time-domain likelihood.
- [x] Add focused unit tests for formulas, parameter handling, and detector options.
- [x] Add a realistic CBC example that runs end-to-end with the new likelihood.
- [x] Run focused verification and record remaining risks.

## Current Design Choice

Use pyRing's covariance-based time-domain likelihood structure:

- residuals are built in bilby per interferometer
- detector noise is described by a Toeplitz covariance from an ACF
- ACF-derived caches provide the quadratic form and log-determinant inputs
- Gaussian, Student-t, and hyperbolic log densities are then evaluated from
  the same quadratic form

Implementation details:

- the new likelihoods live in `bilby.gw.likelihood.time_domain`
- Gaussian, Student-t, and hyperbolic likelihoods share the same covariance
  cache and residual evaluation path
- mixed detector families are supported through
  `MixedTimeDomainGravitationalWaveTransient`
- Student-t and hyperbolic classes support:
  - detector-independent parameters
  - detector-dependent parameters
  - multiple time bands from either an integer or pyRing-style cut times
- the supported covariance backends now match pyRing except for `onsource_ACF`:
  - `direct-inversion`
  - `cholesky-solve-triangular`
  - `toeplitz-inversion`
  - `gohberg-semencul`
- native time-domain waveform generation is supported through
  `prefer_time_domain_waveform=True`
- detector responses still reuse bilby's standard antenna-pattern and delay
  machinery before the residual is evaluated in the time domain

## Alternatives Considered

1. Reuse bilby's frequency-domain likelihood and only accept time-domain source
   models.
   Rejected: this is not a genuine time-domain likelihood.

2. Whiten in the frequency domain and evaluate iid time-domain samples only.
   Simpler, but weaker as a pyRing port and less explicit about the covariance
   model.

3. Port pyRing's full likelihood stack.
   Rejected: too much pyRing-specific plumbing is unrelated to bilby.

## Constraints

- Keep the implementation minimal.
- Prefer explicit constructor arguments so bilby_pipe can pass them through.
- Do not edit the protected LVK config file.
- Avoid touching unrelated untracked research artefacts in the worktree.

## Verification Summary

- Focused unit tests pass:
  `python -m pytest test/gw/time_domain_likelihood_test.py -q`
- A reduced-parameter CBC example runs end-to-end with the Gaussian
  time-domain likelihood:
  `examples/gw_examples/injection_examples/time_domain_cbc_example.py`
- Additional realistic CBC smoke runs completed with:
  - `StudentTTimeDomainGravitationalWaveTransient`
  - `HyperbolicTimeDomainGravitationalWaveTransient`
- A realistic CBC smoke run also completed with the
  `gohberg-semencul` backend.
- A realistic CBC smoke run also completed with mixed detector families:
  Gaussian in `H1` and hyperbolic in `L1`.
- Detector-dependent Student-t and hyperbolic parameter wiring was checked on a
  realistic CBC setup through direct likelihood evaluations.
- Gohberg-Semencul quadratic forms were checked against
  `solve_toeplitz`, and likelihood values were checked against the
  `toeplitz-inversion` backend.
- Mixed detector-family routing was checked against a manual per-detector sum.

## Remaining Risks

- The covariance is built from a PSD-derived finite ACF using large finite PSD
  patches outside the active band. This suppresses out-of-band contributions,
  but it is still an approximation choice.
- The implementation does not port `onsource_ACF`.
- The native time-domain waveform option still reuses bilby's existing detector
  response machinery rather than adding a separate pure time-domain projection
  stack.
