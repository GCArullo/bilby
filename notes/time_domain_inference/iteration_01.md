# Iteration 01

## Findings

- bilby already has time-domain data access and waveform-generator support, but
  no time-domain GW likelihood.
- The best pyRing reference is `../pyring_test`.
- pyRing's reusable core is small:
  - scalar Gaussian / Student-t / hyperbolic formulas from the quadratic form
  - time-band cache construction
  - covariance backends based on Toeplitz ACFs
- bilby_pipe can already import a custom likelihood by dotted path, but it only
  forwards explicit constructor kwargs.

## Implemented

- Added `bilby/gw/likelihood/time_domain.py` with:
  - `TimeDomainGravitationalWaveTransient`
  - `MixedTimeDomainGravitationalWaveTransient`
  - `StudentTTimeDomainGravitationalWaveTransient`
  - `HyperbolicTimeDomainGravitationalWaveTransient`
  - all fixed-covariance pyRing TD backends except `onsource_ACF`:
    `direct-inversion`, `cholesky-solve-triangular`,
    `toeplitz-inversion`, and `gohberg-semencul`
- Exported the new likelihoods from `bilby/gw/likelihood/__init__.py`.
- Added focused tests in `test/gw/time_domain_likelihood_test.py`.
- Added `examples/gw_examples/injection_examples/time_domain_cbc_example.py`.

## Design Decisions Taken

1. Reused bilby's existing waveform-generator and detector-response plumbing
   instead of porting pyRing front to back.
   This kept the change set smaller and preserved bilby's sampler/result
   integration.

2. Shared one covariance-cache implementation across Gaussian, Student-t, and
   hyperbolic likelihoods.
   This avoided duplicating the expensive and error-prone Toeplitz machinery.

3. Supported detector-dependent and time-banded non-Gaussian parameters in the
   constructor and parameter naming, because that is where pyRing's extra
   flexibility matters scientifically.

4. Added a mixed-family class instead of overloading the Gaussian / Student-t /
   hyperbolic classes.
   This kept the existing API stable while matching pyRing's detector-level
   likelihood-family routing.

5. Kept time, phase, distance, and calibration marginalizations unsupported.
   They would need separate derivations in the time-domain likelihood and were
   outside the minimal target.

## Implementation Target

First pass:

- add a new `bilby.gw.likelihood.time_domain` module
- provide Gaussian / Student-t / hyperbolic classes
- support detector-dependent non-Gaussian parameters
- support pyRing-style time bands if the port stays compact
- keep the initial end-to-end example small enough to run locally

## Risks

- Mapping bilby PSD objects to a finite positive-definite time-domain covariance
  needs care.
- A naive full-covariance implementation can be slow for long segments.
- A native CBC time-domain waveform path may require either GWSignal or a new
  LAL wrapper if the existing frequency-domain source route is not sufficient.

## Verification Runbook

- Unit tests:
  `python -m pytest test/gw/time_domain_likelihood_test.py -q`
- Gaussian CBC example:
  `python examples/gw_examples/injection_examples/time_domain_cbc_example.py`
- Realistic CBC sampler smokes also completed for:
  - Student-t likelihood
  - Hyperbolic likelihood
- A realistic CBC sampler smoke also completed for the Gaussian TD likelihood
  with `likelihood_method='gohberg-semencul'`.
- A realistic CBC sampler smoke also completed for a mixed detector network:
  Gaussian in `H1` and hyperbolic in `L1`.
- Realistic CBC direct likelihood evaluations completed for detector-dependent:
  - Student-t `nu_H1_*`, `nu_L1_*`
  - Hyperbolic `alpha_*`, `delta_*`
- Gohberg-Semencul inverse-vector products were checked directly against
  `scipy.linalg.solve_toeplitz`.
- Gaussian and banded Student-t likelihood values with
  `gohberg-semencul` were checked against `toeplitz-inversion`.
- Mixed detector-family routing was checked against a manual per-detector sum.

## Outcome

- The Gaussian likelihood runs a realistic CBC example end-to-end in the time
  domain using a native time-domain waveform generator path.
- Student-t and hyperbolic likelihoods both work with bilby's waveform,
  detector, and sampler stack.
- Mixed detector families now work in one likelihood object.
- Detector-dependent and detector-independent non-Gaussian parameters both work.

## Next Steps

- If a production-ready example is needed rather than a local smoke example,
  increase sampler settings and widen the astrophysical parameter set.
- If stricter "all operations in the time domain" semantics are required, add a
  dedicated time-domain detector-response path instead of reusing bilby's
  standard response machinery.
- If pyRing feature parity is pushed further, the next missing TD item is
  `onsource_ACF`.
