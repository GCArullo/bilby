"""Measure constraint support and bounded waveform-validity sampling of the event.

No evidence correction is applied. Bilby's externally supplied initial live
points make this a wrapper-normalization audit, not a direct demonstration of
the upstream Dynesty initialization-volume bug.
"""
import argparse
import inspect
import json
from pathlib import Path

import bilby
import numpy as np
from scipy.stats import beta

from sampling_event import load_target, sha, write


def interval(successes, total):
    return [float(beta.ppf(.025, successes, total - successes + 1)) if successes else 0.,
            float(beta.ppf(.975, successes + 1, total - successes)) if successes < total else 1.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=261114701)
    parser.add_argument('--prior-draws', type=int, default=65536)
    parser.add_argument('--likelihood-draws', type=int, default=256)
    args = parser.parse_args()
    assert args.prior_draws >= args.likelihood_draws > 0
    inputs, likelihood, priors, provenance = load_target(args.settings, args.outdir, args.seed)
    assert not likelihood.distance_marginalization
    keys = [key for key, prior in priors.items() if not getattr(prior, 'is_fixed', True)
            and not isinstance(prior, bilby.core.prior.Constraint)]
    assert len(keys) == 60 and 'luminosity_distance' in keys and 'sine_gaussian_0_hrss' in keys
    fixed = {key: float(prior.peak) for key, prior in priors.items() if getattr(prior, 'is_fixed', False)}
    baseline = provenance['target']['baseline']['parameters']
    point = {key: baseline[key] for key in keys}
    point.update(fixed)
    # Derived columns must not mask conversion errors when physical inputs vary.
    baseline_loglr = float(likelihood.log_likelihood_ratio(point.copy()))
    np.testing.assert_allclose(baseline_loglr, provenance['baseline_log_likelihood_ratio'], atol=1e-8, rtol=0)
    rng = np.random.default_rng(args.seed)
    cube = rng.uniform(size=(len(keys), args.prior_draws))
    values = np.asarray(priors.rescale(keys, cube))
    draws = dict(zip(keys, values))
    draws.update({key: np.full(args.prior_draws, value) for key, value in fixed.items()})
    valid = np.asarray(priors.evaluate_constraints(draws), dtype=bool)
    if valid.ndim == 0:
        valid = np.full(args.prior_draws, valid)
    assert valid.shape == (args.prior_draws,)
    count = int(np.count_nonzero(valid))
    assert count >= args.likelihood_draws, 'Insufficient constraint-valid draws; increase declared prior sample'
    fraction = count / args.prior_draws
    constraints = {key: dict(minimum=prior.minimum, maximum=prior.maximum)
                   for key, prior in priors.items() if isinstance(prior, bilby.core.prior.Constraint)}
    sampler = bilby.core.sampler.get_sampler_class('dynesty')
    initialization_source = inspect.getsource(sampler.run_sampler)
    prior_initialization_source = inspect.getsource(sampler.get_initial_points_from_prior)
    report = dict(status='running', provenance=provenance, worker_sha256=sha(__file__),
                  keys=keys, fixed_keys=list(fixed), constraints=constraints,
                  baseline_filtered_log_likelihood_ratio=baseline_loglr,
                  prior_draws=args.prior_draws, constraint_valid=count, constraint_fraction=fraction,
                  constraint_fraction_95_interval=interval(count, args.prior_draws),
                  log_constraint_fraction=float(np.log(fraction)),
                  actual_initialization=dict(wrapper_supplies_live_points='get_initial_points_from_prior' in initialization_source,
                                             rejection_checks_finite='check_draw' in prior_initialization_source,
                                             sampler_source_sha256=sha(inspect.getsourcefile(sampler)),
                                             prior_source_sha256=sha(inspect.getsourcefile(type(priors))),
                                             initial_livepoint_function=prior_initialization_source),
                  waveform_scan=[], applied_evidence_correction=False)
    # Use the actual runtime's normalization API and inspect its numerical effect
    # on a valid point. Preserve both the API output and independent support count.
    bilby.core.utils.random.seed(args.seed + 1)
    try:
        factor = float(priors.normalize_constraint_factor(tuple(keys), nrepeats=4))
        normalized = float(priors.ln_prob({key: point[key] for key in keys}, normalized=True))
        unnormalized = float(priors.ln_prob({key: point[key] for key in keys}, normalized=False))
        report['prior_normalization_api'] = dict(status='completed', factor=factor,
            independent_inverse_fraction=1 / fraction, log_factor=float(np.log(factor)),
            actual_log_prior_difference=normalized - unnormalized,
            difference_from_log_factor=float(normalized - unnormalized - np.log(factor)))
    except Exception as error:
        report['prior_normalization_api'] = dict(status='api_failure', error_type=type(error).__name__, error=str(error))
    path = args.outdir / 'support_summary.json'
    write(path, report)
    selected = np.flatnonzero(valid)[:args.likelihood_draws]
    for offset, index in enumerate(selected):
        parameters = {key: float(value[index]) for key, value in draws.items()}
        try:
            value = float(likelihood.log_likelihood_ratio(parameters.copy()))
            usable = np.isfinite(value) and value > -np.finfo(float).max
            record = dict(index=int(index), finite=bool(usable),
                          log_likelihood_ratio=value if np.isfinite(value) else str(value))
        except Exception as error:
            record = dict(index=int(index), finite=False, error_type=type(error).__name__, error=str(error))
        if not record['finite']:
            record['sampled_parameters'] = {key: parameters[key] for key in keys}
        report['waveform_scan'].append(record)
        if offset % 16 == 0:
            write(path, report)
    finite = sum(record['finite'] for record in report['waveform_scan'])
    report.update(status='completed', likelihood_draws=args.likelihood_draws,
                  finite_likelihood_count=finite,
                  finite_given_constraints_95_interval=interval(finite, args.likelihood_draws),
                  interpretation='Constraint support and finite waveform coverage only; no sampled evidence was corrected or certified',
                  normalization_caveat='An intentionally normalized constrained prior is not the same target as an unconstrained prior with zero likelihood outside support. Determine the declared target before applying log-fraction corrections.',
                  waveform_caveat='A bounded random scan cannot exclude rare waveform-invalid prior regions; errors and confidence limits are retained')
    write(path, report)
    print(json.dumps(dict(constraint_fraction=fraction, finite_likelihood=f'{finite}/{args.likelihood_draws}',
                          prior_normalization=report['prior_normalization_api'])), flush=True)


if __name__ == '__main__':
    main()
