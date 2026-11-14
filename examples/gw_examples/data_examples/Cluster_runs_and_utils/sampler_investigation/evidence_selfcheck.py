"""Exercise the estimator audit on one deterministic completed trajectory."""
import argparse

import numpy as np

from evidence_postprocess import compare_chain, load_upstream, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-utils', required=True)
    args = parser.parse_args()
    nlive, niter = 40, 240
    count = nlive + niter
    live = np.minimum(np.arange(count, 0, -1), nlive)
    logvol = np.cumsum(np.log(live / (live + 1)))
    logl = np.linspace(-12, 0, count)
    wt, z, var, info = old.compute_integrals(logl=logl, logvol=logvol)
    samples = np.column_stack([np.linspace(10, 10000, count), np.exp(np.linspace(-55, -48, count))])
    result = old.Results(dict(logl=logl, logvol=logvol, logwt=wt, logz=z, logzerr=np.sqrt(var),
                              information=info, samples=samples, samples_u=samples / samples[-1],
                              samples_id=np.arange(count) % nlive, nlive=nlive, niter=niter,
                              ncall=np.ones(count, dtype=int), samples_it=np.arange(count),
                              blob=np.full(count, None), eff=100.))
    new = load_upstream(args.upstream_utils)
    report = compare_chain(result, ['luminosity_distance', 'sine_gaussian_0_hrss'], new, 121, 8)
    assert abs(report['delta_logz']) < 1e-12
    before, after = [report['versions'][version] for version in ['3.1.0', '3.2.0']]
    assert abs(before['known_support_shift']['result']['residual'] - np.log(2)) < 1e-12
    assert abs(after['known_support_shift']['result']['residual']) < 1e-12
    assert before['jitter']['result']['max_information_bookkeeping_residual'] > 1e-5
    assert after['jitter']['result']['max_information_bookkeeping_residual'] < 1e-12
    assert abs(before['reweight']['result']['information_bookkeeping_residual']) > 1e-5
    assert abs(after['reweight']['result']['information_bookkeeping_residual']) < 1e-12
    assert max(abs(report['quantile_differences']['luminosity_distance'])) > 0
    assert before['quantile_integer_probe']['status'] == 'api_failure'
    assert after['quantile_integer_probe']['status'] == 'completed'
    assert after['quantile_integer_probe']['result']['max_residual'] < 1e-9
    merged_old = old.merge_runs([result, result], print_progress=False)
    merged_new = new.merge_runs([new.Results(result.asdict()), new.Results(result.asdict())], print_progress=False)
    assert abs(merged_old.logz[-1] - merged_new.logz[-1]) < 1e-12
    print('PASS: shared weights, corrected information, known support shift, quantiles and merge')


if __name__ == '__main__':
    main()
