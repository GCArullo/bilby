"""Compare upstream evidence utilities on unchanged completed event trajectories.

Run inside the original Dynesty 3.1 image. The supplied, hash-checked upstream
3.2 utils.py is loaded in an isolated namespace; no installed module is replaced.
The output is diagnostic and never overwrites an event result.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import dill
import dynesty
from dynesty import utils as old
import h5py
import numpy as np
from scipy.special import logsumexp

UPSTREAM_SHA256 = 'a5f8ba4949f62376c1f289162466ead5cbf016645f11010d23b4b158cf75736f'
PARAMETERS = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn',
              'sine_gaussian_0_hrss', 'sine_gaussian_0_Q',
              'sine_gaussian_0_frequency', 'sine_gaussian_0_phase_offset']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_upstream(path):
    assert sha(path) == UPSTREAM_SHA256, 'Unexpected upstream utils.py bytes'
    name = 'dynesty_investigation_3_2_0'
    package = types.ModuleType(name)
    package.__version__ = '3.2.0'
    package.__path__ = [str(Path(path).parent)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.utils', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert sys.modules['dynesty.utils'] is old
    return module


def convert(module, result, **changes):
    data = {key: value for key, value in result.items()
            if key in module.Results._ALLOWED}
    data.update(changes)
    return module.Results(data)


def totals(result):
    return dict(logz=float(result.logz[-1]), logzerr=float(result.logzerr[-1]),
                information=float(result.information[-1]))


def checked(operation):
    try:
        return dict(status='completed', result=operation())
    except Exception as error:
        return dict(status='api_failure', error_type=type(error).__name__, error=str(error))


def compare_chain(result, names, new, seed, repetitions):
    assert not result.isdynamic(), 'This worker is declared for static event runs'
    assert len(result.logl) == result.niter + result.nlive, 'Completed result required'
    assert np.all(np.diff(result.logl) >= 0)
    assert np.all(np.diff(result.logvol) < 0)
    assert np.all(np.isfinite(result.logl))
    weights = np.exp(result.logwt - logsumexp(result.logwt))
    report = dict(native=totals(result), nlive=int(result.nlive), niter=int(result.niter),
                  samples=len(result.logl), posterior_weight_ess=float(1 / np.sum(weights**2)),
                  logvol_init_recorded='logvol_init' in result.keys(), versions={})
    for label, module in [('3.1.0', old), ('3.2.0', new)]:
        res = convert(module, result)
        wt, z, variance, information = module.compute_integrals(logl=res.logl, logvol=res.logvol)
        output = dict(recomputed_logz=float(z[-1]), recomputed_logzerr=float(np.sqrt(variance[-1])),
                      information=float(information[-1]), native_logz_residual=float(z[-1] - result.logz[-1]),
                      weighted_quantiles={})
        for key in PARAMETERS:
            if key in names:
                column = result.samples[:, names.index(key)]
                output['weighted_quantiles'][key] = module.quantile(column, [.05, .5, .95], weights)

        def jitter():
            values = []
            residuals = []
            for offset in range(repetitions):
                trial = module.jitter_run(res, rstate=np.random.default_rng(seed + offset))
                _, _, _, expected_h = module.compute_integrals(logl=trial.logl, logvol=trial.logvol)
                values.append(float(trial.logz[-1]))
                residuals.append(float(trial.information[-1] - expected_h[-1]))
            return dict(logz=values, shrinkage_only_sd=float(np.std(values, ddof=1)),
                        max_information_bookkeeping_residual=float(np.max(np.abs(residuals))),
                        interpretation='Conditional shrinkage uncertainty; does not test mode loss or mixing')

        output['jitter'] = checked(jitter)

        def reweight():
            # A bounded, declared tilt probes bookkeeping on these event samples.
            # It is not adopted as a new physical prior or a scientific result.
            key = 'luminosity_distance'
            column = result.samples[:, names.index(key)]
            factor = 0.2 * np.tanh((column - np.median(column)) / np.std(column))
            trial = module.reweight_run(res, logp_new=res.logl + factor)
            _, _, _, expected_h = module.compute_integrals(logl=res.logl, logvol=res.logvol, reweight=factor)
            return dict(**totals(trial), information_bookkeeping_residual=float(trial.information[-1] - expected_h[-1]),
                        target='diagnostic exp(0.2*tanh((distance-median)/SD)) likelihood tilt', adopted=False)

        output['reweight'] = checked(reweight)

        def resampling(integer=False):
            column = result.samples[:, names.index('luminosity_distance')]
            if integer:
                # Deliberately integer weights test the API using event values;
                # they do not replace the posterior weights in the real result.
                column = column[np.linspace(0, len(column) - 1, 100, dtype=int)]
                sample_weights = np.ones(len(column), dtype=int)
            else:
                sample_weights = weights
            values = module.resample_equal(column, sample_weights, rstate=np.random.default_rng(seed))
            return dict(rows=len(values), mean=float(np.mean(values)),
                        sha256=hashlib.sha256(np.asarray(values).tobytes()).hexdigest(),
                        weighting='unit integer API probe' if integer else 'actual posterior weights')

        output['resample_equal_actual_weights'] = checked(resampling)
        output['resample_equal_integer_probe'] = checked(lambda: resampling(integer=True))

        def support_shift():
            # Known-fraction regression built from the actual event trajectory.
            # This deliberately imposed fraction is not an estimate for the event.
            log_fraction = np.log(0.5)
            changes = dict(logvol=res.logvol + log_fraction, logwt=res.logwt + log_fraction,
                           logz=res.logz + log_fraction)
            if 'logvol_init' in module.Results._ALLOWED:
                changes['logvol_init'] = log_fraction
            shifted = convert(module, res, **changes)
            original_jitter = module.jitter_run(res, rstate=np.random.default_rng(seed))
            shifted_jitter = module.jitter_run(shifted, rstate=np.random.default_rng(seed))
            measured = float(shifted_jitter.logz[-1] - original_jitter.logz[-1])
            return dict(imposed_log_fraction=float(log_fraction), measured_jitter_logz_shift=measured,
                        residual=float(measured - log_fraction),
                        interpretation='Controlled bookkeeping test; event support fraction is not inferred')

        output['known_support_shift'] = checked(support_shift)
        report['versions'][label] = output
    before, after = [report['versions'][version] for version in ['3.1.0', '3.2.0']]
    report['delta_logz'] = after['recomputed_logz'] - before['recomputed_logz']
    report['logzerr_ratio_3_2_over_3_1'] = after['recomputed_logzerr'] / before['recomputed_logzerr']
    report['quantile_differences'] = {key: np.asarray(after['weighted_quantiles'][key]) - values
                                    for key, values in before['weighted_quantiles'].items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', nargs='+', required=True, type=Path)
    parser.add_argument('--upstream-utils', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native-convention', required=True, choices=['log_bayes_factor', 'log_evidence'])
    parser.add_argument('--jitter-repetitions', type=int, default=64)
    parser.add_argument('--seed', type=int, default=261114301)
    args = parser.parse_args()
    assert dynesty.__version__ == '3.1.0'
    assert args.jitter_repetitions >= 2
    assert len(set(map(str, args.native))) == len(args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert args.output.resolve() not in [path.resolve() for path in args.native]
    new = load_upstream(args.upstream_utils)
    report = dict(status='running', worker_sha256=sha(__file__), native_dynesty=dynesty.__version__,
                  native_utils_sha256=sha(inspect.getsourcefile(old)), upstream_utils_sha256=sha(args.upstream_utils),
                  upstream_url='https://github.com/joshspeagle/dynesty/blob/v3.2.0/py/dynesty/utils.py',
                  native_convention=args.native_convention, jitter_repetitions=args.jitter_repetitions,
                  seed=args.seed, chains=[])
    results = []
    names_reference = None
    for index, path in enumerate(args.native):
        assert path.name.endswith('_dynesty.pickle')
        hdf5 = path.with_name(path.name.replace('_dynesty.pickle', '_result.hdf5'))
        with hdf5.open('rb') as stream, h5py.File(stream) as stored:
            names = [key.decode() if isinstance(key, bytes) else str(key) for key in stored['search_parameter_keys'][()]]
            expected = float(stored[args.native_convention][()])
        if names_reference is None:
            names_reference = names
        assert names == names_reference
        assert 'sine_gaussian_0_hrss' in names and 'luminosity_distance' in names
        with path.open('rb') as stream:
            result = dill.load(stream)
        assert len(names) == result.samples.shape[1]
        assert np.isclose(result.logz[-1], expected, rtol=0, atol=1e-7), 'Declared evidence convention does not match'
        item = compare_chain(result, names, new, args.seed + 1000 * index, args.jitter_repetitions)
        item.update(native_path=str(path), native_sha256=sha(path), result_path=str(hdf5), result_sha256=sha(hdf5),
                    declared_evidence_residual=float(result.logz[-1] - expected), parameters=names)
        report['chains'].append(item)
        results.append(result)
        args.output.write_text(json.dumps(report, indent=2, default=encode) + '\n')
        print(json.dumps(dict(chain=path.name, delta_logz=item['delta_logz'],
                              error_ratio=item['logzerr_ratio_3_2_over_3_1'])), flush=True)
    if len(results) > 1:
        report['merge'] = {}
        for label, module in [('3.1.0', old), ('3.2.0', new)]:
            report['merge'][label] = checked(lambda: totals(module.merge_runs(
                [convert(module, result) for result in results], print_progress=False)))
        report['independent_chain_logz_sd'] = float(np.std([result.logz[-1] for result in results], ddof=1))
        report['chain_scatter_note'] = 'Few chains and possible shared mode loss do not calibrate the evidence error estimator'
    report['status'] = 'completed'
    report['interpretation'] = 'Estimator sensitivity on unchanged event samples; no posterior, evidence or paper result was replaced'
    args.output.write_text(json.dumps(report, indent=2, default=encode) + '\n')


if __name__ == '__main__':
    main()
