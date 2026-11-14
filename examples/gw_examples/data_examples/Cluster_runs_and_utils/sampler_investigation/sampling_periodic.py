"""One-angle conditional event controls with independent periodic quadrature."""
import argparse
from email.parser import Parser
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import dynesty
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp

from sampling_event import PERIODIC, load_target, sha, write


class InvalidUnitPoint(ValueError):
    pass


class Transform:
    def __init__(self, prior, origin):
        self.prior = prior
        self.origin = origin
        self.invalid = []

    def __call__(self, unit):
        unit = np.asarray(unit)
        assert unit.shape == (2,)
        if not np.all(np.isfinite(unit)) or np.any(unit < -1e-12) or np.any(unit > 1 + 1e-12):
            self.invalid.append(unit.tolist())
            raise InvalidUnitPoint("Native sampler passed an unwrapped coordinate to the prior transform")
        bounded = np.clip(unit, 0, 1)
        physical_unit = np.mod(bounded[0] + self.origin, 1.)
        return np.array([self.prior.rescale(physical_unit), bounded[1]])


class Target:
    def __init__(self, likelihood, parameters, angle):
        self.likelihood = likelihood
        self.parameters = parameters
        self.angle = angle
        self.calls = 0
        self.seconds = 0.

    def __call__(self, values):
        # Any second coordinate is an independent Uniform(0, 1) auxiliary.
        # Its integral is exactly one; only the physical angle enters L.
        start = time.monotonic()
        value = float(self.likelihood.log_likelihood_ratio(dict(self.parameters, **{self.angle: values[0]})))
        self.calls += 1
        self.seconds += time.monotonic() - start
        return value


def dynesty32(directory):
    """Load the pinned upstream package without replacing installed Dynesty."""
    directory = Path(directory)
    metadata = Parser().parsestr((directory.parent / "dynesty-3.2.0.dist-info/METADATA").read_text())
    assert metadata["Name"].lower() == "dynesty" and metadata["Version"] == "3.2.0"
    spec = importlib.util.spec_from_file_location("dynesty32", directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["dynesty32"] = package
    spec.loader.exec_module(package)
    # Upstream __init__ queries installed distribution metadata by its original
    # name, which would label this isolated source with the installed 3.1 version.
    package.__version__ = metadata["Version"]
    package.utils.DYNESTY_VERSION = metadata["Version"]
    return package


def reference(target, prior, outdir):
    values = {}
    last = None
    stages = []
    for n in [128, 256, 512, 1024, 2048, 4096]:
        unit = np.arange(n + 1) / n
        for u in unit:
            if float(u) not in values:
                values[float(u)] = target([prior.rescale(float(u))])
        logl = np.array([values[float(u)] for u in unit])
        assert np.all(np.isfinite(logl)), "Nonfinite angular target requires separate support audit"
        np.testing.assert_allclose(logl[0], logl[-1], atol=1e-7, rtol=0)
        scale = float(np.max(logl))
        density = np.exp(logl - scale)
        integral = float(trapezoid(density, unit))
        logI = scale + np.log(integral)
        cdf = cumulative_trapezoid(density, unit, initial=0.) / integral
        circular = trapezoid(density * np.exp(2j * np.pi * unit), unit) / integral
        change = None if last is None else abs(logI - last)
        stages.append(dict(intervals=n, log_conditional_integral=logI, change=change))
        last = logI
        # Require agreement of two successive refinements, not a single lucky grid.
        if len(stages) >= 3 and all(stage["change"] < 1e-4 for stage in stages[-2:]):
            break
    converged = len(stages) >= 3 and all(stage["change"] < 1e-4 for stage in stages[-2:])
    np.savez_compressed(outdir / "reference.npz", unit=unit, logl=logl, cdf=cdf)
    write(outdir / "summary.json", dict(mode="reference", converged=converged,
        log_conditional_integral=logI, endpoint_logLR_difference=float(logl[-1] - logl[0]),
        circular_cos=float(circular.real), circular_sin=float(circular.imag),
        true_calls=target.calls, likelihood_seconds=target.seconds, stages=stages,
        interpretation="One-angle conditional integral at frozen event parameters; not full event evidence"))


def nested(target, transform, package, sampler, outdir, seed):
    if sampler == "bilby":
        assert package.__version__ == "3.1.0"
        from bilby.core.sampler.dynesty_utils import EnsembleWalkSampler
        sample = EnsembleWalkSampler(ndim=2, ncdim=2, periodic=[0], reflective=None,
            nonbounded=None, proposals=["diff"], walks=100, naccept=60, maxmcmc=50000)
        bound = "none"
    else:
        sample = sampler
        bound = "multi"
    try:
        run = package.NestedSampler(target, transform, 2, nlive=200, sample=sample,
            bound=bound, periodic=[0], rstate=np.random.default_rng(seed),
            first_update=dict(min_ncall=0, min_eff=100), queue_size=1)
        run.run_nested(dlogz=.1, print_progress=False,
            checkpoint_file=str(outdir / "checkpoint.pickle"), checkpoint_every=600)
    except InvalidUnitPoint as error:
        write(outdir / "summary.json", dict(mode="nested", status="invalid_native_unit_point",
            error=str(error), invalid_inputs=transform.invalid, true_calls=target.calls,
            likelihood_seconds=target.seconds,
            interpretation="Recorded native periodic-prior contract failure; no evidence estimate"))
        return
    result = run.results
    with (outdir / "native_results.pickle").open("wb") as stream:
        pickle.dump(result, stream)
    weights = np.exp(result.logwt - logsumexp(result.logwt))
    prior = transform.prior
    physical_unit = (result.samples[:, 0] - prior.minimum) / (prior.maximum - prior.minimum)
    circular = weights @ np.exp(2j * np.pi * physical_unit)
    np.savez_compressed(outdir / "native_history.npz", samples=result.samples[:, :1],
        samples_u=result.samples_u[:, :1], auxiliary_samples=result.samples[:, 1],
        auxiliary_samples_u=result.samples_u[:, 1], physical_unit=physical_unit, weights=weights,
        logl=result.logl, logwt=result.logwt, logz=result.logz, logzerr=result.logzerr,
        ncall=result.ncall)
    write(outdir / "summary.json", dict(mode="nested", status="completed",
        log_conditional_integral=float(result.logz[-1]), quoted_error=float(result.logzerr[-1]),
        circular_cos=float(circular.real), circular_sin=float(circular.imag),
        true_calls=target.calls, native_calls=int(np.sum(result.ncall)),
        likelihood_seconds=target.seconds, effective_weight_count=float(1 / np.sum(weights ** 2)),
        physical_dimensions=1, sampling_dimensions=2,
        auxiliary="Independent bounded Uniform(0,1), omitted from physical summaries; likelihood independent",
        interpretation="One-angle conditional integral; compare with quadrature and independent seeds"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "reference", "nested"])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--angle", choices=PERIODIC, required=True)
    parser.add_argument("--origin", choices=["default", "seam"], default="default")
    parser.add_argument("--sampler", choices=["bilby", "rslice", "unif"], default="bilby")
    parser.add_argument("--dynesty-version", choices=["3.1", "3.2"], default="3.1")
    parser.add_argument("--dynesty32-dir", type=Path)
    parser.add_argument("--seed", type=int, default=261114601)
    args = parser.parse_args()
    _, likelihood, priors, provenance = load_target(args.settings, args.outdir, args.seed)
    parameters = provenance["sampled_baseline_parameters"]
    prior = priors[args.angle]
    assert type(prior).__name__ == "Uniform" and prior.boundary == "periodic"
    origin = 0. if args.origin == "default" else ((parameters[args.angle] - prior.minimum) / (prior.maximum - prior.minimum)) % 1
    transform = Transform(prior, origin)
    target = Target(likelihood, parameters, args.angle)
    package = dynesty if args.dynesty_version == "3.1" else dynesty32(args.dynesty32_dir)
    assert not (args.sampler == "bilby" and args.dynesty_version == "3.2")
    provenance.update(angle=args.angle, coordinate_origin=origin, origin_name=args.origin,
        native_version=package.__version__, sampler=args.sampler, mode=args.mode,
        native_package_source_sha256=sha(package.__file__), nlive=200, dlogz=.1,
        scope="Single periodic coordinate varied; all other event parameters fixed")
    if args.mode != "reference":
        provenance.update(physical_dimensions=1, sampling_dimensions=2,
            auxiliary="Independent bounded Uniform(0,1); likelihood independent; normalized integral=1")
    provenance["native_source_sha256"] = {
        name: sha(Path(package.__file__).parent / name)
        for name in ["dynesty.py", "internal_samplers.py", "utils.py", "bounding.py"]}
    path = args.outdir / "provenance.json"
    if path.exists():
        assert json.loads(path.read_text()) == provenance, "Output belongs to another test"
    else:
        write(path, provenance)
    if (args.outdir / "summary.json").exists():
        return
    if args.mode == "reference":
        assert args.origin == "default"
        reference(target, prior, args.outdir)
    elif args.mode == "nested":
        nested(target, transform, package, args.sampler, args.outdir, args.seed)
    else:
        u = ((parameters[args.angle] - prior.minimum) / (prior.maximum - prior.minimum) - origin) % 1
        replay = target(transform([u, .5]))
        np.testing.assert_allclose(replay, provenance["baseline_log_likelihood_ratio"], atol=1e-8, rtol=0)
        write(args.outdir / "preflight.json", dict(passed=True, coordinate_replay=replay,
            native_version=package.__version__, target_calls=target.calls,
            production_sampling_started=False))


if __name__ == "__main__":
    main()
