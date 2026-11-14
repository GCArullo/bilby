"""Process-local, source-guarded controls for the two Bilby sampler edits."""
import hashlib
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import dynesty
import numpy as np
from bilby.core.sampler import dynesty_utils as kernel

SOURCE_SHA256 = "aa4b166873ebb3244ab94ea85a9ca56c8bcca3099f5fca4a40fab48e9890fd2f"
CASES = {
    "paper": dict(seed_retention=False, donor_exclusion=False, reflection=True, maxmcmc=5000),
    "seed_only": dict(seed_retention=True, donor_exclusion=False, reflection=True, maxmcmc=5000),
    "donor_only": dict(seed_retention=False, donor_exclusion=True, reflection=True, maxmcmc=5000),
    "reflection_only": dict(seed_retention=False, donor_exclusion=False, reflection=False, maxmcmc=5000),
    "walkcap_only": dict(seed_retention=False, donor_exclusion=False, reflection=True, maxmcmc=50000),
}
SAMPLE_SOURCE = textwrap.dedent(inspect.getsource(kernel.EnsembleWalkSampler.sample))
DONOR_SOURCE = textwrap.dedent(inspect.getsource(kernel._get_proposal_kwargs))


def apply_variant(case, outdir):
    assert dynesty.__version__ == "3.1.0"
    assert hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest() == SOURCE_SHA256
    settings = CASES[case]
    sample = SAMPLE_SOURCE.removeprefix("@staticmethod\n")
    donor = DONOR_SOURCE
    if not settings["seed_retention"]:
        start = sample.index("    if naccept == 0:\n")
        end = sample.index("    sampling_info = {", start)
        sample = sample[:start] + '''    if naccept == 0:
        logger.debug(
            "Unable to find a new point using walk: returning a random point. "
            "If this warning occurs often, increase naccept."
        )
        # Exact pre-88f321fd fallback, including its original call accounting.
        current_u = common_kwargs["rstate"].uniform(0, 1, len(current_u))
        current_v = args.prior_transform(current_u)
        logl = args.loglikelihood(current_v)

''' + sample[end:]
    if not settings["donor_exclusion"]:
        before = "live = np.delete(live, matches, 0)"
        assert donor.count(before) == 1
        donor = donor.replace(before, "np.delete(live, matches, 0)")
    # The installed package remains untouched. Forked workers inherit these
    # guarded process-local functions; checkpoints must resume with this script.
    source = sample + "\n\n" + donor
    path = Path(outdir) / "effective_sampler_source.py"
    if path.exists():
        assert path.read_text() == source, "Different sampler source in existing output"
    else:
        path.write_text(source)
    exec(compile(source, str(path), "exec"), kernel.__dict__)
    kernel.EnsembleWalkSampler.sample = staticmethod(kernel.sample)
    behavior = check_variant(settings)
    return dict(case=case, **settings, installed_module=str(Path(kernel.__file__).resolve()),
                installed_sha256=SOURCE_SHA256,
                effective_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                behavior=behavior)


def check_variant(settings):
    """Exercise both intended changes without an event-likelihood call."""
    initial = np.array([.2, .4])
    args = SimpleNamespace(u=initial.copy(), loglstar=-1., axes=np.eye(2),
        live_points=np.array([[.2, .4], [.1, .1], [.8, .8]]),
        prior_transform=lambda u: u.copy(), loglikelihood=lambda v: 0.,
        rseed=np.random.default_rng(261114991),
        kwargs=dict(walks=3, periodic=None, reflective=None, proposals=["diff"], ncdim=2))
    _, _, proposal = kernel._get_proposal_kwargs(args)
    includes_seed = bool(np.any(np.all(proposal["diff"]["live"] == initial, axis=1)))
    function = kernel.proposal_funcs["diff"]
    try:
        kernel.proposal_funcs["diff"] = lambda **kwargs: np.array([1.2, .4])
        result = kernel.EnsembleWalkSampler.sample(args)
    finally:
        kernel.proposal_funcs["diff"] = function
    retains_seed = bool(np.array_equal(result.u, initial))
    assert retains_seed == settings["seed_retention"]
    assert includes_seed != settings["donor_exclusion"]
    assert result.tuning_info["accept"] == 0
    return dict(zero_acceptance_retains_seed=retains_seed,
                donor_pool_includes_seed=includes_seed,
                returned_point=result.u.tolist())
