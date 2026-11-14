"""Check the controlled changes before spending event-likelihood evaluations."""
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/gw_examples/data_examples/Cluster_runs_and_utils/sampler_investigation"
sys.path.insert(0, str(SCRIPTS))

from sampling_periodic import InvalidUnitPoint, Target, Transform, dynesty32
from sampling_variants import CASES, apply_variant, kernel


def test_each_full_event_case_changes_one_ingredient(tmp_path):
    original_sample = kernel.EnsembleWalkSampler.sample
    original_donor = kernel._get_proposal_kwargs
    expected = {"seed_only": "seed_retention", "donor_only": "donor_exclusion",
                "reflection_only": "reflection", "walkcap_only": "maxmcmc"}
    try:
        for case, settings in CASES.items():
            directory = tmp_path / case
            directory.mkdir()
            result = apply_variant(case, directory)
            changed = [key for key in settings if settings[key] != CASES["paper"][key]]
            assert changed == ([] if case == "paper" else [expected[case]])
            assert result["behavior"]["zero_acceptance_retains_seed"] == settings["seed_retention"]
            assert result["behavior"]["donor_pool_includes_seed"] != settings["donor_exclusion"]
    finally:
        kernel.EnsembleWalkSampler.sample = staticmethod(original_sample)
        kernel._get_proposal_kwargs = original_donor


def test_shifted_periodic_transform_preserves_physical_point_and_rejects_native_overflow():
    from bilby.core.prior import Uniform
    prior = Uniform(0, 2 * np.pi, boundary="periodic")
    default = Transform(prior, 0.)
    seam = Transform(prior, .3)
    np.testing.assert_allclose(default([.3, .7]), [prior.rescale(.3), .7], rtol=0, atol=1e-15)
    np.testing.assert_allclose(default([.3, .7]), seam([0., .7]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(seam([0., .7]), seam([1., .7]), rtol=0, atol=1e-14)
    with pytest.raises(InvalidUnitPoint):
        seam([1.1, .7])
    assert seam.invalid == [[1.1, .7]]


@pytest.mark.parametrize("version", ["3.1", "3.2"])
def test_native_periodic_controls_with_bounded_auxiliary(tmp_path, version):
    import dynesty
    from bilby.core.prior import Uniform
    from scipy.integrate import trapezoid
    from scipy.special import i0
    package = dynesty
    if version == "3.2":
        wheel = os.environ.get("DYNESTY32_TEST_WHEEL")
        if wheel is None:
            pytest.skip("Set DYNESTY32_TEST_WHEEL to run the pinned optional upstream comparison")
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(tmp_path / "upstream")
        package = dynesty32(tmp_path / "upstream/dynesty")
    target = Target(SimpleNamespace(log_likelihood_ratio=lambda p: .8 * np.cos(p["angle"])),
                    {}, "angle")
    prior = Uniform(0, 2 * np.pi, boundary="periodic")
    unit = np.linspace(0, 1, 257)
    angular = np.exp([target([prior.rescale(u), .5]) for u in unit])
    joint = np.array([[np.exp(target([prior.rescale(u), auxiliary]))
                       for auxiliary in [0., .5, 1.]] for u in unit])
    marginalized = trapezoid(joint, [0., .5, 1.], axis=1)
    np.testing.assert_array_equal(marginalized, angular)
    np.testing.assert_allclose(trapezoid(marginalized, unit), i0(.8), atol=1e-14, rtol=0)
    for method in (["unif", "rslice", "bilby"] if version == "3.1" else ["unif", "rslice"]):
        transform = Transform(prior, .17)
        sample = method
        if method == "bilby":
            sample = kernel.EnsembleWalkSampler(ndim=2, ncdim=2, periodic=[0], reflective=None,
                nonbounded=None, proposals=["diff"], walks=20, naccept=10, maxmcmc=1000)
        try:
            run = package.NestedSampler(target, transform, 2, nlive=40, sample=sample,
                bound="none" if method == "bilby" else "multi", periodic=[0], first_update=dict(min_ncall=0, min_eff=100),
                rstate=np.random.default_rng(7319))
            run.run_nested(dlogz=.3, print_progress=False)
        except InvalidUnitPoint:
            # Preserve the actual upstream periodic-proposal defect as a
            # diagnostic, but never accept the irrelevant all-periodic crash.
            assert version == "3.1" and transform.invalid
            print(version, method, "recorded unwrapped proposal")
        else:
            assert run.results.samples.shape[1] == 2
            assert np.isfinite(run.results.logz[-1])
            print(version, method, "completed", run.results.logz[-1])
