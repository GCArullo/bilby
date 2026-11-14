"""Check the controlled changes before spending event-likelihood evaluations."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/gw_examples/data_examples/Cluster_runs_and_utils/sampler_investigation"
sys.path.insert(0, str(SCRIPTS))

from sampling_periodic import InvalidUnitPoint, Transform
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
    np.testing.assert_allclose(default([.3]), seam([0.]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(seam([0.]), seam([1.]), rtol=0, atol=1e-14)
    with pytest.raises(InvalidUnitPoint):
        seam([1.1])
    assert seam.invalid == [[1.1]]
