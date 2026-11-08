import pickle
import unittest

import bilby
import numpy as np


def test_legacy_interped_pickle():
    from bilby.core.utils.calculus import WrappedInterp1d

    prior = bilby.core.prior.Interped(xx=np.linspace(0, 1, 11), yy=np.ones(11))
    prior.probability_density = WrappedInterp1d(prior.xx, prior.yy)
    prior.cumulative_distribution = WrappedInterp1d(prior.xx, prior.YY)
    prior.inverse_cumulative_distribution = WrappedInterp1d(prior.YY, prior.xx)
    prior = pickle.loads(pickle.dumps(prior))
    for value in (0.25, np.array([0.25, 0.75])):
        np.testing.assert_allclose(prior.prob(value), np.ones_like(value))
        np.testing.assert_allclose(prior.ln_prob(value), np.zeros_like(value))
        np.testing.assert_allclose(prior.cdf(value), value)
        np.testing.assert_allclose(prior.rescale(value), value)


if __name__ == "__main__":
    unittest.main()
