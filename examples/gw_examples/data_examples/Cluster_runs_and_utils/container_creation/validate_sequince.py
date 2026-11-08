"""Validate the production Sequince settings on a constrained analytic target."""

import json
import tempfile
from pathlib import Path

import bilby
import numpy as np
from scipy.special import ndtr


class Gaussian(bilby.Likelihood):
    def __init__(self):
        super().__init__()

    def log_likelihood(self, parameters):
        return -0.5 * (((parameters['x'] - 0.25) / 0.07) ** 2
                       + ((parameters['y'] - 0.65) / 0.10) ** 2)


def constrain(sample):
    return dict(sample, cut=sample['x'])


def main():
    priors = bilby.core.prior.PriorDict(
        dict(x=bilby.core.prior.Uniform(0, 1),
             y=bilby.core.prior.Uniform(0, 1),
             cut=bilby.core.prior.Constraint(0, 0.5)),
        conversion_function=constrain,
    )
    # Analytic acceptance fraction removes Monte Carlo normalization noise.
    priors._cached_normalizations[('x', 'y')] = 2.0
    assert np.isneginf(priors.ln_prob(dict(x=0.75, y=0.65)))
    assert np.isclose(priors.ln_prob(dict(x=0.25, y=0.65)), np.log(2))
    integral_x = np.sqrt(2 * np.pi) * 0.07 * (ndtr((0.5 - 0.25) / 0.07) - ndtr(-0.25 / 0.07))
    integral_y = np.sqrt(2 * np.pi) * 0.10 * (ndtr((1 - 0.65) / 0.10) - ndtr(-0.65 / 0.10))
    expected = np.log(2 * integral_x * integral_y)
    bilby.core.utils.random.seed(231123)
    with tempfile.TemporaryDirectory(prefix='sequince-validation-') as outdir:
        options = dict(
            likelihood=Gaussian(), priors=priors,
            sampler='bilby.sequince-ps', outdir=outdir, label='constrained',
            n_particles=2048, mutation_n_steps=40,
            target_ess=4096, target_total_ess=10000, dynamic=False,
            sampler_backend='numpy', mutation='differential-evolution',
            particle_transform='affine', npool=2,
            checkpoint=True, checkpoint_frequency=1, resume=True,
            plot=False, save=False,
        )
        result = bilby.run_sampler(**options)
        resumed = bilby.run_sampler(**options)
        assert resumed.log_evidence == result.log_evidence
        np.testing.assert_array_equal(resumed.samples, result.samples)
        posterior = result.posterior
        assert ((posterior.x > 0) & (posterior.x < 0.5)).all()
        assert abs(posterior.x.mean() - 0.25) < 0.015
        assert abs(posterior.y.mean() - 0.65) < 0.015
        assert abs(result.log_evidence - expected) < 0.15
        checkpoint = Path(outdir) / 'constrained_sequince_ps_checkpoint.h5'
        assert checkpoint.stat().st_size > 0
        from bilby.core.sampler.sequince import SequincePersistentSampling
        restored = SequincePersistentSampling._load_checkpoint(checkpoint)
        assert restored is not None
        print(json.dumps(dict(log_evidence=result.log_evidence,
                              expected_log_evidence=expected,
                              posterior_samples=len(posterior),
                              mean_x=posterior.x.mean(), mean_y=posterior.y.mean(),
                              checkpoint_bytes=checkpoint.stat().st_size)))


if __name__ == '__main__':
    main()
