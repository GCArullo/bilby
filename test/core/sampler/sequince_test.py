import signal
from unittest.mock import patch

import bilby
import numpy as np
import pytest

pytest.importorskip("sequince_bilby")
from bilby.core.sampler.sequince import SequincePersistentSampling


def model(x, mu):
    return np.full_like(x, mu)


def make_sampler(tmp_path, **kwargs):
    likelihood = bilby.core.likelihood.GaussianLikelihood(
        np.array([0.0]), np.array([0.3]), model, sigma=0.1
    )
    priors = bilby.core.prior.PriorDict(
        dict(mu=bilby.core.prior.Uniform(-10, 10))
    )
    return SequincePersistentSampling(
        likelihood, priors, outdir=str(tmp_path), label="resume",
        n_particles=64, mutation_n_steps=4, target_ess=128,
        target_total_ess=256, sampling_seed=1234, exit_code=77, **kwargs,
    )


@pytest.mark.parametrize("custom_checkpoint", [False, True])
def test_completed_checkpoint_regenerates_existing_result(tmp_path, custom_checkpoint):
    kwargs = dict(checkpoint_filename=str(tmp_path / "custom.h5")) if custom_checkpoint else {}
    original = make_sampler(tmp_path, **kwargs).run_sampler()
    resumed = make_sampler(tmp_path, **kwargs).run_sampler()
    assert resumed.log_evidence == original.log_evidence
    np.testing.assert_array_equal(resumed.samples, original.samples)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGALRM])
def test_signals_exit_with_checkpoint_code_and_restore_handlers(tmp_path, signum):
    sampler = make_sampler(tmp_path)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

    def interrupt(self):
        handler = signal.getsignal(signum)
        assert handler != previous[signum]
        handler(signum, None)

    with patch("sequince_bilby.plugin.SequincePersistentSampling.run_sampler", interrupt):
        with pytest.raises(SystemExit) as error:
            sampler.run_sampler()
    assert error.value.code == 77
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
