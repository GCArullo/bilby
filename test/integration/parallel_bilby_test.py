"""Exercise the runbook analysis executable with real MPI workers."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bilby


@pytest.mark.parametrize("student", [False, True])
def test_parallel_sine_gaussian_analysis(tmp_path, student):
    pytest.importorskip("parallel_bilby.schwimmbad_fast")
    pytest.importorskip("mpi4py")
    if shutil.which("mpiexec") is None:
        pytest.skip("MPI runtime is not installed")
    from bilby_pipe.utils import DataDump

    interferometers = bilby.gw.detector.InterferometerList(["H1"])
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=256, duration=1, start_time=1000000000,
    )
    priors = bilby.core.prior.PriorDict(dict(
        ra=0.2, dec=0.3, psi=0.4, geocent_time=1000000000.5,
        sine_gaussian_0_hrss=bilby.core.prior.Uniform(1e-24, 1e-22),
        sine_gaussian_0_Q=9, sine_gaussian_0_frequency=80,
        sine_gaussian_0_time_offset=0, sine_gaussian_0_phase_offset=0,
    ))
    dump = DataDump(
        label="sg", outdir=str(tmp_path), trigger_time=1000000000.5,
        likelihood_lookup_table=None, likelihood_roq_weights=None,
        likelihood_roq_params=None, likelihood_multiband_weights=None,
        priors_dict=dict(priors), priors_class=type(priors),
        interferometers=interferometers,
        meta_data={"injection_parameters": None}, idx=0,
    )
    dump.to_pickle()
    likelihood = "StudentTGravitationalWaveTransient" if student else "GravitationalWaveTransient"
    config = tmp_path / "sg.ini"
    config.write_text("\n".join([
        "label=sg", f"outdir={tmp_path}", f"data-dump-file={dump.filename}",
        "detectors=[H1]", "sampler=dynesty", "request-cpus=3", "local=True",
        "sampling-seed=1234", "reference-frame=sky", "time-reference=geocent",
        "minimum-frequency=20", "maximum-frequency=128", "sampling-frequency=256",
        "waveform-generator=bilby.gw.waveform_generator.WaveformGenerator",
        "frequency-domain-source-model=bilby.gw.source.sine_gaussians",
        "conversion-function=bilby.gw.conversion.convert_to_sine_gaussian_parameters",
        "generation-function=bilby.gw.conversion.identity_map_generation",
        f"likelihood-type=bilby.gw.likelihood.{likelihood}",
        "extra-likelihood-kwargs={'nu': 8}" if student else "extra-likelihood-kwargs=None",
        "result-format=json", "enforce-signal-duration=False",
        "sampler-kwargs={'nlive': 20, 'naccept': 5, 'dlogz': 1, "
        "'check_point_plot': False, 'print_progress': False}",
    ]) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(bilby.__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "cli_bilby.parallel_bilby", str(config)]
    result_file = tmp_path / "result" / "sg_result.json"
    generator = bilby.gw.WaveformGenerator(
        duration=1, sampling_frequency=256, start_time=1000000000,
        frequency_domain_source_model=bilby.gw.source.sine_gaussians,
        parameter_conversion=bilby.gw.conversion.convert_to_sine_gaussian_parameters,
    )
    likelihood_class = getattr(bilby.gw.likelihood, likelihood)
    serial_likelihood = likelihood_class(
        interferometers, generator, **({"nu": 8} if student else {}),
    )
    for stage in ("sample", "resume", "cached"):
        if stage == "resume":
            result_file.unlink()
        completed = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90)
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        if stage == "resume":
            assert "Resume file successfully loaded" in output, output
        if stage == "cached":
            assert "Using cached result" in output, output
        result = bilby.result.read_in_result(str(result_file))
        assert result.search_parameter_keys == ["sine_gaussian_0_hrss"]
        assert np.isfinite(result.log_evidence)
        assert np.isfinite(result.posterior.H1_sine_gaussian_optimal_snr).all()
        assert len(result.posterior) > 0
        expected = [serial_likelihood.log_likelihood_ratio(row.to_dict())
                    for _, row in result.posterior.iterrows()]
        np.testing.assert_allclose(result.posterior.log_likelihood, expected, atol=1e-9)
    if not student:
        interrupted_outdir = tmp_path / "interrupted"
        interrupted = config.read_text().replace(
            f"outdir={tmp_path}", f"outdir={interrupted_outdir}",
        ).replace("local=True", "local=False").replace(
            "'nlive': 20", "'nlive': 500",
        ).replace("'dlogz': 1,", "'dlogz': 1e-100,")
        config.write_text(interrupted + "periodic-restart-time=5\n")
        completed = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 77, completed.stdout + completed.stderr
        assert (interrupted_outdir / "result" / "sg_resume.pickle").exists()
