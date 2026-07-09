import importlib.util
import ast
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "gw_examples"
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "submit_runs_injection.py"
)


def load_submit_runs_injection_module():
    script_dir = SCRIPT_PATH.parent
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "submit_runs_injection_test_module",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("submit_runs_injection_test_module", None)
        sys.path.pop(0)


def test_num_frequency_bands_defaults_to_one():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args([])

    assert args.injection_noise == "student"
    assert args.num_frequency_bands is None
    assert args.noise_generation_seed is None
    assert args.maxmcmc is None
    assert args.require_epnfs is False
    assert module.hypothesis_list(args) == ["gaussian"]


def test_accounting_user_defaults_to_home_basename(monkeypatch):
    monkeypatch.setenv("HOME", "/home/name.surname")

    module = load_submit_runs_injection_module()
    parser = module.build_parser()
    args = parser.parse_args([])

    assert module.DEFAULT_ACCOUNTING_USER == "name.surname"
    assert args.accounting_user == "name.surname"


def test_zero_gaussian_injection_noise_is_available():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--injection-noise", "zero-gaussian"])

    assert args.injection_noise == "zero-gaussian"


def test_noise_generation_seed_is_available():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--noise-generation-seed", "98765"])

    assert args.noise_generation_seed == 98765


def test_injection_duration_is_available_and_rendered_into_ini(tmp_path):
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--injection-duration", "12.5"])
    args.accounting_user = "acct"
    args.require_epnfs = False
    args.test_injection = False
    args.nlive = 10
    args.naccept = 3
    args.frequency_domain_injection = False

    ini_template = "\n".join(
        [
            "accounting-user=old",
            "queue=None",
            "create-summary=False",
            "summarypages-arguments=None",
            "duration=8.0",
            "data-dict=None",
            "data-format=gwf",
            "calibration-model=CubicSpline",
            "calibration-correction-type=data",
            "spline-calibration-envelope-dict={H1: old.dat}",
            "channel-dict=None",
            "psd-dict=None",
            "additional-transfer-paths=None",
            "sampler-kwargs={'nlive': 1}",
            "likelihood-type=old",
            "extra-likelihood-kwargs=old",
            "",
        ]
    )
    template_settings = dict(
        detectors=("H1",),
        duration=args.injection_duration,
        minimum_frequency={"H1": 20.0, "waveform": 10.0},
        maximum_frequency=448.0,
        reference_frequency=10.0,
        waveform_approximant="NRSur7dq4",
        sampler_kwargs={"nlive": 1},
    )

    rendered = module.render_ini(
        ini_template,
        args=args,
        template_settings=template_settings,
        num_frequency_bands=1,
        detector_dependent_nu=False,
        likelihood_nu=None,
        label="label",
        outdir=tmp_path / "out",
        webdir=tmp_path / "web",
        prior_path=tmp_path / "prior.prior",
        data_paths={"H1": str(tmp_path / "H1.hdf5")},
        psd_paths={"H1": str(tmp_path / "H1_psd.dat")},
        stage_dir=tmp_path / "staged_data",
        hypothesis="gaussian",
        sine_gaussian_config=type(
            "Config",
            (),
            dict(enabled=False, total_components=0),
        )(),
    )

    assert args.injection_duration == 12.5
    assert "duration=12.5\n" in rendered


def test_injected_sine_gaussian_values_are_loaded_from_json_and_within_bounds():
    module = load_submit_runs_injection_module()

    values = module.load_injected_sine_gaussian_values()
    assert values["coherent"][1][0] == dict(
        hrss=pytest.approx(1e-22),
        Q=pytest.approx(8.0),
        frequency=pytest.approx(130.0),
        time_offset=pytest.approx(0.0),
        phase_offset=pytest.approx(0.0),
    )

    coherent_components = module.load_injected_sine_gaussian_component_series(
        mode="coherent",
        count=2,
    )
    assert coherent_components == [
        dict(
            hrss=1e-22,
            Q=8.0,
            frequency=40.0,
            time_offset=-0.05,
            phase_offset=0.0,
        ),
        dict(
            hrss=8.5e-23,
            Q=9.0,
            frequency=220.0,
            time_offset=0.05,
            phase_offset=0.5,
        ),
    ]

    incoherent_components = module.load_injected_sine_gaussian_component_series(
        mode="incoherent",
        detector="L1",
        count=1,
    )
    assert incoherent_components == [
        dict(
            hrss=1e-22,
            Q=8.0,
            frequency=135.0,
            time_offset=0.0,
            phase_offset=0.3,
        )
    ]

    for component in coherent_components + incoherent_components:
        module.validate_injected_sine_gaussian_component(
            component,
            frequency_minimum=20.0,
            frequency_maximum=448.0,
        )


def test_injected_sine_gaussian_validation_rejects_out_of_bounds_component():
    module = load_submit_runs_injection_module()

    with pytest.raises(ValueError, match="outside prior bounds"):
        module.validate_injected_sine_gaussian_component(
            dict(
                hrss=1e-22,
                Q=8.0,
                frequency=500.0,
                time_offset=0.0,
                phase_offset=0.0,
            ),
            frequency_minimum=20.0,
            frequency_maximum=448.0,
        )


def test_student_likelihood_runs_student_only_by_default():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--likelihood", "student"])

    assert args.num_frequency_bands is None
    assert module.hypothesis_list(args) == ["student", "gaussian"]


def test_student_likelihood_can_disable_default_gaussian():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--likelihood", "student", "--no-add-gaussian"])

    assert module.hypothesis_list(args) == ["student"]


def test_gaussian_likelihood_rejects_explicit_num_frequency_bands():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--likelihood", "gaussian", "--num-frequency-bands", "2"])

    with pytest.raises(
        ValueError,
        match="--likelihood gaussian cannot be combined with --num-frequency-bands",
    ):
        module.hypothesis_list(args)


def test_add_gaussian_requires_student_likelihood():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--add-gaussian"])

    with pytest.raises(
        ValueError,
        match="--add-gaussian requires --likelihood student",
    ):
        module.hypothesis_list(args)


def test_student_likelihood_keeps_gaussian_companion_for_multiple_frequency_bands():
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--likelihood", "student", "--num-frequency-bands", "4"])

    assert module.hypothesis_list(args) == ["student", "gaussian"]


def test_main_allows_gaussian_default_band_count_with_dry_run(monkeypatch, tmp_path):
    module = load_submit_runs_injection_module()
    base_dir = tmp_path / "runs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--likelihood",
            "gaussian",
            "--num-sine-gaussians",
            "1",
            "--sine-gaussian-mode",
            "coherent",
            "--dry-run",
            "--base-dir",
            str(base_dir),
            "--require-epnfs",
        ],
    )

    assert module.main() == 0
    ini_paths = list(base_dir.rglob("*.ini"))
    assert ini_paths
    assert list(base_dir.rglob("*.prior"))

    gaussian_ini = ini_paths[0].read_text(encoding="utf-8")
    assert "calibration-model=None" in gaussian_ini
    assert "calibration-correction-type=None" in gaussian_ini
    assert "spline-calibration-envelope-dict=None" in gaussian_ini
    assert (
        "frequency-domain-source-model="
        "bilby.gw.source.cbc_plus_sine_gaussians\n"
    ) in gaussian_ini
    assert "distance-marginalization=False\n" in gaussian_ini
    assert "generation-function=None\n" in gaussian_ini
    assert "queue=EPNFS\n" in gaussian_ini


def test_render_ini_writes_maxmcmc_override(tmp_path):
    module = load_submit_runs_injection_module()
    parser = module.build_parser()

    args = parser.parse_args(["--maxmcmc", "10000"])
    args.accounting_user = "acct"
    args.require_epnfs = False
    args.test_injection = False
    args.nlive = 10
    args.naccept = 3
    args.frequency_domain_injection = False

    ini_template = "\n".join(
        [
            "accounting-user=old",
            "queue=None",
            "create-summary=False",
            "environment-variables={}",
            "summarypages-arguments=None",
            "data-dict=None",
            "data-format=gwf",
            "calibration-model=CubicSpline",
            "calibration-correction-type=data",
            "spline-calibration-envelope-dict={H1: old.dat}",
            "channel-dict=None",
            "psd-dict=None",
            "additional-transfer-paths=None",
            "sampler-kwargs={'nlive': 1, 'maxmcmc': 5000}",
            "likelihood-type=old",
            "extra-likelihood-kwargs=old",
            "",
        ]
    )
    template_settings = dict(
        detectors=("H1",),
        minimum_frequency={"H1": 20.0, "waveform": 10.0},
        maximum_frequency=448.0,
        reference_frequency=10.0,
        waveform_approximant="NRSur7dq4",
        sampler_kwargs={"nlive": 1, "maxmcmc": 5000},
    )

    rendered = module.render_ini(
        ini_template,
        args=args,
        template_settings=template_settings,
        num_frequency_bands=1,
        detector_dependent_nu=False,
        likelihood_nu=None,
        label="label",
        outdir=tmp_path / "out",
        webdir=tmp_path / "web",
        prior_path=tmp_path / "prior.prior",
        data_paths={"H1": str(tmp_path / "H1.hdf5")},
        psd_paths={"H1": str(tmp_path / "H1_psd.dat")},
        stage_dir=tmp_path / "staged_data",
        hypothesis="gaussian",
        sine_gaussian_config=type(
            "Config",
            (),
            dict(enabled=False, total_components=0),
        )(),
    )

    sampler_line = next(
        line for line in rendered.splitlines()
        if line.startswith("sampler-kwargs=")
    )
    sampler_kwargs = ast.literal_eval(sampler_line.split("=", 1)[1])
    assert sampler_kwargs["maxmcmc"] == 10000


def test_main_creates_summarypages_without_recalib_parameters_by_default(
    monkeypatch, tmp_path
):
    module = load_submit_runs_injection_module()
    base_dir = tmp_path / "runs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--likelihood",
            "gaussian",
            "--dry-run",
            "--base-dir",
            str(base_dir),
            "--injection-noise",
            "gaussian",
        ],
    )

    assert module.main() == 0

    gaussian_ini = next(base_dir.rglob("*.ini")).read_text(encoding="utf-8")
    summary_line = next(
        line for line in gaussian_ini.splitlines()
        if line.startswith("summarypages-arguments=")
    )
    summary_arguments = ast.literal_eval(summary_line.split("=", 1)[1])
    assert "create-summary=True\n" in gaussian_ini
    assert summary_arguments["ignore_parameters"] == ["recalib*"]
    assert summary_arguments["disable_interactive"] is True
    assert summary_arguments["f_ref"] == 10.0
    assert summary_arguments["f_low"] == 20
    assert summary_arguments["f_start"] == 10.0
    assert summary_arguments["f_final"] == 448.0
    assert summary_arguments["approximant"] == ["NRSur7dq4"]
    assert "calibration" not in summary_arguments
    assert set(summary_arguments["psd"]) == {"H1", "L1"}


def test_stage_injection_bundle_writes_psd_on_staged_frequency_grid(
    monkeypatch, tmp_path
):
    module = load_submit_runs_injection_module()
    args = type(
        "Args",
        (),
        dict(
            label_prefix="label",
            injection_noise="gaussian",
            likelihood="gaussian",
            num_frequency_bands=1,
            detector_dependent_nu=False,
            nu_injection="2.1",
            noise_generation_seed=98765,
            frequency_domain_injection=True,
        ),
    )()
    template_settings = dict(
        detectors=("H1",),
        duration=12.0,
        trigger_time=100.0,
        post_trigger_duration=2.0,
        sampling_frequency=4.0,
        sampling_seed=123,
        minimum_frequency={"H1": 0.0, "waveform": 0.0},
        maximum_frequency=2.0,
        reference_frequency=1.0,
        waveform_approximant="NRSur7dq4",
    )
    sine_gaussian_config = type(
        "Config",
        (),
        dict(
            enabled=False,
            mode="none",
            total_components=0,
            detector_counts=(),
            label_suffix="",
        ),
    )()
    staged_frequencies = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    staged_psd = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    class FakeInterferometer:
        name = "H1"
        frequency_array = staged_frequencies
        power_spectral_density_array = staged_psd
        frequency_domain_strain = np.zeros_like(staged_frequencies, dtype=complex)
        start_time = 90.0
        duration = 12.0
        sampling_frequency = 4.0

    class FakeInterferometers(list):
        def inject_signal(self, parameters, waveform_generator):
            self.injected_parameters = parameters
            self.waveform_generator = waveform_generator

    seed_calls = []
    monkeypatch.setattr(module.bilby.core.utils.random, "seed", seed_calls.append)
    monkeypatch.setattr(
        module,
        "load_maximum_likelihood_injection",
        lambda posterior_path: ({"mass_1": 30.0}, 1.0, 0),
    )
    monkeypatch.setattr(
        module,
        "load_psds",
        lambda posterior_path, detectors: {
            "H1": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        },
    )
    monkeypatch.setattr(
        module,
        "build_interferometers",
        lambda *args, **kwargs: FakeInterferometers([FakeInterferometer()]),
    )
    monkeypatch.setattr(
        module,
        "build_waveform_generator",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "load_test_injection_chirp_mass_bounds",
        lambda posterior_path: (1.0, 2.0),
    )

    bundle = module.stage_injection_bundle(
        tmp_path,
        args,
        template_settings,
        tmp_path / "posterior.h5",
        sine_gaussian_config,
    )

    psd = np.loadtxt(bundle["psd_paths"]["H1"])
    assert seed_calls == [98765]
    np.testing.assert_array_equal(psd[:, 0], staged_frequencies)
    np.testing.assert_array_equal(psd[:, 1], staged_psd)


def test_main_student_multi_band_writes_single_gaussian_companion(
    monkeypatch, tmp_path
):
    module = load_submit_runs_injection_module()
    base_dir = tmp_path / "runs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--likelihood",
            "student",
            "--num-frequency-bands",
            "4",
            "--dry-run",
            "--base-dir",
            str(base_dir),
            "--injection-noise",
            "gaussian",
        ],
    )

    assert module.main() == 0

    gaussian_ini_paths = sorted(base_dir.rglob("*_gaussian.ini"))
    student_ini_paths = sorted(base_dir.rglob("*_student.ini"))

    assert len(gaussian_ini_paths) == 1
    assert len(student_ini_paths) == 1

    gaussian_ini = gaussian_ini_paths[0].read_text(encoding="utf-8")
    student_ini = student_ini_paths[0].read_text(encoding="utf-8")
    assert "__NUM_FREQUENCY_BANDS__" not in gaussian_ini
    assert "num-frequency-bands=4" not in gaussian_ini
    assert "'num_frequency_bands': 4" not in gaussian_ini
    for ini_text in (gaussian_ini, student_ini):
        assert "calibration-model=None" in ini_text
        assert "calibration-correction-type=None" in ini_text
        assert "spline-calibration-envelope-dict=None" in ini_text


def test_main_submits_by_default(monkeypatch, tmp_path):
    module = load_submit_runs_injection_module()
    ini_path = tmp_path / "ini_files" / "run.ini"
    submitted = {}

    def fake_prepare_runs(args):
        assert args.dry_run is False
        return [ini_path]

    def fake_submit_runs(ini_paths, executable):
        submitted["ini_paths"] = ini_paths
        submitted["executable"] = executable

    monkeypatch.setattr(module, "prepare_runs", fake_prepare_runs)
    monkeypatch.setattr(module, "submit_runs", fake_submit_runs)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    assert module.main() == 0
    assert submitted == {
        "ini_paths": [ini_path],
        "executable": "bilby_pipe",
    }


def test_main_dry_run_skips_submit(monkeypatch, tmp_path):
    module = load_submit_runs_injection_module()
    ini_path = tmp_path / "ini_files" / "run.ini"
    submitted = {}

    def fake_prepare_runs(args):
        assert args.dry_run is True
        return [ini_path]

    def fake_submit_runs(ini_paths, executable):
        submitted["ini_paths"] = ini_paths
        submitted["executable"] = executable

    monkeypatch.setattr(module, "prepare_runs", fake_prepare_runs)
    monkeypatch.setattr(module, "submit_runs", fake_submit_runs)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--dry-run"])

    assert module.main() == 0
    assert submitted == {}
