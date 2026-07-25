import importlib.util
import ast
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "gw_examples"
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "submit_runs_real_data.py"
)


def load_submit_runs_real_data_module():
    script_dir = SCRIPT_PATH.parent
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "submit_runs_real_data_test_module",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("submit_runs_real_data_test_module", None)
        sys.path.pop(0)


def test_num_frequency_bands_defaults_to_one():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args([])

    assert args.num_frequency_bands is None
    assert args.maxmcmc is None
    assert args.require_epnfs is False
    assert module.hypothesis_list(args) == ["gaussian"]


@pytest.mark.parametrize(
    ("catalog", "count", "examples"),
    [
        ("GWTC-2.1", 54, {"GW150914", "GW190425_081805", "GW190930_133541"}),
        ("GWTC-3", 36, {"GW191103_012549", "GW200115_042309", "GW200322_091133"}),
    ],
)
def test_all_catalog_manifest_events_are_available(catalog, count, examples):
    module = load_submit_runs_real_data_module()
    manifest_path = (
        SCRIPT_PATH.parent / module.CATALOG_CONFIGS_DIR / catalog / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = {item["event"] for item in manifest["events"]}

    assert len(events) == count
    assert examples <= events
    assert events <= module.EVENT_DEFAULTS.keys()
    if catalog == "GWTC-3":
        assert all(
            item["filename"].endswith("_mixed_nocosmo.h5")
            for item in manifest["events"]
        )
        low_spin = {
            item["event"]
            for item in manifest["events"]
            if item["selected_run"] == "C01:IMRPhenomXPHM:LowSpin"
        }
        assert low_spin == {"GW191219_163120", "GW200115_042309"}
        gw200306 = next(
            item for item in manifest["events"] if item["event"] == "GW200306_093714"
        )
        assert set(gw200306["psd_detectors"]) == {"H1", "L1"}
    for event in events:
        defaults = module.EVENT_DEFAULTS[event]
        assert defaults.run_subdir == f"GWTC_parametric_noise/Runs/{event}"
        assert defaults.working_directory == f"{module.CATALOG_CONFIGS_DIR}/{catalog}"


def test_special_event_templates_and_priors_are_grouped_together():
    module = load_submit_runs_real_data_module()
    special_dir = SCRIPT_PATH.parent / module.SPECIAL_EVENTS_CONFIGS_DIR

    assert {path.name for path in (special_dir / "templates").iterdir()} == {
        "GW150914_t_student_igwn_template.ini",
        "GW200129_065458_Hannam_NRSur7dq4.ini",
        "GW230814_t_student_pSEOB_template.ini",
        "GW231123_t_student_template.ini",
    }
    assert {path.name for path in (special_dir / "priors").iterdir()} == {
        "GW150914_igwn_template.prior",
        "GW200129_065458_Hannam_NRSur7dq4.prior",
        "GW230814_gr_template.prior",
        "GW230814_template.prior",
        "GW231123_template.prior",
    }


def test_accounting_user_defaults_to_home_basename(monkeypatch):
    monkeypatch.setenv("HOME", "/home/name.surname")

    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)
    args = parser.parse_args([])

    assert module.DEFAULT_ACCOUNTING_USER == "name.surname"
    assert args.accounting_user == "name.surname"


def test_gw200129_hannam_profile_generates_nrsur_reproduction(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW200129_065458_Hannam",
            "--likelihood",
            "gaussian",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--home-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    ini_text = next(ini_dir.glob("*.ini")).read_text(encoding="utf-8")
    prior_text = next(prior_dir.glob("*.prior")).read_text(encoding="utf-8")

    assert "waveform-approximant=NRSur7dq4\n" in ini_text
    assert (
        "mode-array=[[2, -2], [2, -1], [2, 0], [2, 1], [2, 2], "
        "[3, -3], [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [3, 3]]\n"
    ) in ini_text
    assert (
        "additional-transfer-paths="
        "[/home/pe.o4/GWTC4-fogg/NRSur7dq4_v1.0.h5]\n"
    ) in ini_text
    assert "minimum=14.5, maximum=49.0" in prior_text
    assert "total_mass = Constraint(minimum=68, maximum=500" in prior_text
    assert "mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(" in prior_text
    assert "minimum=0.25, maximum=1.0" in prior_text


def test_default_output_bases_are_under_public_event_directory(tmp_path):
    module = load_submit_runs_real_data_module()

    outdir_base, webdir_base = module.default_output_bases(
        tmp_path,
        "GW231123/t_Student/Runs",
    )

    event_dir = tmp_path / "public_html" / "GW231123" / "t_Student"
    assert Path(outdir_base) == event_dir / "Runs"
    assert Path(webdir_base) == event_dir / "Runs"
    assert Path(webdir_base) / "run-name" / "web" == (
        Path(outdir_base) / "run-name" / "web"
    )


def test_likelihood_run_subdir_uses_requested_model_likelihood():
    module = load_submit_runs_real_data_module()
    run_subdir = "GW230814/t_Student_pSEOB/Runs"

    assert module.likelihood_run_subdir(run_subdir, "student") == run_subdir
    assert module.likelihood_run_subdir(run_subdir, "hyperbolic") == (
        "GW230814/hyperbolic_pSEOB/Runs"
    )
    assert module.likelihood_run_subdir(run_subdir, "gaussian") == (
        "GW230814/gaussian_pSEOB/Runs"
    )


def test_main_creates_missing_output_bases_before_submission(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    outdir_base = tmp_path / "new" / "out"
    webdir_base = tmp_path / "new" / "web"
    submitted = []

    monkeypatch.setattr(
        module,
        "submit_run",
        lambda *args, **kwargs: submitted.append(1),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "gaussian",
            "--ini-dir",
            str(tmp_path / "ini"),
            "--prior-dir",
            str(tmp_path / "prior"),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0
    assert outdir_base.is_dir()
    assert webdir_base.is_dir()
    assert submitted == [1]


def test_student_likelihood_runs_student_and_gaussian_by_default():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "student"])

    assert args.num_frequency_bands is None
    assert module.hypothesis_list(args) == ["student", "gaussian"]


def test_student_likelihood_can_disable_default_gaussian():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "student", "--no-add-gaussian"])

    assert module.hypothesis_list(args) == ["student"]


def test_hyperbolic_likelihood_runs_hyperbolic_and_gaussian_by_default():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "hyperbolic"])

    assert args.num_frequency_bands is None
    assert module.hypothesis_list(args) == ["hyperbolic", "gaussian"]


def test_hyperbolic_likelihood_can_disable_default_gaussian():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "hyperbolic", "--no-add-gaussian"])

    assert module.hypothesis_list(args) == ["hyperbolic"]


def test_noise_only_inference_requires_student_likelihood():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--noise-only-inference"])

    with pytest.raises(
        ValueError,
        match="--noise-only-inference requires --likelihood student or hyperbolic",
    ):
        module.hypothesis_list(args)


def test_gaussian_likelihood_rejects_explicit_num_frequency_bands():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "gaussian", "--num-frequency-bands", "2"])

    with pytest.raises(
        ValueError,
        match="--likelihood gaussian cannot be combined with --num-frequency-bands",
    ):
        module.hypothesis_list(args)


def test_add_gaussian_requires_student_likelihood():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--add-gaussian"])

    with pytest.raises(
        ValueError,
        match="--add-gaussian requires --likelihood student or hyperbolic",
    ):
        module.hypothesis_list(args)


def test_student_likelihood_keeps_gaussian_companion_for_multiple_frequency_bands():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "student", "--num-frequency-bands", "4"])

    assert module.hypothesis_list(args) == ["student", "gaussian"]


def test_hyperbolic_accepts_detector_dependent_noise():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(
        ["--likelihood", "hyperbolic", "--detector-dependent-noise"]
    )

    assert module.hypothesis_list(args) == ["hyperbolic", "gaussian"]


def test_main_allows_gaussian_default_band_count_with_dry_run(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "gaussian",
            "--num-sine-gaussians",
            "1",
            "--sine-gaussian-mode",
            "coherent",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--home-dir",
            str(tmp_path),
            "--require-epnfs",
        ],
    )

    assert module.main() == 0
    ini_paths = list(ini_dir.glob("*.ini"))
    assert ini_paths
    assert list(prior_dir.glob("*.prior"))

    ini_text = ini_paths[0].read_text(encoding="utf-8")
    assert (
        "frequency-domain-source-model="
        "bilby.gw.source.cbc_plus_sine_gaussians\n"
    ) in ini_text
    assert "distance-marginalization=False\n" in ini_text
    assert "request-cpus=28\n" in ini_text
    assert "request-memory=24.0\n" in ini_text
    assert "request-memory-generation=24.0\n" in ini_text
    assert "transfer-files=True\n" in ini_text
    assert "osg=True\n" in ini_text
    assert "desired-sites=None\n" in ini_text
    assert (
        "generation-function="
        "bilby.gw.conversion.generate_all_cbc_plus_sine_gaussian_parameters\n"
    ) in ini_text
    assert "queue=EPNFS\n" in ini_text

    ini_settings = dict(
        line.split("=", maxsplit=1) for line in ini_text.splitlines() if "=" in line
    )
    outdir = Path(ini_settings["outdir"])
    assert outdir.parent == (
        tmp_path / "public_html" / "GW231123" / "gaussian" / "Runs"
    )
    assert Path(ini_settings["webdir"]) == outdir / "web"

    prior_text = next(prior_dir.glob("*.prior")).read_text(encoding="utf-8")
    assert (
        "luminosity_distance =  bilby.gw.prior.UniformSourceFrame("
        "name='luminosity_distance', cosmology=Planck15, maximum=15000.0, "
        "minimum=10, unit='Mpc')"
    ) in prior_text
    assert (
        "sine_gaussian_0_time_offset = Uniform("
        "name='sine_gaussian_0_time_offset', minimum=-0.15, maximum=0.15)"
    ) in prior_text


def test_render_prior_qualifies_gw_prior_classes():
    module = load_submit_runs_real_data_module()
    prior_template = "\n".join(
        [
            "chirp_mass = UniformInComponentsChirpMass(minimum=10, maximum=20)",
            "mass_ratio = UniformInComponentsMassRatio(minimum=0.1, maximum=1)",
            "__NU_PRIORS__",
        ]
    )
    rendered = module.render_prior(
        prior_template,
        1,
        hypothesis="gaussian",
        template_settings={
            "minimum_frequency": 20,
            "maximum_frequency": 1024,
        },
        sine_gaussian_config=type(
            "Config",
            (),
            dict(enabled=False, total_components=0),
        )(),
    )
    assert (
        "chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(" in rendered
    )
    assert (
        "mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(" in rendered
    )


def test_render_ini_writes_maxmcmc_override():
    module = load_submit_runs_real_data_module()
    ini_template = "\n".join(
        [
            "accounting-user=old",
            "container=None",
            "queue=None",
            "create-summary=False",
            "environment-variables={}",
            "summarypages-arguments=None",
            "sampler-kwargs={'nlive': 2000, 'maxmcmc': 5000}",
            "likelihood-type=old",
            "extra-likelihood-kwargs=old",
            "waveform-approximant=old",
            "minimum-frequency=20",
            "",
        ]
    )
    template_settings = dict(
        minimum_frequency={"H1": 20.0, "waveform": 10.0},
        maximum_frequency=448.0,
        reference_frequency=10.0,
        waveform_approximant="NRSur7dq4",
        spline_calibration_envelope_dict=None,
        psd_dict=None,
        sampler_kwargs={"nlive": 2000, "maxmcmc": 5000},
    )

    rendered = module.render_ini(
        ini_template,
        hypothesis="gaussian",
        label="label",
        outdir="out",
        webdir="web",
        prior_file=Path("prior.prior"),
        band_count=1,
        detector_dependent_noise=False,
        working_directory=Path("working"),
        accounting_user="acct",
        container_image=None,
        require_epnfs=False,
        maxmcmc=10000,
        template_settings=template_settings,
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


@pytest.mark.parametrize("likelihood", ["gaussian", "student", "hyperbolic"])
def test_main_creates_summarypages_without_recalib_parameters_for_all_likelihoods(
    monkeypatch, tmp_path, likelihood
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    argv = [
        str(SCRIPT_PATH),
        "--event",
        "GW231123",
        "--likelihood",
        likelihood,
        "--dry-run",
        "--ini-dir",
        str(ini_dir),
        "--prior-dir",
        str(prior_dir),
        "--outdir-base",
        str(outdir_base),
        "--webdir-base",
        str(webdir_base),
    ]
    if likelihood in module.HEAVY_TAILED_LIKELIHOODS:
        argv.append("--no-add-gaussian")
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    assert module.main() == 0

    ini_paths = list(ini_dir.glob("*.ini"))
    assert len(ini_paths) == 1
    ini_text = ini_paths[0].read_text(encoding="utf-8")
    summary_line = next(
        line for line in ini_text.splitlines()
        if line.startswith("summarypages-arguments=")
    )
    summary_arguments = ast.literal_eval(summary_line.split("=", 1)[1])
    assert "create-summary=True\n" in ini_text
    assert summary_arguments["ignore_parameters"] == ["recalib*"]
    assert summary_arguments["disable_expert"] is True
    assert summary_arguments["f_ref"] == 10.0
    assert summary_arguments["f_low"] == 20
    assert summary_arguments["f_start"] == 10.0
    assert summary_arguments["f_final"] == 448.0
    assert summary_arguments["approximant"] == ["NRSur7dq4"]
    assert summary_arguments["calibration"] == {
        "H1": "/home/pe.o4/GWTC4-fogg/project/working/S231123cg/get-data/calibration/H1.txt",
        "L1": "/home/pe.o4/GWTC4-fogg/project/working/S231123cg/get-data/calibration/L1.txt",
    }


def test_main_disable_calibration_applies_to_hyperbolic_and_gaussian(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW150914",
            "--likelihood",
            "hyperbolic",
            "--num-frequency-bands",
            "1",
            "--disable-calibration",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--home-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    ini_paths = sorted(ini_dir.glob("*.ini"))
    assert [path.name for path in ini_paths] == [
        "GW150914_IGWN_C01_IMRPhenomXPHM_gaussian.ini",
        "GW150914_IGWN_C01_IMRPhenomXPHM_hyperbolic_N1.ini",
    ]
    for ini_path in ini_paths:
        ini_text = ini_path.read_text(encoding="utf-8")
        assert "calibration-model=None\n" in ini_text
        assert "spline-calibration-envelope-dict=None\n" in ini_text
        assert "calibration-marginalization=False\n" in ini_text
        assert "calibration-lookup-table=None\n" in ini_text
        summary_line = next(
            line for line in ini_text.splitlines()
            if line.startswith("summarypages-arguments=")
        )
        summary_arguments = ast.literal_eval(summary_line.split("=", 1)[1])
        assert "calibration" not in summary_arguments


def test_main_student_range_writes_single_gaussian_run_without_n_suffix(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "student",
            "--range",
            "--num-frequency-bands",
            "4",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    gaussian_ini_paths = sorted(ini_dir.glob("*_gaussian*.ini"))
    student_ini_paths = sorted(ini_dir.glob("*_t_student*.ini"))

    assert [path.name for path in gaussian_ini_paths] == ["GW231123_gaussian.ini"]
    assert len(student_ini_paths) == 4
    assert sorted(path.name for path in student_ini_paths) == [
        "GW231123_t_student_N1.ini",
        "GW231123_t_student_N2.ini",
        "GW231123_t_student_N3.ini",
        "GW231123_t_student_N4.ini",
    ]

    gaussian_ini = gaussian_ini_paths[0].read_text(encoding="utf-8")
    assert "gaussian_N" not in gaussian_ini


def test_main_hyperbolic_range_writes_single_gaussian_run_without_n_suffix(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "hyperbolic",
            "--range",
            "--num-frequency-bands",
            "4",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    gaussian_ini_paths = sorted(ini_dir.glob("*_gaussian*.ini"))
    hyperbolic_ini_paths = sorted(ini_dir.glob("*_hyperbolic*.ini"))

    assert [path.name for path in gaussian_ini_paths] == ["GW231123_gaussian.ini"]
    assert len(hyperbolic_ini_paths) == 4
    assert sorted(path.name for path in hyperbolic_ini_paths) == [
        "GW231123_hyperbolic_N1.ini",
        "GW231123_hyperbolic_N2.ini",
        "GW231123_hyperbolic_N3.ini",
        "GW231123_hyperbolic_N4.ini",
    ]

    hyperbolic_ini = hyperbolic_ini_paths[-1].read_text(encoding="utf-8")
    gaussian_ini = gaussian_ini_paths[0].read_text(encoding="utf-8")
    assert (
        "likelihood-type=bilby.gw.likelihood.HyperbolicGravitationalWaveTransient\n"
        in hyperbolic_ini
    )
    assert "'infer_alpha': True" in hyperbolic_ini
    assert "'infer_delta': True" in hyperbolic_ini
    assert "'num_frequency_bands': 4" in hyperbolic_ini
    assert "gaussian_N" not in gaussian_ini


def test_main_student_detector_dependent_noise_writes_detector_specific_priors(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "student",
            "--num-frequency-bands",
            "2",
            "--detector-dependent-noise",
            "--no-add-gaussian",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    ini_path = ini_dir / "GW231123_t_student_detector_dependent_noise_N2.ini"
    prior_path = prior_dir / "GW231123_detector_dependent_noise_N2.prior"
    student_ini = ini_path.read_text(encoding="utf-8")
    student_prior = prior_path.read_text(encoding="utf-8")

    assert "'detector_dependent_noise': True" in student_ini
    assert "nu_H1_1 = Uniform(name='nu_H1_1', minimum=2.1, maximum=1000)" in student_prior
    assert "nu_L1_2 = Uniform(name='nu_L1_2', minimum=2.1, maximum=1000)" in student_prior


def test_main_hyperbolic_detector_dependent_noise_writes_detector_specific_priors(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "hyperbolic",
            "--num-frequency-bands",
            "2",
            "--detector-dependent-noise",
            "--no-add-gaussian",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    ini_path = ini_dir / "GW231123_hyperbolic_detector_dependent_noise_N2.ini"
    prior_path = prior_dir / "GW231123_hyperbolic_detector_dependent_noise_N2.prior"
    hyperbolic_ini = ini_path.read_text(encoding="utf-8")
    hyperbolic_prior = prior_path.read_text(encoding="utf-8")

    assert "'detector_dependent_noise': True" in hyperbolic_ini
    assert (
        "alpha_H1_1 = Uniform(name='alpha_H1_1', minimum=1e-06, maximum=30.0)"
        in hyperbolic_prior
    )
    assert (
        "delta_L1_2 = Uniform(name='delta_L1_2', minimum=1e-06, maximum=30.0)"
        in hyperbolic_prior
    )


def test_main_noise_only_inference_writes_zero_waveform_student_run(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "student",
            "--noise-only-inference",
            "--num-frequency-bands",
            "2",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    student_ini_paths = sorted(ini_dir.glob("*_t_student*.ini"))
    gaussian_ini_paths = sorted(ini_dir.glob("*_gaussian*.ini"))
    prior_paths = sorted(prior_dir.glob("*.prior"))

    assert len(student_ini_paths) == 1
    assert gaussian_ini_paths == []
    assert len(prior_paths) == 1

    student_ini = student_ini_paths[0].read_text(encoding="utf-8")
    prior_text = prior_paths[0].read_text(encoding="utf-8")

    assert "default-prior=bilby.core.prior.PriorDict\n" in student_ini
    assert "frequency-domain-source-model=bilby.gw.source.zero_waveform\n" in student_ini
    assert "create-summary=False\n" in student_ini
    assert "calibration-model=None\n" in student_ini
    assert "spline-calibration-envelope-dict=None\n" in student_ini
    assert "jitter-time=False\n" in student_ini
    assert "chirp_mass =" not in prior_text
    assert "luminosity_distance =" not in prior_text
    assert "nu_1 = Uniform(name='nu_1', minimum=2.1, maximum=1000)" in prior_text
    assert "nu_2 = Uniform(name='nu_2', minimum=2.1, maximum=1000)" in prior_text
    assert "L1_time = DeltaFunction(name='L1_time'" in prior_text


def test_main_noise_only_inference_writes_zero_waveform_hyperbolic_run(
    monkeypatch, tmp_path
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"
    prior_dir = tmp_path / "prior"
    outdir_base = tmp_path / "out"
    webdir_base = tmp_path / "web"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW231123",
            "--likelihood",
            "hyperbolic",
            "--noise-only-inference",
            "--num-frequency-bands",
            "2",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(prior_dir),
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
        ],
    )

    assert module.main() == 0

    hyperbolic_ini_paths = sorted(ini_dir.glob("*_hyperbolic*.ini"))
    gaussian_ini_paths = sorted(ini_dir.glob("*_gaussian*.ini"))
    prior_paths = sorted(prior_dir.glob("*_hyperbolic*.prior"))

    assert len(hyperbolic_ini_paths) == 1
    assert gaussian_ini_paths == []
    assert len(prior_paths) == 1

    hyperbolic_ini = hyperbolic_ini_paths[0].read_text(encoding="utf-8")
    prior_text = prior_paths[0].read_text(encoding="utf-8")

    assert "default-prior=bilby.core.prior.PriorDict\n" in hyperbolic_ini
    assert "frequency-domain-source-model=bilby.gw.source.zero_waveform\n" in hyperbolic_ini
    assert "create-summary=False\n" in hyperbolic_ini
    assert "calibration-model=None\n" in hyperbolic_ini
    assert "spline-calibration-envelope-dict=None\n" in hyperbolic_ini
    assert "jitter-time=False\n" in hyperbolic_ini
    assert (
        "likelihood-type=bilby.gw.likelihood.HyperbolicGravitationalWaveTransient\n"
        in hyperbolic_ini
    )
    assert "chirp_mass =" not in prior_text
    assert "luminosity_distance =" not in prior_text
    assert "alpha_1 = Uniform(name='alpha_1', minimum=1e-06, maximum=30.0)" in prior_text
    assert "alpha_2 = Uniform(name='alpha_2', minimum=1e-06, maximum=30.0)" in prior_text
    assert "delta_1 = Uniform(name='delta_1', minimum=1e-06, maximum=30.0)" in prior_text
    assert "delta_2 = Uniform(name='delta_2', minimum=1e-06, maximum=30.0)" in prior_text
    assert "L1_time = DeltaFunction(name='L1_time'" in prior_text
