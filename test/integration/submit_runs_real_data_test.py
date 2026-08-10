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
    assert args.joint is False
    assert args.high_mass_catalog is False
    assert module.hypothesis_list(args) == ["gaussian"]


@pytest.mark.parametrize(
    ("catalog", "count", "examples"),
    [
        ("GWTC-2.1", 54, {"GW150914", "GW190425_081805", "GW190930_133541"}),
        ("GWTC-3", 36, {"GW191103_012549", "GW200115_042309", "GW200322_091133"}),
        ("GWTC-4", 86, {"GW230529_181500", "GW231028_153006", "GW240109_050431"}),
        ("GWTC-5", 104, {"GW240413_022019", "GW240925_005809", "GW250119_190238"}),
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


def test_mass_classification_covers_catalog_manifests():
    module = load_submit_runs_real_data_module()
    classification_path = (
        SCRIPT_PATH.parent
        / module.CATALOG_CONFIGS_DIR
        / module.MASS_CLASSIFICATION_FILE
    )
    rows = []
    for line in classification_path.read_text(encoding="utf-8").splitlines():
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) == 4 and columns[-1] in {"high", "low"}:
            rows.append(columns)

    manifest_events = set()
    for catalog in ("GWTC-2.1", "GWTC-3", "GWTC-4", "GWTC-5"):
        manifest_path = (
            SCRIPT_PATH.parent
            / module.CATALOG_CONFIGS_DIR
            / catalog
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_events.update(item["event"] for item in manifest["events"])

    assert len(rows) == 280
    assert {row[1] for row in rows} == manifest_events
    assert all(
        (float(row[2]) > 50) == (row[-1] == "high")
        for row in rows
    )
    assert tuple(row[1] for row in rows if row[-1] == "high") == (
        module.HIGH_MASS_EVENTS
    )
    assert len(module.HIGH_MASS_EVENTS) == 174


def test_high_mass_catalog_flag_runs_every_documented_event(monkeypatch):
    module = load_submit_runs_real_data_module()
    commands = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--high-mass-catalog",
            "--likelihood",
            "hyperbolic",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    assert module.main() == 0
    assert [command[-1] for command in commands] == list(
        module.HIGH_MASS_EVENTS
    )
    assert all(command[-2] == "--event" for command in commands)
    assert all("--high-mass-catalog" not in command for command in commands)


def test_special_event_templates_and_priors_are_grouped_together():
    module = load_submit_runs_real_data_module()
    special_dir = SCRIPT_PATH.parent / module.SPECIAL_EVENTS_CONFIGS_DIR

    assert module.EVENT_DEFAULTS["GW241127_SEOB"].working_directory == str(
        Path.home() / "LVK_posteriors/GW241127_061008"
    )
    assert module.EVENT_DEFAULTS["GW241127_pSEOB"].working_directory == str(
        Path.home() / "LVK_posteriors/GW241127_061008"
    )
    assert {path.name for path in (special_dir / "templates").iterdir()} == {
        "GW150914_t_student_igwn_template.ini",
        "GW150914_welch_template.ini",
        "GW190521_030229_LVK_NRSur7dq4.ini",
        "GW191109_010717_no_glitch_subtraction_template.ini",
        "GW200129_065458_Hannam_NRSur7dq4.ini",
        "GW230814_t_student_pSEOB_template.ini",
        "GW231123_t_student_template.ini",
        "GW241127_t_student_SEOB_template.ini",
        "GW241127_t_student_pSEOB_template.ini",
    }
    assert {path.name for path in (special_dir / "priors").iterdir()} == {
        "GW150914_igwn_template.prior",
        "GW190521_030229_LVK_NRSur7dq4.prior",
        "GW200129_065458_Hannam_NRSur7dq4.prior",
        "GW230814_gr_template.prior",
        "GW230814_template.prior",
        "GW231123_template.prior",
        "GW241127_SEOB_template.prior",
        "GW241127_pSEOB_template.prior",
    }
    assert {path.name for path in (special_dir / "source_configs").iterdir()} == {
        "GW190521_030229_LVK_NRSur7dq4.ini",
    }


@pytest.mark.parametrize(
    ("event", "source_model", "deviation_prior", "model_directory"),
    [
        (
            "GW241127_SEOB",
            "bilby.gw.source.gwsignal_binary_black_hole",
            None,
            "SEOB",
        ),
        (
            "GW241127_pSEOB",
            "bilby_tgr.pseob.source.gwsignal_binary_black_hole",
            "domega220 = Uniform(name='domega220', minimum=-0.8, maximum=2.0)",
            "pSEOB",
        ),
    ],
)
def test_gw241127_seob_profiles_generate_released_setup(
    monkeypatch,
    tmp_path,
    event,
    source_model,
    deviation_prior,
    model_directory,
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
            event,
            "--likelihood",
            "gaussian",
            "--no-container",
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

    assert "detectors=['H1', 'L1', 'V1']\n" in ini_text
    assert "trigger-time=1416723026.229858\n" in ini_text
    assert "waveform-approximant=SEOBNRv5PHM\n" in ini_text
    assert f"frequency-domain-source-model={source_model}\n" in ini_text
    assert (
        "minimum-frequency={'H1': 20, 'L1': 30, 'V1': 20, 'waveform': 13.33}\n"
        in ini_text
    )
    assert "maximum-frequency={'H1': 448, 'L1': 448, 'V1': 448}\n" in ini_text
    for setting in (
        "data-dict",
        "psd-dict",
        "spline-calibration-envelope-dict",
    ):
        assert "'L1': '" in next(
            line for line in ini_text.splitlines()
            if line.startswith(f"{setting}=")
        )
    assert (
        f"outdir={tmp_path}/public_html/GWTC_parametric_noise/Runs/"
        f"GW241127_061008/{model_directory}/"
        "gaussian_detector_independent_noise_N1\n"
    ) in ini_text
    if deviation_prior is None:
        assert "domega220" not in prior_text
        assert "dtau220" not in prior_text
    else:
        assert deviation_prior in prior_text
        assert (
            "dtau220 = Uniform(name='dtau220', minimum=-0.8, maximum=2.0)"
            in prior_text
        )


@pytest.mark.parametrize(
    ("minimum_frequency", "suffix"),
    [(20, "fmin20"), (30, "fmin30")],
)
def test_gw241127_minimum_frequency_override_is_collision_safe(
    monkeypatch,
    tmp_path,
    minimum_frequency,
    suffix,
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW241127_pSEOB",
            "--likelihood",
            "gaussian-parametric",
            "--no-add-gaussian",
            "--minimum-frequency",
            str(minimum_frequency),
            "--no-container",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(tmp_path / "prior"),
            "--home-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    ini_path = next(ini_dir.glob("*.ini"))
    ini_text = ini_path.read_text(encoding="utf-8")

    assert (
        ini_path.name
        == f"GW241127_061008_pSEOB_gaussian_parametric_N1_{suffix}.ini"
    )
    assert (
        f"minimum-frequency={{'H1': {minimum_frequency}.0, "
        f"'L1': {minimum_frequency}.0, 'V1': {minimum_frequency}.0, "
        "'waveform': 13.33}\n"
    ) in ini_text
    assert "distance-marginalization=False\n" in ini_text
    assert (
        f"/pSEOB/gaussian-parametric_detector_independent_noise_N1_{suffix}\n"
        in ini_text
    )


@pytest.mark.parametrize(
    "event",
    ["GW230814", "GW241127_SEOB", "GW241127_pSEOB"],
)
@pytest.mark.parametrize("likelihood", ["gaussian", "student", "hyperbolic"])
def test_special_event_submission_disables_distance_marginalization(
    monkeypatch,
    tmp_path,
    event,
    likelihood,
):
    module = load_submit_runs_real_data_module()
    ini_dir = tmp_path / "ini"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            event,
            "--likelihood",
            likelihood,
            "--no-add-gaussian",
            "--no-container",
            "--dry-run",
            "--ini-dir",
            str(ini_dir),
            "--prior-dir",
            str(tmp_path / "prior"),
            "--home-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0
    ini_text = next(ini_dir.glob("*.ini")).read_text(encoding="utf-8")
    assert "distance-marginalization=False\n" in ini_text


def test_gw190521_lvk_nrsur_profile_generates_released_setup(
    monkeypatch,
    tmp_path,
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
            "GW190521_030229_LVK_NRSur7dq4",
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
    assert "duration=8.0\n" in ini_text
    assert "sampling-frequency=1024.0\n" in ini_text
    assert "reference-frequency=11.0\n" in ini_text
    assert (
        "minimum-frequency={'H1': 11.0, 'L1': 11.0, 'V1': 11.0, 'waveform': 11.0}\n"
    ) in ini_text
    assert (
        "additional-transfer-paths="
        "[/home/pe.o4/GWTC4-fogg/NRSur7dq4_v1.0.h5]\n"
    ) in ini_text
    assert "minimum=70.0, maximum=150.0" in prior_text
    assert "minimum=0.17, maximum=1.0" in prior_text
    assert "total_mass = Constraint(name='total_mass', minimum=200.0" in prior_text
    assert (
        f"outdir={tmp_path}/public_html/GWTC_parametric_noise/Runs/"
        "GW190521_030229/gaussian_detector_independent_noise_N1\n"
    ) in ini_text


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
    assert (
        f"outdir={tmp_path}/public_html/GW200129_065458_Hannam/Runs/"
        "gaussian_detector_independent_noise_N1\n"
    ) in ini_text


def test_default_output_bases_are_under_public_event_directory(tmp_path):
    module = load_submit_runs_real_data_module()

    outdir_base, webdir_base = module.default_output_bases(
        tmp_path,
        "GW231123/Runs",
    )

    event_dir = tmp_path / "public_html" / "GW231123"
    assert Path(outdir_base) == event_dir / "Runs"
    assert Path(webdir_base) == event_dir / "Runs"
    assert Path(webdir_base) / "run-name" / "web" == (
        Path(outdir_base) / "run-name" / "web"
    )


@pytest.mark.parametrize(
    "likelihood",
    ["student", "hyperbolic", "gaussian-parametric", "gaussian"],
)
def test_all_likelihoods_share_the_event_run_directory(
    monkeypatch,
    tmp_path,
    likelihood,
):
    module = load_submit_runs_real_data_module()
    expected = tmp_path / "public_html" / "GW230814" / "Runs"
    ini_dir = tmp_path / "ini"
    arguments = [
        str(SCRIPT_PATH),
        "--event",
        "GW230814",
        "--likelihood",
        likelihood,
        "--dry-run",
        "--ini-dir",
        str(ini_dir),
        "--prior-dir",
        str(tmp_path / "prior"),
        "--home-dir",
        str(tmp_path),
    ]
    if likelihood in module.PARAMETRIC_NOISE_LIKELIHOODS:
        arguments.append("--no-add-gaussian")
    monkeypatch.setattr(sys, "argv", arguments)

    assert module.main() == 0
    for ini_path in ini_dir.glob("*.ini"):
        ini_text = ini_path.read_text(encoding="utf-8")
        assert f"outdir={expected}/" in ini_text
        assert f"webdir={expected}/" in ini_text


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
        module,
        "preflight_local_data",
        lambda *args, **kwargs: None,
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


def test_preflight_downloads_missing_catalog_data(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    glitch_directory = tmp_path / "glitch_data"
    missing_frame = glitch_directory / "frame.gwf"
    psd = tmp_path / "psd.dat"
    psd.write_text("psd", encoding="utf-8")
    downloader = tmp_path / "download_glitch_data.py"
    downloader.write_text("", encoding="utf-8")
    calls = []

    def download(command, **kwargs):
        calls.append((command, kwargs))
        glitch_directory.mkdir()
        missing_frame.write_text("frame", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", download)

    module.preflight_local_data(
        {
            "data_dict": {"H1": str(missing_frame)},
            "psd_dict": {"H1": str(psd)},
            "spline_calibration_envelope_dict": None,
        },
        event="GW191109_010717",
        working_directory=tmp_path,
    )

    assert calls == [
        (
            [
                sys.executable,
                str(downloader),
                "--event",
                "GW191109_010717",
            ],
            {"check": True, "cwd": tmp_path},
        )
    ]


def test_preflight_reports_local_data_still_missing(tmp_path):
    module = load_submit_runs_real_data_module()
    missing_psd = tmp_path / "psd.dat"

    with pytest.raises(
        FileNotFoundError,
        match=f"Missing required local data for GW231123:\\n  - {missing_psd}",
    ):
        module.preflight_local_data(
            {
                "data_dict": None,
                "psd_dict": {"H1": str(missing_psd)},
                "spline_calibration_envelope_dict": None,
            },
            event="GW231123",
            working_directory=tmp_path,
        )


def test_existing_gaussian_companion_is_not_resubmitted(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    run_base = tmp_path / "public_html" / "GW230814" / "Runs"
    gaussian_run = run_base / "gaussian_detector_independent_noise_N1"
    gaussian_run.mkdir(parents=True)
    submitted = []

    monkeypatch.setattr(
        module,
        "submit_run",
        lambda ini_path, **kwargs: submitted.append(ini_path),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--event",
            "GW230814",
            "--likelihood",
            "hyperbolic",
            "--home-dir",
            str(tmp_path),
            "--ini-dir",
            str(tmp_path / "ini"),
            "--prior-dir",
            str(tmp_path / "prior"),
        ],
    )

    assert module.main() == 0
    assert len(submitted) == 1
    assert "hyperbolic" in submitted[0].name


def test_submit_run_preflights_local_inputs(monkeypatch, tmp_path):
    module = load_submit_runs_real_data_module()
    ini_path = tmp_path / "run.ini"
    ini_path.write_text(
        f"psd-dict={{ H1:{tmp_path / 'missing_psd.dat'} }}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("bilby_pipe should not be called"),
    )

    with pytest.raises(FileNotFoundError, match="missing_psd.dat"):
        module.submit_run(ini_path, submit_directory=tmp_path)


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


def test_gaussian_parametric_runs_with_gaussian_companion_by_default():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(
        [
            "--likelihood",
            "gaussian-parametric",
            "--num-frequency-bands",
            "4",
        ]
    )

    assert module.hypothesis_list(args) == ["gaussian-parametric", "gaussian"]


def test_noise_only_inference_requires_parametric_noise_likelihood():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--noise-only-inference"])

    with pytest.raises(
        ValueError,
        match="--noise-only-inference requires a parametric-noise likelihood",
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


def test_add_gaussian_requires_parametric_noise_likelihood():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--add-gaussian"])

    with pytest.raises(
        ValueError,
        match="--add-gaussian requires a parametric-noise likelihood",
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


def test_joint_requires_detector_independent_heavy_tailed_likelihood():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "gaussian", "--joint"])
    with pytest.raises(
        ValueError,
        match="--joint requires --likelihood student or hyperbolic",
    ):
        module.hypothesis_list(args)

    args = parser.parse_args(
        [
            "--likelihood",
            "hyperbolic",
            "--joint",
            "--detector-dependent-noise",
        ]
    )
    with pytest.raises(
        ValueError,
        match="--joint cannot be combined with --detector-dependent-noise",
    ):
        module.hypothesis_list(args)


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
    assert "request-cpus=16\n" in ini_text
    assert "request-memory=24.0\n" in ini_text
    assert "request-memory-generation=24.0\n" in ini_text
    assert "transfer-files=True\n" in ini_text
    assert "osg=True\n" in ini_text
    assert "desired-sites=None\n" in ini_text
    assert (
        "generation-function="
        "bilby.gw.conversion.generate_all_cbc_plus_sine_gaussian_parameters\n"
    ) in ini_text
    assert "queue=None\n" in ini_text

    ini_settings = dict(
        line.split("=", maxsplit=1) for line in ini_text.splitlines() if "=" in line
    )
    outdir = Path(ini_settings["outdir"])
    assert outdir.parent == tmp_path / "public_html" / "GW231123" / "Runs"
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
            "distance-marginalization=True",
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


def test_render_ini_writes_per_run_submission_controls():
    module = load_submit_runs_real_data_module()
    ini_template = "\n".join(
        [
            "accounting-user=old",
            "container=None",
            "queue=None",
            "detectors=['H1', 'L1']",
            "reference-frame=H1L1",
            "time-reference=H1L1",
            "create-summary=False",
            "environment-variables={}",
            "summarypages-arguments=None",
            "sampler-kwargs={'nlive': 2000}",
            "likelihood-type=old",
            "extra-likelihood-kwargs=old",
            "distance-marginalization=True",
            "waveform-approximant=old",
            "waveform-arguments-dict=None",
            "minimum-frequency=20",
            "",
        ]
    )
    template_settings = dict(
        minimum_frequency={"H1": 20.0, "waveform": 10.0},
        maximum_frequency=448.0,
        reference_frequency=10.0,
        waveform_approximant="IMRPhenomXPHM",
        frequency_domain_source_model="bilby.gw.source.lal_binary_black_hole",
        spline_calibration_envelope_dict=None,
        psd_dict={"L1": "__WORKING_DIRECTORY__/psds/L1.dat"},
        sampler_kwargs={"nlive": 2000},
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
        detectors=["L1"],
        working_directory=Path("/base/dir"),
        accounting_user="acct",
        container_image=None,
        require_epnfs=False,
        condor_job_priority=10,
        maxmcmc=None,
        waveform_arguments={"PhenomXPrecVersion": 320},
        template_settings=template_settings,
        sine_gaussian_config=type(
            "Config",
            (),
            dict(enabled=False, total_components=0),
        )(),
    )

    lines = rendered.splitlines()
    # The template omits condor-job-priority, so it must be appended.
    assert "condor-job-priority=10" in lines
    assert "detectors=['L1']" in lines
    # One detector cannot triangulate, so the zenith/azimuth frame is dropped.
    assert "reference-frame=sky" in lines
    assert "time-reference=L1" in lines
    assert "waveform-arguments-dict={'PhenomXPrecVersion': 320}" in lines
    assert "__WORKING_DIRECTORY__" not in rendered


def test_resolve_spin_taylor_approximant_maps_to_prec_version():
    module = load_submit_runs_real_data_module()

    assert module.resolve_spin_taylor_approximant("IMRPhenomXPHM") == (
        "IMRPhenomXPHM",
        None,
    )
    assert module.resolve_spin_taylor_approximant("IMRPhenomXPHM_SpinTaylor") == (
        "IMRPhenomXPHM",
        {"PhenomXPrecVersion": 320},
    )


@pytest.mark.parametrize(
    "likelihood",
    ["gaussian", "student", "hyperbolic", "gaussian-parametric"],
)
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
    if likelihood in module.PARAMETRIC_NOISE_LIKELIHOODS:
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
    # Naming the page after the run directory keeps pesummary's
    # '<label>_<label>_<parameter>.html' names inside the 255 byte limit.
    outdir_line = next(
        line for line in ini_text.splitlines() if line.startswith("outdir=")
    )
    assert summary_arguments["labels"] == [Path(outdir_line.split("=", 1)[1]).name]
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
    student_ini = student_ini_paths[-1].read_text(encoding="utf-8")
    assert "'joint': False" in student_ini
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
    assert "'joint': False" in hyperbolic_ini
    assert "gaussian_N" not in gaussian_ini


@pytest.mark.parametrize(
    ("likelihood", "filename"),
    [
        ("student", "GW231123_t_student_joint_N1.ini"),
        ("hyperbolic", "GW231123_hyperbolic_joint_N1.ini"),
    ],
)
def test_main_joint_opt_in_writes_distinct_run(
    monkeypatch, tmp_path, likelihood, filename
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
            likelihood,
            "--joint",
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

    ini_path = ini_dir / filename
    joint_ini = ini_path.read_text(encoding="utf-8")
    assert "'joint': True" in joint_ini
    assert (
        f"{likelihood}_detector_independent_noise_joint_N1"
        in joint_ini
    )


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


def test_main_gaussian_parametric_range_writes_shared_psd_scale_priors(
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
            "GW231123",
            "--likelihood",
            "gaussian-parametric",
            "--range",
            "--num-frequency-bands",
            "2",
            "--no-add-gaussian",
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

    assert sorted(path.name for path in ini_dir.glob("*.ini")) == [
        "GW231123_gaussian_parametric_N1.ini",
        "GW231123_gaussian_parametric_N2.ini",
    ]
    ini_text = (
        ini_dir / "GW231123_gaussian_parametric_N2.ini"
    ).read_text(encoding="utf-8")
    prior_text = (
        prior_dir / "GW231123_gaussian_parametric_N2.prior"
    ).read_text(encoding="utf-8")

    assert (
        "likelihood-type="
        "bilby.gw.likelihood.GaussianParametricGravitationalWaveTransient\n"
        in ini_text
    )
    assert "'num_psd_frequency_bands': 2" in ini_text
    assert "'detector_dependent_noise': False" in ini_text
    assert (
        "log_psd_scale_1 = Uniform(name='log_psd_scale_1', "
        "minimum=-1.0, maximum=1.0)"
        in prior_text
    )
    assert (
        "log_psd_scale_2 = Uniform(name='log_psd_scale_2', "
        "minimum=-1.0, maximum=1.0)"
        in prior_text
    )
    assert "log_psd_scale_H1" not in prior_text


def test_main_gaussian_parametric_detector_dependent_noise_writes_priors(
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
            "GW231123",
            "--likelihood",
            "gaussian-parametric",
            "--num-frequency-bands",
            "2",
            "--detector-dependent-noise",
            "--no-add-gaussian",
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

    ini_path = (
        ini_dir
        / "GW231123_gaussian_parametric_detector_dependent_noise_N2.ini"
    )
    prior_path = (
        prior_dir
        / "GW231123_gaussian_parametric_detector_dependent_noise_N2.prior"
    )
    ini_text = ini_path.read_text(encoding="utf-8")
    prior_text = prior_path.read_text(encoding="utf-8")

    assert "'detector_dependent_noise': True" in ini_text
    assert (
        "log_psd_scale_H1_1 = Uniform(name='log_psd_scale_H1_1', "
        "minimum=-1.0, maximum=1.0)"
        in prior_text
    )
    assert (
        "log_psd_scale_L1_2 = Uniform(name='log_psd_scale_L1_2', "
        "minimum=-1.0, maximum=1.0)"
        in prior_text
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
