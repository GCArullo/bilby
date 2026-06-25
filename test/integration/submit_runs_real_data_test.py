import importlib.util
import ast
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
    assert args.require_epnfs is False
    assert module.hypothesis_list(args) == ["gaussian"]


def test_accounting_user_defaults_to_home_basename(monkeypatch):
    monkeypatch.setenv("HOME", "/home/name.surname")

    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)
    args = parser.parse_args([])

    assert module.DEFAULT_ACCOUNTING_USER == "name.surname"
    assert args.accounting_user == "name.surname"


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
        match="--add-gaussian requires --likelihood student",
    ):
        module.hypothesis_list(args)


def test_student_likelihood_keeps_gaussian_companion_for_multiple_frequency_bands():
    module = load_submit_runs_real_data_module()
    parser = module.build_argument_parser(SCRIPT_PATH.parent)

    args = parser.parse_args(["--likelihood", "student", "--num-frequency-bands", "4"])

    assert module.hypothesis_list(args) == ["student", "gaussian"]


def test_main_allows_gaussian_default_band_count_with_dry_run(monkeypatch, tmp_path):
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
            "--outdir-base",
            str(outdir_base),
            "--webdir-base",
            str(webdir_base),
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
    assert "generation-function=None\n" in ini_text
    assert "queue=EPNFS\n" in ini_text

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


def test_main_creates_summarypages_without_recalib_parameters_by_default(
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
            "gaussian",
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

    ini_text = next(ini_dir.glob("*.ini")).read_text(encoding="utf-8")
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
