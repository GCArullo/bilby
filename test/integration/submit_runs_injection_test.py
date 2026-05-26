import importlib.util
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
