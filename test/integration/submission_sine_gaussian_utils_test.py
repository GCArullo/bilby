import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = (
    REPO_ROOT
    / "examples"
    / "gw_examples"
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "submission_sine_gaussian_utils.py"
)


def load_submission_sine_gaussian_utils_module():
    spec = importlib.util.spec_from_file_location(
        "submission_sine_gaussian_utils_test_module",
        UTILS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("submission_sine_gaussian_utils_test_module", None)


def replace_line(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Unable to find config key {key!r}")


def test_effective_nlive_uses_runbook_sine_gaussian_schedule():
    module = load_submission_sine_gaussian_utils_module()

    baseline = module.SineGaussianConfiguration()
    coherent_1 = module.SineGaussianConfiguration(total_components=1, mode="coherent")
    incoherent_1 = module.SineGaussianConfiguration(
        total_components=1,
        mode="incoherent",
        detector_counts=(("H1", 1),),
    )
    coherent_2 = module.SineGaussianConfiguration(total_components=2, mode="coherent")
    coherent_3 = module.SineGaussianConfiguration(total_components=3, mode="coherent")
    independent_1 = module.SineGaussianConfiguration(
        total_components=1,
        mode="coherent-independent",
    )
    independent_2 = module.SineGaussianConfiguration(
        total_components=2,
        mode="coherent-independent",
    )
    independent_3 = module.SineGaussianConfiguration(
        total_components=3,
        mode="coherent-independent",
    )
    incoherent_h1_l1 = module.SineGaussianConfiguration(
        total_components=2,
        mode="incoherent",
        detector_counts=(("H1", 1), ("L1", 1)),
    )

    assert module.effective_nlive(2000, baseline) == 2000
    assert module.effective_nlive(2000, coherent_1) == 2500
    assert module.effective_nlive(2000, incoherent_1) == 2500
    assert module.effective_nlive(2000, coherent_2) == 3000
    assert module.effective_nlive(2000, coherent_3) == 3000
    assert module.effective_nlive(2000, independent_1) == 3000
    assert module.effective_nlive(2000, independent_2) == 3500
    assert module.effective_nlive(2000, independent_3) == 3500
    assert module.effective_nlive(2000, incoherent_h1_l1) == 3000


def test_sine_gaussian_submission_settings_disable_distance_and_generation():
    module = load_submission_sine_gaussian_utils_module()
    config = module.SineGaussianConfiguration(total_components=1, mode="coherent")
    ini_text = "\n".join(
        [
            "frequency-domain-source-model=lal_binary_black_hole",
            "conversion-function=None",
            "generation-function=bilby.gw.conversion.generate_all_bbh_parameters",
            "distance-marginalization=True",
        ]
    ) + "\n"

    rendered = module.apply_sine_gaussian_waveform_settings(
        ini_text,
        config,
        replace_line=replace_line,
    )

    assert (
        "frequency-domain-source-model="
        "bilby.gw.source.cbc_plus_sine_gaussians\n"
    ) in rendered
    assert (
        "conversion-function="
        "bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters\n"
    ) in rendered
    assert (
        "generation-function="
        "bilby.gw.conversion.generate_all_cbc_plus_sine_gaussian_parameters\n"
    ) in rendered
    assert "distance-marginalization=False\n" in rendered


def test_sine_gaussian_distance_validation_rejects_enabled_marginalization():
    module = load_submission_sine_gaussian_utils_module()
    config = module.SineGaussianConfiguration(total_components=1, mode="coherent")

    with pytest.raises(
        ValueError,
        match="require distance-marginalization=False",
    ):
        module.validate_sine_gaussian_distance_marginalization(
            "distance-marginalization=True\n",
            config,
        )


def test_submission_preflight_accepts_local_and_remote_inputs(tmp_path):
    module = load_submission_sine_gaussian_utils_module()
    frame = tmp_path / "frame_data.gwf"
    psd = tmp_path / "psd.dat"
    calibration = tmp_path / "calibration.txt"
    transfer_directory = tmp_path / "staged_data"
    for path in (frame, psd, calibration):
        path.write_text("data", encoding="utf-8")
    transfer_directory.mkdir()

    module.validate_submission_local_paths(
        "\n".join(
            [
                f"data-dict={{ H1:file://localhost{frame}, "
                "L1:osdf:///igwn/frame.gwf }}",
                f"psd-dict={{ H1:{psd} }}",
                f"spline-calibration-envelope-dict={{ H1:{calibration} }}",
                f"additional-transfer-paths=[{transfer_directory}]",
            ]
        ),
        base_directory=tmp_path,
    )


def test_submission_preflight_reports_every_missing_input(tmp_path):
    module = load_submission_sine_gaussian_utils_module()
    paths = {
        "data-dict": tmp_path / "frame.gwf",
        "psd-dict": tmp_path / "psd.dat",
        "spline-calibration-envelope-dict": tmp_path / "calibration.txt",
        "additional-transfer-paths": tmp_path / "staged_data",
    }
    ini_text = "\n".join(
        [
            f"data-dict={{ H1:{paths['data-dict']} }}",
            f"psd-dict={{ H1:{paths['psd-dict']} }}",
            "spline-calibration-envelope-dict="
            f"{{ H1:{paths['spline-calibration-envelope-dict']} }}",
            f"additional-transfer-paths=[{paths['additional-transfer-paths']}]",
        ]
    )

    with pytest.raises(FileNotFoundError) as exc:
        module.validate_submission_local_paths(
            ini_text,
            base_directory=tmp_path,
        )

    for setting, path in paths.items():
        assert f"{setting}: {path}" in str(exc.value)


def test_independently_localized_coherent_configuration_and_priors():
    module = load_submission_sine_gaussian_utils_module()

    configurations = module.resolve_sine_gaussian_configurations(
        num_sine_gaussians=1,
        range_mode=False,
        mode="coherent-independent",
        incoherent_detectors=None,
        incoherent_counts_spec=None,
        detectors=("H1", "L1"),
    )

    assert configurations == [
        module.SineGaussianConfiguration(
            total_components=1,
            mode="coherent-independent",
        )
    ]
    assert configurations[0].label_suffix == "_sg_coherent_independent_1"
    prior = module.build_sine_gaussian_prior_block(
        configurations[0],
        minimum_frequency=20.0,
        maximum_frequency=448.0,
    )
    assert "independent_sine_gaussian_ra = Uniform(" in prior
    assert "independent_sine_gaussian_dec = Cosine(" in prior
    assert "independent_sine_gaussian_psi = Uniform(" in prior
    assert "independent_sine_gaussian_0_hrss = LogUniform(" in prior
