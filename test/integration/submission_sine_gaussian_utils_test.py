import importlib.util
import sys
from pathlib import Path


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
    assert "generation-function=None\n" in rendered
    assert "distance-marginalization=False\n" in rendered
