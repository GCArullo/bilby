import importlib.util
import math
import re
import sys
from pathlib import Path

from scipy.special import kve


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "gw_examples"
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "submit_runs_real_data.py"
)
GAUSSIAN_LIMIT_TEST_QUADRATIC_FORMS = (0.0, 1.0, 4.0, 10.0, 25.0)


def load_submit_runs_real_data_module():
    script_dir = SCRIPT_PATH.parent
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "submit_runs_real_data_prior_limits_test_module",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("submit_runs_real_data_prior_limits_test_module", None)
        sys.path.pop(0)


def prior_maximum(prior_block, parameter):
    match = re.search(
        rf"{parameter} = Uniform\(name='{parameter}', minimum=[^,]+, maximum=([^)]+)\)",
        prior_block,
    )
    assert match is not None
    return float(match.group(1))


def student_t_minus_gaussian_log_density(quadratic_form, nu):
    return (
        -0.5 * (nu + 2.0) * math.log1p(quadratic_form / nu)
        + 0.5 * quadratic_form
    )


def hyperbolic_minus_gaussian_log_density(quadratic_form, alpha, delta):
    hyperbolic_radius = math.sqrt(delta ** 2 + quadratic_form)
    log_bessel_norm = math.log(kve(1, alpha * delta)) - alpha * delta
    log_bessel_radius = (
        math.log(kve(0, alpha * hyperbolic_radius))
        - alpha * hyperbolic_radius
    )
    hyperbolic_log_density = (
        math.log(alpha)
        - math.log(2.0 * math.pi * delta)
        - log_bessel_norm
        + log_bessel_radius
    )
    gaussian_log_density = -math.log(2.0 * math.pi) - 0.5 * quadratic_form
    return hyperbolic_log_density - gaussian_log_density


def test_generated_prior_upper_edges_recover_gaussian_limit_to_expected_accuracy():
    module = load_submit_runs_real_data_module()
    nu_prior_block = module.build_nu_priors(1)
    hyperbolic_prior_block = module.build_hyperbolic_priors(1)

    nu_max = prior_maximum(nu_prior_block, "nu")
    alpha_max = prior_maximum(hyperbolic_prior_block, "alpha")
    delta_max = prior_maximum(hyperbolic_prior_block, "delta")

    student_t_accuracy = max(
        abs(student_t_minus_gaussian_log_density(quadratic_form, nu_max))
        for quadratic_form in GAUSSIAN_LIMIT_TEST_QUADRATIC_FORMS
    )
    hyperbolic_accuracy = max(
        abs(
            hyperbolic_minus_gaussian_log_density(
                quadratic_form,
                alpha_max,
                delta_max,
            )
        )
        for quadratic_form in GAUSSIAN_LIMIT_TEST_QUADRATIC_FORMS
    )

    assert student_t_accuracy < 0.13
    assert hyperbolic_accuracy < 0.08
