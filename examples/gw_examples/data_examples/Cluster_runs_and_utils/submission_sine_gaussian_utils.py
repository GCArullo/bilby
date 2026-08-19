from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


NLIVE_ONE_SINE_GAUSSIAN_UPLIFT = 500
NLIVE_MULTI_SINE_GAUSSIAN_UPLIFT = 1000
NLIVE_COHERENT_INDEPENDENT_UPLIFT = 500
SINE_GAUSSIAN_HRSS_BOUNDS = (1e-24, 1e-20)
SINE_GAUSSIAN_Q_BOUNDS = (0.1, 30.0)
SINE_GAUSSIAN_TIME_OFFSET_BOUNDS = (-0.15, 0.15)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_sine_gaussian_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str = "",
    subject: str = "recovery waveform",
) -> None:
    prefix = prefix.strip()
    if prefix and not prefix.endswith("-"):
        prefix = f"{prefix}-"

    def option(name: str) -> str:
        return f"--{prefix}{name}"

    parser.add_argument(
        option("num-sine-gaussians"),
        type=int,
        default=0,
        help=(
            f"Total number of sine-Gaussian components in the {subject}. "
            "Use 0 for the existing CBC-only behaviour. With "
            f"{option('sine-gaussian-range')} this is treated as the maximum count."
        ),
    )
    parser.add_argument(
        option("sine-gaussian-range"),
        action="store_true",
        help=(
            "Generate every sine-Gaussian configuration up to "
            f"{option('num-sine-gaussians')}. In either coherent mode this "
            "expands to counts 1..N. In incoherent mode it expands to every "
            "detector-count partition for each total count."
        ),
    )
    parser.add_argument(
        option("sine-gaussian-mode"),
        choices=("coherent", "coherent-independent", "incoherent"),
        default="coherent",
        help=(
            f"How the additional sine-Gaussians are added to the {subject}. "
            "Coherent components use the CBC sky position; coherent-independent "
            "components have a shared, independently sampled sky position; "
            "incoherent components are detector-local."
        ),
    )
    parser.add_argument(
        option("incoherent-detectors"),
        nargs="+",
        default=None,
        help=(
            "Optional detector subset used for incoherent sine-Gaussian runs. "
            "Defaults to the full detector list in the template/event."
        ),
    )
    parser.add_argument(
        option("incoherent-sg-counts"),
        nargs="+",
        default=None,
        help=(
            "Detector allocation for incoherent sine-Gaussians in single mode, "
            "for example `H1=1 L1=2` or `H1=1,L1=2`. The counts must sum to "
            f"{option('num-sine-gaussians')}."
        ),
    )


@dataclass(frozen=True)
class SineGaussianConfiguration:
    total_components: int = 0
    mode: str = "none"
    detector_counts: tuple[tuple[str, int], ...] = ()

    @property
    def enabled(self) -> bool:
        return self.total_components > 0

    @property
    def is_incoherent(self) -> bool:
        return self.mode == "incoherent"

    @property
    def label_suffix(self) -> str:
        if not self.enabled:
            return ""
        if self.mode == "coherent":
            return f"_sg_coherent_{self.total_components}"
        if self.mode == "coherent-independent":
            return f"_sg_coherent_independent_{self.total_components}"
        detector_suffix = "_".join(
            f"{detector}x{count}" for detector, count in self.detector_counts
        )
        return f"_sg_incoherent_{detector_suffix}"

    @property
    def description(self) -> str:
        if not self.enabled:
            return "CBC only"
        if self.mode == "coherent":
            return f"coherent SGs={self.total_components}"
        if self.mode == "coherent-independent":
            return f"independently localized coherent SGs={self.total_components}"
        detector_summary = ", ".join(
            f"{detector}={count}" for detector, count in self.detector_counts
        )
        return f"incoherent SGs ({detector_summary})"


def parse_template_value(raw_value: str):
    raw_value = raw_value.strip()
    if raw_value in {"None", ""}:
        return None
    if raw_value in {"True", "False"}:
        return raw_value == "True"
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def parse_ini_dict_string(raw_value: str) -> dict[str, object]:
    normalized = raw_value.strip()
    normalized = normalized.replace("=", ":")
    normalized = normalized.replace(" ", "")
    normalized = re.sub(
        r'([A-Za-z_/\.0-9\-\+][^\[\],:"}]*)',
        r'"\g<1>"',
        normalized,
    )
    normalized = normalized.replace('""', '"')
    parsed = ast.literal_eval(normalized)
    if not isinstance(parsed, dict):
        raise ValueError(f"Unable to parse ini dict: {raw_value}")
    return parsed


def _ini_setting(ini_text: str, key: str) -> str | None:
    for line in ini_text.splitlines():
        stripped = line.strip()
        if "=" not in stripped:
            continue
        candidate, value = stripped.split("=", 1)
        if candidate.strip() == key:
            return value.strip()
    return None


def _path_values(raw_value: str) -> list[str]:
    parsed = parse_template_value(raw_value)
    stripped = raw_value.strip()
    if isinstance(parsed, str) and stripped.startswith("{"):
        inner = stripped[1:-1].strip()
        parsed = (
            []
            if not inner
            else [
                parse_template_value(item.split(":", 1)[1])
                for item in inner.split(",")
                if item.strip()
            ]
        )
    elif (
        isinstance(parsed, str)
        and stripped.startswith("[")
        and stripped.endswith("]")
    ):
        inner = stripped[1:-1].strip()
        parsed = (
            []
            if not inner
            else [parse_template_value(item) for item in inner.split(",")]
        )

    if parsed is None:
        return []
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, dict):
        values = parsed.values()
    elif isinstance(parsed, (list, tuple)):
        values = parsed
    else:
        raise ValueError(f"Local input paths must be strings, found {parsed!r}")

    paths = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Local input paths must be strings, found {value!r}")
        paths.append(value)
    return paths


def validate_submission_local_paths(
    ini_text: str,
    *,
    base_directory: Path,
) -> None:
    settings = (
        ("data-dict", False),
        ("psd-dict", False),
        ("spline-calibration-envelope-dict", False),
        ("additional-transfer-paths", True),
    )
    missing = []
    for setting, allow_directory in settings:
        raw_value = _ini_setting(ini_text, setting)
        if raw_value is None:
            continue
        for value in _path_values(raw_value):
            url = urlparse(value)
            if url.scheme and url.scheme != "file":
                continue
            if url.scheme == "file":
                value = url.path
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base_directory / path
            path = path.resolve()
            valid = path.exists() if allow_directory else path.is_file()
            if not valid:
                missing.append((setting, path))

    if missing:
        formatted = "\n".join(
            f"  - {setting}: {path}" for setting, path in missing
        )
        raise FileNotFoundError(
            "Missing local input paths required for grid submission:\n"
            f"{formatted}"
        )


def validate_sine_gaussian_distance_marginalization(
    ini_text: str,
    config: SineGaussianConfiguration,
) -> None:
    if not config.enabled:
        return
    value = _ini_setting(ini_text, "distance-marginalization")
    if value is None or parse_template_value(value) is not False:
        raise ValueError(
            "CBC+sine-Gaussian configs require distance-marginalization=False"
        )


def read_template_settings(ini_template: str) -> dict[str, object]:
    parsed = {}
    for line in ini_template.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = parse_template_value(value)

    required_keys = (
        "detectors",
        "trigger-time",
        "duration",
        "post-trigger-duration",
        "sampling-frequency",
        "maximum-frequency",
        "minimum-frequency",
        "reference-frequency",
        "waveform-approximant",
        "sampler-kwargs",
        "frequency-domain-source-model",
        "generation-function",
        "conversion-function",
    )
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise ValueError(
            f"Template ini is missing required keys: {', '.join(missing)}"
        )

    sampler_kwargs = parsed["sampler-kwargs"]
    if not isinstance(sampler_kwargs, dict):
        raise ValueError("sampler-kwargs must parse to a dictionary")

    calibration_envelopes = parsed.get("spline-calibration-envelope-dict")
    if isinstance(calibration_envelopes, str):
        try:
            calibration_envelopes = parse_ini_dict_string(calibration_envelopes)
        except (ValueError, SyntaxError):
            # Some templates still contain unresolved placeholders at parse time.
            pass

    psd_dict = parsed.get("psd-dict")
    if isinstance(psd_dict, str):
        try:
            psd_dict = parse_ini_dict_string(psd_dict)
        except (ValueError, SyntaxError):
            # Some templates still contain unresolved placeholders at parse time.
            pass

    data_dict = parsed.get("data-dict")
    if isinstance(data_dict, str):
        try:
            data_dict = parse_ini_dict_string(data_dict)
        except (ValueError, SyntaxError):
            # Some templates still contain unresolved placeholders at parse time.
            pass

    return dict(
        detectors=tuple(parsed["detectors"]),
        reference_frame=parsed.get("reference-frame", "sky"),
        time_reference=parsed.get("time-reference", "geocent"),
        trigger_time=float(parsed["trigger-time"]),
        duration=float(parsed["duration"]),
        post_trigger_duration=float(parsed["post-trigger-duration"]),
        sampling_frequency=float(parsed["sampling-frequency"]),
        maximum_frequency=parsed["maximum-frequency"],
        minimum_frequency=parsed["minimum-frequency"],
        reference_frequency=float(parsed["reference-frequency"]),
        waveform_approximant=str(parsed["waveform-approximant"]),
        sampler_kwargs=sampler_kwargs,
        sampling_seed=parsed.get("sampling-seed"),
        data_dict=data_dict,
        calibration_model=parsed.get("calibration-model"),
        spline_calibration_nodes=parsed.get("spline-calibration-nodes"),
        spline_calibration_envelope_dict=calibration_envelopes,
        psd_dict=psd_dict,
        frequency_domain_source_model=str(parsed["frequency-domain-source-model"]),
        conversion_function=parsed["conversion-function"],
        generation_function=parsed["generation-function"],
    )


def require_supported_sine_gaussian_source_model(
    template_settings: dict[str, object],
    config: SineGaussianConfiguration,
) -> None:
    if not config.enabled:
        return

    source_model = str(template_settings["frequency_domain_source_model"])
    supported_models = {
        "lal_binary_black_hole",
        "bilby.gw.source.lal_binary_black_hole",
    }
    if source_model not in supported_models:
        raise ValueError(
            "CBC+sine-Gaussian runs currently require a LAL BBH template source "
            f"model. Found frequency-domain-source-model={source_model!r}."
        )


def resolve_sine_gaussian_configurations(
    *,
    num_sine_gaussians: int,
    range_mode: bool,
    mode: str,
    incoherent_detectors: Iterable[str] | None,
    incoherent_counts_spec: str | Iterable[str] | None,
    detectors: Iterable[str],
) -> list[SineGaussianConfiguration]:
    if num_sine_gaussians < 0:
        raise ValueError("--num-sine-gaussians must be non-negative")

    available_detectors = tuple(detectors)
    if incoherent_detectors is None:
        selected_detectors = available_detectors
    else:
        selected_detectors = tuple(dict.fromkeys(incoherent_detectors))
        unknown = sorted(set(selected_detectors).difference(available_detectors))
        if unknown:
            raise ValueError(
                "Unknown incoherent detector(s): {}. Available detectors: {}.".format(
                    ", ".join(unknown), ", ".join(available_detectors)
                )
            )
        if not selected_detectors:
            raise ValueError("At least one incoherent detector must be specified")

    if num_sine_gaussians == 0:
        if range_mode:
            raise ValueError(
                "--sine-gaussian-range requires --num-sine-gaussians >= 1"
            )
        if mode != "coherent":
            raise ValueError(
                "--sine-gaussian-mode only applies when --num-sine-gaussians >= 1"
            )
        if incoherent_counts_spec is not None or incoherent_detectors is not None:
            raise ValueError(
                "Incoherent sine-Gaussian options require --num-sine-gaussians >= 1"
            )
        return [SineGaussianConfiguration()]

    if mode in {"coherent", "coherent-independent"}:
        if incoherent_counts_spec is not None or incoherent_detectors is not None:
            raise ValueError(
                "Incoherent sine-Gaussian options cannot be used with "
                f"--sine-gaussian-mode {mode}"
            )
        if range_mode:
            return [
                SineGaussianConfiguration(total_components=count, mode=mode)
                for count in range(1, num_sine_gaussians + 1)
            ]
        return [
            SineGaussianConfiguration(
                total_components=num_sine_gaussians,
                mode=mode,
            )
        ]

    if mode != "incoherent":
        raise ValueError(f"Unknown sine-Gaussian mode: {mode}")

    if range_mode:
        if incoherent_counts_spec is not None:
            raise ValueError(
                "--incoherent-sg-counts cannot be combined with --sine-gaussian-range"
            )
        configurations = []
        for total_components in range(1, num_sine_gaussians + 1):
            for partition in _enumerate_incoherent_partitions(
                total_components,
                selected_detectors,
            ):
                configurations.append(
                    SineGaussianConfiguration(
                        total_components=total_components,
                        mode="incoherent",
                        detector_counts=partition,
                    )
                )
        return configurations

    detector_counts = _parse_incoherent_counts(
        incoherent_counts_spec,
        detectors=selected_detectors,
    )
    if not detector_counts:
        if len(selected_detectors) == 1:
            detector_counts = {selected_detectors[0]: num_sine_gaussians}
        else:
            raise ValueError(
                "In incoherent single mode, specify --incoherent-sg-counts "
                "(for example `H1=1` or `H1=1 L1=2`)."
            )

    total_components = sum(detector_counts.values())
    if total_components != num_sine_gaussians:
        raise ValueError(
            "--incoherent-sg-counts must sum to --num-sine-gaussians "
            f"({num_sine_gaussians}), got {total_components}."
        )

    ordered_counts = tuple(
        (detector, detector_counts[detector])
        for detector in available_detectors
        if detector_counts.get(detector, 0) > 0
    )
    return [
        SineGaussianConfiguration(
            total_components=num_sine_gaussians,
            mode="incoherent",
            detector_counts=ordered_counts,
        )
    ]


def build_sine_gaussian_prior_block(
    config: SineGaussianConfiguration,
    *,
    minimum_frequency,
    maximum_frequency,
):
    return _build_sine_gaussian_prior_block(
        config,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )


def combine_prior_blocks(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)


def apply_sine_gaussian_waveform_settings(
    ini_text: str,
    config: SineGaussianConfiguration,
    *,
    replace_line,
) -> str:
    if not config.enabled:
        return ini_text

    updated = replace_line(
        ini_text,
        "frequency-domain-source-model",
        "bilby.gw.source.cbc_plus_sine_gaussians",
    )
    updated = replace_line(
        updated,
        "conversion-function",
        "bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters",
    )
    updated = replace_line(
        updated,
        "generation-function",
        "bilby.gw.conversion.generate_all_cbc_plus_sine_gaussian_parameters",
    )
    updated = replace_line(updated, "distance-marginalization", "False")
    validate_sine_gaussian_distance_marginalization(updated, config)
    return updated


def effective_nlive(base_nlive: int, config: SineGaussianConfiguration) -> int:
    if config.total_components <= 0:
        return base_nlive
    if config.total_components == 1:
        uplift = NLIVE_ONE_SINE_GAUSSIAN_UPLIFT
    else:
        uplift = NLIVE_MULTI_SINE_GAUSSIAN_UPLIFT
    if config.mode == "coherent-independent":
        uplift += NLIVE_COHERENT_INDEPENDENT_UPLIFT
    return base_nlive + uplift


def sine_gaussian_frequency_bounds(minimum_frequency, maximum_frequency) -> tuple[float, float]:
    return (
        _resolve_frequency_minimum(minimum_frequency),
        _resolve_frequency_maximum(maximum_frequency),
    )


def _parse_incoherent_counts(
    raw_spec: str | Iterable[str] | None,
    *,
    detectors: tuple[str, ...],
) -> dict[str, int]:
    if raw_spec is None:
        return {}

    if isinstance(raw_spec, str):
        raw_items = [raw_spec]
    else:
        raw_items = list(raw_spec)

    tokens = []
    for item in raw_items:
        tokens.extend(token for token in item.split(",") if token)

    counts = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(
                "Invalid --incoherent-sg-counts entry {!r}; expected DETECTOR=COUNT.".format(
                    token
                )
            )
        detector, raw_count = token.split("=", 1)
        detector = detector.strip()
        if detector not in detectors:
            raise ValueError(
                "Unknown incoherent detector {!r}. Available detectors: {}.".format(
                    detector, ", ".join(detectors)
                )
            )
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sine-Gaussian count for detector {detector!r}: {raw_count!r}"
            ) from exc
        if count < 0:
            raise ValueError(
                f"Sine-Gaussian counts must be non-negative, got {count} for {detector}"
            )
        counts[detector] = count

    counts = {detector: count for detector, count in counts.items() if count > 0}
    return counts


def _enumerate_incoherent_partitions(
    total_components: int,
    detectors: tuple[str, ...],
) -> list[tuple[tuple[str, int], ...]]:
    if not detectors:
        raise ValueError("At least one detector is required for incoherent SG runs")

    partitions = []

    def recurse(index: int, remaining: int, current: list[tuple[str, int]]) -> None:
        detector = detectors[index]
        is_last = index == len(detectors) - 1
        if is_last:
            final_current = list(current)
            if remaining > 0:
                final_current.append((detector, remaining))
            partitions.append(tuple(final_current))
            return

        for count in range(remaining, -1, -1):
            next_current = list(current)
            if count > 0:
                next_current.append((detector, count))
            recurse(index + 1, remaining - count, next_current)

    recurse(0, total_components, [])
    return partitions


def _build_sine_gaussian_prior_block(
    config: SineGaussianConfiguration,
    *,
    minimum_frequency,
    maximum_frequency,
) -> str:
    if not config.enabled:
        return ""

    frequency_minimum = _resolve_frequency_minimum(minimum_frequency)
    frequency_maximum = _resolve_frequency_maximum(maximum_frequency)

    if frequency_minimum >= frequency_maximum:
        raise ValueError(
            "Invalid sine-Gaussian frequency prior bounds: "
            f"{frequency_minimum} >= {frequency_maximum}"
        )

    lines = []
    if config.mode == "coherent":
        for index in range(config.total_components):
            lines.extend(
                _sine_gaussian_prior_lines(
                    prefix=f"sine_gaussian_{index}_",
                    upper_bound_name=(
                        None if index == 0 else f"sine_gaussian_{index - 1}_hrss"
                    ),
                    frequency_minimum=frequency_minimum,
                    frequency_maximum=frequency_maximum,
                )
            )
    elif config.mode == "coherent-independent":
        lines.extend([
            "independent_sine_gaussian_ra = Uniform("
            "name='independent_sine_gaussian_ra', minimum=0, "
            "maximum=2 * np.pi, boundary='periodic')",
            "independent_sine_gaussian_dec = Cosine("
            "name='independent_sine_gaussian_dec')",
            "independent_sine_gaussian_psi = Uniform("
            "name='independent_sine_gaussian_psi', minimum=0, "
            "maximum=np.pi, boundary='periodic')",
        ])
        for index in range(config.total_components):
            lines.extend(
                _sine_gaussian_prior_lines(
                    prefix=f"independent_sine_gaussian_{index}_",
                    upper_bound_name=(
                        None
                        if index == 0
                        else f"independent_sine_gaussian_{index - 1}_hrss"
                    ),
                    frequency_minimum=frequency_minimum,
                    frequency_maximum=frequency_maximum,
                )
            )
    else:
        component_index = 0
        for detector, count in config.detector_counts:
            previous_hrss_name = None
            for _ in range(count):
                prefix = f"sine_gaussian_{component_index}_{detector}_"
                lines.extend(
                    _sine_gaussian_prior_lines(
                        prefix=prefix,
                        upper_bound_name=previous_hrss_name,
                        frequency_minimum=frequency_minimum,
                        frequency_maximum=frequency_maximum,
                    )
                )
                previous_hrss_name = f"{prefix}hrss"
                component_index += 1

    return "\n".join(lines)


def _sine_gaussian_prior_lines(
    *,
    prefix: str,
    upper_bound_name: str | None,
    frequency_minimum: float,
    frequency_maximum: float,
) -> list[str]:
    if upper_bound_name is None:
        hrss_line = (
            f"{prefix}hrss = LogUniform(name='{prefix}hrss', "
            f"minimum={SINE_GAUSSIAN_HRSS_BOUNDS[0]}, maximum={SINE_GAUSSIAN_HRSS_BOUNDS[1]})"
        )
    else:
        hrss_line = (
            f"{prefix}hrss = bilby.gw.prior.ConditionalUpperBoundedLogUniform("
            f"name='{prefix}hrss', minimum={SINE_GAUSSIAN_HRSS_BOUNDS[0]}, "
            f"maximum={SINE_GAUSSIAN_HRSS_BOUNDS[1]}, "
            f"upper_bound_name='{upper_bound_name}')"
        )
    return [
        hrss_line,
        (
            f"{prefix}Q = Uniform(name='{prefix}Q', "
            f"minimum={SINE_GAUSSIAN_Q_BOUNDS[0]}, maximum={SINE_GAUSSIAN_Q_BOUNDS[1]})"
        ),
        (
            f"{prefix}frequency = Uniform(name='{prefix}frequency', "
            f"minimum={frequency_minimum}, maximum={frequency_maximum})"
        ),
        (
            f"{prefix}time_offset = Uniform(name='{prefix}time_offset', "
            f"minimum={SINE_GAUSSIAN_TIME_OFFSET_BOUNDS[0]}, "
            f"maximum={SINE_GAUSSIAN_TIME_OFFSET_BOUNDS[1]})"
        ),
        (
            f"{prefix}phase_offset = Uniform(name='{prefix}phase_offset', "
            "minimum=-np.pi, maximum=np.pi, boundary='periodic')"
        ),
    ]


def _resolve_frequency_minimum(minimum_frequency) -> float:
    if isinstance(minimum_frequency, dict):
        values = [
            float(value)
            for key, value in minimum_frequency.items()
            if key != "waveform"
        ]
        positive_values = [value for value in values if value > 0]
        if positive_values:
            return min(positive_values)
        if values:
            return min(values)
        waveform_value = minimum_frequency.get("waveform", 20.0)
        return float(waveform_value)
    return float(minimum_frequency)


def _resolve_frequency_maximum(maximum_frequency) -> float:
    if isinstance(maximum_frequency, dict):
        return min(float(value) for value in maximum_frequency.values())
    return float(maximum_frequency)
