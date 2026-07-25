#!/usr/bin/env python3

"""Extract GWTC-2.1 nocosmo configs and make bilby_pipe run templates.

The Zenodo release stores configuration files, PSDs, calibration envelopes,
and priors inside per-event HDF5 files.  This script reads only those objects
using HTTP range requests; it does not download the posterior samples.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np


RECORD_ID = 6513631
ROOT = Path(__file__).resolve().parent
BASE_TEMPLATE = (
    ROOT.parent
    / "Initialisation_file_templates"
    / "GW150914_t_student_igwn_template.ini"
)
EVENT_PATTERN = re.compile(r"v2-(GW\d{6}(?:_\d{6})?)_PEDataRelease_mixed_nocosmo\.h5$")
GWTC1_PREFIXES = ("GW15", "GW17")
GLITCH_MODELS = {
    "GW190413_134308": "glitch-only",
    "GW190425_081805": "glitch-only",
    "GW190503_185404": "glitch-only",
    "GW190513_205428": "glitch-only",
    "GW190514_065416": "glitch-only",
    "GW190701_203306": "glitch+signal",
    "GW190924_021846": "glitch-only",
}
GLITCH_FRAMES = {
    "GW190413_134308": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1239195648-4096.gwf",
    "GW190425_081805": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1240211456-4096.gwf",
    "GW190503_185404": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1240944640-4096.gwf",
    "GW190513_205428": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1241812992-4096.gwf",
    "GW190514_065416": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1241849856-2591.gwf",
    "GW190701_203306": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1246048256-4096.gwf",
    "GW190924_021846": "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1253322752-4096.gwf",
}
LOW_FREQUENCY_MITIGATION = {
    "GW190727_060333": {"L1": 50},
    "GW190814_211039": {"L1": 30},
}
# GWOSC GWTC-2.1 event times for releases whose embedded LALInference config
# does not carry a trigger-time setting.
TRIGGER_TIME_OVERRIDES = {
    "GW170608_020116": 1180922494.5,
    "GW190707_093326": 1246527224.1,
    "GW190720_000836": 1247616534.7,
    "GW190725_174728": 1248112066.4,
    "GW190728_064510": 1248331528.5,
    "GW190814_211039": 1249852257.0,
    "GW190917_114630": 1252756008.0,
    "GW190924_021846": 1253326744.8,
}
OTHER_DATA_NOTES = {
    "GW190403_051519": "Virgo data newly included in the GWTC-2.1 PE update.",
    "GW190413_052954": "Virgo data newly included in the GWTC-2.1 PE update.",
    "GW190426_190642": "Virgo data newly included in the GWTC-2.1 PE update.",
    "GW190805_211137": "Virgo data newly included in the GWTC-2.1 PE update.",
    "GW190814_211039": "H1 was out of observing mode; detector-characterization-selected H1 data were used.",
    "GW190915_235702": "Virgo used additionally cleaned V1O3Repro1A strain.",
    "GW190924_021846": "Virgo used additionally cleaned V1O3Repro1A strain.",
    "GW190929_012149": "Virgo used additionally cleaned V1O3Repro1A strain.",
}
SOURCES = {
    "release": "https://zenodo.org/records/6513631",
    "paper": "https://dcc.ligo.org/public/0173/P2100063/014/o3a_final.pdf",
    "glitches": "https://zenodo.org/records/6477075",
    "documentation": "https://gwosc.org/GWTC-2.1/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event", action="append", help="Record event ID to prepare; default: all"
    )
    parser.add_argument(
        "--local-h5-dir",
        type=Path,
        help="Optional directory containing release HDF5 files",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--event-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-event", help=argparse.SUPPRESS)
    parser.add_argument("--index-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def fetch_record() -> list[dict[str, object]]:
    url = f"https://zenodo.org/api/records/{RECORD_ID}"
    with urllib.request.urlopen(url) as response:
        record = json.load(response)
    events = []
    for item in record["files"]:
        match = EVENT_PATTERN.search(item["key"])
        if match:
            events.append(
                {
                    "record_event": match.group(1),
                    "filename": item["key"],
                    "checksum": item["checksum"],
                    "size": item["size"],
                    "url": f"https://zenodo.org/records/{RECORD_ID}/files/{item['key']}?download=1",
                }
            )
    return sorted(events, key=lambda item: item["record_event"])


def event_name(record_event: str) -> str:
    if record_event.startswith(GWTC1_PREFIXES):
        return record_event.split("_", maxsplit=1)[0]
    return record_event


def event_gps(record_event: str) -> float:
    date_text, time_text = record_event.removeprefix("GW").split("_")
    utc = datetime.strptime(date_text + time_text, "%y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    leap_seconds = 17 if utc < datetime(2017, 1, 1, tzinfo=timezone.utc) else 18
    return utc.timestamp() - 315964800 + leap_seconds


def decode(value):
    if isinstance(value, np.ndarray):
        values = [decode(item) for item in value.tolist()]
        return values[0] if len(values) == 1 else values
    if isinstance(value, np.generic):
        return decode(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_dataset(dataset: h5py.Dataset):
    return decode(dataset[()])


def flatten(group: h5py.Group) -> dict[str, object]:
    values = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            values[name] = read_dataset(obj)

    group.visititems(visitor)
    return values


def value_text(value) -> str:
    if isinstance(value, list):
        return repr(value)
    return str(value)


def write_source_config(config_group: h5py.Group, path: Path) -> None:
    children = list(config_group)
    if children == ["config"]:
        values = flatten(config_group["config"])
        text = "\n".join(
            f"{key} = {value_text(value)}" for key, value in sorted(values.items())
        )
    else:
        blocks = []
        for section in sorted(children):
            values = flatten(config_group[section])
            lines = [f"[{section}]"]
            lines.extend(
                f"{key} = {value_text(value)}" for key, value in sorted(values.items())
            )
            blocks.append("\n".join(lines))
        text = "\n\n".join(blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def sanitized_run_name(run_name: str) -> str:
    return run_name.removeprefix("C01:").replace(":", "__").replace("/", "_")


def select_run(h5_file: h5py.File, record_event: str) -> str:
    if record_event == "GW190425_081805":
        return "C01:IMRPhenomPv2_NRTidal:LowSpin"
    preferred = "C01:IMRPhenomXPHM"
    if preferred in h5_file:
        return preferred
    runs = [key for key in h5_file if key.startswith("C01:") and key != "C01:Mixed"]
    if not runs:
        raise RuntimeError(f"No analysis run found for {record_event}")
    return sorted(runs)[0]


def parse_literal(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def first(values: dict[str, object], *keys: str, default=None):
    for key in keys:
        if key in values and values[key] not in (None, "None", ""):
            return values[key]
    return default


def number(value, default: float) -> float:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        parsed = parsed[0]
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return float(default)


def parse_detectors(value) -> list[str]:
    parsed = parse_literal(value)
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    detectors = []
    for item in parsed:
        detector = str(parse_literal(item)).strip("'\"")
        if detector in {"H1", "L1", "V1"}:
            detectors.append(detector)
    return detectors


def detector_setting(value, detectors: list[str], default: float):
    parsed = parse_literal(value)
    if isinstance(parsed, dict):
        return {str(key): float(item) for key, item in parsed.items()}
    scalar = number(parsed, default)
    return {detector: scalar for detector in detectors}


def product_detectors(run: h5py.Group) -> tuple[list[str], list[str]]:
    psd_detectors = sorted(run["psds"]) if "psds" in run else []
    calibration_detectors = (
        sorted(run["calibration_envelope"]) if "calibration_envelope" in run else []
    )
    return psd_detectors, calibration_detectors


def extract_products(
    run: h5py.Group,
    output_dir: Path,
    maximum_frequency: dict[str, float],
    detectors: list[str],
) -> None:
    selected_detectors = set(detectors)
    if "psds" in run:
        psd_dir = output_dir / "psds"
        for detector, dataset in run["psds"].items():
            if detector not in selected_detectors:
                continue
            psd_dir.mkdir(parents=True, exist_ok=True)
            values = np.asarray(dataset[:], dtype=float)
            values = values[values[:, 0] <= maximum_frequency[detector]]
            np.savetxt(
                psd_dir / f"{detector}.dat",
                values,
                fmt="%.12e",
                header="f psd(f)",
            )
    if "calibration_envelope" in run:
        calibration_dir = output_dir / "calibration"
        for detector, dataset in run["calibration_envelope"].items():
            if detector not in selected_detectors:
                continue
            calibration_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                calibration_dir / f"{detector}.txt",
                np.asarray(dataset[:], dtype=float),
                fmt="%.12e",
                header="f amp_median phase_median amp_lower amp_upper phase_lower phase_upper",
            )


def replace_setting(text: str, key: str, value) -> str:
    prefix = f"{key}="
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{prefix}{value}"
            return "\n".join(lines) + "\n"
    raise KeyError(f"Base template has no {key} setting")


def mapping_text(mapping: dict[str, object]) -> str:
    return repr(mapping)


def source_values(
    run: h5py.Group,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = {}
    raw_config = {}
    if "config_file" in run:
        raw_config = flatten(run["config_file"])
        if "config" in run["config_file"]:
            config = flatten(run["config_file/config"])
    metadata = {}
    metadata_path = "meta_data/other/command_line_args"
    if metadata_path in run:
        metadata = flatten(run[metadata_path])
    return config, raw_config, metadata


def get_settings(
    run: h5py.Group,
    record_event: str,
    psd_detectors: list[str],
    calibration_detectors: list[str],
) -> dict[str, object]:
    config, raw, metadata = source_values(run)
    detectors = parse_detectors(
        first(
            config,
            "detectors",
            default=first(metadata, "detectors", default=first(raw, "analysis/ifos")),
        )
    )
    if not detectors:
        detectors = sorted(set(psd_detectors) | set(calibration_detectors))
    trigger_time = number(
        first(config, "trigger-time", default=metadata.get("trigger_time")), 0
    )
    if trigger_time == 0:
        trigger_time = TRIGGER_TIME_OVERRIDES.get(record_event, event_gps(record_event))
    duration = number(
        first(
            config,
            "duration",
            default=first(metadata, "duration", default=raw.get("engine/seglen")),
        ),
        4,
    )
    sampling_frequency = number(
        first(
            config,
            "sampling-frequency",
            default=first(
                metadata, "sampling_frequency", default=raw.get("engine/srate")
            ),
        ),
        2048,
    )
    post_trigger_duration = number(
        first(
            config,
            "post-trigger-duration",
            default=metadata.get("post_trigger_duration"),
        ),
        2,
    )
    psd_length = number(
        first(config, "psd-length", default=metadata.get("psd_length")),
        duration,
    )
    f_ref = number(
        first(
            config,
            "reference-frequency",
            default=first(
                metadata, "reference_frequency", default=raw.get("engine/fref")
            ),
        ),
        20,
    )
    minimum = detector_setting(
        first(
            config,
            "minimum-frequency",
            default=first(
                raw, "lalinference/flow", default=metadata.get("minimum_frequency")
            ),
        ),
        detectors,
        20,
    )
    for detector, cutoff in LOW_FREQUENCY_MITIGATION.get(record_event, {}).items():
        minimum[detector] = float(cutoff)
    maximum = detector_setting(
        first(
            config,
            "maximum-frequency",
            default=first(
                raw, "lalinference/fhigh", default=metadata.get("maximum_frequency")
            ),
        ),
        detectors,
        0.875 * sampling_frequency / 2,
    )
    approximant_object = run.get("approximant")
    if isinstance(approximant_object, h5py.Dataset):
        approximant = decode(approximant_object[()])
    elif "meta_data/meta_data/approximant" in run:
        approximant = read_dataset(run["meta_data/meta_data/approximant"])
    else:
        approximant = run.name.rsplit(":", 1)[-1]
    if isinstance(approximant, list):
        approximant = approximant[0]
    reference_frame = first(
        config, "reference-frame", "reference_frame", default="sky"
    )
    time_reference = first(
        config, "time-reference", "time_reference", default="geocent"
    )
    return {
        "detectors": detectors,
        "trigger_time": trigger_time,
        "duration": duration,
        "sampling_frequency": sampling_frequency,
        "post_trigger_duration": post_trigger_duration,
        "psd_length": psd_length,
        "reference_frequency": f_ref,
        "minimum_frequency": minimum,
        "maximum_frequency": maximum,
        "approximant": str(approximant),
        "reference_frame": str(reference_frame),
        "time_reference": str(time_reference),
    }


def make_template(
    event: str,
    record_event: str,
    run_name: str,
    settings: dict[str, object],
    psd_detectors: list[str],
    calibration_detectors: list[str],
) -> str:
    text = BASE_TEMPLATE.read_text(encoding="utf-8")
    detectors = settings["detectors"]
    run_dir = sanitized_run_name(run_name)
    data_root = f"__WORKING_DIRECTORY__/data/{event}/{run_dir}"
    psds = {detector: f"{data_root}/psds/{detector}.dat" for detector in psd_detectors}
    calibration = {
        detector: f"{data_root}/calibration/{detector}.txt"
        for detector in calibration_detectors
    }
    channels = {detector: "GWOSC" for detector in detectors}
    data_dict = None
    if record_event in GLITCH_FRAMES:
        channels["L1"] = "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4"
        data_dict = {
            "L1": f"__WORKING_DIRECTORY__/glitch_data/{GLITCH_FRAMES[record_event]}"
        }
    source_model = (
        "lal_binary_neutron_star"
        if record_event == "GW190425_081805"
        else "lal_binary_black_hole"
    )
    values = {
        "reference-frame": settings["reference_frame"],
        "time-reference": settings["time_reference"],
        "calibration-model": "CubicSpline" if calibration else "None",
        "spline-calibration-envelope-dict": mapping_text(calibration)
        if calibration
        else "None",
        "trigger-time": settings["trigger_time"],
        "data-dict": mapping_text(data_dict) if data_dict else "None",
        "channel-dict": mapping_text(channels),
        "fetch-open-data-kwargs": mapping_text(
            {"sample_rate": max(4096, int(settings["sampling_frequency"]))}
        ),
        "detectors": repr(detectors),
        "duration": settings["duration"],
        "psd-dict": mapping_text(psds) if psds else "None",
        "psd-length": int(settings["psd_length"]),
        "post-trigger-duration": settings["post_trigger_duration"],
        "sampling-frequency": settings["sampling_frequency"],
        "maximum-frequency": mapping_text(settings["maximum_frequency"]),
        "minimum-frequency": mapping_text(settings["minimum_frequency"]),
        "reference-frequency": settings["reference_frequency"],
        "waveform-approximant": settings["approximant"],
        "default-prior": "BNSPriorDict"
        if record_event == "GW190425_081805"
        else "BBHPriorDict",
        "frequency-domain-source-model": source_model,
    }
    for key, value in values.items():
        text = replace_setting(text, key, value)
    return text


def make_gw190425_prior() -> str:
    return """# Reconstructed low-spin BNS prior: the release contains no analytic prior datasets.
chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=1.48, maximum=1.50, unit='$M_{\\odot}$')
mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1.0)
mass_1 = Constraint(name='mass_1', minimum=1.0, maximum=5.31)
mass_2 = Constraint(name='mass_2', minimum=1.0, maximum=5.31)
a_1 = Uniform(name='a_1', minimum=0, maximum=0.05)
a_2 = Uniform(name='a_2', minimum=0, maximum=0.05)
tilt_1 = Sine(name='tilt_1')
tilt_2 = Sine(name='tilt_2')
phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
lambda_1 = Uniform(name='lambda_1', minimum=0, maximum=5000)
lambda_2 = Uniform(name='lambda_2', minimum=0, maximum=10000)
luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=10, maximum=1000, unit='Mpc')
theta_jn = Sine(name='theta_jn')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
dec = Cosine(name='dec')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
__NU_PRIORS__
"""


def make_config_bbh_prior(run: h5py.Group) -> str:
    _, raw, metadata = source_values(run)
    chirp_min = number(
        first(raw, "engine/chirpmass-min", default=metadata.get("chirp_mass_min")), 2
    )
    chirp_max = number(
        first(raw, "engine/chirpmass-max", default=metadata.get("chirp_mass_max")), 200
    )
    q_min = number(
        first(raw, "engine/q-min", default=metadata.get("minimum_mass_ratio")), 0.05
    )
    component_min = number(
        first(raw, "engine/comp-min", default=metadata.get("minimum_component_mass")), 1
    )
    component_max = number(
        first(raw, "engine/comp-max", default=metadata.get("maximum_component_mass")),
        1000,
    )
    spin_max = number(
        first(
            raw, "engine/a_spin1-max", default=metadata.get("maximum_spin_magnitude")
        ),
        0.99,
    )
    distance_max = number(
        first(raw, "engine/distance-max", default=metadata.get("maximum_distance")),
        10000,
    )
    return f"""# Reconstructed from the bounds in the embedded LALInference configuration.
chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum={chirp_min}, maximum={chirp_max}, unit='$M_{{\\odot}}$')
mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum={q_min}, maximum=1.0)
mass_1 = Constraint(name='mass_1', minimum={component_min}, maximum={component_max})
mass_2 = Constraint(name='mass_2', minimum={component_min}, maximum={component_max})
a_1 = Uniform(name='a_1', minimum=0, maximum={spin_max})
a_2 = Uniform(name='a_2', minimum=0, maximum={spin_max})
tilt_1 = Sine(name='tilt_1')
tilt_2 = Sine(name='tilt_2')
phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=10, maximum={distance_max}, unit='Mpc')
theta_jn = Sine(name='theta_jn')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
dec = Cosine(name='dec')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
__NU_PRIORS__
"""


def make_prior(run: h5py.Group, record_event: str) -> tuple[str, bool]:
    analytic_path = "priors/analytic"
    if analytic_path not in run or not list(run[analytic_path]):
        if record_event == "GW190425_081805":
            return make_gw190425_prior(), True
        return make_config_bbh_prior(run), True
    excluded = {"azimuth", "dec", "geocent_time", "ra", "time_jitter", "zenith"}
    values = []
    for key, dataset in sorted(run[analytic_path].items()):
        if key in excluded or key.endswith("_time") or key.startswith("recalib_"):
            continue
        value = read_dataset(dataset)
        values.append(f"{key} = {value}")
    values.extend(
        [
            "dec = Cosine(name='dec')",
            "ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')",
            "__NU_PRIORS__",
        ]
    )
    return "\n".join(values) + "\n", False


def prepare_event(
    item: dict[str, object], local_h5_dir: Path | None
) -> dict[str, object]:
    record_event = str(item["record_event"])
    event = event_name(record_event)
    local_path = local_h5_dir / str(item["filename"]) if local_h5_dir else None
    if local_path and local_path.is_file():
        handle = local_path.open("rb")
    else:
        try:
            import fsspec
        except ImportError as error:
            raise RuntimeError(
                "Remote extraction requires fsspec and aiohttp"
            ) from error
        handle = fsspec.open(
            str(item["url"]), mode="rb", block_size=1024 * 1024, cache_type="blockcache"
        ).open()

    with handle, h5py.File(handle, "r") as h5_file:
        source_dir = ROOT / "source_configs" / event
        source_files = []
        for run_name in sorted(key for key in h5_file if key.startswith("C01:")):
            config_group = h5_file[run_name].get("config_file")
            if config_group is None or not list(config_group):
                continue
            path = source_dir / f"{sanitized_run_name(run_name)}.ini"
            write_source_config(config_group, path)
            source_files.append(str(path.relative_to(ROOT)))

        selected_run = select_run(h5_file, record_event)
        run = h5_file[selected_run]
        metadata_snapshot = None
        if not source_files and "meta_data/other/command_line_args" in run:
            metadata_path = (
                source_dir
                / f"{sanitized_run_name(selected_run)}__command_line_args.ini"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            values = flatten(run["meta_data/other/command_line_args"])
            metadata_path.write_text(
                "\n".join(
                    f"{key} = {value_text(value)}"
                    for key, value in sorted(values.items())
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_snapshot = str(metadata_path.relative_to(ROOT))
        output_dir = ROOT / "data" / event / sanitized_run_name(selected_run)
        psd_detectors, calibration_detectors = product_detectors(run)
        settings = get_settings(run, record_event, psd_detectors, calibration_detectors)
        selected_detectors = set(settings["detectors"])
        psd_detectors = [
            detector for detector in psd_detectors if detector in selected_detectors
        ]
        calibration_detectors = [
            detector
            for detector in calibration_detectors
            if detector in selected_detectors
        ]
        extract_products(
            run,
            output_dir,
            settings["maximum_frequency"],
            settings["detectors"],
        )
        prior, reconstructed_prior = make_prior(run, record_event)
        template = make_template(
            event,
            record_event,
            selected_run,
            settings,
            psd_detectors,
            calibration_detectors,
        )

        template_path = ROOT / "templates" / f"{event}.ini"
        prior_path = ROOT / "priors" / f"{event}.prior"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(template, encoding="utf-8")
        prior_path.write_text(prior, encoding="utf-8")

        _, raw_config, metadata = source_values(run)
        source_text = " ".join(
            value_text(value) for value in (*raw_config.values(), *metadata.values())
        )
        special_channel = any(
            channel in source_text
            for channel in ("T1700406_v4", "P1800169_v4")
        )
        result = {
            **item,
            "event": event,
            "selected_run": selected_run,
            "approximant": settings["approximant"],
            "detectors": settings["detectors"],
            "minimum_frequency": settings["minimum_frequency"],
            "maximum_frequency": settings["maximum_frequency"],
            "trigger_time": settings["trigger_time"],
            "duration": settings["duration"],
            "sampling_frequency": settings["sampling_frequency"],
            "source_configs": source_files,
            "embedded_config_available": bool(source_files),
            "metadata_snapshot": metadata_snapshot,
            "psd_detectors": psd_detectors,
            "calibration_detectors": calibration_detectors,
            "reconstructed_prior": reconstructed_prior,
            "source_selects_glitch_subtracted_channel": special_channel,
            "template": str(template_path.relative_to(ROOT)),
            "prior": str(prior_path.relative_to(ROOT)),
        }
        event_metadata = ROOT / "metadata" / f"{event}.json"
        event_metadata.parent.mkdir(parents=True, exist_ok=True)
        event_metadata.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result


def make_manifest(events: list[dict[str, object]]) -> None:
    manifest = {
        "record_id": RECORD_ID,
        "record_url": SOURCES["release"],
        "events": events,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def row_link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'


def make_html(events: list[dict[str, object]]) -> None:
    rows = []
    for item in events:
        record_event = str(item["record_event"])
        event = str(item["event"])
        glitch_model = GLITCH_MODELS.get(record_event)
        if glitch_model:
            glitch = (
                f'<span class="badge yes">Yes</span> L1; {glitch_model} BayesWave model'
            )
            if item["source_selects_glitch_subtracted_channel"]:
                provenance = (
                    "config"
                    if item["embedded_config_available"]
                    else "command-line metadata"
                )
                selected_data = f'<span class="badge yes">Yes</span> released {provenance} selects the <code>T1700406_v4</code> channel'
            else:
                selected_data = '<span class="badge warn">Not explicit</span> use the prepared template and public glitch frame'
        else:
            glitch = '<span class="badge no">No</span> no event-specific transient-glitch subtraction reported'
            selected_data = '<span class="badge na">N/A</span> standard released strain; <code>SUB60HZ</code> is line cleaning'
        mitigation = LOW_FREQUENCY_MITIGATION.get(record_event)
        if mitigation:
            cutoffs = ", ".join(
                f"{detector} {frequency} Hz"
                for detector, frequency in mitigation.items()
            )
            low_frequency = f'<span class="badge warn">Yes</span> {cutoffs} (nominally 20 Hz elsewhere)'
        else:
            configured = {
                detector: frequency
                for detector, frequency in item["minimum_frequency"].items()
                if frequency > 21
            }
            if configured:
                cutoffs = ", ".join(
                    f"{detector} {frequency:g} Hz"
                    for detector, frequency in configured.items()
                )
                low_frequency = (
                    f'<span class="badge na">Not attributed to DQ</span> config has {cutoffs}; '
                    "GWTC-2.1 Table I does not identify it as candidate-specific cutoff mitigation"
                )
            else:
                low_frequency = (
                    '<span class="badge no">No</span> nominal ≈20 Hz analysis cutoff'
                )
        source_configs = item["source_configs"]
        if source_configs:
            links = [row_link(path, Path(path).stem) for path in source_configs]
            config = "<br>".join(links)
        else:
            snapshot = item.get("metadata_snapshot")
            link = (
                f"; {row_link(snapshot, 'command-line metadata snapshot')}"
                if snapshot
                else ""
            )
            config = f'<span class="badge warn">Missing</span> no config datasets in release; template reconstructed from metadata{link}'
        notes = OTHER_DATA_NOTES.get(record_event, "None reported.")
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(event)}</strong><br><small>{html.escape(record_event)}</small></td>"
            f"<td>{html.escape(', '.join(item['detectors']))}</td>"
            f"<td>{glitch}</td>"
            f"<td>{selected_data}</td>"
            f"<td>{low_frequency}</td>"
            f"<td>{html.escape(notes)}</td>"
            f"<td>{config}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GWTC-2.1 event data quality</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f5f7fa; --card:#fff; --ink:#172033; --muted:#5c667a; --line:#dbe1ea; --accent:#274c77; }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#111723; --card:#192232; --ink:#eef3fb; --muted:#aeb9ca; --line:#344156; --accent:#8ecae6; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }}
    main {{ max-width:1700px; margin:auto; padding:32px 24px 64px; }} h1 {{ margin-bottom:.25rem; }} .lede {{ max-width:1000px; color:var(--muted); }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin:24px 0; }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
    .number {{ font-size:2rem; font-weight:750; }} .label {{ color:var(--muted); }} .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
    table {{ border-collapse:collapse; width:100%; min-width:1450px; }} th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ position:sticky; top:0; background:var(--card); color:var(--muted); font-size:.82rem; letter-spacing:.03em; text-transform:uppercase; }}
    tr:hover td {{ background:color-mix(in srgb,var(--accent) 6%,transparent); }} .badge {{ display:inline-block; border-radius:999px; padding:1px 7px; font-size:.78rem; font-weight:700; }} .yes {{ background:#d7f5e5; color:#14532d; }} .no,.na {{ background:#e8edf4; color:#334155; }} .warn {{ background:#fff0c2; color:#713f12; }}
    code {{ font-size:.85em; }} a {{ color:var(--accent); }} section {{ margin-top:28px; }} small {{ color:var(--muted); }}
    @media (max-width:800px) {{ .summary {{ grid-template-columns:repeat(2,1fr); }} main {{ padding:20px 12px 40px; }} }}
  </style>
</head>
<body><main>
  <h1>GWTC-2.1 event data quality</h1>
  <p class="lede">Event-by-event audit for the 54 events in the GWTC-2.1 parameter-estimation release (Zenodo record 6513631, v2). “Glitch subtraction” below means transient BayesWave subtraction. The <code>CLEAN_SUB60HZ</code> channel used for O3a is standard 60 Hz line cleaning and must not be interpreted as transient-glitch subtraction.</p>
  <div class="summary">
    <div class="card"><div class="number">54</div><div class="label">events audited</div></div>
    <div class="card"><div class="number">7</div><div class="label">L1 glitch-subtracted</div></div>
    <div class="card"><div class="number">2</div><div class="label">raised DQ low-frequency cutoffs</div></div>
    <div class="card"><div class="number">1</div><div class="label">release config gap (GW190425)</div></div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Event</th><th>Detectors</th><th>Transient glitch subtraction?</th><th>Does released config select subtracted strain?</th><th>DQ low-frequency cutoff?</th><th>Other released-data note</th><th>Embedded source configs</th></tr></thead>
    <tbody>{chr(10).join(rows)}</tbody>
  </table></div>
  <section><h2>Interpretation and assumptions</h2>
    <ul>
      <li>The catalog paper’s Table I is treated as the authoritative list of candidate-specific artifact mitigation: seven L1 BayesWave subtractions, L1 at 50 Hz for GW190727_060333, and L1 at 30 Hz for GW190814.</li>
      <li>GW170608 (H1 30 Hz), GW190413_134308 (L1 35 Hz), and GW190514_065416 (L1 50 Hz) also have raised cutoffs in the embedded configs. They are displayed but not classified as DQ-cutoff mitigation because Table I does not attribute those settings to data quality.</li>
      <li>For the seven subtracted cases, the public glitch release says the <code>..._T1700406_v4</code> channel is the calibrated strain with the glitch model removed and was used for parameter estimation. The prepared templates select that channel.</li>
      <li>All other events are marked “no event-specific mitigation reported”; this does not claim their data are artifact-free.</li>
      <li>GW190425 has empty <code>config_file</code>, PSD, calibration-envelope, and analytic-prior groups in the v2 HDF5 release. Its template uses released command-line metadata, estimates the PSD from strain, disables calibration marginalization, and uses a documented reconstructed low-spin BNS prior.</li>
    </ul>
  </section>
  <section><h2>Sources</h2><ul>
    <li><a href="{SOURCES["release"]}">GWTC-2.1 parameter-estimation release</a> — event HDF5 files and embedded metadata.</li>
    <li><a href="{SOURCES["paper"]}">GWTC-2.1 catalog paper</a> — Table I and Sections II B/V.</li>
    <li><a href="{SOURCES["glitches"]}">GWTC-2.1 glitch-model release</a> — seven events, channel definitions, and public frames.</li>
    <li><a href="{SOURCES["documentation"]}">GWOSC GWTC-2.1 documentation</a> — calibration channels and special cases.</li>
  </ul></section>
</main></body></html>
"""
    (ROOT / "data_quality.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.worker:
        item = json.loads(args.event_json)
        result = prepare_event(item, args.local_h5_dir)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.worker_event:
        item = next(
            item
            for item in fetch_record()
            if item["record_event"] == args.worker_event
            or event_name(item["record_event"]) == args.worker_event
        )
        prepare_event(item, args.local_h5_dir)
        print(item["record_event"], flush=True)
        return 0

    if args.index_only:
        metadata = []
        for item in fetch_record():
            path = ROOT / "metadata" / f"{event_name(item['record_event'])}.json"
            if path.is_file():
                metadata.append(json.loads(path.read_text(encoding="utf-8")))
        metadata.sort(key=lambda item: item["record_event"])
        make_manifest(metadata)
        make_html(metadata)
        print(f"Indexed {len(metadata)} events in {ROOT}")
        return 0

    requested = set(args.event or [])
    items = fetch_record()
    if requested:
        items = [
            item
            for item in items
            if item["record_event"] in requested
            or event_name(item["record_event"]) in requested
        ]
        found = {item["record_event"] for item in items} | {
            event_name(item["record_event"]) for item in items
        }
        missing = requested - found
        if missing:
            raise ValueError(f"Unknown event(s): {', '.join(sorted(missing))}")

    for index, item in enumerate(items, start=1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--event-json",
            json.dumps(item),
        ]
        if args.local_h5_dir:
            command.extend(["--local-h5-dir", str(args.local_h5_dir.resolve())])
        print(f"[{index:02d}/{len(items):02d}] {item['record_event']}", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    metadata = []
    for item in fetch_record():
        path = ROOT / "metadata" / f"{event_name(item['record_event'])}.json"
        if path.is_file():
            metadata.append(json.loads(path.read_text(encoding="utf-8")))
    metadata.sort(key=lambda item: item["record_event"])
    make_manifest(metadata)
    make_html(metadata)
    print(f"Prepared {len(metadata)} events in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
