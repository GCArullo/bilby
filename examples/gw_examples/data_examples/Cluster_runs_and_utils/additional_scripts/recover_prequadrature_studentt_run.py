#!/usr/bin/env python3

"""Recover an older bilby_pipe Student-t run after the quadrature fix.

This utility is intended for run directories created before the Student-t
noise-evidence quadrature path existed. It completes any pending
``log_noise_evidence`` calculation from the saved shard results, updates the
stored likelihood metadata so bilby can use those shard results as cache hits
again, and then performs the normal downstream bilby_pipe steps:

1. merge parallel shard results when required
2. write the lightweight final result file
3. make the configured bilby_pipe plots
4. execute configured post-processing hooks

The script assumes the current Python environment can import both the local
repository checkout of ``bilby`` and the installed ``bilby_pipe`` package.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bilby
import bilby_pipe.data_analysis
import bilby_pipe.main
from bilby.core.utils import logger


RESULT_NAME_REGEX = re.compile(r"(?P<label>.+)_result\.(?P<extension>[^.]+)$")
PARALLEL_LABEL_REGEX = re.compile(r"^(?P<base>.+)_par\d+$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the bilby_pipe run directory to recover.",
    )
    parser.add_argument(
        "--ini",
        type=Path,
        default=None,
        help=(
            "Path to the *_config_complete.ini file. If omitted, the script "
            "looks for exactly one such file inside the run directory."
        ),
    )
    parser.add_argument(
        "--data-dump-file",
        type=Path,
        default=None,
        help=(
            "Path to the generation data dump. If omitted, the script looks "
            "for exactly one *generation_data_dump.pickle file in run_dir/data."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the bilby_pipe plot step.",
    )
    parser.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Skip configured post-processing executables.",
    )
    parser.add_argument(
        "--skip-final-result",
        action="store_true",
        help="Skip writing the lightweight final_result file.",
    )
    return parser


def _resolve_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


def _find_unique_file(run_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Unable to find {description} in {run_dir} using pattern {pattern!r}"
        )
    raise RuntimeError(
        f"Found multiple {description} files in {run_dir}: "
        + ", ".join(str(path) for path in matches)
    )


def _load_main_inputs(run_dir: Path, ini_path: Path):
    parser = bilby_pipe.main.create_main_parser()
    args, unknown_args = bilby_pipe.main.parse_args([str(ini_path)], parser)

    if args.analysis_executable_parser is not None:
        module = ".".join(args.analysis_executable_parser.split(".")[:-1])
        function = args.analysis_executable_parser.split(".")[-1]
        parser = getattr(__import__(module, fromlist=[function]), function)()
        args, unknown_args = bilby_pipe.main.parse_args([str(ini_path)], parser)

    args.outdir = str(run_dir)
    args.result_directory = str((run_dir / "result").resolve())
    args.final_result_directory = str((run_dir / "final_result").resolve())
    Path(args.result_directory).mkdir(parents=True, exist_ok=True)
    Path(args.final_result_directory).mkdir(parents=True, exist_ok=True)
    args.plot_node_needed = any(
        getattr(args, f"plot_{plot_type}", False)
        for plot_type in ["calibration", "corner", "marginal", "skymap", "waveform"]
    )
    return args


def _result_label_from_path(result_path: Path) -> str:
    match = RESULT_NAME_REGEX.match(result_path.name)
    if match is None:
        raise ValueError(f"Unrecognized result filename format: {result_path.name}")
    return match.group("label")


def _discover_analysis_result_files(result_dir: Path, extension: str) -> list[Path]:
    candidates = sorted(result_dir.glob(f"*_result.{extension}"))
    analysis_results = []
    for path in candidates:
        stem = path.stem
        if "_analysis_" not in stem:
            continue
        if "_merge_result" in stem or "_merge_" in stem:
            continue
        if "_reweighted_result" in stem:
            continue
        analysis_results.append(path.resolve())

    if not analysis_results:
        raise FileNotFoundError(
            f"No analysis shard results matching *analysis*_result.{extension} "
            f"were found in {result_dir}"
        )
    return analysis_results


def _merge_label(shard_result_paths: list[Path]) -> str:
    labels = [_result_label_from_path(path) for path in shard_result_paths]
    base_labels = set()
    for label in labels:
        match = PARALLEL_LABEL_REGEX.match(label)
        if match is None:
            base_labels.add(label)
        else:
            base_labels.add(match.group("base"))

    if len(base_labels) != 1:
        raise RuntimeError(
            "Unable to infer a unique merge label from shard results: "
            + ", ".join(labels)
        )
    return f"{base_labels.pop()}_merge"


def _coerce_command_arguments(arguments) -> list[str]:
    if arguments in [None, "None"]:
        return []
    if isinstance(arguments, str):
        return shlex.split(arguments)
    return [str(arg) for arg in arguments]


def _run_command(command: list[str], cwd: Path) -> None:
    executable = shutil.which(command[0]) if not os.path.isabs(command[0]) else command[0]
    if executable is None:
        raise FileNotFoundError(f"Unable to locate executable {command[0]!r}")

    full_command = [executable, *command[1:]]
    logger.info("Running command: %s", " ".join(shlex.quote(part) for part in full_command))
    subprocess.run(full_command, cwd=str(cwd), check=True)


def _is_finite(value) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _update_result_evidence(result, noise_log_evidence: float) -> None:
    result.log_noise_evidence = float(noise_log_evidence)
    if _is_finite(getattr(result, "log_evidence", None)):
        result.log_bayes_factor = float(result.log_evidence - result.log_noise_evidence)
    elif _is_finite(getattr(result, "log_bayes_factor", None)):
        result.log_evidence = float(result.log_bayes_factor + result.log_noise_evidence)
    else:
        raise RuntimeError(
            f"Unable to reconstruct evidences for result {result.label}: "
            "both log_evidence and log_bayes_factor are non-finite"
        )


def _build_analysis_input(
    ini_path: Path,
    run_dir: Path,
    data_dump_file: Path,
    label: str,
    detectors: Iterable[str],
    sampler_name: str,
):
    arguments = [
        str(ini_path),
        "--outdir",
        str(run_dir),
        "--label",
        label,
        "--data-dump-file",
        str(data_dump_file),
        "--sampler",
        sampler_name,
    ]
    for detector in detectors:
        arguments.extend(["--detectors", detector])

    parser = bilby_pipe.data_analysis.create_analysis_parser(usage=__doc__)
    args, unknown_args = bilby_pipe.data_analysis.parse_args(arguments, parser)
    return bilby_pipe.data_analysis.DataAnalysisInput(args, unknown_args)


def _recover_analysis_result(
    result_path: Path,
    ini_path: Path,
    run_dir: Path,
    data_dump_file: Path,
    sampler_name: str,
) -> Path:
    label = _result_label_from_path(result_path)
    result = bilby.core.result.read_in_result(filename=str(result_path))
    detectors = result.meta_data.get("likelihood", {}).get("interferometers", None)
    if not detectors:
        raise RuntimeError(
            f"Unable to determine detector list from {result_path}"
        )

    analysis_input = _build_analysis_input(
        ini_path=ini_path,
        run_dir=run_dir,
        data_dump_file=data_dump_file,
        label=label,
        detectors=detectors,
        sampler_name=sampler_name,
    )
    likelihood, priors = analysis_input.get_likelihood_and_priors()
    if likelihood.__class__.__name__ != "StudentTGravitationalWaveTransient":
        raise RuntimeError(
            f"Run recovery only supports StudentTGravitationalWaveTransient, got "
            f"{likelihood.__class__.__name__} for {label}"
        )

    meta_data = copy.deepcopy(getattr(result, "meta_data", {}) or {})
    meta_data["likelihood"] = copy.deepcopy(likelihood.meta_data)
    meta_data["data_dump"] = str(data_dump_file.resolve())
    needs_noise_evidence = bool(meta_data.get("noise_evidence_pending", False)) or not _is_finite(
        getattr(result, "log_noise_evidence", None)
    )
    meta_data.pop("noise_evidence_pending", None)
    result.meta_data = meta_data

    if needs_noise_evidence:
        logger.info("Computing missing noise evidence for %s", label)
        noise_log_evidence = likelihood.noise_log_evidence(priors=priors)
        _update_result_evidence(result=result, noise_log_evidence=noise_log_evidence)
    elif (
        _is_finite(getattr(result, "log_evidence", None))
        and _is_finite(getattr(result, "log_noise_evidence", None))
        and not _is_finite(getattr(result, "log_bayes_factor", None))
    ):
        result.log_bayes_factor = float(result.log_evidence - result.log_noise_evidence)

    result.save_to_file(
        overwrite=True,
        extension=result_path.suffix.lstrip("."),
        outdir=str(result_path.parent),
    )
    logger.info(
        "Recovered %s: log_evidence=%s log_noise_evidence=%s log_bayes_factor=%s",
        label,
        result.log_evidence,
        result.log_noise_evidence,
        result.log_bayes_factor,
    )
    return result_path


def _apply_max_samples(result, max_samples: int | None):
    if max_samples is not None and len(result.posterior) > max_samples:
        result.posterior = result.posterior.sample(max_samples).sort_index()
    return result


def _apply_lightweight(result):
    for key in ["_nested_samples", "log_likelihood_evaluations", "log_prior_evaluations"]:
        setattr(result, key, None)
    return result


def _write_final_result(
    parent_result_path: Path,
    outdir: Path,
    max_samples: int | None,
    extension: str,
) -> Path:
    result = bilby.core.result.read_in_result(filename=str(parent_result_path))
    result = _apply_max_samples(result=result, max_samples=max_samples)
    result = _apply_lightweight(result=result)
    result.save_to_file(overwrite=True, extension=extension, outdir=str(outdir))
    return outdir / f"{result.label}_result.{extension}"


def _run_plot_step(inputs, parent_result_path: Path, run_dir: Path) -> None:
    if not inputs.plot_node_needed:
        logger.info("Plotting not requested in the run configuration; skipping")
        return

    command = [
        "bilby_pipe_plot",
        "--result",
        str(parent_result_path),
        "--outdir",
        str(inputs.result_directory),
    ]
    for plot_type in ["calibration", "corner", "marginal", "skymap", "waveform"]:
        if getattr(inputs, f"plot_{plot_type}", False):
            command.append(f"--{plot_type}")
    command.extend(["--format", inputs.plot_format])
    _run_command(command, cwd=run_dir)


def _run_postprocessing(inputs, parent_result_path: Path, run_dir: Path) -> None:
    if inputs.single_postprocessing_executable:
        arguments = _coerce_command_arguments(inputs.single_postprocessing_arguments)
        arguments = [argument.replace("$RESULT", str(parent_result_path)) for argument in arguments]
        _run_command([inputs.single_postprocessing_executable, *arguments], cwd=run_dir)

    if inputs.postprocessing_executable:
        arguments = _coerce_command_arguments(inputs.postprocessing_arguments)
        _run_command([inputs.postprocessing_executable, *arguments], cwd=run_dir)


def recover_run(
    run_dir: Path,
    ini_path: Path | None,
    data_dump_file: Path | None,
    skip_plots: bool,
    skip_postprocessing: bool,
    skip_final_result: bool,
) -> None:
    run_dir = _resolve_run_dir(run_dir)
    ini_path = (ini_path.expanduser().resolve() if ini_path else _find_unique_file(
        run_dir, "*_config_complete.ini", "complete bilby_pipe config"
    ))
    data_dump_file = (
        data_dump_file.expanduser().resolve()
        if data_dump_file
        else _find_unique_file(run_dir / "data", "*generation_data_dump.pickle", "generation data dump")
    )

    inputs = _load_main_inputs(run_dir=run_dir, ini_path=ini_path)
    if inputs.likelihood_type != "bilby.gw.likelihood.StudentTGravitationalWaveTransient":
        raise RuntimeError(
            "This recovery script only supports "
            "bilby.gw.likelihood.StudentTGravitationalWaveTransient runs"
        )

    result_dir = Path(inputs.result_directory).resolve()
    shard_result_paths = _discover_analysis_result_files(
        result_dir=result_dir,
        extension=inputs.result_format,
    )
    logger.info("Found %d analysis result file(s) to recover", len(shard_result_paths))

    recovered_paths = [
        _recover_analysis_result(
            result_path=path,
            ini_path=ini_path,
            run_dir=run_dir,
            data_dump_file=data_dump_file,
            sampler_name=inputs.sampler,
        )
        for path in shard_result_paths
    ]

    if len(recovered_paths) == 1:
        parent_result_path = recovered_paths[0]
        logger.info("Single analysis result found; merge step is not required")
    else:
        merge_label = _merge_label(recovered_paths)
        merged = bilby.core.result.ResultList([str(path) for path in recovered_paths]).combine(
            consistency_level="warning"
        )
        merged.label = merge_label
        merged.outdir = str(result_dir)
        merged.save_to_file(overwrite=True, extension=inputs.result_format, outdir=str(result_dir))
        parent_result_path = result_dir / f"{merge_label}_result.{inputs.result_format}"
        logger.info("Wrote merged result to %s", parent_result_path)

    if inputs.final_result and not skip_final_result:
        final_result_path = _write_final_result(
            parent_result_path=parent_result_path,
            outdir=Path(inputs.final_result_directory).resolve(),
            max_samples=inputs.final_result_nsamples,
            extension=inputs.result_format,
        )
        logger.info("Wrote final lightweight result to %s", final_result_path)
    elif inputs.final_result:
        logger.info("Final-result step requested but skipped by command line")
    else:
        logger.info("Final-result step not requested in the run configuration")

    if skip_plots:
        logger.info("Skipping plots by command line request")
    else:
        _run_plot_step(inputs=inputs, parent_result_path=parent_result_path, run_dir=run_dir)

    if skip_postprocessing:
        logger.info("Skipping post-processing by command line request")
    else:
        _run_postprocessing(
            inputs=inputs,
            parent_result_path=parent_result_path,
            run_dir=run_dir,
        )


def main() -> None:
    args = build_parser().parse_args()
    recover_run(
        run_dir=args.run_dir,
        ini_path=args.ini,
        data_dump_file=args.data_dump_file,
        skip_plots=args.skip_plots,
        skip_postprocessing=args.skip_postprocessing,
        skip_final_result=args.skip_final_result,
    )


if __name__ == "__main__":
    main()
