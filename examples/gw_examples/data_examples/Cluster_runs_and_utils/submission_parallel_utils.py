"""Select parallel_bilby's MPI pool for the runbook submission launchers."""

from submission_sine_gaussian_utils import positive_int


def add_parallel_arguments(parser):
    parser.add_argument(
        "--parallel-bilby", action="store_true",
        help="Use parallel_bilby's MPI pool with the current Bilby Dynesty driver.",
    )
    parser.add_argument(
        "--mpi-cpus", type=positive_int, default=8,
        help="MPI ranks per analysis job, including the master (default: 8).",
    )


def apply_parallel_settings(ini, *, enabled, cpus, replace_line):
    if not enabled:
        return ini
    if cpus < 2:
        raise ValueError("--parallel-bilby requires --mpi-cpus >= 2")
    for key, value in {
        "sampler": "dynesty",
        "analysis-executable": "bilby_parallel_analysis",
        "request-cpus": str(cpus),
        "htcondor-strict-cpu-request": "True",
    }.items():
        ini = replace_line(ini, key, value)
    return ini
