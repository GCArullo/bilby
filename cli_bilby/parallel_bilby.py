"""Launch a bilby_pipe analysis with parallel_bilby's MPI pool."""

import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    from bilby_pipe.data_analysis import create_analysis_parser
    from bilby_pipe.main import parse_args
    from bilby_pipe.utils import SAMPLER_SETTINGS, convert_string_to_dict

    arguments = sys.argv[1:]
    if arguments[:1] == ["--worker"]:
        from bilby.core.utils import setup_logger
        from bilby_pipe.data_analysis import main as analysis_main
        from mpi4py import MPI

        sys.argv = [sys.argv[0], *arguments[1:]]
        if MPI.COMM_WORLD.rank != 0:
            # Only the master checkpoints on Bilby-pipe's periodic eviction.
            sys.argv.extend(["--periodic-restart-time", "0"])
        setup_logger()
        try:
            analysis_main()
        except BaseException as error:
            code = error.code if isinstance(error, SystemExit) else 1
            if code not in (None, 0):
                if not isinstance(error, SystemExit):
                    traceback.print_exc()
                    sys.stderr.flush()
                # Workers may be sending results when the master checkpoints.
                # Abort after the checkpoint to stop every rank without hanging.
                MPI.COMM_WORLD.Abort(code if isinstance(code, int) else 1)
            raise
        return

    parser = create_analysis_parser(usage=__doc__)
    args, _ = parse_args(arguments, parser)
    if args.sampler != "dynesty":
        parser.error("parallel_bilby requires sampler=dynesty")
    if args.request_cpus < 2:
        parser.error("parallel_bilby requires request-cpus >= 2 (master plus workers)")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        parser.error("mpiexec is not installed; install an MPI runtime")
    import parallel_bilby.schwimmbad_fast  # noqa: F401

    presets = {
        "default": "DynestyDefault",
        "dynestydefault": "DynestyDefault",
        "fasttest": "FastTest",
    }
    kwargs = dict(SAMPLER_SETTINGS["DynestyDefault"])
    if args.sampler_kwargs:
        preset = presets.get(args.sampler_kwargs.lower())
        kwargs.update(
            SAMPLER_SETTINGS[preset] if preset else
            convert_string_to_dict(args.sampler_kwargs, "sampler-kwargs")
        )
    else:
        kwargs = {}
    kwargs.update(pool="parallel_bilby")
    (Path(args.outdir) / "result").mkdir(parents=True, exist_ok=True)
    command = [
        mpiexec, "-n", str(args.request_cpus), sys.executable,
        "-m", "cli_bilby.parallel_bilby", "--worker", *arguments,
        "--sampler-kwargs", repr(kwargs),
    ]
    code = subprocess.run(command).returncode
    result_file = Path(args.outdir) / "result" / f"{args.label}_result.{args.result_format}"
    if code == 0 and (not result_file.is_file() or result_file.stat().st_size == 0):
        # MPIPoolFast can abort a failed worker with exit code zero.
        print("MPI exited without a completed result; marking the analysis as failed.", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
