import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "gw_examples"
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "monitor_runs.py"
)


def load_monitor_runs_module():
    spec = importlib.util.spec_from_file_location(
        "monitor_runs_test_module",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resubmitted_job_is_matched_by_outdir(monkeypatch, tmp_path, capsys):
    module = load_monitor_runs_module()
    root = tmp_path / "event" / "SEOB"
    run_dir = root / "student_detector_independent_noise_N1"
    label = "event_student_analysis_H1L1_par0"
    submit_dir = run_dir / "submit"
    submit_dir.mkdir(parents=True)
    (submit_dir / f"{label}.submit").touch()

    queued = {
        "ClusterId": 20,
        "ProcId": 0,
        "JobStatus": 2,
        "Iwd": str(tmp_path),
        "Args": (
            "bilby_pipe_analysis config.ini "
            f"--outdir event/SEOB/{run_dir.name} --label {label}"
        ),
    }
    old_history = {
        "ClusterId": 10,
        "ProcId": 0,
        "ExitCode": 77,
        "ExitBySignal": False,
        "Iwd": str(root),
        "Args": (
            "bilby_pipe_analysis config.ini "
            f"--outdir {run_dir.name} --label {label}"
        ),
    }

    monkeypatch.setattr(module, "latest_dz", lambda *_: "dZ 1 (target 0.1)")
    module.print_snapshot(root, [queued], [old_history], colour=False)

    output = capsys.readouterr().out
    assert "RUNNING" in output
    assert "FAILED" not in output
    assert module.active_roots([queued]) == [root]
