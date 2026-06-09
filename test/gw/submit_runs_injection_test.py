import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / (
    "examples/gw_examples/data_examples/Cluster_runs_and_utils/"
    "submit_runs_injection.py"
)


def import_submit_runs_injection(monkeypatch, module_name="submit_runs_injection_test"):
    script_path = SCRIPT_PATH
    monkeypatch.syspath_prepend(str(script_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_import_does_not_require_gwpy(monkeypatch):
    saved_gwpy_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "gwpy" or name.startswith("gwpy.")
    }
    for name in saved_gwpy_modules:
        del sys.modules[name]

    class BlockGwpyImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "gwpy" or fullname.startswith("gwpy."):
                raise ImportError("blocked gwpy import")
            return None

    finder = BlockGwpyImports()
    sys.meta_path.insert(0, finder)
    try:
        import_submit_runs_injection(monkeypatch)
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved_gwpy_modules)


def test_write_time_series_writes_gwpy_hdf5_layout(monkeypatch, tmp_path):
    module = import_submit_runs_injection(
        monkeypatch,
        module_name="submit_runs_injection_write_test",
    )
    path = tmp_path / "H1_test.hdf5"

    module.write_time_series(
        path,
        detector="H1",
        strain=np.array([1.0, 2.0, 3.0]),
        start_time=123.5,
        sampling_frequency=4.0,
    )

    with h5py.File(path, "r") as h5_file:
        dataset = h5_file["H1_SIM"]
        np.testing.assert_array_equal(dataset[()], [1.0, 2.0, 3.0])
        assert dataset.attrs["dx"] == 0.25
        assert dataset.attrs["name"] == "H1_SIM"
        assert dataset.attrs["unit"] == ""
        assert dataset.attrs["x0"] == 123.5
        assert dataset.attrs["xunit"] == "s"


def test_write_frequency_domain_strain_writes_hdf5_layout(monkeypatch, tmp_path):
    module = import_submit_runs_injection(
        monkeypatch,
        module_name="submit_runs_injection_fd_write_test",
    )
    path = tmp_path / "H1_fd_test.hdf5"

    strain = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    frequencies = np.array([0.0, 0.25])
    module.write_frequency_domain_strain(
        path,
        detector="H1",
        frequency_domain_strain=strain,
        frequencies=frequencies,
        start_time=123.5,
        duration=4.0,
        sampling_frequency=4.0,
    )

    with h5py.File(path, "r") as h5_file:
        np.testing.assert_array_equal(h5_file["frequency_array"][()], frequencies)
        np.testing.assert_array_equal(h5_file["frequency_domain_strain"][()], strain)
        assert h5_file.attrs["detector"] == "H1"
        assert h5_file.attrs["duration"] == 4.0
        assert h5_file.attrs["sampling_frequency"] == 4.0
        assert h5_file.attrs["start_time"] == 123.5


def test_render_ini_frequency_domain_injection_uses_native_fd_loader(monkeypatch, tmp_path):
    module = import_submit_runs_injection(
        monkeypatch,
        module_name="submit_runs_injection_fd_ini_test",
    )
    ini_template = "\n".join(
        [
            "accounting-user=old",
            "queue=None",
            "create-summary=False",
            "summarypages-arguments=None",
            "data-dict=None",
            "data-format=gwf",
            "calibration-model=CubicSpline",
            "calibration-correction-type=data",
            "spline-calibration-envelope-dict={H1: old.dat}",
            "channel-dict=None",
            "psd-dict=None",
            "additional-transfer-paths=None",
            "sampler-kwargs={'nlive': 1}",
            "likelihood-type=old",
            "extra-likelihood-kwargs=old",
            "environment-variables={'OMP_NUM_THREADS'=1}",
            "plot-data=True",
            "plot-spectrogram=True",
            "",
        ]
    )
    args = type(
        "Args",
        (),
        dict(
            accounting_user="acct",
            require_epnfs=False,
            test_injection=False,
            nlive=10,
            naccept=3,
            frequency_domain_injection=True,
        ),
    )()
    template_settings = dict(
        detectors=("H1",),
        minimum_frequency={"H1": 20.0, "waveform": 10.0},
        maximum_frequency=448.0,
        reference_frequency=10.0,
        waveform_approximant="NRSur7dq4",
        sampler_kwargs={"nlive": 1},
    )
    stage_dir = tmp_path / "staged_data"

    rendered = module.render_ini(
        ini_template,
        args=args,
        template_settings=template_settings,
        num_frequency_bands=1,
        detector_dependent_nu=False,
        likelihood_nu=None,
        label="label",
        outdir=tmp_path / "out",
        webdir=tmp_path / "web",
        prior_path=tmp_path / "prior.prior",
        data_paths={"H1": str(tmp_path / "H1_fd.hdf5")},
        psd_paths={"H1": str(tmp_path / "H1_psd.dat")},
        stage_dir=stage_dir,
        hypothesis="gaussian",
        sine_gaussian_config=type(
            "Config",
            (),
            dict(enabled=False, total_components=0),
        )(),
    )

    assert "data-format=bilby_frequency_domain_hdf5" in rendered
    assert "plot-data=False" in rendered
    assert "plot-spectrogram=False" in rendered
    assert "BILBY_FD_DATA_PATCH" not in rendered
    assert "PYTHONPATH" not in rendered
