"""Full-event one-factor sampler controls on one immutable CBC+1SG target."""
import argparse
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path

import bilby
import bilby_pipe
import bilby_pipe.data_analysis as analysis
import dynesty
import numpy as np

PERIODIC = ["phi_12", "phi_jl", "psi", "phase", "sine_gaussian_0_phase_offset", "azimuth"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def encode(item):
        return item.tolist() if isinstance(item, np.ndarray) else item.item()
    Path(path).write_text(json.dumps(value, indent=2, default=encode) + "\n")


def sampled_point(priors, supplied):
    """Discard derived posterior columns before changing sampled coordinates."""
    point = {}
    for key, prior in priors.items():
        if isinstance(prior, bilby.core.prior.Constraint):
            continue
        if getattr(prior, "is_fixed", False):
            point[key] = prior.peak
        elif isinstance(prior, (int, float)):
            point[key] = prior
        else:
            point[key] = supplied[key]
    return point


def load_target(settings_path, outdir, seed):
    settings = json.loads(Path(settings_path).read_text())
    target = settings["target"]
    outdir.mkdir(parents=True, exist_ok=True)
    for key in ["complete_config", "data_dump"]:
        assert sha(target[key]) == target[key + "_sha256"], key
    args, unknown = analysis.parse_args(
        [target["complete_config"], "--data-dump-file", target["data_dump"],
         "--label", outdir.name, "--outdir", str(outdir),
         "--sampling-seed", str(seed)], analysis.create_analysis_parser())
    inputs = analysis.DataAnalysisInput(args, unknown)
    likelihood, priors = inputs.get_likelihood_and_priors()
    assert inputs.waveform_approximant == "NRSur7dq4"
    assert bool(likelihood.distance_marginalization) == target["distance_marginalization"]
    assert all(key in priors for key in PERIODIC)
    assert len([key for key in priors if key.startswith("sine_gaussian_")]) == 5
    distance = likelihood.priors["luminosity_distance"] if likelihood.distance_marginalization else priors["luminosity_distance"]
    assert isinstance(distance, bilby.gw.prior.UniformSourceFrame)
    assert (distance.minimum, distance.maximum) == (10, 20000)
    point = sampled_point(priors, target["baseline"]["parameters"])
    lnLR = float(likelihood.log_likelihood_ratio(dict(point)))
    np.testing.assert_allclose(lnLR, target["baseline"]["log_likelihood_ratio"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(likelihood.noise_log_likelihood(), target["noise_log_likelihood"], atol=1e-8, rtol=0)
    replays = []
    for record in target.get("likelihood_replays", []):
        replay = float(likelihood.log_likelihood_ratio(sampled_point(priors, record["parameters"])))
        expected = record.get("log_likelihood_ratio", record.get("logLR"))
        np.testing.assert_allclose(replay, expected, atol=1e-8, rtol=0)
        replays.append(dict(expected=expected, actual=replay))
    provenance = dict(settings_sha256=sha(settings_path), seed=seed, target=target,
        baseline_log_likelihood_ratio=lnLR, noise_log_likelihood=float(likelihood.noise_log_likelihood()),
        bilby=bilby.__version__, bilby_pipe=bilby_pipe.__version__, dynesty=dynesty.__version__,
        likelihood_source_sha256=sha(inspect.getsourcefile(type(likelihood))))
    provenance["likelihood_replays"] = replays
    provenance["sampled_baseline_parameters"] = point
    return inputs, likelihood, priors, provenance


def main():
    from sampling_variants import CASES, apply_variant
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    multiprocessing.set_start_method("fork", force=True)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    variant = apply_variant(args.case, outdir)
    inputs, likelihood, priors, provenance = load_target(args.settings, outdir, args.seed)
    assert not likelihood.distance_marginalization
    calibration = [key for key, prior in priors.items() if key.startswith("recalib_") and not prior.is_fixed]
    assert len(calibration) == 40
    for key in calibration:
        priors[key].boundary = "reflective" if variant["reflection"] else None
    periodic = [key for key, prior in priors.items() if getattr(prior, "boundary", None) == "periodic"]
    assert periodic == PERIODIC
    kwargs = dict(inputs.sampler_kwargs)
    kwargs.update(nlive=2500, naccept=60, npool=8, sample="acceptance-walk",
                  maxmcmc=variant["maxmcmc"], sampling_seed=args.seed)
    assert "ncdim" not in kwargs
    kwargs.pop("seed", None)
    # Instantiate the real wrapper to verify the selected internal sampler.
    cls = bilby.core.sampler.get_sampler_class("dynesty")
    wrapper = cls(likelihood, priors, outdir=str(outdir / "preflight"),
                  label="sampler_preflight", use_ratio=True,
                  skip_import_verification=True, soft_init=True, **kwargs)
    from dynesty.dynesty import _get_internal_sampler
    native_kwargs = wrapper.sampler_init_kwargs
    internal = _get_internal_sampler(native_kwargs["sample"], wrapper.ndim,
        native_kwargs["ncdim"] or wrapper.ndim, native_kwargs["periodic"],
        native_kwargs["reflective"], native_kwargs["walks"], native_kwargs["slices"], native_kwargs["facc"])
    assert type(internal).__name__ == "EnsembleWalkSampler"
    assert wrapper.sampler_init_kwargs["bound"] == "none"
    assert internal.naccept == 60 and internal.maxmcmc == variant["maxmcmc"]
    reflected = internal.sampler_kwargs["reflective"]
    assert (0 if reflected is None else len(reflected)) == (40 if variant["reflection"] else 0)
    assert len(internal.sampler_kwargs["periodic"]) == 6
    assert wrapper.ndim == 60
    provenance.update(variant=variant, sampler_kwargs=kwargs, dimensions=wrapper.ndim,
                      use_ratio=True, native_evidence_convention="log_bayes_factor",
                      calibration_parameters=calibration, periodic_parameters=periodic,
                      scope="full real CBC+coherent1SG event; one sampler ingredient differs from common control")
    path = outdir / "provenance.json"
    if path.exists():
        assert json.loads(path.read_text()) == provenance, "Output belongs to another case or source"
    else:
        write(path, provenance)
    if args.preflight:
        write(outdir / "preflight.json", dict(passed=True, native_sampler_checked=True,
              event_point_replayed=True, production_sampling_started=False))
        return
    inputs.meta_data["sampler_investigation"] = provenance
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler="dynesty",
        use_ratio=True,
        label=outdir.name, outdir=str(outdir / "result"),
        conversion_function=inputs.parameter_generation,
        injection_parameters=inputs.meta_data.get("injection_parameters"),
        meta_data=inputs.meta_data, result_class=inputs.result_class,
        save=inputs.result_format, **kwargs)
    write(outdir / "summary.json", dict(case=args.case, seed=args.seed,
        log_evidence=float(result.log_evidence), log_evidence_error=float(result.log_evidence_err),
        log_bayes_factor=float(result.log_bayes_factor), posterior_rows=len(result.posterior),
        convergence="Pending independent-chain and trajectory assessment"))


if __name__ == "__main__":
    main()
