"""Phase 5 -- run both models at real chain length and cache the results.

    python scripts/run_full_fits.py              # both models
    python scripts/run_full_fits.py national     # just the national one
    python scripts/run_full_fits.py geo --cores 1

4 chains x 1000 tune x 1000 draws by default. Four chains because R-hat compares
between-chain to within-chain variance and two chains estimate the former very badly -- every "R-hat >
1.01" note in Phases 2-4 came from a 2-chain run and could not be read as more than a
hint. 1000/1000 is pymc's own default, chosen here before seeing whether the
diagnostics pass rather than raised until they did.

`--draws` exists for the geo model, which is run at 500 rather than 1000 on this
machine. Not a statistical choice: its `channel_contribution` and `control_contribution`
arrays are (draws x 113 dates x 26 geos x channels), about 1.7GB of deterministics at
1000 draws, and the run gets OOM-killed. pymc-marketing samples the free RVs and then
recomputes deterministics in one vectorised pass, so that peak cannot be avoided by
restricting `var_names`. Four chains at 500 draws still gives 2,000 posterior draws, and
the ESS threshold rather than the draw count is what decides whether that was enough.

THE `__main__` GUARD IS LOAD-BEARING
------------------------------------
Windows has no `fork`, so pymc's parallel sampling spawns fresh interpreters that
re-import the calling module. Without the guard below, each child re-executes the
sampling call and pymc raises

    RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

Hit while testing whether the `cores=1` workaround in docs/CHALLENGES.md #1 was still
needed. It is not a pymc bug and it is not the ZeroDivisionError that workaround was
written for -- see docs/CHALLENGES.md #8.
"""
import argparse
import time

from mmm_bayes.config import GEO_FIT_NC, GEO_FIT_TARGET_REL_NC, NATIONAL_FIT_NC
from mmm_bayes.diagnostics import (
    FULL_CHAINS,
    FULL_DRAWS,
    FULL_TUNE,
    divergence_locations,
    energy_summary,
    save_idata,
    summarise,
)
from mmm_bayes.geo_model import run_geo_fit
from mmm_bayes.model import run_skeleton_fit

# name -> (fit function, extra kwargs, cache path, drop per-observation deterministics)
# The geo fit is cached without its deterministics: they are 1.1GB of a 1.1GB file and
# nothing reads them back (attribution refuses panel fits). The national fit keeps
# everything, because its contributions are the deliverable.
FITS = {
    "national": (run_skeleton_fit, {"anchored": True}, NATIONAL_FIT_NC, False),
    "geo": (run_geo_fit, {"pooled": True}, GEO_FIT_NC, True),
    # Phase 6. Same model, same priors, same sampler settings -- the ONLY difference is
    # the channel divisor, so a difference in the identification report is attributable
    # to the scaling and to nothing else.
    "geo-target-relative": (
        run_geo_fit,
        {"pooled": True, "channel_scaling": "target-relative"},
        GEO_FIT_TARGET_REL_NC,
        True,
    ),
}


def run_one(name: str, cores: int, target_accept: float, draws: int, tune: int) -> None:
    fit_fn, kwargs, path, drop_derived = FITS[name]

    print("=" * 78)
    print(f"{name.upper()} -- {FULL_CHAINS} chains x {draws} draws "
          f"({tune} tune), target_accept={target_accept}, cores={cores}")
    print("=" * 78)

    started = time.perf_counter()
    _, result = fit_fn(
        draws=draws,
        tune=tune,
        chains=FULL_CHAINS,
        target_accept=target_accept,
        cores=cores,
        progressbar=False,
        **kwargs,
    )
    elapsed = time.perf_counter() - started
    idata = result["idata"]

    print(f"\nSampled in {elapsed / 60:.1f} minutes.\n")
    report = summarise(idata, label=name)
    print(report)

    print("\nper-chain energy:")
    print(energy_summary(idata).round(4).to_string())

    if report.divergences:
        print("\nwhere the divergences fell (largest shift in posterior sd units):")
        print(divergence_locations(idata).round(3).to_string())

    print("\nworst entries by R-hat:")
    print(report.worst.round(4).to_string())

    # `drop_derived` was unpacked from FITS and then never passed here, so every
    # cache this script regenerated kept its per-observation deterministics: the geo
    # fit came back at 1,113MB against the 79MB the footprint fix in commit 11b6260
    # documented and docs/AUDIT.md still claims. Ruff does not flag an unused
    # tuple-unpacking target, and no test regenerates a cache, so nothing caught it --
    # the size of the file this script wrote did.
    print(f"\nSaved to {save_idata(idata, path, drop_derived=drop_derived)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=list(FITS), choices=list(FITS) + [],
                        help="which models to fit (default: both)")
    parser.add_argument("--cores", type=int, default=1,
                        help="parallel chains. 1 is safest here: this machine has 2 "
                             "physical cores and the geo model's four chains at "
                             "cores=2 have been OOM-killed.")
    parser.add_argument("--target-accept", type=float, default=0.99,
                        help="0.99 by default -- see docs/CHALLENGES.md #4")
    parser.add_argument("--draws", type=int, default=FULL_DRAWS)
    parser.add_argument("--tune", type=int, default=FULL_TUNE)
    args = parser.parse_args()

    for model_name in args.models or list(FITS):
        run_one(model_name, cores=args.cores, target_accept=args.target_accept,
                draws=args.draws, tune=args.tune)
