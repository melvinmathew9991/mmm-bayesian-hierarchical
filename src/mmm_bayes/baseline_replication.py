"""Phase 1 -- replicate the ridge project's fit inside this repo.

This is the anchor every later Bayesian claim gets compared against. It's not
busywork: it catches data-loading drift early (are we handed the same national
frame, the same controls, the same channels?) rather than discovering a mismatch
after weeks of Bayesian modelling, and it gives Phase 9 (ridge vs. Bayesian
comparison) a same-repo, same-environment ridge fit to compare against instead of
numbers copied from a README.

Imports the ridge project directly (installed editable from ../mmm-marketing-project as the
`mmm` package) rather than reimplementing fit_constrained_ridge, run_mmm, etc. here.
Duplicating already-validated code would be worse practice than importing it, and
it would defeat the point of the comparison: if this repo's copy of the ridge model
ever diverges from the real one, Phase 9's comparison would be comparing against a
fork, not the actual validated project.
"""
import pandas as pd

# `mmm` is the ridge project's own package, installed editable from the sibling repo.
# build_controls is imported but unused on purpose: Phase 9's comparison must build
# controls from the same source the ridge fit used, and naming it here is what makes
# that dependency visible instead of rediscovered later.
from mmm.features import build_controls  # noqa: F401
from mmm.loaders import PAID_CHANNELS, load_national
from mmm.model import run_mmm

from mmm_bayes.loaders import load_national as load_national_local


def verify_data_matches() -> None:
    """Confirm this repo's copied CSV is byte-identical in shape/content to the
    ridge project's own load, before trusting any comparison built on it."""
    local = load_national_local()
    upstream = load_national()

    assert local.shape == upstream.shape, (
        f"Shape mismatch: local {local.shape} vs. ridge project {upstream.shape}. "
        "The copied CSV in this repo's data/raw/ may be stale -- re-copy it from "
        "mmm-marketing-project/data/raw/national_weekly.csv."
    )
    pd.testing.assert_frame_equal(local, upstream, check_exact=True)
    print(f"Data verified identical: {local.shape[0]} weeks, {local.shape[1]} columns.")


def replicate_ridge_baseline(verbose: bool = True) -> dict:
    """Refit the ridge project's exact model against the data as loaded in THIS repo.

    Returns the same result dict as mmm.model.run_mmm: fit, media, controls, decays,
    halves, alpha, r2_in_sample, r2_oos, mape_oos, lift, nested.
    """
    df = load_national_local()  # local copy, not the ridge project's own load
    result = run_mmm(df=df, verbose=verbose, nested=True)
    return result


def summarize(result: dict) -> pd.DataFrame:
    """Channel-level summary table: ridge point estimates, to be placed side by
    side with Bayesian posterior means/intervals in Phase 9."""
    fit = result["fit"]
    media = result["media"]
    contrib = fit.contributions(media.values)
    total = contrib.sum(axis=0)

    return pd.DataFrame({
        "channel": PAID_CHANNELS,
        "decay": [result["decays"][c] for c in PAID_CHANNELS],
        "half_point": [result["halves"][c] for c in PAID_CHANNELS],
        "ridge_coef": fit.media_coef,
        "ridge_contribution": total,
    }).set_index("channel")


if __name__ == "__main__":
    pd.set_option("display.width", 130)
    print("=" * 74)
    print("PHASE 1 -- BASELINE REPLICATION")
    print("=" * 74)
    verify_data_matches()
    print()
    result = replicate_ridge_baseline(verbose=True)
    print("\n" + "=" * 74)
    print("PHASE 1 SUMMARY TABLE (anchor for later Bayesian comparison)")
    print("=" * 74)
    print(summarize(result).round(3).to_string())
