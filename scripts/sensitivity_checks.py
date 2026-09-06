"""Phase 5 -- the open items Phases 2-4 explicitly handed forward.

    python scripts/sensitivity_checks.py init          # CHALLENGES #2
    python scripts/sensitivity_checks.py accept        # CHALLENGES #4
    python scripts/sensitivity_checks.py vidtr         # PRIORS.md
    python scripts/sensitivity_checks.py trend         # HIERARCHY.md
    python scripts/sensitivity_checks.py decay-pooling # HIERARCHY.md

Each of these was written down as an open question rather than guessed at, and each is
answered here by running the comparison rather than by argument. They use SHORTER chains
than `run_full_fits.py` on purpose: a sensitivity check asks whether a posterior MOVES
when an assumption changes, and that comparison is paired -- both arms see the same seed,
the same data and the same chain length, so the sampling noise largely cancels.
Establishing that either arm has converged is a different question, and
`run_full_fits.py` is where it is asked.

Read every result through `diagnostics.compare_posteriors`, whose unit is the reference
posterior's own standard deviation. An absolute shift means nothing without knowing how
wide the posterior is; 0.2 sd is noise and 2 sd is the conclusion changing.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior

from mmm_bayes.diagnostics import DERIVED_VARS, compare_posteriors, summarise
from mmm_bayes.geo_model import build_geo_model, hierarchical_config, run_geo_fit
from mmm_bayes.loaders import GEO_CHANNELS, GEO_DIM, PAID_CHANNELS
from mmm_bayes.model import run_skeleton_fit
from mmm_bayes.priors import (
    DECAY_CONCENTRATION,
    adstock_alpha_prior,
    anchor_table,
    saturation_lam_prior,
)

# Paired-comparison settings. Same seed both arms, so a difference is the assumption
# and not the random stream.
#
# Full chain length for the national checks, not the shorter runs Phases 2-4 used. That
# is not thoroughness for its own sake: at 2 chains x 500 the national model showed 2
# divergences, and at 4 x 1000 it shows 48. Divergences are counted per draw, so a short
# run does not just estimate the rate imprecisely -- it makes a real problem look like
# rounding error. Anything asking a question about divergences has to run at length.
PAIRED = {"draws": 1000, "tune": 1000, "chains": 4, "cores": 1,
          "random_seed": 42, "progressbar": False}

# The geo model is ~8x the national one's cost per draw. These checks ask whether a
# posterior MOVES, which is a paired comparison and tolerates a shorter chain; they do
# not ask whether either arm has converged.
PAIRED_GEO = {"draws": 500, "tune": 500, "chains": 2, "cores": 1,
              "random_seed": 42, "progressbar": False}


def _caught_overflow(record) -> int:
    return sum(1 for w in record if "overflow" in str(w.message).lower())


def slim(idata):
    """Drop the per-observation derived arrays from a posterior, in place.

    A 4 x 1000 national fit carries `channel_contribution` (4000 x 209 x 10) and
    `control_contribution` (4000 x 209 x 19) -- ~200MB per fit before anything else. A
    check that holds three arms at once to compare them was being OOM-killed on this
    machine. Every comparison below is over model PARAMETERS, so the derived arrays can
    go as soon as whatever needed them has been computed.
    """
    for name in DERIVED_VARS:
        if name in idata.posterior:
            del idata.posterior[name]
    return idata


def check_init() -> None:
    """docs/CHALLENGES.md #2: is the overflow RuntimeWarning the jitter phase?

    The working conclusion since Phase 2 has been that `RuntimeWarning: overflow
    encountered in dot` is a benign transient from NUTS's `jitter+adapt_diag` start,
    after two scaling fixes failed to remove it. That conclusion was never tested
    directly -- the obvious test is to remove the jitter and see whether the warning
    goes with it. `init="adapt_diag"` does exactly that and nothing else.
    """
    print("=" * 78)
    print("CHALLENGES #2 -- does the overflow warning come from the jitter phase?")
    print("=" * 78)

    results = {}
    for init in ("jitter+adapt_diag", "adapt_diag"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, result = run_skeleton_fit(anchored=True, init=init, **PAIRED)
        idata = result["idata"]
        print(f"\ninit={init!r}")
        print(f"  overflow warnings : {_caught_overflow(record)}")
        print(f"  divergences       : {int(idata.sample_stats['diverging'].sum())}")
        print(f"  {summarise(idata, init)}".replace("\n", "\n  "))
        results[init] = slim(idata)

    print("\nposterior shift, adapt_diag vs jitter+adapt_diag (top rows):")
    print(compare_posteriors(
        results["jitter+adapt_diag"], results["adapt_diag"],
        ["adstock_alpha", "saturation_lam"],
        "jitter", "no_jitter",
    ).head(8).round(4).to_string())


def check_target_accept() -> None:
    """docs/CHALLENGES.md #4: pick a target_accept on evidence, not on the count.

    Phase 3 found 2 divergences at 0.9, 3 at 0.95 and 0 at 0.99 on a 2-chain run, and
    deliberately left the default alone for Phase 5 to settle. Settling it means
    checking two things, not one: that the divergences go, AND that the posterior does
    not move when they do. A step size small enough to hide a real geometry problem
    would clear the count while shifting the answer.
    """
    print("=" * 78)
    print("CHALLENGES #4 -- target_accept for the national model")
    print("=" * 78)

    fits = {}
    for target_accept in (0.9, 0.95, 0.99):
        _, result = run_skeleton_fit(anchored=True, target_accept=target_accept, **PAIRED)
        idata = result["idata"]
        report = summarise(idata, f"target_accept={target_accept}")
        step = float(idata.sample_stats["step_size"].mean())
        print(f"\n{report}")
        print(f"  mean step size {step:.5f}")
        fits[target_accept] = slim(idata)

    print("\nposterior shift, 0.99 vs 0.9 (top rows):")
    print(compare_posteriors(
        fits[0.9], fits[0.99], ["adstock_alpha", "saturation_lam", "saturation_beta"],
        "ta_0.9", "ta_0.99",
    ).head(8).round(4).to_string())


def check_vidtr_prior() -> None:
    """docs/PRIORS.md: `vidtr` carries the tightest decay prior of the ten.

    Its ridge decay came back at exactly 0.0, the lower edge of the search grid, and a
    Beta with a mean clipped to 0.05 cannot be wide -- a 90% interval of [0.00, 0.23]
    against roughly [0.02, 0.63] for the library default. PRIORS.md accepted that on the
    argument that zero is a hard edge of the parameter space rather than censoring, and
    flagged it as the one channel most deserving a sensitivity check. This is it: refit
    with `vidtr` alone moved to the library default and see whether its posterior was
    being held there by the prior.
    """
    print("=" * 78)
    print("PRIORS.md -- is vidtr's decay posterior prior-driven?")
    print("=" * 78)

    table = anchor_table()
    index = PAID_CHANNELS.index("vidtr")

    relaxed = table.copy()
    relaxed.iloc[index, relaxed.columns.get_loc("beta_a")] = 1.0
    relaxed.iloc[index, relaxed.columns.get_loc("beta_b")] = 3.0
    print(f"vidtr decay prior: Beta({table['beta_a'].iloc[index]:.2f}, "
          f"{table['beta_b'].iloc[index]:.2f})  ->  Beta(1, 3) [library default]")
    print(f"(every other channel unchanged, concentration {DECAY_CONCENTRATION:g})")

    from pymc_marketing.mmm import MMM, GeometricAdstock, InverseScaledLogisticSaturation
    from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling

    from mmm_bayes.model import ADSTOCK_L_MAX, build_dataframe

    X, y = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]
    fits = {}
    for label, anchors in (("anchored", table), ("vidtr_relaxed", relaxed)):
        model = MMM(
            date_column="date", channel_columns=PAID_CHANNELS,
            control_columns=control_columns, target_column="y",
            adstock=GeometricAdstock(l_max=ADSTOCK_L_MAX,
                                     priors={"alpha": adstock_alpha_prior(anchors)}),
            saturation=InverseScaledLogisticSaturation(
                priors={"lam": saturation_lam_prior(anchors)}),
            scaling=Scaling(target=DataDerivedScaling(method="max", dims=()),
                            channel=DataDerivedScaling(method="max", dims=())),
        )
        idata = model.fit(X=X, y=y.rename("y"), **PAIRED)
        print(summarise(idata, label))
        fits[label] = slim(idata)

    # compare_posteriors labels rows by channel and sorts by |shift|, so the channel
    # that moved most is the first row -- which is the question being asked.
    shift = compare_posteriors(fits["anchored"], fits["vidtr_relaxed"],
                               ["adstock_alpha"], "anchored", "relaxed")
    print("\ndecay posteriors under both priors:")
    print(shift.round(4).to_string())


def check_trend() -> None:
    """docs/HIERARCHY.md: `Google_Impressions` decay came back at 0.979 (sd 0.012).

    A geometric decay that close to 1 makes the adstocked series behave like a slow
    level rather than a carryover, which is what a channel absorbing an unmodelled trend
    looks like. The geo panel has no trend term -- the national model's controls include
    one, the geo model's do not, because the geo data ships with no controls at all.
    Adding a linear trend is the direct test: if the decay drops, the 0.979 was the
    channel standing in for the trend.
    """
    print("=" * 78)
    print("HIERARCHY.md -- is Google_Impressions' 0.979 decay an unmodelled trend?")
    print("=" * 78)

    from pymc_marketing.mmm import MMM, GeometricAdstock, InverseScaledLogisticSaturation
    from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling

    from mmm_bayes.geo_features import PEAK_WEEK_COLS
    from mmm_bayes.geo_model import (
        GEO_ADSTOCK_L_MAX,
        GEO_YEARLY_SEASONALITY,
        build_geo_dataframe,
    )

    X, y = build_geo_dataframe()
    dates = pd.DatetimeIndex(X["date"])
    ordinal = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    X = X.assign(trend=(ordinal - ordinal.min()) / (ordinal.max() - ordinal.min()))

    config = hierarchical_config()
    controls = PEAK_WEEK_COLS + ["trend"]
    config["gamma_control"] = Prior(
        "Normal",
        mu=Prior("Normal", mu=0, sigma=0.18, dims="control"),
        sigma=Prior("HalfNormal", sigma=0.05, dims="control"),
        dims=(GEO_DIM, "control"), centered=False,
    )
    with_trend = MMM(
        date_column="date", channel_columns=GEO_CHANNELS, control_columns=controls,
        target_column="y", dims=(GEO_DIM,), yearly_seasonality=GEO_YEARLY_SEASONALITY,
        adstock=GeometricAdstock(l_max=GEO_ADSTOCK_L_MAX),
        saturation=InverseScaledLogisticSaturation(),
        scaling=Scaling(target=DataDerivedScaling(method="max", dims=()),
                        channel=DataDerivedScaling(method="max", dims=())),
        model_config=config,
    )

    _, baseline = run_geo_fit(pooled=True, **PAIRED_GEO)
    trend_idata = with_trend.fit(X=X, y=y, **PAIRED_GEO)

    print("\nadstock decay, without vs with a linear trend control:")
    shift = compare_posteriors(baseline["idata"], trend_idata, ["adstock_alpha"],
                               "no_trend", "with_trend")
    print(shift.round(4).to_string())

    trend_coef = trend_idata.posterior["gamma_control"].sel(control="trend")
    print(f"\ntrend coefficient: mean {float(trend_coef.mean()):.4f}, "
          f"sd {float(trend_coef.std()):.4f} (across geos and draws)")


def check_decay_pooling() -> None:
    """docs/HIERARCHY.md: complete pooling of decay was argued for, not tested.

    The argument was a priori -- carryover is a property of the medium, not the division
    -- and it saved 150 parameters. An a priori argument is still an assumption. The test
    is to let decay vary by division under a hierarchical prior and look at the fitted
    across-geo scale: if it shrinks toward zero, the divisions really do share a decay
    and complete pooling loses nothing.
    """
    print("=" * 78)
    print("HIERARCHY.md -- does decay actually vary by division?")
    print("=" * 78)

    # Beta cannot be given a non-centred hierarchical form, so the partially pooled
    # arm uses a logit-scale Normal squashed back into (0, 1) -- the same trick the
    # positive parameters use, with a different link.
    hierarchical_decay = Prior(
        "Normal",
        mu=Prior("Normal", mu=-1.0, sigma=1.0, dims="channel"),
        sigma=Prior("HalfNormal", sigma=0.5, dims="channel"),
        dims=(GEO_DIM, "channel"), centered=False, transform="sigmoid",
    )
    partially_pooled = build_geo_model(pooled=True)
    # Set on the adstock TRANSFORMATION, not via model_config. `model_config` seeds the
    # transformation's priors at construction time; assigning it afterwards changes the
    # dict and leaves the transformation holding its original prior, so the fit runs
    # happily on the completely pooled decay and the comparison silently compares
    # nothing. Verified by checking the built graph before spending a fit on it.
    partially_pooled.adstock.function_priors = {
        **partially_pooled.adstock.function_priors,
        "alpha": hierarchical_decay,
    }

    _, complete = run_geo_fit(pooled=True, **PAIRED_GEO)
    _, partial = run_geo_fit(pooled=True, model=partially_pooled, **PAIRED_GEO)

    posterior = partial["idata"].posterior
    scale = posterior["adstock_alpha_raw_sigma"].mean(("chain", "draw"))
    print("\nacross-division scale of decay, per channel (partially pooled arm):")
    for channel, value in zip(GEO_CHANNELS, np.asarray(scale).ravel(), strict=True):
        print(f"  {channel:24s} {value:.4f}")

    completely = complete["idata"].posterior["adstock_alpha"].mean(("chain", "draw"))
    partially = posterior["adstock_alpha"].mean(("chain", "draw", GEO_DIM))
    print("\nchannel decay: completely pooled vs the mean of the partially pooled:")
    for channel, c, p in zip(GEO_CHANNELS, np.asarray(completely).ravel(),
                             np.asarray(partially).ravel(), strict=True):
        print(f"  {channel:24s} {c:.4f}   {p:.4f}   diff {p - c:+.4f}")


def check_intercept() -> None:
    """docs/PRIORS.md listed the intercept prior as "not yet reviewed". It is wrong.

    The full national fit attributes **105% of sales to media** (94% HDI 62%-164%)
    against the ridge project's 12.4%. Decomposing the posterior shows why: controls
    contribute ~0% (they are z-scored, so they explain deviations rather than level) and
    the intercept contributes -5%. Media is carrying the entire baseline.

    The cause is pymc-marketing's default `Normal(0, 2)` intercept. That default assumes
    a centred target. Here the target is max-scaled sales, which averages 0.28 and is
    strictly positive, so a prior centred on zero actively pushes the baseline to nothing
    -- and the HalfNormal media coefficients are then the only always-on, non-negative
    component available to lift the fitted line up to the data. They take everything.

    The fix is to centre the intercept where the data's baseline actually is. Note what
    this prior does NOT say: it does not say how much media contributes. It says the
    baseline is near the average sales level, which is a statement about the target's
    units, not about marketing.
    """
    from pymc_extras.prior import Prior as P

    from mmm_bayes.loaders import load_national
    from mmm_bayes.model import build_dataframe, build_model

    print("=" * 78)
    print("PRIORS.md -- the intercept prior, and the 105%-of-sales attribution")
    print("=" * 78)

    sales = load_national()["sales"]
    scaled_mean = float((sales / sales.max()).mean())
    print(f"mean of max-scaled sales: {scaled_mean:.4f}  "
          f"(the library default centres the intercept at 0.0)")

    X, y = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]

    fits = {}
    for label, intercept in (
        ("default N(0,2)", None),
        (f"centred N({scaled_mean:.2f},0.15)", P("Normal", mu=scaled_mean, sigma=0.15)),
    ):
        model = build_model(control_columns, anchored=True)
        if intercept is not None:
            model.model_config = {**model.model_config, "intercept": intercept}
        idata = model.fit(X=X, y=y.rename("y"), **PAIRED)
        fits[label] = idata
        print(f"\n{summarise(idata, label)}")

        total_sales = float(sales.sum())
        scale = float(sales.max())
        media = float((idata.posterior["channel_contribution"].sum(("date", "channel"))
                       * scale).mean())
        base = float((idata.posterior["intercept_contribution"] * len(sales) * scale).mean())
        controls = float((idata.posterior["control_contribution"].sum(("date", "control"))
                          * scale).mean())
        print(f"  media     {media / total_sales:8.1%} of sales")
        print(f"  controls  {controls / total_sales:8.1%}")
        print(f"  intercept {base / total_sales:8.1%}")
        print("  (ridge project's point estimate for media: 12.4%)")
        fits[label] = slim(idata)

    labels = list(fits)
    print(f"\nposterior shift, {labels[1]} vs {labels[0]}:")
    print(compare_posteriors(fits[labels[0]], fits[labels[1]],
                             ["saturation_beta", "adstock_alpha"],
                             "default", "centred").head(10).round(4).to_string())


def check_beta_scale() -> None:
    """The rest of the 105%: `saturation_beta ~ HalfNormal(2)` on a [0, 1]-scaled target.

    Centring the intercept moves media from 104% of sales to 84% -- real, but only a
    fifth of the gap to ridge's 12.4%. This is the rest of it, and it is a scale
    mismatch rather than a modelling subtlety.

    The target is max-scaled, so it lives in [0, 1] and averages 0.31. Each channel's
    contribution is `beta * saturation(x)` with the saturation in [0, 1), so the ten
    betas together set how much of that 0.31 media can claim. For media to be ~12% of
    sales, they need to sum to roughly 0.1. pymc-marketing's default `HalfNormal(2)`
    has a prior MEAN of 1.6 *per channel* -- sixteen times the whole budget, ten times
    over. It places almost no mass in the region ridge's answer occupies.

    That is not an uninformative prior. It is an informative prior for enormous media
    effects, and it is uninformative-looking only because nobody checked it against the
    scale of the thing being modelled.

    Note carefully what this check is NOT. Phase 3 left `saturation_beta` unanchored on
    the argument that anchoring it to ridge's fitted coefficients would prejudge Phase
    9's comparison. That argument still stands and nothing here violates it: narrowing a
    prior to respect the target's scale says nothing about which channel works, or about
    what the total should be. It only stops the prior from insisting the total is huge.
    """
    print("=" * 78)
    print("DIAGNOSTICS.md -- is the media share set by the beta prior rather than data?")
    print("=" * 78)

    from mmm_bayes.loaders import load_national
    from mmm_bayes.model import build_dataframe, build_model

    sales = load_national()["sales"]
    total_sales, scale = float(sales.sum()), float(sales.max())
    X, y = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]

    print(f"target is max-scaled: mean {float((sales / scale).mean()):.3f}, max 1.0")
    print("ridge's answer needs the ten betas to sum to roughly 0.1")
    print()

    rows = []
    for sigma in (2.0, 0.5, 0.1):
        model = build_model(control_columns, anchored=True)
        config = dict(model.model_config)
        config["intercept"] = Prior("Normal", mu=float((sales / scale).mean()), sigma=0.15)
        model.model_config = config
        model.saturation.function_priors = {
            **model.saturation.function_priors,
            "beta": Prior("HalfNormal", sigma=sigma, dims="channel"),
        }
        idata = model.fit(X=X, y=y.rename("y"), **PAIRED)

        media = float((idata.posterior["channel_contribution"].sum(("date", "channel"))
                       * scale).mean()) / total_sales
        base = float((idata.posterior["intercept_contribution"]
                      * len(sales) * scale).mean()) / total_sales
        beta_sum = float(idata.posterior["saturation_beta"].sum("channel").mean())
        report = summarise(idata, f"HalfNormal({sigma})")
        rows.append({
            "beta_prior_sigma": sigma,
            "sum_of_betas": beta_sum,
            "media_share": media,
            "intercept_share": base,
            "divergences": report.divergences,
            "max_rhat": report.max_rhat,
        })
        print(f"HalfNormal({sigma}): media {media:.1%}, intercept {base:.1%}, "
              f"sum(beta) {beta_sum:.3f}, {report.divergences} divergences")
        slim(idata)

    print()
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("ridge point estimate: media 12.4% of sales")


CHECKS = {
    "beta-scale": check_beta_scale,
    "init": check_init,
    "accept": check_target_accept,
    "vidtr": check_vidtr_prior,
    "intercept": check_intercept,
    "trend": check_trend,
    "decay-pooling": check_decay_pooling,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=list(CHECKS))
    CHECKS[parser.parse_args().check]()
