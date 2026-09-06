"""Phase 4 -- hierarchical structure: partial pooling across geographies.

READ THIS FIRST: this is a SECOND model, not a geo-aware version of the national one.

The build plan reads as though Phase 4 adds a geo dimension to the Phases 1-3 model.
It cannot. The geo dataset is a different business:

* different channels -- 6 impression/view series (Paid_Views, Google_Impressions, ...)
  against the national model's 10 dollar-spend channels;
* **no dollar spend at all**, so nothing in this model can produce a ROAS;
* different period -- geo runs 2018-01-06 to 2020-02-29, national 2014-08-03 to
  2018-07-29, with zero overlapping week-start dates;
* different scale -- geo total sales run ~$3.7M/week against national ~$99.6M/week.

So the Phase 3 anchors do not transfer (there is no channel in common to anchor), and
this model does not put credible intervals on the national model's contributions.
What it does deliver is the thing Phase 4 is actually for -- a hierarchical MMM with
partial pooling across 26 divisions -- and a model on the same data as the ridge
project's geo-DiD estimator, which is what makes Phase 6's calibration coherent.
docs/HIERARCHY.md is the long-form version of this.

WHAT PARTIAL POOLING REQUIRED
-----------------------------
`MMM(dims=("geo",))` alone does NOT pool. Every parameter simply gains a geo axis with
an independent prior per division -- 936 free parameters over 2,938 observations
with nothing shared between divisions (624 before the holiday-week controls).
pymc-marketing's own MMM docstring says the dims "share hierarchical priors so that
information is partially pooled across geographies", which is not what the defaults do
in 1.1.0 -- see docs/CHALLENGES.md #6. The pooling here is therefore explicit, via
`model_config`, and the choices are per parameter rather than global:

* **adstock decay -- pooled completely across geos, one per channel.** Carryover is a
  property of the medium, not of the division: an email keeps working for as long as
  it keeps working whether it lands in division A or division T. Phase 3 also showed
  decay barely moves off its prior even with 209 national weeks, so 156 per-geo decays
  on 113 weeks each would be noise with a geo label on it. 156 parameters -> 6.
* **media coefficient and half-point -- partially pooled**, non-centred lognormal with
  a per-channel location and scale. These are where real division-level differences
  plausibly live, and where the pooling actually does work.
* **intercept -- partially pooled, and constrained positive.** Baseline sales cannot be
  negative. An unconstrained hierarchical Normal intercept left 12.5% of the prior
  predictive below zero; exponentiating it and tightening the symmetric seasonality
  spread brought that to 4.9% (see `prior_predictive_summary`).
* **seasonality and holiday weeks -- partially pooled.** Measured across divisions, the
  Fourier coefficients differ by an sd of ~0.01 on values of ~0.1, so pooling should
  shrink these almost to a single shared seasonal curve. That is a prediction this
  structure makes, and `tests/test_geo_fit.py` checks it comes true.

Counting parameters is the wrong way to judge any of this. Non-centred pooling swaps
an independent prior per group for an offset per group plus a couple of hyperparameters,
so the raw count barely moves (936 -> 845, almost all of it the adstock change). What
pooling buys is a smaller EFFECTIVE number of parameters -- group offsets shrunk toward
zero by a scale the data estimates -- which is why the tests check the fitted
across-geo sigmas rather than the parameter count.

Non-centred throughout (`centered=False`): hierarchical models with 26 groups and
weakly identified scales are the textbook funnel case, and Phase 3 already saw the
sampler produce divergences on far simpler geometry (docs/CHALLENGES.md #4).
"""
import sys

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, InverseScaledLogisticSaturation
from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling

from mmm_bayes.geo_features import PEAK_WEEK_COLS, build_geo_controls
from mmm_bayes.loaders import GEO_CHANNELS, GEO_DIM, load_geo
from mmm_bayes.model import console_supports_progressbar

# Same 8-week carryover window as the national model, for the same reason -- see
# model.ADSTOCK_L_MAX. Kept identical so a later comparison of decay estimates between
# the two models is not confounded by a different truncation.
GEO_ADSTOCK_L_MAX = 8

# Two harmonics, matching features.fourier_terms in the national model. Two harmonics
# cannot represent this data's sharp Q4 spike on their own -- fitted, they span only
# -0.15 to +0.25 of max-scaled sales against an observed peak of 1.0 -- which is
# exactly why the holiday-week dummies in geo_features are not optional here.
GEO_YEARLY_SEASONALITY = 2


def build_geo_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Long-format (X, y) for a multidimensional MMM: one row per division-week.

    pymc-marketing's `to_mmm_dataset` pivots this into a (date x geo x channel) array
    itself, so the job here is only to name the date and geo columns and attach the
    holiday-week controls.
    """
    geo = load_geo().rename(columns={"Division": GEO_DIM, "week": "date"})
    controls = build_geo_controls(geo["date"])

    X = pd.concat([geo[["date", GEO_DIM] + GEO_CHANNELS], controls], axis=1)
    y = geo["Sales"].astype(float).rename("y")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def hierarchical_config(
    intercept_median: float = 0.30,
    beta_location: float = -3.5,
    half_point_median: float = 0.3,
) -> dict[str, Prior]:
    """The partial-pooling prior structure. Every scale here was set by prior
    predictive check against the observed data, not by taste -- see
    `prior_predictive_summary` and docs/HIERARCHY.md for the numbers each one is
    answering.

    All of these live on the SCALED axes: pymc-marketing divides the target and each
    channel by their own per-geo max before any prior applies, so an intercept of 0.30
    means 30% of that division's own peak week, not a dollar amount.
    """
    return {
        # Baseline sales, positive by construction. Observed max-scaled sales average
        # 0.246 across divisions with an sd of only 0.014, so the location is well
        # determined and the across-geo scale can be tight. Centred at 0.30 rather
        # than at the observed 0.246 on purpose: the prior predictive trades a higher
        # baseline against less impossible mass, and 0.30 halves P(sales < 0) from
        # 8.2% to 4.9% at the cost of a predictive median 2.4x the observed one. A
        # prior wider and higher than the data is the acceptable side of that trade;
        # 8% of the prior on negative sales is not. See docs/HIERARCHY.md.
        "intercept": Prior(
            "Normal",
            mu=Prior("Normal", mu=np.log(intercept_median), sigma=0.3),
            sigma=Prior("HalfNormal", sigma=0.15),
            dims=GEO_DIM,
            centered=False,
            transform="exp",
        ),
        # Media effect. exp(Normal(-3.5, 1.0)) puts a channel's coefficient between
        # about 0.006 and 0.16 with 90% probability; six channels then span roughly
        # 4% to 90% of scaled sales, which is wide enough not to prejudge the media
        # share -- the same reasoning that kept the national model's beta unanchored.
        "saturation_beta": _hierarchical_positive(beta_location, 1.0, 0.4),
        # Half-saturation point, in scaled-and-adstocked channel units, so it lives in
        # roughly [0, 1]. Centred at 0.3 with a wide log spread.
        "saturation_lam": _hierarchical_positive(np.log(half_point_median), 0.7, 0.4),
        # Pooled completely across geos -- see the module docstring. This is the
        # library default Beta(1, 3), just with the geo axis dropped.
        "adstock_alpha": Prior("Beta", alpha=1, beta=3, dims="channel"),
        # Seasonality. An OLS fit of these four modes on the real panel gives
        # coefficients no larger than 0.13 with an across-division sd of ~0.01, so
        # sigma 0.15 on the shared mean is generous and HalfNormal(0.05) on the
        # across-geo spread is the pooling doing its job.
        "gamma_fourier": Prior(
            "Normal",
            mu=Prior("Normal", mu=0, sigma=0.15, dims="fourier_mode"),
            sigma=Prior("HalfNormal", sigma=0.05, dims="fourier_mode"),
            dims=(GEO_DIM, "fourier_mode"),
            centered=False,
        ),
        # Holiday weeks. Wider than the Fourier prior because these carry the Q4 spike
        # the harmonics cannot, and that spike is large -- weeks 47-50 average 3-4x
        # baseline. Left free to take either sign, matching the national model's
        # treatment of controls: a lift-shaped prior on a control is a stronger claim
        # than this model needs to make.
        "gamma_control": Prior(
            "Normal",
            mu=Prior("Normal", mu=0, sigma=0.18, dims="control"),
            sigma=Prior("HalfNormal", sigma=0.05, dims="control"),
            dims=(GEO_DIM, "control"),
            centered=False,
        ),
        "likelihood": Prior(
            "Normal",
            sigma=Prior("HalfNormal", sigma=Prior("HalfNormal", sigma=0.04), dims=GEO_DIM),
            dims=("date", GEO_DIM),
        ),
    }


def _hierarchical_positive(location: float, location_sigma: float,
                           scale_sigma: float, group: str = "channel") -> Prior:
    """Non-centred hierarchical prior for a strictly positive per-(geo, channel)
    parameter: `exp(mu[channel] + sigma[channel] * z[geo, channel])`.

    Non-centred rather than centred because this is the shape that funnels. With 26
    divisions and a scale the data only weakly informs, the centred version couples
    each group's value to the group scale and NUTS has to take tiny steps in the neck
    -- the standard cause of divergences in a hierarchical model, and Phase 3 already
    showed this sampler is close to that edge (docs/CHALLENGES.md #4).

    `transform="exp"` rather than a LogNormal because pymc-extras only supports
    `centered=False` on Normal; exponentiating a non-centred Normal gives the same
    lognormal marginal with the geometry that samples.
    """
    return Prior(
        "Normal",
        mu=Prior("Normal", mu=location, sigma=location_sigma, dims=group),
        sigma=Prior("HalfNormal", sigma=scale_sigma, dims=group),
        dims=(GEO_DIM, group),
        centered=False,
        transform="exp",
    )


def build_geo_model(pooled: bool = True) -> MMM:
    """Construct the (unfitted) hierarchical geo MMM.

    `pooled=False` gives pymc-marketing's default multidimensional model -- a geo axis
    on every parameter with independent priors, no sharing. It exists so the pooling
    can be compared against its own absence rather than asserted to help, which is what
    `tests/test_geo_model.py::test_unpooled_model_has_no_shared_hyperparameters` and the
    Phase 5 comparison need.
    """
    return MMM(
        date_column="date",
        channel_columns=GEO_CHANNELS,
        control_columns=PEAK_WEEK_COLS,
        target_column="y",
        dims=(GEO_DIM,),
        yearly_seasonality=GEO_YEARLY_SEASONALITY,
        adstock=GeometricAdstock(l_max=GEO_ADSTOCK_L_MAX),
        saturation=InverseScaledLogisticSaturation(),
        scaling=Scaling(
            # dims=() reduces over date only, so these are per-geo (and, for channels,
            # per geo AND channel) maxima -- verified in tests/test_geo_model.py. That
            # matters: divisions differ ~20x in size, and scaling them jointly would
            # leave the smallest ones numerically invisible.
            target=DataDerivedScaling(method="max", dims=()),
            channel=DataDerivedScaling(method="max", dims=()),
        ),
        model_config=hierarchical_config() if pooled else None,
    )


def count_free_parameters(model: MMM) -> dict[str, int]:
    """Free parameters per random variable of a BUILT model, plus a "total" key.

    Used to state the pooling's cost concretely rather than by adjective.
    """
    counts = {}
    for rv in model.model.free_RVs:
        dims = tuple(model.model.named_vars_to_dims.get(rv.name, ()))
        counts[rv.name] = int(np.prod([len(model.model.coords[d]) for d in dims])) if dims else 1
    counts["total"] = sum(counts.values())
    return counts


def prior_predictive_summary(model: MMM, draws: int = 400, random_seed: int = 1) -> pd.DataFrame:
    """Prior predictive quantiles for the target, next to the observed ones.

    The check that matters for a hierarchical MMM is not whether the prior looks
    reasonable in isolation but whether what it implies about SALES brackets the sales
    actually seen. Building this made two problems visible that reading the priors did
    not: the default `Normal(0, 2)` intercept put 12.5% of the prior predictive below
    zero, and the seasonality prior, not the noise prior, was what put it there.
    """
    import pymc as pm

    with model.model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)

    predicted = np.asarray(prior.prior_predictive["y"]).ravel()
    target = model.xarray_dataset["_target"]
    observed = (target / target.max("date")).values.ravel()

    quantiles = [0.01, 0.25, 0.5, 0.95, 0.99]
    out = pd.DataFrame(
        {
            "prior_predictive": [float(np.quantile(predicted, q)) for q in quantiles],
            "observed": [float(np.quantile(observed, q)) for q in quantiles],
        },
        index=[f"q{int(q * 100):02d}" for q in quantiles],
    )
    out.loc["P(y<0)"] = [float((predicted < 0).mean()), float((observed < 0).mean())]
    return out


def run_geo_fit(draws: int = 500, tune: int = 500, chains: int = 2,
                target_accept: float = 0.99, random_seed: int = 42, cores: int = 1,
                pooled: bool = True, progressbar: bool | None = None,
                init: str = "jitter+adapt_diag", model: MMM | None = None,
                **sample_kwargs) -> tuple[MMM, dict]:
    """Fit the geo model with a SHORT chain. Same caveat as the national model's
    `run_skeleton_fit`: this proves the hierarchy compiles and samples, it does not
    establish inference quality. Phase 5 owns that.

    `init` and `target_accept` are what Phase 5 measured on the national model --
    pymc's own `jitter+adapt_diag`, and 0.99. See docs/CHALLENGES.md #2 and #4, and note
    in particular that dropping the jitter looked like an improvement in isolation and
    was not one in combination.
    """
    if progressbar is None:
        progressbar = console_supports_progressbar()

    X, y = build_geo_dataframe()
    # `model` lets a caller pass a variant (a trend term, a different pooling scheme)
    # without duplicating the fit call -- Phase 5's sensitivity checks all do this.
    supplied = model is not None
    model = build_geo_model(pooled=pooled) if model is None else model

    idata = model.fit(
        X=X,
        y=y,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        random_seed=random_seed,
        progressbar=progressbar,
        cores=cores,  # explicit -- see docs/CHALLENGES.md #1 and #8
        init=init,
        **sample_kwargs,
    )
    # `pooled` describes what THIS function built. A caller-supplied model may pool
    # differently -- the decay-pooling check passes one that does -- so it is reported
    # as unknown rather than as whatever `pooled` happened to be set to.
    return model, {"idata": idata, "pooled": None if supplied else pooled}


if __name__ == "__main__":
    pd.set_option("display.width", 130)
    print("=" * 78)
    print("PHASE 4 -- HIERARCHICAL GEO MODEL (partial pooling across 26 divisions)")
    print("=" * 78)

    X, y = build_geo_dataframe()
    print(f"X: {X.shape}, y: {y.shape}  ({X[GEO_DIM].nunique()} divisions x "
          f"{X['date'].nunique()} weeks)")
    print(f"Channels ({len(GEO_CHANNELS)}): {GEO_CHANNELS}")
    print(f"Holiday-week controls ({len(PEAK_WEEK_COLS)}): {PEAK_WEEK_COLS}")

    print("\n--- free parameters, pooled vs. not ---")
    rows = {}
    for label, pooled in (("unpooled", False), ("pooled", True)):
        model = build_geo_model(pooled=pooled)
        model.build_model(X=X, y=y)
        rows[label] = count_free_parameters(model)
    table = pd.DataFrame(rows).fillna(0).astype(int)
    print(table.to_string())

    print("\n--- prior predictive check (max-scaled sales) ---")
    print(prior_predictive_summary(model).round(3).to_string())

    if "--fit" in sys.argv:
        print("\nSampling (500 draws, 500 tune, 2 chains)...\n")
        model, result = run_geo_fit()
        print("\nSampling completed without error.")
    else:
        print("\n(Pass --fit to sample. Structure and priors above need no MCMC.)")
