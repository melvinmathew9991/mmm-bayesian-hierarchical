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
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, InverseScaledLogisticSaturation
from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling

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


def log_hierarchical_config(
    intercept_location: float = -1.70,
    beta_location: float = -3.5,
    half_point_median: float = 0.3,
) -> dict[str, Prior]:
    """The same pooling structure re-specified for `link="log"`.

    NOT a variant of `hierarchical_config` with two numbers changed. Under
    `link="log"` pymc-marketing puts a `LogNormal` likelihood on the target and
    `median(y) = exp(mu) * target_scale`, so **the entire linear predictor moves to log
    space** and every prior in `hierarchical_config` changes meaning. Two of them break
    outright, which is why this is a separate function rather than a flag:

    * **The intercept prior becomes unsatisfiable.** `hierarchical_config` builds it with
      `transform="exp"`, so it is strictly positive -- correct under the identity link,
      where the intercept is a share of the division's peak week and a negative baseline
      is nonsense. In log space the required value is `log(0.19) ~ -1.7`, and a strictly
      positive intercept cannot reach it at all: the smallest baseline it can express is
      `exp(0) = 1.0`, i.e. every week at or above the division's own maximum. Measured by
      OLS on `log(max-scaled sales)` per division: mean **-1.705**, across-geo sd 0.071.
      So the exp transform is dropped and the location moved.

      Worth stating, because it is the point of the whole change: the transform existed
      only to keep baseline sales positive, and under a LogNormal likelihood positivity
      is guaranteed by construction. The structural fix makes the workaround obsolete
      rather than merely relocating it.

    * **The holiday-week prior is ~5x too tight.** Those dummies carry the Q4 spike, and
      in log space that spike is a *multiplier*, not an offset. The same OLS gives
      coefficients up to **1.704** (`peak_wk48` averages 1.432 across divisions) against
      `hierarchical_config`'s `Normal(0, 0.18)`. A prior that puts the true value 9 sd
      out is not a weak prior, it is a wrong one.

    The rest follow the same measurement:

    * seasonality -- log-space Fourier coefficients reach 0.161 with an across-geo sd up
      to 0.060, an order of magnitude more cross-division variation than the identity
      link's ~0.01, so the pooling here is looser than Phase 4's;
    * likelihood scale -- residual log sd after the deterministic part is **0.197**, so
      `HalfNormal(0.25)` on the across-geo scale brackets it; `hierarchical_config`'s
      `HalfNormal(0.04)` describes additive noise on max-scaled sales and means nothing
      here;
    * `saturation_beta` keeps its location. That is arithmetic, not laziness: to produce
      a media share `s` the six channels must supply a total log-lift of `log(1/(1-s))`,
      so `s` between 3% and 61% wants per-channel values of roughly 0.006 to 0.156 --
      which is exactly the range `exp(Normal(-3.5, 1.0))` already spans. The prior is
      left uncommitted about the media share for the same reason it was under the
      identity link.
    * `saturation_lam` and `adstock_alpha` are untouched, because both live in
      *channel* units and the link changes the response scale only.
    """
    return {
        # Log-space baseline. No `transform="exp"`: the value must be negative, and
        # positivity of sales now comes from the LogNormal likelihood instead.
        "intercept": Prior(
            "Normal",
            mu=Prior("Normal", mu=intercept_location, sigma=0.3),
            sigma=Prior("HalfNormal", sigma=0.15),
            dims=GEO_DIM,
            centered=False,
        ),
        "saturation_beta": _hierarchical_positive(beta_location, 1.0, 0.4),
        "saturation_lam": _hierarchical_positive(np.log(half_point_median), 0.7, 0.4),
        "adstock_alpha": Prior("Beta", alpha=1, beta=3, dims="channel"),
        # Wider than the identity-link version on both levels: 0.161 observed against a
        # 0.15 prior sd is snug, and the across-geo spread is 0.060 rather than ~0.01.
        "gamma_fourier": Prior(
            "Normal",
            mu=Prior("Normal", mu=0, sigma=0.25, dims="fourier_mode"),
            sigma=Prior("HalfNormal", sigma=0.10, dims="fourier_mode"),
            dims=(GEO_DIM, "fourier_mode"),
            centered=False,
        ),
        # The Q4 multiplier. Left free to take either sign, matching the identity-link
        # treatment: a lift-shaped prior on a control is a stronger claim than this model
        # needs to make, even when the lift is known to be large.
        "gamma_control": Prior(
            "Normal",
            mu=Prior("Normal", mu=0, sigma=1.0, dims="control"),
            sigma=Prior("HalfNormal", sigma=0.30, dims="control"),
            dims=(GEO_DIM, "control"),
            centered=False,
        ),
        "likelihood": Prior(
            "LogNormal",
            sigma=Prior("HalfNormal", sigma=Prior("HalfNormal", sigma=0.25), dims=GEO_DIM),
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


CHANNEL_SCALING_MODES = ("per-channel", "target-relative")


def target_relative_channel_scaling(X: pd.DataFrame, y: pd.Series) -> xr.DataArray:
    """Per-(geo, channel) channel divisors that PRESERVE the cross-division media
    intensity contrast, as `max_y[geo] * k[channel]`.

    WHY THIS EXISTS -- the default scaling destroys the only variation a DiD could use.
    ------------------------------------------------------------------------------
    `DataDerivedScaling(method="max", dims=())` divides each channel by *its own* per-geo
    maximum, and the target by *its own* per-geo maximum. Both divisors are then
    per-division constants, so what the model sees is

        x_scaled / y_scaled = (x / max_x[g, c]) / (y / max_y[g])

    and the media-to-sales ratio -- how heavily a division is media-supported, which is
    exactly the cross-sectional contrast Phase 6 wants -- is divided straight out. What
    survives is only the within-geo shape over time, which is the same information the
    national model already has, replicated 26 times rather than added to.

    Measured on this panel, that is not a small effect. Raw media-per-sales spans 11.17x
    across divisions (CV 0.271) and is close to orthogonal to division size
    (corr(log sales, log intensity) = 0.155) -- near-ideal identifying variation. After
    the default scaling the model sees a span of 1.79x, and worse, an INVERTED one:
    corr(raw intensity, scaled intensity) = -0.457. Division C has the lowest raw
    intensity and the highest scaled one.

    The sharpest case is Google_Impressions. Divisions C and N run 50,708 and 55,527
    impressions against division B's 458,607,393 -- a ~9,000x gap, and genuine rather
    than a zero artefact (no division-channel pair has a single zero week). Dividing C's
    Google by C's own maximum inflates it to the full [0, 1] range, so the model cannot
    tell "this division does not run Google" from "this division saturates it".

    WHAT THIS COMPUTES
    ------------------
    Scale channel `c` in division `g` by `max_y[g] * k[c]`, where `k[c]` is a single
    constant shared across divisions:

        k[c] = median_g( max_t x[g, c] / max_y[g] )

    Because `k` does not vary by geo, the cross-division contrast passes through intact;
    because the divisor is proportional to `max_y[g]`, the media-to-sales ratio is
    preserved up to that per-channel constant. `k` is a median rather than a mean so the
    two Google outliers set none of it.

    The per-channel constant is what keeps the existing priors valid. Scaling by
    `max_y[g]` alone leaves the six channels spanning three orders of magnitude
    (Paid_Views median 0.006 against Google_Impressions median 0.912, max 12.0), which
    the `saturation_lam` prior centred at 0.3 does not describe. With `k[c]` the typical
    division's channel maximum lands at ~1.0 -- the same place the default scaling put
    it -- so `hierarchical_config`'s prior calibration carries over unchanged, while the
    divisions that genuinely differ move off it (C's Google maximum sits at 0.001, N's at
    0.000). Recovered contrast: span 7.00x, corr(raw, scaled) = +0.865.

    Returns a `(geo, channel)` DataArray with labelled coordinates, for
    `FixedScaling(dims=(), value=...)`.

    THE ROW ORDER IS LOAD-BEARING. `FixedScaling` aligns the supplied array to the data
    grid **positionally, not by label**, despite the array carrying coordinate labels and
    despite `MMM._align_fixed_scale_dataarray` being written as an xarray broadcast that
    looks like it would align. Measured: feeding this same array with its 26 divisions
    reversed changes the applied divisors by up to 1.99e7 and raises no error -- every
    division would be scaled by another division's maximum, silently. See
    docs/CHALLENGES.md #14.

    What makes the order right here is that `groupby(GEO_DIM)` sorts the divisions
    alphabetically and pymc-marketing's own pivot does too, and that the channel columns
    are selected as `GEO_CHANNELS`, which is the order the pivot keeps (it is NOT
    alphabetical). Both facts are luck rather than contract, so
    `tests/test_geo_model.py::test_target_relative_scaling_matches_the_model_coord_order`
    asserts them, and a companion test pins the positional behaviour itself so that a
    future library fix to label alignment is detected rather than silently relied on.
    """
    frame = X[[GEO_DIM] + GEO_CHANNELS].copy()
    frame["_y"] = np.asarray(y, dtype=float)

    max_y = frame.groupby(GEO_DIM)["_y"].max()
    max_x = frame.groupby(GEO_DIM)[GEO_CHANNELS].max()
    k = max_x.div(max_y, axis=0).median(axis=0)

    scale = pd.DataFrame(
        np.outer(max_y.to_numpy(), k.to_numpy()),
        index=max_y.index,
        columns=list(k.index),
    )
    return xr.DataArray(
        scale.to_numpy(),
        dims=(GEO_DIM, "channel"),
        coords={GEO_DIM: scale.index.tolist(), "channel": scale.columns.tolist()},
    )


def build_geo_model(pooled: bool = True,
                    channel_scaling: str = "per-channel",
                    link: str = "identity") -> MMM:
    """Construct the (unfitted) hierarchical geo MMM.

    `pooled=False` gives pymc-marketing's default multidimensional model -- a geo axis
    on every parameter with independent priors, no sharing. It exists so the pooling
    can be compared against its own absence rather than asserted to help, which is what
    `tests/test_geo_model.py::test_unpooled_model_has_no_shared_hyperparameters` and the
    Phase 5 comparison need.

    `channel_scaling` selects how the channels are put on a common footing:

    * `"per-channel"` -- each channel divided by its own per-geo maximum. The default
      through Phase 5, and the setting every cached fit and documented geo number was
      produced under. Kept as the default so the Phase 6 comparison is against the
      measured baseline rather than against a moved one.
    * `"target-relative"` -- `max_y[geo] * k[channel]`, which preserves the
      cross-division media intensity contrast the default divides out. See
      `target_relative_channel_scaling` for why that contrast is the whole of Phase 6's
      identifying variation, and what it measures.

    `link` selects the functional form. `"identity"` is the additive Gaussian model of
    Phases 4-6. `"log"` gives pymc-marketing's multiplicative model -- a `LogNormal`
    likelihood with `median(y) = exp(mu) * target_scale` -- which is the structural fix
    for the 4.9% of identity-link prior predictive mass that falls on impossible negative
    sales (docs/HIERARCHY.md's long-standing open item). It selects
    `log_hierarchical_config`, because the log link re-specifies the priors rather than
    reinterpreting them; see that function.

    Under `link="log"` the decomposition is COUNTERFACTUAL, not additive: the library
    reports media as `exp(mu) - exp(mu - mu_media)`, so summing the per-component
    deterministics the way `geo_attribution` does for the identity link is meaningless.
    `geo_attribution` refuses log-link fits for that reason.

    NOTE for anything that replays this model's graph over a cached fit
    (`geo_attribution._built_geo_model`): the `channel_scaling` modes produce IDENTICAL
    free-RV names, so that function's name-based guard cannot tell them apart. A model
    built with the wrong `channel_scaling` will recompute contributions against the wrong
    divisors and return wrong numbers with no error raised. Pass `model=` explicitly.
    """
    if link not in ("identity", "log"):
        raise ValueError(f"link must be 'identity' or 'log', got {link!r}.")
    if channel_scaling not in CHANNEL_SCALING_MODES:
        raise ValueError(
            f"channel_scaling must be one of {CHANNEL_SCALING_MODES}, "
            f"got {channel_scaling!r}."
        )

    if channel_scaling == "per-channel":
        channel_scale = DataDerivedScaling(method="max", dims=())
    else:
        X, y = build_geo_dataframe()
        channel_scale = FixedScaling(
            dims=(), value=target_relative_channel_scaling(X, y)
        )

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
            # dims=() reduces over date only, so the target divisor is a per-geo maximum
            # -- verified in tests/test_geo_model.py. That matters: divisions differ ~23x
            # in size, and scaling them jointly would leave the smallest ones numerically
            # invisible. The CHANNEL divisor is where Phase 6 found a cost; see
            # `channel_scaling` above and `target_relative_channel_scaling`.
            target=DataDerivedScaling(method="max", dims=()),
            channel=channel_scale,
        ),
        model_config=(
            (log_hierarchical_config() if link == "log" else hierarchical_config())
            if pooled else None
        ),
        link=link,
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
                channel_scaling: str = "per-channel", link: str = "identity",
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
    model = (build_geo_model(pooled=pooled, channel_scaling=channel_scaling, link=link)
             if model is None else model)

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
    # `pooled` and `channel_scaling` describe what THIS function built. A
    # caller-supplied model may do either differently -- the decay-pooling check passes
    # one that pools differently -- so they are reported as unknown rather than as
    # whatever the arguments happened to be set to. `channel_scaling` matters especially:
    # it leaves no trace in the free-RV names, so a fit cached without this record cannot
    # be told apart from the other variant afterwards (see build_geo_model's note).
    return model, {
        "idata": idata,
        "pooled": None if supplied else pooled,
        "channel_scaling": None if supplied else channel_scaling,
        "link": None if supplied else link,
    }


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
