"""Structural tests for the Phase 4 hierarchical geo model. No MCMC here.

Two things are being guarded. First, the claims the module docstring and
docs/HIERARCHY.md make about the data -- that the geo panel is a different business
from the national one -- because those claims are the reason Phase 4 is a second model
rather than an extension, and a reader has no way to check them by eye. Second, that
the pooling is actually in the model: `dims=("geo",)` produces something that looks
hierarchical and is not, so "it built without error" proves nothing.
"""
import numpy as np
import pytest

from mmm_bayes.geo_features import PEAK_WEEK_COLS, build_geo_controls
from mmm_bayes.geo_model import (
    GEO_ADSTOCK_L_MAX,
    build_geo_dataframe,
    build_geo_model,
    count_free_parameters,
    hierarchical_config,
    log_hierarchical_config,
    prior_predictive_summary,
    target_relative_channel_scaling,
)
from mmm_bayes.loaders import GEO_CHANNELS, GEO_DIM, load_geo, load_national

N_DIVISIONS = 26
N_GEO_WEEKS = 113


@pytest.fixture(scope="module")
def built_pooled():
    X, y = build_geo_dataframe()
    model = build_geo_model(pooled=True)
    model.build_model(X=X, y=y)
    return model


@pytest.fixture(scope="module")
def built_unpooled():
    X, y = build_geo_dataframe()
    model = build_geo_model(pooled=False)
    model.build_model(X=X, y=y)
    return model


@pytest.fixture(scope="module")
def built_log_link():
    X, y = build_geo_dataframe()
    model = build_geo_model(pooled=True, link="log")
    model.build_model(X=X, y=y)
    return model


@pytest.fixture(scope="module")
def built_target_relative():
    X, y = build_geo_dataframe()
    model = build_geo_model(pooled=True, channel_scaling="target-relative")
    model.build_model(X=X, y=y)
    return model


def _raw_media_intensity():
    """Media impressions per sales dollar, per division, from the raw CSV."""
    geo = load_geo()
    totals = geo.groupby("Division")[GEO_CHANNELS].sum().sum(axis=1)
    return totals / geo.groupby("Division")["Sales"].sum()


def _scaled_media_intensity(model):
    """The same quantity as the MODEL sees it, after its own scaling is applied."""
    ds = model.xarray_dataset
    target, channel = ds["_target"], ds["_channel"]
    scales = model.get_scales_as_xarray()
    scaled_y = target / scales["target_scale"]
    scaled_x = channel / scales["channel_scale"]
    return (scaled_x.sum("date").sum("channel") / scaled_y.sum("date")).to_series()


# --------------------------------------------------------------- the data claim


def test_geo_panel_is_balanced():
    geo = load_geo()
    assert geo["Division"].nunique() == N_DIVISIONS
    assert geo["week"].nunique() == N_GEO_WEEKS
    assert len(geo) == N_DIVISIONS * N_GEO_WEEKS
    assert geo.groupby("Division").size().nunique() == 1


def test_geo_is_a_different_dataset_from_national():
    """The load-bearing claim behind Phase 4 being a second model rather than an
    extension of Phases 1-3. If a future dataset refresh ever made these overlap, the
    whole framing in docs/HIERARCHY.md would need revisiting -- so it fails here rather
    than being quietly outgrown."""
    geo = load_geo()
    national = load_national()

    overlap = set(geo["week"]) & set(national.index)
    assert not overlap, (
        f"{len(overlap)} overlapping week-start dates between the geo and national "
        "panels. docs/HIERARCHY.md argues these are separate businesses partly on "
        "there being none."
    )
    assert not (set(GEO_CHANNELS) & set(national.columns)), (
        "a geo channel name now appears in the national data -- the two channel sets "
        "were disjoint, which is why the Phase 3 anchors do not transfer"
    )


def test_geo_channels_exclude_the_sum_identity_column():
    """Overall_Views = Paid_Views + Organic_Views (docs/DATA.md defect #1). Including
    it makes the design matrix rank deficient, so the loader drops it."""
    geo = load_geo()
    assert "Overall_Views" not in geo.columns
    assert "Overall_Views" not in GEO_CHANNELS


# ------------------------------------------------------------------- the frame


def test_geo_dataframe_is_long_format_with_controls():
    X, y = build_geo_dataframe()
    assert len(X) == len(y) == N_DIVISIONS * N_GEO_WEEKS
    assert X[GEO_DIM].nunique() == N_DIVISIONS
    assert X["date"].nunique() == N_GEO_WEEKS
    for column in GEO_CHANNELS + PEAK_WEEK_COLS:
        assert column in X.columns
    assert (y > 0).all(), "sales must be positive before a max-scaled model sees them"


def test_holiday_dummies_fire_only_in_their_own_week():
    X, _ = build_geo_dataframe()
    week_of_year = X["date"].dt.isocalendar().week.to_numpy()
    for column in PEAK_WEEK_COLS:
        week = int(column.removeprefix("peak_wk"))
        fires = X[column].to_numpy() == 1.0
        assert fires.any(), f"{column} never fires"
        np.testing.assert_array_equal(fires, week_of_year == week)


def test_holiday_controls_reject_a_week_that_never_occurs():
    """A dummy that is constant zero has no identifiable coefficient. Better to fail
    than to hand the sampler a column it cannot learn anything about."""
    import pandas as pd

    january_only = pd.Series(pd.date_range("2020-01-06", periods=8, freq="7D"))
    with pytest.raises(ValueError, match="never fire"):
        build_geo_controls(january_only)


# ------------------------------------------------------------------ the pooling


def test_unpooled_model_has_no_shared_hyperparameters(built_unpooled):
    """pymc-marketing's MMM docstring says `dims` gives partially pooled priors. In
    1.1.0 it does not -- every parameter just gains a geo axis with an independent
    prior. This pins the actual behaviour, so the docstring's claim cannot be taken on
    trust by a future reader (docs/CHALLENGES.md #6)."""
    counts = count_free_parameters(built_unpooled)
    assert counts["adstock_alpha"] == N_DIVISIONS * len(GEO_CHANNELS)
    assert counts["saturation_beta"] == N_DIVISIONS * len(GEO_CHANNELS)
    assert not [name for name in counts if name.endswith(("_mu", "_sigma", "_offset"))
                and name != "y_sigma"], "unpooled model unexpectedly has hyperparameters"


def test_pooled_model_has_a_hyperparameter_for_every_pooled_block(built_pooled):
    counts = count_free_parameters(built_pooled)
    for block in ("saturation_beta_raw", "saturation_lam_raw", "gamma_fourier",
                  "gamma_control", "intercept_contribution_raw"):
        assert f"{block}_mu" in counts, f"{block} has no shared location"
        assert f"{block}_sigma" in counts, f"{block} has no shared scale"
        assert f"{block}_offset" in counts, f"{block} is not non-centred"


def test_adstock_is_pooled_completely_across_divisions(built_pooled, built_unpooled):
    """The one parameter deliberately given no geo axis at all. 156 -> 6."""
    assert count_free_parameters(built_unpooled)["adstock_alpha"] == 156
    assert count_free_parameters(built_pooled)["adstock_alpha"] == len(GEO_CHANNELS)
    dims = built_pooled.model.named_vars_to_dims["adstock_alpha"]
    assert tuple(dims) == ("channel",), f"adstock_alpha has dims {tuple(dims)}"


def test_pooling_is_not_a_parameter_count_saving(built_pooled, built_unpooled):
    """Guards against the wrong claim as much as the wrong code.

    Non-centred pooling trades an independent prior per group for an offset per group
    plus hyperparameters, so the raw count barely moves. If a future edit "improves"
    the model by chasing a smaller parameter count, this says why that is the wrong
    target -- the pooling's effect is shrinkage, measured in tests/test_geo_fit.py.
    """
    pooled = count_free_parameters(built_pooled)["total"]
    unpooled = count_free_parameters(built_unpooled)["total"]
    assert 0.8 * unpooled < pooled < unpooled, (
        f"pooled total {pooled} vs unpooled {unpooled}: the counts should be close, "
        "with the difference almost entirely the adstock change"
    )


def test_every_hierarchical_prior_is_non_centred():
    """Centred hierarchical priors funnel. With 26 groups and weakly identified
    scales that is the standard divergence generator, and Phase 3 already showed this
    sampler sitting near that edge (docs/CHALLENGES.md #4)."""
    config = hierarchical_config()
    for name in ("intercept", "saturation_beta", "saturation_lam",
                 "gamma_fourier", "gamma_control"):
        assert config[name].centered is False, f"{name} is centred"


def test_positive_parameters_are_exponentiated():
    """Baseline sales and media coefficients cannot be negative. `transform="exp"` is
    what enforces that on a non-centred Normal."""
    config = hierarchical_config()
    for name in ("intercept", "saturation_beta", "saturation_lam"):
        assert config[name].transform == "exp", f"{name} is not constrained positive"


# ------------------------------------------------------------------ the scaling


def test_channels_and_target_are_scaled_per_division(built_pooled):
    """Divisions differ ~20x in size. A single shared max would leave the smallest
    ones numerically invisible, so the scalers must carry a geo axis."""
    channel_scale = np.asarray(built_pooled.model.named_vars["channel_scale"].eval())
    target_scale = np.asarray(built_pooled.model.named_vars["target_scale"].eval())
    assert channel_scale.shape == (N_DIVISIONS, len(GEO_CHANNELS))
    assert target_scale.shape == (N_DIVISIONS,)

    geo = load_geo()
    expected = geo.groupby("Division")["Sales"].max().to_numpy(dtype=float)
    np.testing.assert_allclose(target_scale, expected, rtol=1e-12)
    assert target_scale.max() / target_scale.min() > 10, (
        "divisions were expected to differ by more than 10x in size; if they no "
        "longer do, the per-geo scaling argument needs rechecking"
    )


# ------------------------------------------- the channel scaling and what it costs


def test_per_channel_scaling_erases_every_division_difference(built_pooled):
    """The information loss, stated as plainly as it can be: under the default scaling
    EVERY division's EVERY channel has a scaled maximum of exactly 1.0, so a division
    running 50k Google impressions is indistinguishable from one running 459M."""
    ds = built_pooled.xarray_dataset
    scaled_max = (ds["_channel"] / built_pooled.get_scales_as_xarray()["channel_scale"]).max("date")
    np.testing.assert_allclose(np.asarray(scaled_max), 1.0, rtol=1e-12)


def test_target_relative_scaling_preserves_the_cross_division_contrast(
    built_pooled, built_target_relative
):
    """The finding this scaling exists for. The default divides out each division's
    media-to-sales ratio -- the contrast a geo-DiD exploits -- and does not merely
    flatten it but INVERTS it. Guarded by sign, because the sign is the whole point."""
    raw = _raw_media_intensity()
    default = _scaled_media_intensity(built_pooled).reindex(raw.index)
    relative = _scaled_media_intensity(built_target_relative).reindex(raw.index)

    corr_default = np.corrcoef(raw.to_numpy(), default.to_numpy())[0, 1]
    corr_relative = np.corrcoef(raw.to_numpy(), relative.to_numpy())[0, 1]

    assert corr_default < 0, (
        f"the default scaling was measured to invert the contrast (corr -0.457); it is "
        f"now {corr_default:.3f}. If this has become positive the finding in "
        f"docs/DIAGNOSTICS.md needs rechecking, not this assertion relaxing."
    )
    assert corr_relative > 0.8, (
        f"target-relative scaling should carry the raw contrast through nearly intact "
        f"(measured +0.865); got {corr_relative:.3f}"
    )

    span_default = default.max() / default.min()
    span_relative = relative.max() / relative.min()
    assert span_relative > 3 * span_default, (
        f"expected the recovered span (measured 7.00x) to dwarf the default's 1.79x; "
        f"got {span_relative:.2f}x against {span_default:.2f}x"
    )


def test_target_relative_scaling_shares_one_constant_across_divisions():
    """`scale[g, c] = max_y[g] * k[c]` with k shared. If k varied by geo the
    cross-division contrast would be divided out again, which is the bug this whole
    scaling exists to avoid -- so it is asserted rather than assumed."""
    X, y = build_geo_dataframe()
    scale = target_relative_channel_scaling(X, y)

    max_y = load_geo().groupby("Division")["Sales"].max()
    k = scale / scale[GEO_DIM].to_index().map(max_y).to_numpy()[:, None]

    for channel in k["channel"].values:
        column = np.asarray(k.sel(channel=channel))
        np.testing.assert_allclose(column, column[0], rtol=1e-12)


def test_target_relative_scaling_matches_the_model_coord_order(built_target_relative):
    """`FixedScaling` aligns positionally, so this array's row order IS its correctness.
    It is right because groupby and pymc-marketing's pivot both sort divisions
    alphabetically, and because the channel columns are selected in GEO_CHANNELS order,
    which the pivot preserves and which is not alphabetical. Neither is a contract, so
    both are asserted here rather than trusted."""
    X, y = build_geo_dataframe()
    scale = target_relative_channel_scaling(X, y)
    applied = built_target_relative.get_scales_as_xarray()["channel_scale"]

    assert [str(g) for g in scale[GEO_DIM].values] == [str(g) for g in applied[GEO_DIM].values]
    assert [str(c) for c in scale["channel"].values] == list(GEO_CHANNELS)
    assert [str(c) for c in applied["channel"].values] == list(GEO_CHANNELS)

    # and with the order right, the applied divisors are the intended ones exactly
    np.testing.assert_allclose(np.asarray(applied), np.asarray(scale), rtol=1e-12)


def test_fixed_scaling_aligns_positionally_not_by_label(monkeypatch, built_target_relative):
    """Pins the trap itself. Reversing the division order of an array that still carries
    correct labels must change the applied divisors -- which is what makes the order
    check above load-bearing rather than decorative. If this test ever fails because the
    library began aligning by label, that is good news, and the docstring in
    `target_relative_channel_scaling` and docs/CHALLENGES.md #14 should be updated
    rather than this assertion inverted in place."""
    import mmm_bayes.geo_model as geo_model

    original = geo_model.target_relative_channel_scaling

    def reversed_order(X, y):
        scale = original(X, y)
        return scale.isel({GEO_DIM: np.argsort(scale[GEO_DIM].values)[::-1]})

    monkeypatch.setattr(geo_model, "target_relative_channel_scaling", reversed_order)

    X, y = build_geo_dataframe()
    shuffled_model = geo_model.build_geo_model(channel_scaling="target-relative")
    shuffled_model.build_model(X=X, y=y)

    applied = np.asarray(shuffled_model.get_scales_as_xarray()["channel_scale"])
    expected = np.asarray(built_target_relative.get_scales_as_xarray()["channel_scale"])
    assert not np.allclose(applied, expected), (
        "FixedScaling appears to align by label now. That is a fix, not a failure -- "
        "see docs/CHALLENGES.md #14."
    )


def test_both_scalings_leave_the_target_untouched(built_pooled, built_target_relative):
    """Only the channel divisor changes. If the target scaling moved too, a difference
    in the fitted media share could not be attributed to the channel scaling alone."""
    np.testing.assert_allclose(
        np.asarray(built_pooled.get_scales_as_xarray()["target_scale"]),
        np.asarray(built_target_relative.get_scales_as_xarray()["target_scale"]),
        rtol=1e-12,
    )


def test_unknown_channel_scaling_is_rejected():
    with pytest.raises(ValueError, match="channel_scaling must be one of"):
        build_geo_model(channel_scaling="per-geo")


# ------------------------------------------------- the log link (functional form)


def test_log_link_puts_no_prior_mass_on_negative_sales(built_log_link):
    """The open item this variant exists for. The identity link leaves 4.9% of the prior
    predictive below zero and no prior tuning removes it -- an additive Gaussian with
    symmetric seasonality priors always will. Under a LogNormal likelihood it is exactly
    zero by construction, not merely small."""
    summary = prior_predictive_summary(built_log_link)
    assert summary.loc["P(y<0)", "prior_predictive"] == 0.0


def test_log_link_prior_predictive_still_brackets_observed_sales(built_log_link):
    """Zero impossible mass is worthless if the prior stops describing the data. The
    identity link bought its 4.9% by centring the baseline high -- a predictive median
    2.4x the observed one. The log link should not need that trade."""
    summary = prior_predictive_summary(built_log_link)
    predicted, observed = summary["prior_predictive"], summary["observed"]
    assert predicted["q25"] < observed["q50"] < predicted["q95"]
    assert predicted["q50"] < 2 * observed["q50"], (
        "the log-link predictive median should sit far closer to the observed one than "
        "the identity link's 2.4x; if it does not, the intercept location is wrong"
    )


def test_log_link_intercept_is_not_exponentiated_and_is_negative():
    """The structural incompatibility, and the reason this is a separate config rather
    than a flag. In log space the baseline must be near log(0.19) = -1.7. The
    identity-link prior is `transform="exp"` and strictly positive, so the smallest
    baseline it can express is exp(0) = 1.0 -- every week at or above the division's own
    peak. It cannot reach the right region at all."""
    identity = hierarchical_config()["intercept"]
    log_space = log_hierarchical_config()["intercept"]

    assert getattr(identity, "transform", None) == "exp"
    assert getattr(log_space, "transform", None) is None
    assert log_space.parameters["mu"].parameters["mu"] < 0


def test_log_link_control_prior_covers_the_measured_q4_multiplier():
    """In log space the holiday dummies carry a multiplier, not an offset. Measured by
    OLS on log(max-scaled sales), those coefficients reach 1.70. The identity-link prior
    sd of 0.18 would put that 9 sd out, which is a wrong prior rather than a weak one."""
    identity_sd = hierarchical_config()["gamma_control"].parameters["mu"].parameters["sigma"]
    log_sd = log_hierarchical_config()["gamma_control"].parameters["mu"].parameters["sigma"]

    assert identity_sd < 0.25
    assert log_sd >= 1.0, "must comfortably cover the observed 1.70 log-space lift"


def test_log_link_uses_a_lognormal_likelihood(built_log_link):
    """pymc-marketing rejects any other likelihood under link='log', because the
    counterfactual decomposition is only correct for that one."""
    assert log_hierarchical_config()["likelihood"].distribution == "LogNormal"
    assert built_log_link.link == "log"


def test_log_link_registers_y_original_scale_and_it_is_excluded_from_diagnostics(
    built_log_link,
):
    """Finding #14 was a per-observation deterministic missing from DERIVED_VARS being
    counted as a model parameter, which inflated every documented entry count. The log
    link introduces a new one, so the guard is asserted rather than assumed."""
    from mmm_bayes.diagnostics import DERIVED_VARS

    names = {v.name for v in built_log_link.model.deterministics}
    assert "y_original_scale" in names
    assert "y_original_scale" in DERIVED_VARS


def test_identity_link_is_unchanged_by_the_log_link_addition(built_pooled):
    """Regression guard: the default must still be the additive Gaussian model every
    cached fit and documented number was produced under."""
    assert built_pooled.link == "identity"
    assert "y_original_scale" not in {v.name for v in built_pooled.model.deterministics}


def test_unknown_link_is_rejected():
    with pytest.raises(ValueError, match="link must be"):
        build_geo_model(link="logit")


def test_geo_attribution_refuses_log_link_fits(built_log_link):
    """The additive decomposition is meaningless under a multiplicative model, and a
    silent wrong number is the failure mode this module was built to prevent."""
    from mmm_bayes.geo_attribution import _built_geo_model

    with pytest.raises(ValueError, match="decomposes additively"):
        _built_geo_model(idata=None, model=built_log_link)


def test_adstock_window_matches_the_national_model():
    from mmm_bayes.model import ADSTOCK_L_MAX

    assert GEO_ADSTOCK_L_MAX == ADSTOCK_L_MAX, (
        "the two models share a carryover window on purpose, so a later comparison of "
        "decay estimates is not confounded by different truncation"
    )


# --------------------------------------------------------- the prior predictive


def test_prior_predictive_brackets_the_observed_sales(built_pooled):
    """The check that actually caught problems in this phase: reading the priors did
    not reveal that the seasonality term, not the noise term, was putting 12.5% of the
    prior predictive on negative sales."""
    summary = prior_predictive_summary(built_pooled, draws=200, random_seed=1)
    prior, observed = summary["prior_predictive"], summary["observed"]

    # Coverage: the prior's central 98% must contain the observed one, both ends.
    assert prior["q01"] < observed["q01"], "the prior rules out the observed low weeks"
    assert prior["q99"] > observed["q99"], (
        "the prior predictive must cover the observed upper tail, or the Q4 spike is "
        "being ruled out a priori"
    )

    # The prior sits deliberately higher than the data -- see hierarchical_config's
    # note on trading baseline level against impossible mass. Bounded so that
    # "deliberately higher" cannot quietly become "unrelated to the data".
    assert 1 < prior["q50"] / observed["q50"] < 4, (
        f"prior predictive median is {prior['q50'] / observed['q50']:.1f}x the "
        "observed one; the documented trade-off puts it near 2.4x"
    )
    assert prior["P(y<0)"] < 0.06, (
        "too much of the prior predictive is on impossible (negative) sales -- see "
        "docs/HIERARCHY.md on the Gaussian identity link"
    )
