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
    prior_predictive_summary,
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
