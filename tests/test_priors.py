"""Tests for Phase 3 prior anchoring.

The interesting risk in Phase 3 is not "does the code run" -- it is "is the unit
conversion right". A half-point anchored in the wrong units produces a model that
samples happily and is quietly wrong, which is far worse than one that crashes. So
most of what follows checks the arithmetic against ground truth rather than checking
that functions return objects:

* the channel divisor this module computes is the one pymc-marketing actually uses
  (read off the built model's `channel_scale`, not assumed from the docs);
* pymc's normalised adstock really is `(1 - d)` times the ridge project's own
  unnormalised adstock, per channel, on the real data;
* the saturation reaches exactly half at the anchored `lam`, which is the whole
  reason for switching to `InverseScaledLogisticSaturation`.
"""
import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
from pymc_marketing.mmm.transformers import (
    geometric_adstock,
    inverse_scaled_logistic_saturation,
)
from scipy import stats
from scipy.signal import lfilter

from mmm_bayes.loaders import PAID_CHANNELS, SPEND_COLS, load_national
from mmm_bayes.model import ADSTOCK_L_MAX, build_dataframe, build_model
from mmm_bayes.priors import (
    DECAY_CONCENTRATION,
    DECAY_MEAN_CLIP,
    adstock_alpha_prior,
    anchor_table,
    anchored_transforms,
    channel_scales,
    half_point_to_lam,
    load_ridge_anchors,
    national_csv_md5,
    saturation_lam_prior,
)


@pytest.fixture(scope="module")
def table():
    return anchor_table()


def test_anchors_cached_against_the_current_csv():
    """The cache carries an MD5 so anchors fit on other data can't be used silently."""
    anchors = load_ridge_anchors()
    assert anchors["national_csv_md5"] == national_csv_md5()
    assert anchors["n_weeks"] == 209
    assert set(anchors["channels"]) == set(PAID_CHANNELS)


def test_channel_scales_match_the_models_own_scaler():
    """`channel_scales` recomputes the divisor pymc-marketing derives internally.

    That duplication is the load-bearing assumption behind every `lam` anchor, so it
    is pinned against the model's own `channel_scale` variable rather than trusted
    from the `DataDerivedScaling` docstring.
    """
    X, y = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]
    model = build_model(control_columns, anchored=True)
    model.build_model(X=X, y=y.rename("y"))

    from_model = np.asarray(model.model.named_vars["channel_scale"].eval())
    ours = channel_scales().to_numpy()
    np.testing.assert_allclose(ours, from_model, rtol=1e-12)


@pytest.mark.parametrize("channel", PAID_CHANNELS)
def test_normalised_adstock_is_one_minus_decay_times_ridge_adstock(channel, table):
    """The (1 - d) factor in `half_point_to_lam`, checked on the real series.

    pymc-marketing normalises its geometric adstock weights to sum to 1; the ridge
    project's IIR does not (its weights sum to 1/(1-d)). If that factor were missing
    or inverted, half-points on the four d=0.8 channels would be off by 5x and the
    posterior would be biased with no error raised anywhere.

    The tolerance is on the MEDIAN ratio, not the max: the L-lag truncation in pymc's
    adstock leaves a genuine +/-20% spread on the d=0.8 channels (documented in
    priors.py), which the LogNormal(sigma=0.7) prior absorbs. What must not drift is
    the central conversion factor.
    """
    df = load_national()
    col = SPEND_COLS[PAID_CHANNELS.index(channel)]
    decay = float(table.loc[channel, "ridge_decay"])
    scale = float(table.loc[channel, "max_spend"])

    spend = df[col].to_numpy(dtype=float)
    x = ptx.as_xtensor(pt.as_tensor_variable(spend / scale), dims=("date",))
    pymc_adstocked = geometric_adstock(
        x, alpha=decay, l_max=ADSTOCK_L_MAX, dim="date", normalize=True
    ).eval()

    ridge_adstocked = lfilter([1.0], [1.0, -decay], spend)
    predicted = (1.0 - decay) * ridge_adstocked / scale

    nonzero = predicted > 1e-9
    median_ratio = float(np.median(pymc_adstocked[nonzero] / predicted[nonzero]))
    assert median_ratio == pytest.approx(1.0, abs=0.03), (
        f"{channel}: pymc's normalised adstock is {median_ratio:.4f}x the predicted "
        "(1-d)*ridge_adstock/scale. The unit conversion in priors.half_point_to_lam "
        "no longer matches what pymc-marketing computes."
    )


@pytest.mark.parametrize("channel", PAID_CHANNELS)
def test_saturation_is_exactly_half_at_the_anchored_lam(channel, table):
    """`InverseScaledLogisticSaturation`'s `lam` is the half-saturation point.

    This is the property the Phase 3 reparametrisation was made for. Plain
    `LogisticSaturation` would give 0.5 at ln(3)/lam instead, so if a future edit
    switches the saturation back, this fails rather than silently mis-anchoring.
    """
    lam = float(table.loc[channel, "lam_anchor"])
    z = ptx.as_xtensor(pt.as_tensor_variable(np.array([lam])), dims=("date",))
    response = float(inverse_scaled_logistic_saturation(z, lam=lam).eval()[0])
    assert response == pytest.approx(0.5, abs=1e-9)


def test_lam_anchors_land_inside_the_observed_input_range(table):
    """A half-point far outside the data range is the signature of a units error.

    The saturation input is a normalised adstock of max-scaled spend, so it lives in
    [0, 1]. Anchors clustered near 0 or far above 1 would mean the conversion had
    dropped or double-counted a scale factor -- the failure mode that motivated every
    check in this file.
    """
    lams = table["lam_anchor"].to_numpy()
    assert lams.min() > 0.01, f"lam anchors implausibly small: {lams.min():.4g}"
    assert lams.max() < 1.0, f"lam anchors above the input range: {lams.max():.4g}"


def test_half_point_conversion_is_explicit_arithmetic():
    """Spot-check the formula on hand-computed numbers, independent of the real data."""
    assert half_point_to_lam(half_point=1000.0, decay=0.0, scale=2000.0) == pytest.approx(0.5)
    assert half_point_to_lam(half_point=1000.0, decay=0.8, scale=1000.0) == pytest.approx(0.2)


def test_zero_scale_channel_is_rejected_not_silently_divided():
    """Mirrors the ridge project's own guard on a non-positive half-point: a channel
    with no spend must fail loudly rather than produce inf/NaN priors."""
    with pytest.raises(ValueError, match="Channel scale must be positive"):
        half_point_to_lam(half_point=1000.0, decay=0.5, scale=0.0)


def test_decay_prior_is_centred_on_the_ridge_decay(table):
    """Beta(mu*k, (1-mu)*k) must have mean mu = the ridge decay, clipped at the edges."""
    prior = adstock_alpha_prior(table)
    a = np.asarray(prior.parameters["alpha"], dtype=float)
    b = np.asarray(prior.parameters["beta"], dtype=float)
    means = a / (a + b)
    expected = np.clip(table["ridge_decay"].to_numpy(), *DECAY_MEAN_CLIP)
    np.testing.assert_allclose(means, expected, rtol=1e-12)
    np.testing.assert_allclose(a + b, DECAY_CONCENTRATION, rtol=1e-12)


def test_decay_prior_stays_about_as_wide_as_the_library_default(table):
    """Anchoring is meant to relocate the prior, not to shrink it.

    pymc-marketing's default Beta(1, 3) has a 90% interval 0.615 wide. Every channel
    whose ridge decay sits in the interior of [0, 1] must stay broadly comparable, or
    the "soft prior, the data can still move it" claim in docs/PRIORS.md stops being
    true. `vidtr` is excluded and checked separately below -- its mean is clipped to
    the boundary, where the Beta family cannot be wide.
    """
    a = table["beta_a"].to_numpy()
    b = table["beta_b"].to_numpy()
    interior = table["ridge_decay"].to_numpy() > DECAY_MEAN_CLIP[0]

    widths = stats.beta.ppf(0.95, a, b) - stats.beta.ppf(0.05, a, b)
    default_width = stats.beta.ppf(0.95, 1, 3) - stats.beta.ppf(0.05, 1, 3)

    assert widths[interior].min() > 0.45, (
        f"tightest interior decay prior spans only {widths[interior].min():.3f}, "
        f"against {default_width:.3f} for the Beta(1, 3) library default"
    )


def test_decay_prior_concentration_is_uniform_across_channels(table):
    """One concentration for all ten channels. Width differences between channels are
    a property of the Beta family at different means, not per-channel tuning -- which
    would be a much stronger claim about the ridge fit than it can support."""
    total = table["beta_a"].to_numpy() + table["beta_b"].to_numpy()
    np.testing.assert_allclose(total, DECAY_CONCENTRATION, rtol=1e-12)


def test_boundary_decay_of_zero_gives_a_proper_but_deliberately_tight_prior(table):
    """`vidtr` came back at exactly 0.0 from the ridge grid.

    Two things are being pinned. First, Beta(0, k) is not a distribution, so the mean
    is clipped to keep the prior proper. Second, the resulting prior is by some margin
    the tightest of the ten (a 90% interval about half the width of the others),
    because a Beta with mean 0.05 cannot be wide. That is accepted rather than tuned
    away: unlike the four channels pinned at the grid's UPPER edge of 0.8 -- where the
    truth may well lie above the edge, and where the prior's 90% interval duly reaches
    0.98 -- zero is a hard edge of the parameter space, so "the search found no
    carryover" is faithfully represented by mass concentrated near zero. Flagged in
    docs/PRIORS.md as the one channel whose posterior most deserves a Phase 5
    prior-sensitivity check.
    """
    assert float(table.loc["vidtr", "ridge_decay"]) == 0.0
    assert float(table.loc["vidtr", "beta_a"]) > 0.0
    assert float(table.loc["vidtr", "beta_a"]) == pytest.approx(
        DECAY_MEAN_CLIP[0] * DECAY_CONCENTRATION
    )

    a = table["beta_a"].to_numpy()
    b = table["beta_b"].to_numpy()
    widths = stats.beta.ppf(0.95, a, b) - stats.beta.ppf(0.05, a, b)
    vidtr_width = widths[list(table.index).index("vidtr")]
    assert vidtr_width == widths.min()

    upper_pinned = table["ridge_decay"].to_numpy() == 0.8
    assert stats.beta.ppf(0.95, a[upper_pinned], b[upper_pinned]).min() > 0.95, (
        "channels pinned at the ridge grid's upper edge must keep real prior mass "
        "above 0.8 -- that boundary is censoring, not a hard edge"
    )


def test_lam_prior_is_lognormal_centred_on_the_anchor(table):
    prior = saturation_lam_prior(table)
    mu = np.asarray(prior.parameters["mu"], dtype=float)
    np.testing.assert_allclose(np.exp(mu), table["lam_anchor"].to_numpy(), rtol=1e-12)


def test_anchored_transforms_carry_per_channel_priors():
    adstock, saturation = anchored_transforms(ADSTOCK_L_MAX)
    assert adstock.l_max == ADSTOCK_L_MAX
    assert adstock.function_priors["alpha"].dims == ("channel",)
    assert saturation.function_priors["lam"].dims == ("channel",)
    assert np.asarray(adstock.function_priors["alpha"].parameters["alpha"]).shape == (
        len(PAID_CHANNELS),
    )
    # beta is deliberately left at the library default -- see priors.py.
    assert saturation.function_priors["beta"].dims in (None, ())


def test_anchored_and_unanchored_models_both_build():
    """`anchored=False` must keep reproducing the Phase 2 specification, since the
    docs/CHALLENGES.md #2 regression tests are written against that one."""
    X, _ = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]

    anchored = build_model(control_columns, anchored=True)
    skeleton = build_model(control_columns, anchored=False)

    assert type(anchored.saturation).__name__ == "InverseScaledLogisticSaturation"
    assert type(skeleton.saturation).__name__ == "LogisticSaturation"

    # MMM stamps dims onto every prior at construction, so dims alone don't
    # distinguish the two. The parameter VALUES do: the skeleton keeps the library's
    # scalar Beta(1, 3), the anchored model carries one (a, b) pair per channel.
    assert np.ndim(skeleton.adstock.function_priors["alpha"].parameters["alpha"]) == 0
    assert np.shape(anchored.adstock.function_priors["alpha"].parameters["alpha"]) == (
        len(PAID_CHANNELS),
    )
