"""Does the model's contribution decomposition mean anything?

Separate from tests/test_diagnostics.py on purpose. Those tests ask whether the sampler
converged; these ask whether the answer is usable. Phase 5 found the difference matters:
the national model passes every convergence threshold except divergences, and still
attributes 105% of sales to media.

Convergence diagnostics cannot catch that. R-hat and ESS describe how well the chains
explored the posterior, not whether the posterior describes the business. A
misspecified model can be sampled perfectly.
"""
import numpy as np
import pytest

from mmm_bayes.attribution import (
    IMPLAUSIBLE_MEDIA_SHARE,
    decompose,
    media_share_interval,
)
from mmm_bayes.config import NATIONAL_FIT_NC
from mmm_bayes.diagnostics import load_idata
from mmm_bayes.loaders import load_national

pytestmark = pytest.mark.skipif(
    not NATIONAL_FIT_NC.exists(),
    reason="needs the cached full fit: python scripts/run_full_fits.py national",
)


@pytest.fixture(scope="module")
def national():
    return load_idata(NATIONAL_FIT_NC)


def test_components_sum_to_observed_sales(national):
    """The invariant that makes every other number here readable.

    intercept + controls + media must reconstruct total sales. If it does not, the
    scaling is wrong somewhere and a "105% of sales" reading would be a units bug rather
    than a finding. This is what established that it is a finding.
    """
    table = decompose(national, load_national()["sales"])
    reconstructed = table.loc[["intercept", "controls", "media"], "mean"].sum()
    observed = float(load_national()["sales"].sum())

    assert reconstructed == pytest.approx(observed, rel=1e-3), (
        f"components sum to {reconstructed:,.0f} against observed {observed:,.0f} -- "
        "the decomposition is not on the sales scale, so its shares are meaningless"
    )


def test_shares_sum_to_one(national):
    table = decompose(national, load_national()["sales"])
    shares = table.loc[["intercept", "controls", "media"], "share_of_sales"]
    assert shares.sum() == pytest.approx(1.0, abs=1e-3)


def test_media_share_carries_a_credible_interval(national):
    """The gap this project exists to close: the ridge project reported a point
    estimate for media-attributed sales and said plainly that it had no interval
    around it."""
    lower, mean, upper = media_share_interval(national, load_national()["sales"])
    assert lower < mean < upper
    assert upper - lower > 0, "an interval of zero width is a point estimate in disguise"


def test_media_share_is_flagged_when_it_exceeds_what_is_possible(national):
    """A guard, not an assertion of correctness.

    Media cannot plausibly account for most of revenue -- spend is 2.2% of it, so a
    share near 1.0 implies a return no business sees. This test documents the currently
    known-bad state rather than pretending it away: it asserts that the *flag fires*,
    so the day the model is fixed this test fails and has to be rewritten as the
    positive check it should be.
    """
    _, mean, _ = media_share_interval(national, load_national()["sales"])
    implausible = mean > IMPLAUSIBLE_MEDIA_SHARE

    assert implausible, (
        f"media share is now {mean:.1%}, at or below the {IMPLAUSIBLE_MEDIA_SHARE:.0%} "
        "implausibility threshold. If the intercept prior has been fixed, replace this "
        "test with one asserting the share is plausible, and update docs/DIAGNOSTICS.md."
    )


def test_controls_contribute_almost_nothing_in_total(national):
    """Not a defect on its own -- controls are z-scored, so they explain deviations
    around the level rather than the level itself, and their total is ~0 by
    construction. It is recorded because combined with a near-zero intercept it is what
    leaves media carrying the whole baseline."""
    table = decompose(national, load_national()["sales"])
    assert abs(table.loc["controls", "share_of_sales"]) < 0.05


def test_decompose_is_robust_to_a_missing_control_group():
    """A model fit without controls has no `control_contribution` variable at all.
    `decompose` should report zero for it rather than raising."""
    from arviz_base import from_dict

    rng = np.random.default_rng(0)
    idata = from_dict(
        {
            "posterior": {
                "channel_contribution": rng.normal(0.01, 0.001, (2, 50, 4, 3)),
                "intercept_contribution": rng.normal(0.2, 0.01, (2, 50)),
            }
        },
        dims={"channel_contribution": ["date", "channel"]},
        coords={"channel": ["a", "b", "c"], "date": [0, 1, 2, 3]},
    )
    import pandas as pd

    sales = pd.Series([100.0, 120.0, 90.0, 110.0])
    table = decompose(idata, sales)
    assert table.loc["controls", "mean"] == 0.0
    assert set(table.index) == {"intercept", "controls", "media", "total"}


def test_identification_report_flags_the_collinear_split(national):
    """The central Phase 5 result, pinned.

    If media and baseline were separately identified, their sum would be no better
    determined than either part. Here the sum's coefficient of variation is an order of
    magnitude smaller, and the two are almost perfectly anti-correlated -- the data
    determines total sales and leaves the split to the prior.
    """
    from mmm_bayes.attribution import identification_report

    table = identification_report(national, load_national()["sales"])

    assert table.loc["media", "corr_media_baseline"] < -0.9, (
        "media and baseline are no longer strongly anti-correlated -- if the model has "
        "been respecified, docs/DIAGNOSTICS.md's identification section needs redoing"
    )
    assert table.loc["media + baseline", "cv"] < table.loc["media", "cv"] / 5, (
        "the sum is no longer much better determined than the parts, which is the "
        "signature this diagnostic exists to detect"
    )


def test_identification_report_is_content_with_a_well_identified_model():
    """Guard against the diagnostic firing on everything: two independent components
    should show near-zero correlation and no sharpening of the sum."""
    import pandas as pd
    from arviz_base import from_dict

    from mmm_bayes.attribution import identification_report

    rng = np.random.default_rng(7)
    idata = from_dict(
        {
            "posterior": {
                "channel_contribution": rng.normal(0.02, 0.004, (2, 400, 4, 3)),
                "intercept_contribution": rng.normal(0.05, 0.01, (2, 400)),
            }
        },
        dims={"channel_contribution": ["date", "channel"]},
        coords={"channel": ["a", "b", "c"], "date": [0, 1, 2, 3]},
    )
    table = identification_report(idata, pd.Series([100.0, 120.0, 90.0, 110.0]))
    assert abs(table.loc["media", "corr_media_baseline"]) < 0.2
    assert table.loc["media + baseline", "cv"] > table.loc["media", "cv"] / 5


def test_panel_fits_are_refused_rather_than_mis_answered():
    """Audit finding: every function here assumes one series and one target scale.

    Run against the hierarchical geo fit before this guard existed, `decompose`
    reported the intercept at 3051% of sales and the components missed reconciliation
    by a factor of 34 -- with no error. The bug was latent (nothing calls these with a
    panel fit) but the code handled panel dims explicitly, which is worse than not
    supporting them: it looked deliberate.
    """
    from arviz_base import from_dict

    from mmm_bayes.attribution import channel_shares

    rng = np.random.default_rng(3)
    panel = from_dict(
        {
            "posterior": {
                "channel_contribution": rng.normal(0.01, 0.001, (2, 30, 4, 2, 3)),
                "intercept_contribution": rng.normal(0.2, 0.01, (2, 30, 2)),
            }
        },
        dims={"channel_contribution": ["date", "geo", "channel"],
              "intercept_contribution": ["geo"]},
        coords={"channel": ["a", "b", "c"], "date": [0, 1, 2, 3], "geo": ["X", "Y"]},
    )
    import pandas as pd

    sales = pd.Series([100.0, 120.0, 90.0, 110.0])
    for fn in (decompose, media_share_interval):
        with pytest.raises(NotImplementedError, match="panel fits"):
            fn(panel, sales)
    with pytest.raises(NotImplementedError, match="panel fits"):
        channel_shares(panel, sales, ["a", "b", "c"])


def test_channel_shares_reconciles_with_the_aggregate_media_share(national):
    """The per-channel table is the project's headline deliverable and had no test.

    Two properties make it trustworthy: the shares must sum to the aggregate media
    share computed independently, and each mean must lie inside its own interval.
    """
    from mmm_bayes.attribution import channel_shares
    from mmm_bayes.loaders import PAID_CHANNELS

    sales = load_national()["sales"]
    table = channel_shares(national, sales, PAID_CHANNELS)
    _, aggregate, _ = media_share_interval(national, sales)

    assert table["share_mean"].sum() == pytest.approx(aggregate, rel=1e-9)
    assert (table["q03"] <= table["share_mean"]).all()
    assert (table["share_mean"] <= table["q97"]).all()
    assert table["share_mean"].is_monotonic_decreasing, "table should rank by share"
    assert set(table.index) == set(PAID_CHANNELS)
