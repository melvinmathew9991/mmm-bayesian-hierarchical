"""Does the panel decomposition mean anything?

`tests/test_attribution.py` asks this of the national model. The geo model needs it asked
separately for one reason that is not obvious: its contributions are not stored. They are
1.1GB of deterministics that `save_idata(drop_derived=True)` discards, so
`mmm_bayes.geo_attribution` rebuilds them from the free RVs with
`pm.compute_deterministics`.

That makes correctness here a two-part claim, and both parts are tested rather than
argued. The reconstruction must reproduce what the library itself would have stored, and
the per-geo scaling must put the result back in dollars. The first is checked against the
*national* fit, where the deterministics were kept and ground truth therefore exists; the
second against pymc-marketing's own `total_media_contribution_original_scale`.

The reconciliation tests are not ceremony either. Writing this module, the components
were right and `observed_data["y"]` -- what they were being checked against -- was still
max-scaled, and only the reconciliation ratio (3.6e6 rather than 1.0) said so.
"""
import numpy as np
import pandas as pd
import pytest

from mmm_bayes.attribution import IMPLAUSIBLE_MEDIA_SHARE
from mmm_bayes.config import GEO_FIT_NC, NATIONAL_FIT_NC, geo_csv_path
from mmm_bayes.diagnostics import load_idata

pytestmark = [
    pytest.mark.skipif(
        not GEO_FIT_NC.exists(),
        reason="needs the cached geo fit: python scripts/run_full_fits.py geo --draws 500",
    ),
    pytest.mark.skipif(
        not geo_csv_path().exists(),
        reason="needs the geo CSV (non-redistributable; see docs/DATA.md)",
    ),
]


@pytest.fixture(scope="module")
def geo_idata():
    return load_idata(GEO_FIT_NC)


@pytest.fixture(scope="module")
def geo_built():
    """The built geo model, shared across tests -- building and compiling it is the
    expensive part, and every test here would otherwise pay it again."""
    from mmm_bayes.geo_model import build_geo_dataframe, build_geo_model

    X, y = build_geo_dataframe()
    model = build_geo_model(pooled=True)
    model.build_model(X=X, y=y)
    return model


@pytest.fixture(scope="module")
def totals(geo_idata, geo_built):
    from mmm_bayes.geo_attribution import panel_component_totals

    return panel_component_totals(geo_idata, model=geo_built)


def test_recomputation_reproduces_stored_deterministics():
    """The load-bearing assumption of the whole module, checked where it can be.

    The geo fit has no stored contributions to compare against -- that is why they are
    recomputed. The national fit does. If replaying the model graph over cached free RVs
    reproduces the national model's stored `channel_contribution` exactly, then the same
    mechanism applied to the geo model is not an approximation either.

    Run on the national model rather than the geo one on purpose: this is the only place
    in the project where both the inputs and the answer exist side by side.
    """
    import pymc as pm

    if not NATIONAL_FIT_NC.exists():
        pytest.skip("needs the cached national fit")

    from mmm_bayes.model import build_dataframe, build_model

    idata = load_idata(NATIONAL_FIT_NC)
    posterior = idata.posterior.dataset
    constant = idata.constant_data.dataset

    X, y = build_dataframe()
    model = build_model(control_columns=list(constant["control"].values), anchored=True)
    model.build_model(X=X, y=y.rename("y"))

    free = [rv.name for rv in model.model.free_RVs]
    names = ["channel_contribution", "control_contribution"]
    recomputed = pm.compute_deterministics(
        posterior[free].isel(draw=slice(0, 20)), model=model.model,
        var_names=names, merge_dataset=False, progressbar=False,
    )

    for name in names:
        stored = posterior[name].isel(draw=slice(0, 20)).values
        assert np.allclose(recomputed[name].values, stored, rtol=0, atol=1e-12), (
            f"recomputing {name} from the cached free RVs does not reproduce the stored "
            "array. The geo module rebuilds its contributions this way, so if this "
            "drifts its numbers are wrong with no error attached."
        )


def test_scaling_matches_the_librarys_own_original_scale_total(geo_idata, geo_built):
    """Per-geo scaling, checked against pymc-marketing rather than against arithmetic.

    `target_scale` is a 26-entry vector spanning 326k to 3.6M. Multiplying by it and
    summing is the step the audit found `attribution` getting wrong on a panel, and the
    library computes the same quantity itself as
    `total_media_contribution_original_scale` -- so the two must agree exactly.
    """
    import pymc as pm

    from mmm_bayes.geo_attribution import _dataset

    posterior = _dataset(geo_idata, "posterior")
    free = [rv.name for rv in geo_built.model.free_RVs]
    block = pm.compute_deterministics(
        posterior[free].isel(draw=slice(0, 10)), model=geo_built.model,
        var_names=["total_media_contribution_original_scale", "channel_contribution"],
        merge_dataset=False, progressbar=False,
    )

    target_scale = _dataset(geo_idata, "constant_data")["target_scale"]
    manual = (block["channel_contribution"].sum(["date", "channel"]) * target_scale).sum("geo")

    assert np.allclose(manual.values, block["total_media_contribution_original_scale"].values)


def test_components_reconcile_with_observed_sales(geo_idata, totals):
    """intercept + controls + seasonality + media must reconstruct observed panel sales.

    The four-component split matters here. The national model carries seasonality inside
    its controls; the geo model has it as a separate additive term, so a decomposition
    that copied the national model's three components would miss by that term's size and
    look almost right.
    """
    from mmm_bayes.geo_attribution import geo_decompose

    table = geo_decompose(geo_idata, components=totals)
    assert table.loc["total", "share_of_sales"] == pytest.approx(1.0, abs=0.02), (
        f"components reconstruct {table.loc['total', 'share_of_sales']:.3f} of observed "
        "sales -- the panel decomposition is not on the sales scale"
    )


def test_observed_sales_are_read_out_of_max_scaled_units(geo_idata):
    """`observed_data["y"]` is stored max-scaled per division, not in dollars.

    Summing it straight gives ~28 per division. This is the bug the reconciliation row
    caught while the module was being written, so it is pinned: the smallest division
    still has to look like millions of dollars, not tens.
    """
    from mmm_bayes.geo_attribution import observed_sales_by_division

    observed = observed_sales_by_division(geo_idata)
    assert len(observed) == 26
    assert observed.min() > 1e6, (
        f"smallest division has observed sales of {observed.min():,.0f} -- that is "
        "max-scaled units leaking through, not dollars"
    )


def test_every_division_reconciles(geo_idata, totals):
    """Reconciliation division by division, not just in aggregate.

    An aggregate that reconciles can still hide per-division errors that cancel -- and
    the per-geo scale vector is exactly the kind of mistake that would produce them,
    since getting it backwards inflates small divisions and deflates large ones.
    """
    from mmm_bayes.geo_attribution import geo_decompose_by_division

    table = geo_decompose_by_division(geo_idata, components=totals)
    assert len(table) == 26
    assert table["reconciliation"].between(0.95, 1.05).all(), (
        "some divisions do not reconcile:\n"
        f"{table['reconciliation'].sort_values().head().to_string()}"
    )


def test_geo_model_has_the_same_identification_problem(geo_idata, totals):
    """The finding this module was built to be able to state.

    Phase 5 established that the national model's media/baseline split is set by the
    prior, and made Phase 6's calibration against the geo panel load-bearing as a result.
    This test records that the geo model carries the same pathology -- less extreme
    (media at ~55% of sales against the national model's 108%, correlation -0.98 against
    -0.997) but the same signature: the sum is an order of magnitude better determined
    than either part.

    It is a known-bad-state test, like `test_media_share_is_flagged_when_it_exceeds_what
    _is_possible` in tests/test_attribution.py. It fails the day the geo model is
    identified, which is the day docs/DIAGNOSTICS.md's Phase 6 section needs rewriting.
    """
    from mmm_bayes.geo_attribution import geo_identification_report, geo_media_share_interval

    table = geo_identification_report(geo_idata, components=totals)
    _, mean, _ = geo_media_share_interval(geo_idata, components=totals)

    assert mean > IMPLAUSIBLE_MEDIA_SHARE, (
        f"geo media share is now {mean:.1%}, below the implausibility threshold. If the "
        "model was respecified, Phase 6's premise changed with it."
    )
    assert table.loc["media", "corr_media_baseline"] < -0.9
    assert table.loc["media + baseline", "cv"] < table.loc["media", "cv"] / 5


def test_chunking_does_not_change_the_answer(geo_idata, geo_built, totals):
    """The draw-chunking exists to cap memory, so it must be arithmetically invisible.

    Reducing over `date` inside the loop is what keeps the peak at 56MB instead of
    1.1GB; a chunk boundary that dropped or double-counted draws would shift the totals
    slightly and look like sampling noise.

    17 against the fixture's default 50, rather than against a second run at some other
    size: 17 does not divide the 500 draws evenly, so it exercises a short final chunk,
    and reusing the fixture means this costs one recomputation rather than two.
    """
    from mmm_bayes.geo_attribution import panel_component_totals

    ragged = panel_component_totals(geo_idata, model=geo_built, chunk=17)

    for name, values in ragged.items():
        assert np.allclose(values, totals[name]), f"{name} depends on the chunk size"


def test_a_mismatched_model_is_refused(geo_idata):
    """Recomputing against the wrong graph must raise, not answer.

    The unpooled variant has the same channels, dates and divisions but different free
    RVs. Replaying it over this fit's posterior is the one failure mode that would
    produce plausible, wrong numbers in silence.
    """
    from mmm_bayes.geo_attribution import panel_component_totals
    from mmm_bayes.geo_model import build_geo_dataframe, build_geo_model

    X, y = build_geo_dataframe()
    unpooled = build_geo_model(pooled=False)
    unpooled.build_model(X=X, y=y)

    with pytest.raises(ValueError, match="missing free RVs"):
        panel_component_totals(geo_idata, model=unpooled)


def test_media_share_carries_a_credible_interval(geo_idata, totals):
    """Same deliverable as the national model's, on the panel: a share with an interval
    around it rather than a point estimate."""
    from mmm_bayes.geo_attribution import geo_media_share_interval

    lower, mean, upper = geo_media_share_interval(geo_idata, components=totals)
    assert lower < mean < upper
    assert 0.0 < lower and upper < 1.5


def test_shared_reporting_helpers_are_the_same_ones_attribution_uses():
    """The two modules must not drift into two definitions of a share.

    `geo_attribution` imports `_shares_table`, `_interval` and `_identification_table`
    from `attribution` rather than restating them; this pins that, so a future edit that
    copies one across to "decouple" the modules fails here rather than quietly producing
    two subtly different identification reports.
    """
    from mmm_bayes import attribution, geo_attribution

    assert geo_attribution._shares_table is attribution._shares_table
    assert geo_attribution._interval is attribution._interval
    assert geo_attribution._identification_table is attribution._identification_table


def test_shares_table_handles_any_component_set():
    """The shared table takes three components from the national model and four from the
    geo one, so it must not have either count baked in."""
    from mmm_bayes.attribution import _shares_table

    rng = np.random.default_rng(11)
    components = {name: rng.normal(25.0, 1.0, (2, 50)) for name in ("a", "b", "c", "d")}
    table = _shares_table(components, total_sales=100.0)

    assert set(table.index) == {"a", "b", "c", "d", "total"}
    assert table.loc["total", "mean"] == pytest.approx(
        table.loc[["a", "b", "c", "d"], "mean"].sum()
    )
    assert isinstance(table, pd.DataFrame)
