"""Contribution decomposition for the hierarchical geo panel.

`mmm_bayes.attribution` is single-series and says so; it refuses panel fits outright.
This module is the panel answer. It is a separate module rather than a `geo=True` branch
threaded through that one because the two models differ in more than the shape of a
scale factor:

1. **Where the contributions come from.** The national fit stores `channel_contribution`
   and `control_contribution` as posterior deterministics, so `attribution` just reads
   them. The geo fit does not have them: they are (draws x 113 dates x 26 geos x
   channels), 1.1GB of a 1.1GB file, and `diagnostics.save_idata(drop_derived=True)`
   discards them before writing. So the panel path has to *reconstruct* them.
2. **What a share is a share of.** One series has one total; a panel has 26, differing by
   more than 10x, and the quantity Phase 6 needs is per-division, not just the aggregate.

What the two models genuinely share -- turning per-draw component totals into shares,
intervals and an identification report -- is shared, via the `_shares_table`,
`_identification_table` and `_interval` helpers in `attribution`. The duplication a
second module would otherwise cause is all in the reporting arithmetic, and that
arithmetic does not care how many geos it is looking at.

WHY RECONSTRUCTION IS SAFE
--------------------------
The deterministics were dropped, but every free RV they are a function of survived, so
they are recoverable exactly rather than approximately -- `pm.compute_deterministics`
replays pymc-marketing's own graph rather than this module reimplementing geometric
adstock and inverse-scaled logistic saturation, which is the version of this that would
silently drift when the library changes its parametrisation.

That claim is tested rather than asserted. On the *national* fit, where the
deterministics were kept, recomputing them from the cached free RVs reproduces the
stored arrays to 5.6e-17 (`test_recomputation_reproduces_stored_deterministics`). The
per-geo scaling is pinned the same way: multiplying by `target_scale` and summing
reproduces pymc-marketing's own `total_media_contribution_original_scale` exactly
(`test_scaling_matches_the_librarys_own_original_scale_total`).

This also retires the reason the drop was questionable. `save_idata`'s justification for
discarding 1.1GB was that nothing reads it back -- true only because attribution refused
panels, which is circular. The real justification is this module: the arrays are cheap to
rebuild (~2ms per draw-chain once compiled) and expensive to store, so they are
recomputed on demand and never written.

FOUR COMPONENTS, NOT THREE
--------------------------
The national model carries its seasonality as four Fourier columns inside `controls`.
The geo model uses pymc-marketing's own `yearly_seasonality`, which is a separate
additive term. A panel decomposition that copied the national model's three components
would drop it and miss reconciliation -- silently, because seasonality is small here
(0.5% of sales). It gets its own row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_bayes.attribution import _identification_table, _interval, _shares_table

#: Recomputed per chunk of draws, then reduced over `date` before the next chunk. The
#: full `control_contribution` at 4 chains x 500 draws is 1.1GB; at 50 draws it is
#: 56MB, and reducing inside the loop is what keeps the peak there rather than at the
#: total. Not a speed knob -- recomputation costs ~2ms per draw-chain either way.
DRAW_CHUNK = 50

#: The additive terms of the geo model's mean, in the order they are reported.
#: `intercept_contribution` is a level (chain, draw, geo) with no date axis; the other
#: three are per-date and are summed over it.
_PANEL_COMPONENTS = (
    "intercept_contribution",
    "control_contribution",
    "yearly_seasonality_contribution",
    "channel_contribution",
)

_COMPONENT_LABELS = {
    "intercept_contribution": "intercept",
    "control_contribution": "controls",
    "yearly_seasonality_contribution": "seasonality",
    "channel_contribution": "media",
}

COMPONENTS = tuple(_COMPONENT_LABELS.values())


def _dataset(idata, group: str):
    """The `xarray.Dataset` for a group, whether or not arviz handed back a DataTree.

    arviz 1.x returns an `InferenceData` backed by `xarray.DataTree`, whose nodes support
    single-variable lookup but reject `node[[a, b]]` with "Selecting via tags is
    deprecated". This module selects the whole free-RV set at once, so it needs the
    Dataset underneath. Written as a shim rather than a hard `.dataset` because the same
    functions are called in tests with hand-built `from_dict` idata, where the groups are
    already plain Datasets.
    """
    node = getattr(idata, group)
    return getattr(node, "dataset", node)


def _built_geo_model(idata, model=None):
    """A built geo model whose graph matches the cached fit, for replaying deterministics.

    Guarded rather than trusted. Recomputing deterministics against a model that differs
    from the one that produced the fit -- an unpooled variant, a different control set --
    would return numbers with no error attached, which is the exact failure mode this
    module exists to avoid. So the model's free RVs must all be present in the posterior.
    """
    from mmm_bayes.geo_model import build_geo_dataframe, build_geo_model

    if model is None:
        X, y = build_geo_dataframe()
        model = build_geo_model(pooled=True)
        model.build_model(X=X, y=y)
    elif model.model is None:
        raise ValueError("`model` must already be built (call model.build_model(X, y)).")

    # The additive decomposition below is only meaningful under the identity link. Under
    # `link="log"` pymc-marketing reports media as the counterfactual
    # `exp(mu) - exp(mu - mu_media)`, and the per-component deterministics are log-space
    # terms that do not sum to the response at all -- adding them and multiplying by
    # `target_scale` would produce a confident wrong number, which is the failure mode
    # this module exists to prevent (see `attribution._reject_panel_models`).
    #
    # The name guard below does happen to catch a link mismatch when the model is built
    # here, because the log link drops the intercept's exp transform and so renames
    # `intercept_contribution_raw_*` to `intercept_contribution_*`. It does NOT catch a
    # caller passing a matched log-link model and idata together, which is the case this
    # check is for.
    if str(getattr(model, "link", "identity")) != "identity":
        raise ValueError(
            f"geo_attribution decomposes additively and this model uses "
            f"link={model.link!r}. Under a log link the components are "
            "log-space terms and media is a counterfactual "
            "(exp(mu) - exp(mu - mu_media)), so summing them is meaningless. Use the "
            "library's own `total_media_contribution_original_scale` and "
            "`y_original_scale` deterministics instead -- see docs/HIERARCHY.md."
        )

    missing = {rv.name for rv in model.model.free_RVs} - set(_dataset(idata, "posterior").data_vars)
    if missing:
        raise ValueError(
            f"The cached fit is missing free RVs this model needs: {sorted(missing)}. "
            "The idata and the model disagree about the model's structure -- was this "
            "fit produced with a different pooling scheme? Recomputing against a "
            "mismatched graph would return wrong numbers with no error attached."
        )
    return model


def panel_component_totals(idata, model=None, chunk: int = DRAW_CHUNK) -> dict[str, np.ndarray]:
    """Per-draw, per-geo component totals on the original sales scale.

    Returns an array of shape (chain, draw, geo) for each of the four additive
    components, in dollars: what that component contributed to that division across the
    whole 113-week window, on that posterior draw.

    Three scale corrections, all of which the audit caught `attribution` getting wrong on
    a panel because it assumed one series (see `attribution._reject_panel_models`):

    * `target_scale` is per geo -- 26 maxima spanning 326k to 3.6M -- so a single global
      divisor is off by up to 10x for a given division;
    * the intercept applies once per *date* (113), not once per row of the long-format
      panel (2,938);
    * seasonality is its own additive term here, not a control.
    """
    import pymc as pm

    model = _built_geo_model(idata, model)
    posterior = _dataset(idata, "posterior")
    free = [rv.name for rv in model.model.free_RVs]

    # Per-geo scale and the true period count, both read off the fit rather than passed
    # in, so neither can disagree with the model that produced it.
    target_scale = _dataset(idata, "constant_data")["target_scale"]
    n_periods = posterior.sizes["date"]

    accumulated: dict[str, list] = {name: [] for name in _PANEL_COMPONENTS}

    for start in range(0, posterior.sizes["draw"], chunk):
        block = pm.compute_deterministics(
            posterior[free].isel(draw=slice(start, start + chunk)),
            model=model.model,
            var_names=list(_PANEL_COMPONENTS),
            merge_dataset=False,
            progressbar=False,
        )
        for name in _PANEL_COMPONENTS:
            values = block[name]
            reduce_dims = [d for d in values.dims if d not in ("chain", "draw", "geo")]
            totals = values.sum(reduce_dims) if reduce_dims else values
            if "date" not in values.dims:
                # A level, not a series: multiply out by the number of periods before it
                # is comparable to the components that were summed over date.
                totals = totals * n_periods
            # Reduce inside the loop: holding un-reduced arrays for every chunk would
            # rebuild exactly the 1.1GB this avoids.
            accumulated[name].append(np.asarray(totals * target_scale))

    return {
        _COMPONENT_LABELS[name]: np.concatenate(chunks, axis=1)
        for name, chunks in accumulated.items()
    }


def _resolve(idata, components, model, chunk) -> dict[str, np.ndarray]:
    """Use the caller's component totals, or compute them.

    Every report below is a different view of the same four arrays, and rebuilding them
    costs a model build plus a graph compile. A caller producing a full report -- which
    is what Phase 6 will do -- computes `panel_component_totals` once and passes it to
    each, rather than paying for it four times over.
    """
    if components is not None:
        return components
    return panel_component_totals(idata, model=model, chunk=chunk)


def geo_decompose(idata, sales: pd.Series | None = None, model=None,
                  chunk: int = DRAW_CHUNK, components=None) -> pd.DataFrame:
    """Posterior mean contribution of each component across the whole panel.

    Same shape of answer as `attribution.decompose`, aggregated over divisions, so the
    two models' decompositions can be read side by side. `sales` defaults to the fit's
    own observed target, which is what makes the reconciliation row mean anything.
    """
    components = _resolve(idata, components, model, chunk)
    aggregated = {name: values.sum(axis=-1) for name, values in components.items()}
    return _shares_table(aggregated, _observed_total(idata, sales))


def geo_decompose_by_division(idata, model=None, chunk: int = DRAW_CHUNK,
                              components=None) -> pd.DataFrame:
    """The same decomposition, one row per division, as a share of that division's sales.

    This is the form Phase 6 needs. The geo-DiD estimator assigns treatment division by
    division, so a single panel-wide media share is not the quantity to calibrate
    against -- what matters is whether a given division's modelled media response is
    something a holdout in that division could have detected.
    """
    components = _resolve(idata, components, model, chunk)

    table = pd.DataFrame(
        {name: values.reshape(-1, values.shape[-1]).mean(axis=0)
         for name, values in components.items()},
        index=pd.Index(list(_dataset(idata, "posterior").coords["geo"].values), name="geo"),
    )
    table["observed_sales"] = observed_sales_by_division(idata).reindex(table.index).to_numpy()
    table["media_share"] = table["media"] / table["observed_sales"]
    table["reconciliation"] = table[list(COMPONENTS)].sum(axis=1) / table["observed_sales"]
    return table.sort_values("media_share", ascending=False)


def geo_media_share_interval(idata, sales: pd.Series | None = None, model=None,
                             hdi_prob: float = 0.94, chunk: int = DRAW_CHUNK,
                             components=None) -> tuple[float, float, float]:
    """(lower, mean, upper) for media's share of total panel sales."""
    components = _resolve(idata, components, model, chunk)
    media = components["media"].sum(axis=-1).ravel() / _observed_total(idata, sales)
    return _interval(media, hdi_prob)


def geo_identification_report(idata, model=None, chunk: int = DRAW_CHUNK,
                              components=None) -> pd.DataFrame:
    """Is the geo model's media/baseline split identified?

    The same test `attribution.identification_report` runs on the national model, and the
    reason it has to be run here too: Phase 6 plans to calibrate the national model
    against this panel, and a calibration target carrying the national model's own
    pathology would be lending strength it does not have. Answering it needs the panel
    decomposition, which is why the question could not be asked before this module
    existed.
    """
    components = _resolve(idata, components, model, chunk)
    media = components["media"].sum(axis=-1).ravel()
    baseline = sum(components[name].sum(axis=-1).ravel()
                   for name in ("intercept", "controls", "seasonality"))
    return _identification_table(media, baseline)


def observed_sales_by_division(idata) -> pd.Series:
    """Observed sales per division, in dollars.

    `observed_data["y"]` is NOT in dollars: pymc-marketing stores the target as the model
    saw it, divided by each division's own maximum, so summing it gives ~28 per division
    rather than a revenue figure. Multiplying by `target_scale` is what puts it back.

    Caught by the reconciliation row, which is the argument for computing that row at
    all: the components were right and the thing they were being checked against was
    wrong, and the ratio said so immediately (3.6e6 rather than ~1.0).
    """
    observed = _dataset(idata, "observed_data")["y"].sum("date")
    scaled = observed * _dataset(idata, "constant_data")["target_scale"]
    return scaled.to_series()


def _observed_total(idata, sales: pd.Series | None) -> float:
    """Total observed sales across the panel, taken from the fit unless overridden."""
    if sales is not None:
        return float(sales.sum())
    return float(observed_sales_by_division(idata).sum())


def main() -> None:
    """Print the panel decomposition, with the health check above it.

    Deliberately the same shape as `scripts/report_contributions.py`: numbers below a
    warning when the model does not support them, rather than a clean table that reads as
    a result. The geo model earns that warning -- media at 54.6% of panel sales is above
    the threshold `attribution.IMPLAUSIBLE_MEDIA_SHARE` exists to catch.
    """
    from mmm_bayes.attribution import IMPLAUSIBLE_MEDIA_SHARE
    from mmm_bayes.config import GEO_FIT_NC
    from mmm_bayes.diagnostics import load_idata

    pd.set_option("display.width", 140)
    idata = load_idata(GEO_FIT_NC)

    print("=" * 78)
    print("GEO PANEL -- CONTRIBUTION DECOMPOSITION WITH CREDIBLE INTERVALS")
    print("=" * 78)
    print("Recomputing contributions from the cached free RVs (the fit does not store")
    print("them -- see this module's docstring)...\n")

    # Computed once and passed to each report: rebuilding them costs a model build and a
    # graph compile, and all four views below are views of the same four arrays.
    components = panel_component_totals(idata)

    print("--- decomposition (posterior mean, original sales scale) ---")
    table = geo_decompose(idata, components=components)
    print(table.assign(
        mean=lambda d: d["mean"].map("{:,.0f}".format),
        share_of_sales=lambda d: d["share_of_sales"].map("{:.1%}".format),
    ).to_string())

    lower, mean, upper = geo_media_share_interval(idata, components=components)
    print(f"\nmedia-attributed share of panel sales: {mean:.1%} "
          f"[{lower:.1%}, {upper:.1%}] at 94% credibility")

    print("\n--- is the media/baseline split identified? ---")
    identification = geo_identification_report(idata, components=components)
    print(identification.round(4).to_string())

    if mean > IMPLAUSIBLE_MEDIA_SHARE:
        correlation = float(identification["corr_media_baseline"].iloc[0])
        print()
        print("!" * 78)
        print(f"DO NOT USE THESE NUMBERS. Media at {mean:.0%} of sales is not a finding,")
        print("it is the same symptom the national model shows. This panel has no dollar")
        print("spend at all, so there is not even a return to sanity-check it against.")
        print()
        print(f"Media and baseline correlate at {correlation:+.3f} in the posterior, and")
        print("their sum is far better determined than either part -- the data fixes")
        print("total sales and leaves the split to the priors.")
        print()
        print("This matters beyond this model: Phase 6 planned to calibrate the national")
        print("model against this panel. Calibrating an unidentified split against")
        print("another unidentified split transfers a prior, not evidence.")
        print("See docs/DIAGNOSTICS.md.")
        print("!" * 78)

    print("\n--- per division (share of that division's own sales) ---")
    by_division = geo_decompose_by_division(idata, components=components)
    print(by_division[["media_share", "observed_sales", "reconciliation"]].assign(
        media_share=lambda d: d["media_share"].map("{:.1%}".format),
        observed_sales=lambda d: d["observed_sales"].map("{:,.0f}".format),
        reconciliation=lambda d: d["reconciliation"].map("{:.3f}".format),
    ).to_string())

    spread = by_division["media_share"]
    sales = by_division["observed_sales"]
    print(f"\nmedia share spans {spread.min():.1%} to {spread.max():.1%} across 26 "
          f"divisions whose sales span {sales.max() / sales.min():.0f}x.")
    print("That narrowness is NOT the pooling: saturation_beta's across-geo scale sits")
    print("at prior-CDF 0.07-0.46, well inside a prior that left room it did not use.")
    print("It is the CHANNEL SCALING. Dividing each channel by its own per-geo maximum")
    print("removes each division's media-to-sales ratio -- the cross-sectional contrast")
    print("a geo-DiD exploits. Raw intensity spans 11.17x across divisions; scaled, the")
    print("model sees 1.79x, and inverted (corr -0.457). See docs/DIAGNOSTICS.md and")
    print("geo_model.target_relative_channel_scaling for the alternative.")


if __name__ == "__main__":
    main()
