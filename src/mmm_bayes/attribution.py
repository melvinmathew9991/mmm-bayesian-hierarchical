"""Contribution decomposition on the original sales scale.

Phase 5 needed this to answer a question convergence diagnostics cannot: the national
model passes R-hat, ESS and E-BFMI, and still attributes 105% of sales to media. A
sampler can explore a misspecified posterior perfectly well.

(105% is the figure at `target_accept=0.9`, where the diagnosis in docs/DIAGNOSTICS.md
was carried out. The final fit at 0.99 gives 107.6%, which is what
`report_contributions.py` prints. The 105% figures quoted through this module and
`scripts/sensitivity_checks.py` are quoted as they were measured.)

Two things make this module worth having rather than inlining the arithmetic:

1. **The scale.** pymc-marketing's `channel_contribution` is in max-scaled target units,
   so every share computed straight off it is wrong by a factor of
   `max(sales) / sum(sales)` -- about 65x here. Getting that wrong in one direction
   makes media look negligible and in the other makes it look impossible, and neither
   error announces itself.
2. **The reconciliation.** `decompose` returns the components AND their total, so the
   caller can check they add up to observed sales. Without that check, "media is 105% of
   sales" is indistinguishable from a units bug -- which is exactly what had to be ruled
   out before it could be reported as a finding.

**Single time series only.** Every function here assumes one series and one target
scale, which is true of the national model and false of the hierarchical geo one. Panel
fits are rejected rather than mis-answered -- see `_reject_panel_models` -- and
`mmm_bayes.geo_attribution` is where the panel version lives.

The split between the two modules is by responsibility rather than by model. What is
model-specific is *obtaining* the per-draw component totals: the national fit reads
stored deterministics and divides by one scale, the geo fit recomputes dropped
deterministics and applies a per-geo scale vector. What is not model-specific is
everything done to those totals afterwards -- shares, credible intervals, the
identification report -- so `_shares_table`, `_interval` and `_identification_table`
below are shared with the panel module rather than written twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Above this, a media share is not a finding about marketing, it is a symptom.
# Media spend is 2.2% of revenue in this dataset (Phase 1), so a share of 0.5 implies a
# return of roughly 23x across the whole portfolio -- far outside anything a retail MMM
# reports. Set as a smell test, not a target: nothing in the model is tuned against it.
IMPLAUSIBLE_MEDIA_SHARE = 0.5


#: Panel dimensions this module cannot handle. See `_reject_panel_models`.
_PANEL_DIMS = ("geo",)


def _reject_panel_models(idata) -> None:
    """Refuse a multidimensional (panel) fit rather than return a wrong number.

    Found by audit, not by use. Everything in this module assumes a single time series:
    it multiplies the intercept by `len(sales)` and divides by a single `sales.max()`.
    Both are wrong for a panel fit, and wrong in ways that do not announce themselves --
    run against the geo model, `decompose` reported the intercept at 3051% of sales and
    the components failed to reconcile by a factor of 34.

    Two separate errors compound there. `len(sales)` on a long-format panel counts
    division-weeks (2,938) where the intercept applies once per *date* (113). And
    pymc-marketing scales the target per geo -- `target_scale` has one entry per
    division, differing by more than 10x -- so a single global max is not the divisor
    the model used.

    Supporting panels properly means threading the per-geo scale vector and the true
    period count through every function here. That is a feature, and it is not attempted
    under an audit. What is fixed is the silence: the module now says it cannot do this,
    instead of answering anyway.

    That feature now exists, as `mmm_bayes.geo_attribution`, and this refusal stays
    anyway. It is not a stub waiting to be replaced by a branch: the panel path needs a
    built model to recompute the contributions the geo fit does not store, so it cannot
    share these functions' signature even if it shared their arithmetic. Callers get a
    pointer rather than a wrong number.
    """
    panel = [d for d in _PANEL_DIMS if d in idata.posterior.dims]
    if panel:
        raise NotImplementedError(
            f"mmm_bayes.attribution does not support panel fits (found dims {panel}). "
            "Its arithmetic assumes one time series and one target scale; on the geo "
            "model it silently reports an intercept share above 3000% and does not "
            "reconcile. Use it on the national fit only, and "
            "mmm_bayes.geo_attribution for the panel."
        )


def _component_totals(idata, n_periods: int, scale: float) -> dict[str, np.ndarray]:
    """Per-draw totals for each additive component, on the original target scale."""
    _reject_panel_models(idata)
    posterior = idata.posterior

    media = np.asarray(
        posterior["channel_contribution"].sum(
            [d for d in posterior["channel_contribution"].dims if d not in ("chain", "draw")]
        )
    ) * scale

    if "control_contribution" in posterior:
        controls = np.asarray(
            posterior["control_contribution"].sum(
                [d for d in posterior["control_contribution"].dims
                 if d not in ("chain", "draw")]
            )
        ) * scale
    else:
        controls = np.zeros_like(media)

    intercept = posterior["intercept_contribution"]
    extra = [d for d in intercept.dims if d not in ("chain", "draw")]
    # The intercept is a level, not a series: it applies once per period, so it has to
    # be multiplied out by the number of periods before it is comparable to the others.
    intercept = np.asarray(intercept.sum(extra) if extra else intercept) * n_periods * scale

    return {"intercept": intercept, "controls": controls, "media": media}


def decompose(idata, sales: pd.Series) -> pd.DataFrame:
    """Posterior mean contribution of each component, in sales units and as a share.

    `sales` is the observed target on its original scale; its max is the divisor
    pymc-marketing applied, and its sum is what the shares are shares of.
    """
    scale = float(sales.max())
    components = _component_totals(idata, n_periods=len(sales), scale=scale)
    return _shares_table(components, float(sales.sum()))


def media_share_interval(idata, sales: pd.Series,
                         hdi_prob: float = 0.94) -> tuple[float, float, float]:
    """(lower, mean, upper) for media's share of total sales.

    The interval is the deliverable. The ridge project reported 12.4% media-attributed
    and stated in its own README that it had no interval around that number; producing
    one is the reason this repo exists.
    """
    scale = float(sales.max())
    total_sales = float(sales.sum())
    media = _component_totals(idata, len(sales), scale)["media"].ravel() / total_sales
    return _interval(media, hdi_prob)


def identification_report(idata, sales: pd.Series) -> pd.DataFrame:
    """Is the media/baseline split identified, or is the prior deciding it?

    This is the diagnostic that explains Phase 5's whole result, and it is not one any
    convergence check performs. The test is simple: if two components are separately
    identified, their SUM should be no better determined than either part. If the sum is
    sharply determined while the parts are not, the data has pinned down the total and
    left the split to the prior.

    On the national model the answer is unambiguous -- posterior correlation between
    total media contribution and the intercept is -0.997, and their sum has a
    coefficient of variation of 0.021 against 0.256 for media alone. The data determines
    total sales, which is trivially true, and says almost nothing about how much of it
    media caused.

    That is a specification problem, not a sampling problem. Saturated, adstocked,
    always-positive media is nearly a constant plus noise on this data, which makes it
    nearly collinear with a scalar intercept; the ridge model breaks the same tie with
    an L2 penalty, and this model breaks it with the media-coefficient prior. Neither
    breaks it with evidence. Resolving it needs information from outside the time series
    -- an experiment, a lift test, a geo holdout -- which is what Phases 6 and 7 are for.
    """
    scale = float(sales.max())
    components = _component_totals(idata, len(sales), scale)
    media = components["media"].ravel()
    baseline = components["intercept"].ravel() + components["controls"].ravel()
    return _identification_table(media, baseline)


def channel_shares(idata, sales: pd.Series, channels: list[str],
                   hdi_prob: float = 0.94) -> pd.DataFrame:
    """Per-channel contribution as a share of total sales, with a credible interval."""
    scale = float(sales.max())
    total_sales = float(sales.sum())

    _reject_panel_models(idata)
    contribution = idata.posterior["channel_contribution"]
    other = [d for d in contribution.dims if d not in ("chain", "draw", "channel")]
    per_channel = np.asarray(contribution.sum(other)) * scale / total_sales
    flat = per_channel.reshape(-1, per_channel.shape[-1])

    tail = (1 - hdi_prob) / 2
    return pd.DataFrame(
        {
            "share_mean": flat.mean(axis=0),
            f"q{int(tail * 100):02d}": np.quantile(flat, tail, axis=0),
            f"q{int((1 - tail) * 100):02d}": np.quantile(flat, 1 - tail, axis=0),
        },
        index=pd.Index(channels, name="channel"),
    ).sort_values("share_mean", ascending=False)


# --- shared with mmm_bayes.geo_attribution -----------------------------------------
# These three operate on per-draw component totals that are already on the original
# sales scale. Reaching those totals is what differs between a single series and a
# 26-division panel -- one scale factor or twenty-six, a stored deterministic or a
# recomputed one. What happens afterwards does not differ at all, so the panel module
# imports these rather than restating them and letting the two drift.


def _shares_table(components: dict[str, np.ndarray], total_sales: float) -> pd.DataFrame:
    """Posterior mean and share of sales per component, plus a `total` row.

    The `total` row is the point. Without it, "media is 105% of sales" is
    indistinguishable from a units bug -- it is the reconciliation against observed sales
    that lets one be ruled out before the other is reported as a finding.
    """
    rows = {
        name: {
            "mean": float(values.mean()),
            "share_of_sales": float(values.mean()) / total_sales,
        }
        for name, values in components.items()
    }
    stacked = sum(components.values())
    rows["total"] = {
        "mean": float(stacked.mean()),
        "share_of_sales": float(stacked.mean()) / total_sales,
    }
    return pd.DataFrame(rows).T[["mean", "share_of_sales"]]


def _interval(values: np.ndarray, hdi_prob: float) -> tuple[float, float, float]:
    """(lower, mean, upper) from a flat array of per-draw values."""
    tail = (1 - hdi_prob) / 2
    return (
        float(np.quantile(values, tail)),
        float(values.mean()),
        float(np.quantile(values, 1 - tail)),
    )


def _identification_table(media: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """Media, baseline and their sum, with the coefficient of variation of each.

    The comparison that matters is the third row against the first two: if the sum is
    sharply determined while the parts are not, the data has fixed total sales and left
    the split to the prior. `identification_report` is the long-form version.
    """
    combined = media + baseline

    def cv(values: np.ndarray) -> float:
        mean = values.mean()
        return float(values.std() / abs(mean)) if mean else np.nan

    return pd.DataFrame(
        {
            "mean": [media.mean(), baseline.mean(), combined.mean()],
            "sd": [media.std(), baseline.std(), combined.std()],
            "cv": [cv(media), cv(baseline), cv(combined)],
        },
        index=pd.Index(["media", "baseline", "media + baseline"], name="component"),
    ).assign(
        corr_media_baseline=[float(np.corrcoef(media, baseline)[0, 1])] * 3,
    )
