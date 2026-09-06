"""Per-channel media contribution with credible intervals, from a cached full fit.

    python scripts/report_contributions.py

This is the artefact the whole project exists to produce. The ridge project's README
states its own gap in one line -- *"No posterior. Non-negative ridge stands in for a
hierarchical Bayesian MMM, so there are no credible intervals on any contribution."*
Every number below has an interval.

Read the health check at the top before reading the numbers. As of Phase 5 the national
model attributes an implausible share of sales to media, and printing the intervals
without that warning attached would be presenting a broken decomposition as a result.
"""
import argparse

import pandas as pd

from mmm_bayes.attribution import (
    IMPLAUSIBLE_MEDIA_SHARE,
    channel_shares,
    decompose,
    identification_report,
    media_share_interval,
)
from mmm_bayes.config import NATIONAL_FIT_NC
from mmm_bayes.diagnostics import load_idata, summarise
from mmm_bayes.loaders import PAID_CHANNELS, load_national
from mmm_bayes.priors import load_ridge_anchors

RIDGE_MEDIA_SHARE = 0.124  # Phase 1 replication, docs/PRIORS.md


def main(hdi_prob: float) -> None:
    pd.set_option("display.width", 130)
    idata = load_idata(NATIONAL_FIT_NC)
    sales = load_national()["sales"]

    print("=" * 78)
    print("NATIONAL MODEL -- MEDIA CONTRIBUTION WITH CREDIBLE INTERVALS")
    print("=" * 78)
    print(summarise(idata, "sampling"))

    print("\n--- decomposition (posterior mean, original sales scale) ---")
    table = decompose(idata, sales)
    print(table.assign(
        mean=lambda d: d["mean"].map("{:,.0f}".format),
        share_of_sales=lambda d: d["share_of_sales"].map("{:.1%}".format),
    ).to_string())

    lower, mean, upper = media_share_interval(idata, sales, hdi_prob)
    print(f"\nmedia-attributed share of sales: {mean:.1%} "
          f"[{lower:.1%}, {upper:.1%}] at {hdi_prob:.0%} credibility")
    print(f"ridge project's point estimate  : {RIDGE_MEDIA_SHARE:.1%} (no interval)")

    print()
    print("--- is the media/baseline split identified? ---")
    identification = identification_report(idata, sales)
    print(identification.round(4).to_string())

    if mean > IMPLAUSIBLE_MEDIA_SHARE:
        correlation = float(identification["corr_media_baseline"].iloc[0])
        print()
        print("!" * 78)
        print(f"DO NOT USE THESE NUMBERS. Media at {mean:.0%} of sales is not a finding,")
        print("it is a symptom -- media spend is 2.2% of revenue, so this implies a")
        print("return no business sees.")
        print()
        print(f"Media and baseline correlate at {correlation:+.3f} in the posterior, and")
        print("their sum is far better determined than either part. The data fixes total")
        print("sales and leaves the split to the priors; no prior setting tested comes")
        print("near the ridge estimate. That makes this a specification problem, not a")
        print("tuning one -- it needs information from outside the time series.")
        print("See docs/DIAGNOSTICS.md.")
        print("!" * 78)

    print(f"\n--- per channel, share of total sales ({hdi_prob:.0%} interval) ---")
    shares = channel_shares(idata, sales, PAID_CHANNELS, hdi_prob)

    anchors = load_ridge_anchors()["channels"]
    total_sales = float(sales.sum())
    shares["ridge_share"] = [
        anchors[c]["ridge_contribution"] / total_sales for c in shares.index
    ]
    print(shares.map("{:.2%}".format).to_string())

    print("\nThe ridge column is a point estimate from the same data and the same "
          "channels\n(Phase 1). Phase 9 is where the two are compared properly; it is "
          "here only so\nthe scale of the disagreement is visible next to the intervals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdi-prob", type=float, default=0.94)
    main(parser.parse_args().hdi_prob)
