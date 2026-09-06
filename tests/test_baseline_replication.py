import pandas as pd

from mmm_bayes.baseline_replication import (
    replicate_ridge_baseline,
    summarize,
    verify_data_matches,
)


def test_verify_data_matches_does_not_raise():
    verify_data_matches()


def test_replicate_ridge_baseline_matches_known_headline_numbers():
    """Regression test against the ridge project's documented headline result
    (12.4% media-attributed, 2.2% spend share) -- not just "did it run"."""
    result = replicate_ridge_baseline(verbose=False)
    fit = result["fit"]
    media = result["media"]
    total_contribution = fit.contributions(media.values).sum()

    from mmm_bayes.loaders import SPEND_COLS, load_national
    df = load_national()
    media_share = total_contribution / df["sales"].sum()
    spend_share = df[SPEND_COLS].sum().sum() / df["sales"].sum()

    assert abs(media_share - 0.124) < 0.005, f"expected ~12.4% media share, got {media_share:.1%}"
    assert abs(spend_share - 0.022) < 0.002, f"expected ~2.2% spend share, got {spend_share:.1%}"


def test_summarize_returns_all_channels():
    result = replicate_ridge_baseline(verbose=False)
    table = summarize(result)
    assert len(table) == 10
    assert isinstance(table, pd.DataFrame)
    assert "ridge_contribution" in table.columns
