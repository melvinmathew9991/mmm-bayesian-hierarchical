"""Slow (MCMC) checks on the Phase 4 hierarchical geo model. Run with `pytest -m slow`.

The tiny chains here say nothing about convergence -- Phase 5 owns that. What they can
establish is that the hierarchy is real: that the pooled parameters have a shared scale
the sampler actually estimates, and that the scale comes out where the raw data says it
should. A hierarchical model that samples but shrinks nothing is indistinguishable from
an unpooled one, and no structural test can tell the difference.
"""
import numpy as np
import pytest

from mmm_bayes.geo_model import run_geo_fit
from mmm_bayes.loaders import GEO_CHANNELS


@pytest.fixture(scope="module")
def geo_fit():
    # target_accept=0.9, not Phase 5's 0.99 default: at 0.99 the step size is small
    # enough that a 20-draw chain barely moves, which turns these structural checks into
    # tests of the sampler. See tests/test_priors_fit.py::anchored_fit.
    return run_geo_fit(draws=20, tune=20, chains=1, pooled=True, target_accept=0.9,
                       progressbar=False)


@pytest.mark.slow
def test_hierarchical_geo_model_samples_end_to_end(geo_fit):
    _, result = geo_fit
    assert result["idata"] is not None
    assert result["pooled"] is True


@pytest.mark.slow
def test_a_caller_supplied_model_is_not_labelled_with_this_functions_pooling():
    """`pooled` describes what run_geo_fit built. When a caller passes its own model --
    as the decay-pooling sensitivity check does, with a differently pooled adstock --
    reporting `pooled=True` would attach this function's label to someone else's model.
    It reports None instead."""
    from mmm_bayes.geo_model import build_geo_model

    supplied = build_geo_model(pooled=True)
    _, result = run_geo_fit(draws=15, tune=15, chains=1, model=supplied,
                            target_accept=0.9, progressbar=False)
    assert result["pooled"] is None


@pytest.mark.slow
def test_posterior_carries_the_hierarchy(geo_fit):
    """Every partially pooled block must appear in the posterior with its shared
    location and scale. If a future edit drops a hyperprior, the model still samples
    and still looks hierarchical from the outside -- this is what notices."""
    posterior = geo_fit[1]["idata"].posterior

    for block in ("saturation_beta_raw", "saturation_lam_raw", "gamma_fourier",
                  "gamma_control", "intercept_contribution_raw"):
        assert f"{block}_sigma" in posterior, f"{block} lost its across-geo scale"
        assert f"{block}_mu" in posterior, f"{block} lost its shared location"


@pytest.mark.slow
def test_adstock_stays_pooled_in_the_posterior(geo_fit):
    """Decay is deliberately given no geo axis. A posterior with one would mean the
    pooling silently stopped applying."""
    posterior = geo_fit[1]["idata"].posterior
    assert "geo" not in posterior["adstock_alpha"].dims
    assert posterior["adstock_alpha"].sizes["channel"] == len(GEO_CHANNELS)


@pytest.mark.slow
def test_seasonality_shrinks_further_than_media_response():
    """The substantive prediction the pooling structure makes.

    An OLS fit on the raw panel puts the across-division sd of the Fourier
    coefficients at ~0.01 against coefficient magnitudes of ~0.1 -- divisions of this
    business share a seasonal shape almost exactly. Media response has no such reason
    to be identical. So the fitted `gamma_fourier_sigma` should shrink much closer to
    zero than `saturation_beta_raw_sigma`.

    This is the test that would fail if the pooling were decorative: with no real
    shrinkage both scales would simply track their priors, which are within 8x of each
    other, not the order of magnitude asserted here.
    """
    _, result = run_geo_fit(draws=150, tune=250, chains=1, pooled=True,
                            target_accept=0.9, progressbar=False)
    posterior = result["idata"].posterior

    seasonal = float(posterior["gamma_fourier_sigma"].mean())
    media = float(posterior["saturation_beta_raw_sigma"].mean())

    assert seasonal < media / 5, (
        f"across-geo seasonality scale {seasonal:.4f} is not much smaller than the "
        f"media-response scale {media:.4f}. Either the divisions stopped sharing a "
        "seasonal shape, or the pooling is not shrinking anything."
    )
    assert seasonal < 0.02, (
        f"gamma_fourier_sigma {seasonal:.4f} is larger than the ~0.01 across-division "
        "spread measured directly from the panel by OLS"
    )


@pytest.mark.slow
def test_media_contributions_are_non_negative(geo_fit):
    """The `transform="exp"` on saturation_beta exists to guarantee this. Spending
    money cannot reduce sales -- the same constraint the ridge project enforces with a
    coefficient floor and the national model with a HalfNormal."""
    contributions = geo_fit[1]["idata"].posterior["channel_contribution"]
    assert float(contributions.min()) >= 0.0, (
        "a negative media contribution means the positivity transform is no longer "
        "being applied to saturation_beta"
    )


@pytest.mark.slow
def test_every_division_gets_its_own_media_coefficient(geo_fit):
    """Partial pooling, not complete pooling: the whole point is that divisions are
    allowed to differ. If every geo's coefficient were identical the model would have
    collapsed to a single national one with extra steps."""
    beta = geo_fit[1]["idata"].posterior["saturation_beta"].mean(("chain", "draw"))
    assert "geo" in beta.dims
    spread = float(beta.std("geo").mean())
    assert spread > 0, "media coefficients are identical across all 26 divisions"
    assert np.isfinite(spread)
