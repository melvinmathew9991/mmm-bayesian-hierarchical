"""Slow (MCMC / ridge-refit) checks on the Phase 3 anchored model.

Split out from tests/test_priors.py so the fast suite stays fast: everything here
either samples or refits the ridge model, and both are minutes-scale on a machine
without a C compiler for pytensor. Run with `pytest -m slow`.
"""
import numpy as np
import pytest

from mmm_bayes.loaders import PAID_CHANNELS
from mmm_bayes.model import run_skeleton_fit
from mmm_bayes.priors import anchor_table, load_ridge_anchors


@pytest.fixture(scope="module")
def anchored_fit():
    """One short anchored chain, shared across the tests below.

    This says nothing about convergence -- that is Phase 5's job, and the answer is in
    docs/DIAGNOSTICS.md. What it does prove is that the anchored specification compiles,
    that per-channel priors line up with the model's channel dimension, and that the
    posterior lands in the region the priors point at rather than somewhere the
    reparametrisation mangled.

    `target_accept=0.9`, not the 0.99 Phase 5 made the default. That default is right
    for a real fit and wrong here: at 0.99 the step size is small enough that a 20-draw
    chain hits max tree depth and barely moves, and the neighbourhood test below then
    fails on a sampler artefact rather than on anything about the anchoring. Raising
    target_accept buys fewer divergences at real chain length and costs exploration at
    tiny ones -- worth knowing, and the reason this fixture states its own settings
    instead of inheriting them.
    """
    return run_skeleton_fit(draws=200, tune=200, chains=1, anchored=True,
                            target_accept=0.9)


@pytest.mark.slow
def test_anchored_model_samples_end_to_end(anchored_fit):
    _, result = anchored_fit
    assert result["idata"] is not None


@pytest.mark.slow
def test_anchored_model_has_per_channel_decay_and_half_point(anchored_fit):
    """The whole point of Phase 3: alpha and lam are estimated per channel now.

    Under the Phase 2 defaults these were scalars shared across all ten channels. If a
    future edit dropped `dims="channel"` from the priors, the model would still sample
    and still look anchored in the docs -- this is what catches that.
    """
    _, result = anchored_fit
    posterior = result["idata"].posterior

    for var in ("adstock_alpha", "saturation_lam"):
        assert var in posterior, f"{var} missing from the posterior: {list(posterior)}"
        assert "channel" in posterior[var].dims, (
            f"{var} has dims {posterior[var].dims} -- it is not per-channel, so the "
            "anchored priors are not reaching the model."
        )
        assert posterior[var].sizes["channel"] == len(PAID_CHANNELS)


@pytest.mark.slow
def test_anchored_posterior_stays_in_the_neighbourhood_of_its_anchors(anchored_fit):
    """A units error in the half-point conversion would not crash -- it would just
    push the posterior somewhere the anchors never pointed. At this chain length the
    posterior is still dominated by the prior, so the check is deliberately loose: every
    channel's posterior-mean half-point should sit within the prior's own 90% band
    (the anchor times [0.32, 3.16] for LogNormal sigma = 0.7).
    """
    _, result = anchored_fit
    lam_post = result["idata"].posterior["saturation_lam"].mean(("chain", "draw"))
    table = anchor_table()

    for ch in PAID_CHANNELS:
        anchor = float(table.loc[ch, "lam_anchor"])
        got = float(lam_post.sel(channel=ch))
        assert anchor / 3.2 < got < anchor * 3.2, (
            f"{ch}: posterior-mean half-point {got:.4f} is outside the prior's own 90% "
            f"band around its anchor {anchor:.4f}. Check the unit conversion in "
            "priors.half_point_to_lam before trusting this fit."
        )


@pytest.mark.slow
def test_cached_anchors_still_match_a_fresh_ridge_fit():
    """The cache in data/derived/ is a convenience, not a second source of truth.

    Its MD5 guard catches the data changing underneath it, but not the ridge project
    itself changing -- a new version of `mmm` installed from the sibling repo could
    fit different decays and the cache would keep serving the old ones. Refitting here
    is the only thing that catches that, which is why it is worth 90 seconds.
    """
    from mmm_bayes.baseline_replication import replicate_ridge_baseline, summarize

    cached = load_ridge_anchors()["channels"]
    fresh = summarize(replicate_ridge_baseline(verbose=False))

    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(
            [cached[ch]["decay"], cached[ch]["half_point"]],
            [fresh.loc[ch, "decay"], fresh.loc[ch, "half_point"]],
            rtol=1e-9,
            err_msg=(
                f"{ch}: cached anchors disagree with a fresh ridge fit. The ridge "
                "project's estimator has changed -- regenerate with "
                "`python scripts/build_ridge_anchors.py`."
            ),
        )
