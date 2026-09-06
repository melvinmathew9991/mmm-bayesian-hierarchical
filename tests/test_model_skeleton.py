"""Tests for the Phase 2 skeleton.

The MCMC smoke test uses a deliberately tiny chain (20 draws, 20 tune, 1 chain) --
enough to prove the model compiles and samples without the cores/ZeroDivisionError
or a hard crash, not enough to say anything about convergence. Real diagnostics
(R-hat, ESS, divergences at a chain length that means something) are Phase 5's job,
not this smoke test's.

The two MCMC tests below pass `anchored=False` on purpose. They were written against
the Phase 2 specification -- library-default priors and plain LogisticSaturation --
and the docs/CHALLENGES.md #2 conclusion they pin was reached on that model. Letting
them silently follow `build_model`'s new anchored default would change what they test
while leaving their names and docstrings claiming otherwise. The anchored model gets
its own end-to-end test in tests/test_priors_fit.py.

A caution these tests earned the hard way. Phase 5 found the national model showing 2
divergences at 2 chains x 500 draws and 48 at 4 x 1000. A tiny chain does not estimate
a divergence rate imprecisely -- it makes a real problem look like rounding error. The
zero-divergence assertion below is a smoke test, not evidence about geometry; that lives
in docs/DIAGNOSTICS.md.
"""
import pytest

from mmm_bayes.loaders import PAID_CHANNELS
from mmm_bayes.model import build_dataframe, build_model, run_skeleton_fit


def test_build_dataframe_shape():
    X, y = build_dataframe()
    assert len(X) == len(y) == 209
    assert "date" in X.columns
    for ch in PAID_CHANNELS:
        assert ch in X.columns


@pytest.mark.parametrize("anchored", [False, True])
def test_build_model_constructs_without_fitting(anchored):
    X, _ = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]
    model = build_model(control_columns, anchored=anchored)
    assert model is not None


@pytest.mark.slow
def test_skeleton_fit_runs_without_error():
    """Regression test for the cores=None ZeroDivisionError (docs/CHALLENGES.md #1).
    Tiny chain -- this checks the pipeline runs end-to-end, not inference quality."""
    model, result = run_skeleton_fit(draws=20, tune=20, chains=1, anchored=False)
    assert result["idata"] is not None


@pytest.mark.slow
def test_skeleton_fit_has_zero_divergences():
    """docs/CHALLENGES.md #2, now resolved.

    Written in Phase 2 as `..._despite_overflow_warning`, when an overflow
    RuntimeWarning survived two scaling fixes and the working conclusion was that it was
    a benign transient of NUTS's jitter phase. Phase 5 confirmed that directly:
    `init="adapt_diag"` removes the jitter and the warning disappears, with no posterior
    movement. The warning is no longer part of what this test observes, so it is no
    longer part of its name.

    What the test still pins is the divergence count on the Phase 2 specification, at a
    chain far too short to be informative about it -- see the module docstring. Phase 5
    is where divergences are actually measured.
    """
    model, result = run_skeleton_fit(draws=20, tune=20, chains=1, anchored=False)
    idata = result["idata"]
    n_divergent = int(idata.sample_stats["diverging"].sum())
    assert n_divergent == 0, (
        f"Expected 0 divergences on the Phase 2 specification, got {n_divergent}."
    )


def test_progressbar_is_disabled_on_a_console_that_cannot_encode_it():
    """docs/CHALLENGES.md #5: PyMC's rich progress bar writes U+2009, which a stock
    Windows cp1252 console cannot encode -- killing the run AFTER sampling finishes.
    The guard has to key off the actual stdout encoding, not the platform."""
    import io
    import sys

    from mmm_bayes.model import console_supports_progressbar

    original = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        assert console_supports_progressbar() is False
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        assert console_supports_progressbar() is True
    finally:
        sys.stdout = original
