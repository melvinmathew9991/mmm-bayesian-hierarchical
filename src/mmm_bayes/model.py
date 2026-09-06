"""Phases 2-3 -- the Bayesian core model.

Builds a pymc-marketing MMM on the same national dataset, the same 10 paid channels,
and the same control set as the Phase 1 ridge replication, so any later difference
between the two models' results traces to the Bayesian-vs-frequentist distinction
this project exists to study, not to a different baseline specification.

STATUS: Phase 3 done. `build_model(anchored=True)` (the default) carries per-channel
priors anchored to the ridge project's fitted decays and half-points -- the unit
conversion and the reasoning live in `mmm_bayes.priors`, and docs/PRIORS.md is the
prose version. `anchored=False` reproduces the Phase 2 skeleton exactly (library
defaults, `LogisticSaturation`), which is what the two Phase 2 regression tests still
pin, so the earlier verified result stays reproducible rather than being overwritten.

pymc-marketing's API has moved fast (its own docs note LightweightMMM being deprecated
in favor of it). Checked against the installed version at build time:
    pymc              6.2.0
    pymc-marketing    1.1.0
Re-check `MMM.__init__` and the `GeometricAdstock` / `LogisticSaturation` /
`InverseScaledLogisticSaturation` signatures before reusing this code against a
different installed version.
"""
import sys

import pandas as pd
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling

from mmm_bayes.features import build_controls, standardize_controls
from mmm_bayes.loaders import PAID_CHANNELS, SPEND_COLS, load_national
from mmm_bayes.priors import anchored_transforms

# l_max: 8 weeks of carryover. The ridge project's own coordinate-ascent search found
# decays up to 0.8 for several channels; at 0.8, the adstock weight at lag 8 is
# 0.8**8 ~= 0.168 -- not yet negligible, but pymc-marketing's own examples typically use
# l_max in the 4-8 range and this keeps the model's compute tractable for a skeleton
# run. Revisit if the anchored-prior version (see PRIORS.md) pushes decay estimates
# higher and 8 weeks starts truncating real carryover.
ADSTOCK_L_MAX = 8


def build_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Reshape the national data into the (X, y) format pymc-marketing's MMM.fit
    expects: date and channel/control columns as regular columns, not an index.

    Controls are z-scored (see features.standardize_controls) -- pymc-marketing's
    scaling= only covers target/channel, so this is done by hand here, same reasoning
    as the ridge project's own control standardization in transforms.py.
    """
    df = load_national()
    controls = standardize_controls(build_controls(df))

    X = df[SPEND_COLS].copy()
    X.columns = PAID_CHANNELS  # pymc-marketing wants channel_columns to match X's columns
    X = X.join(controls)
    X["date"] = df.index
    X = X.reset_index(drop=True)

    y = df["sales"].reset_index(drop=True)
    return X, y


def build_model(control_columns: list[str], anchored: bool = True) -> MMM:
    """Construct the (unfitted) MMM. Kept separate from fitting so the model can be
    sanity-checked (prior predictive, graph structure) before spending compute on a
    full MCMC run.

    `anchored=True` (default, Phase 3) uses per-channel priors centred on the ridge
    project's fitted decays and half-points, via `mmm_bayes.priors.anchored_transforms`.
    Note that this also switches the saturation to `InverseScaledLogisticSaturation`:
    the same logistic curve, reparametrised so its `lam` IS the half-saturation point,
    which is what lets the ridge half-point be stated as a prior in its own units
    instead of through a hidden `ln(3)/lam` conversion. See priors.py for the full
    unit-conversion argument.

    `anchored=False` is the Phase 2 skeleton verbatim -- library-default priors and
    plain `LogisticSaturation`. Kept because docs/CHALLENGES.md #2's overflow-warning
    finding and its two regression tests were established against that exact
    specification; folding them into the anchored model would quietly change what
    those tests test.

    `scaling` divides target (sales, ~$45M-$352M) and each channel (spend up to
    ~$158M for `dm`, near-zero for others) by their own max, before any prior touches
    them. Added to fix docs/CHALLENGES.md #2 (RuntimeWarning: overflow encountered in
    dot from PyMC's mass-matrix computation) -- the same five-order-of-magnitude
    problem the ridge project's own transforms.py standardizes against for its ridge
    penalty, now addressed on the Bayesian side via pymc-marketing's built-in scaling
    rather than by hand-rolled standardization.

    NOTE: `Scaling` only covers `target` and `channel`, not `control_columns` --
    pymc-marketing has no built-in control scaling. Controls here (store_count ~600-
    700, consumer_sentiment ~80-95, gas_price ~1.8-4, vs. seasonality/peak indicators
    already in [0, 1] or [-1, 1]) are far closer in scale to each other and to a
    scaled target than raw media spend was, so they're left unscaled for now. Revisit
    if diagnostics in Phase 5 point back to a control-scale problem.
    """
    if anchored:
        adstock, saturation = anchored_transforms(ADSTOCK_L_MAX)
    else:
        adstock, saturation = GeometricAdstock(l_max=ADSTOCK_L_MAX), LogisticSaturation()

    return MMM(
        date_column="date",
        channel_columns=PAID_CHANNELS,
        control_columns=control_columns,
        target_column="y",
        adstock=adstock,
        saturation=saturation,
        scaling=Scaling(
            target=DataDerivedScaling(method="max", dims=()),
            channel=DataDerivedScaling(method="max", dims=()),
        ),
    )


def console_supports_progressbar() -> bool:
    """Whether stdout can encode the characters PyMC's rich progress bar emits.

    On a stock Windows console stdout is cp1252, and rich's progress display uses a
    thin space (U+2009). PyMC writes it on exiting the sampler, so the whole run dies
    with `UnicodeEncodeError: 'charmap' codec can't encode character '\\u2009'` AFTER
    sampling has finished -- the fit is lost at the last moment, with a traceback deep
    in rich that says nothing about encodings being the cause. Hit while running
    `python -m mmm_bayes.model`, the exact command the README gives. See
    docs/CHALLENGES.md #5.

    Detected rather than hardcoded off: the progress bar is genuinely useful on a
    UTF-8 terminal, and `PYTHONIOENCODING=utf-8` makes any terminal one.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        # Written as an escape, not the literal character: an invisible thin
        # space in source is one "tidy up the whitespace" away from becoming an
        # ordinary space, which every encoding accepts, silently disabling this
        # guard.
        "\u2009".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def run_skeleton_fit(draws: int = 500, tune: int = 500, chains: int = 2,
                     target_accept: float = 0.99, random_seed: int = 42,
                     cores: int = 1, anchored: bool = True,
                     progressbar: bool | None = None, init: str = "jitter+adapt_diag",
                     **sample_kwargs) -> tuple[MMM, dict]:
    """Fit with a SHORT chain -- enough to prove the model compiles and samples
    end-to-end, not to produce publication-quality inference. Phase 5 (full fit +
    diagnostics) is where draws/tune go up and R-hat/ESS/divergences get properly
    checked and reported.

    `anchored` is passed straight through to `build_model`; the Phase 2 regression
    tests call this with `anchored=False` to keep testing the specification they were
    written against.

    `progressbar=None` (the default) turns the bar on only when stdout can encode it
    -- see `console_supports_progressbar`.

    Two defaults here were set by Phase 5 measurement rather than inherited -- and the
    second one is the reason the first is NOT what entry #2 of docs/CHALLENGES.md might
    lead you to expect.

    * `target_accept=0.99` rather than 0.9. At real chain length the divergence count
      falls monotonically, 48 -> 11 -> 1 across 0.9 / 0.95 / 0.99, and the posterior
      moves at most 0.07 sd -- so the smaller step is resolving the geometry rather than
      stepping around a problem. docs/CHALLENGES.md #4.
    * `init="jitter+adapt_diag"`, pymc's default, stated explicitly because Phase 5
      tried to change it and the measurement said no. Dropping the jitter does remove
      the `overflow encountered in dot` warning and, on its own at `target_accept=0.9`,
      cuts divergences from 48 to 12. Combined with `target_accept=0.99` it is a
      disaster -- 280 divergences, R-hat 1.11, bulk ESS 25. The jitter is what gives
      four chains different starting points on a posterior with a near-collinear ridge
      (see docs/DIAGNOSTICS.md); without it, and with a tiny step size, they crawl along
      that ridge from the same point and never decorrelate. docs/CHALLENGES.md #2.

    The warning is therefore still emitted, and that is now a choice rather than an
    unsolved problem: its cause is confirmed, and the alternative samples worse.

    Anything else in `sample_kwargs` goes straight to `model.fit`.
    """
    if progressbar is None:
        progressbar = console_supports_progressbar()
    X, y = build_dataframe()
    control_columns = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]

    model = build_model(control_columns, anchored=anchored)
    Xy = X.copy()
    Xy["y"] = y.values

    idata = model.fit(
        X=Xy.drop(columns=["y"]),
        y=Xy["y"],
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        random_seed=random_seed,
        progressbar=progressbar,
        # cores explicit, not left to PyMC's default: on a single-core machine
        # (os.cpu_count() == 1), PyMC's internal setup_cores_blas_cores does
        # blas_cores // cores with cores resolved to 0, raising ZeroDivisionError
        # before sampling starts. See docs/CHALLENGES.md #1.
        cores=cores,
        init=init,
        **sample_kwargs,
    )
    return model, {"idata": idata, "control_columns": control_columns}


if __name__ == "__main__":
    print("=" * 74)
    print("PHASE 3 -- BAYESIAN CORE, ANCHORED PRIORS (short chain)")
    print("=" * 74)
    print(f"Channels: {PAID_CHANNELS}")
    X, y = build_dataframe()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    control_cols = [c for c in X.columns if c not in PAID_CHANNELS + ["date"]]
    print(f"Control columns ({len(control_cols)}): {control_cols}")
    print("\nPriors: per-channel, anchored to the ridge fit. Run "
          "`python -m mmm_bayes.priors` for the anchor table.")
    print("\nRunning a SHORT chain (500 draws, 500 tune, 2 chains) to verify the model "
          "compiles and samples end-to-end. This is not a real fit -- see Phase 5.\n")

    model, result = run_skeleton_fit(anchored=True)
    print("\nSampling completed without error.")
