"""Baseline control features. Ported from the ridge project's mmm.features.

Kept unchanged deliberately: the ridge project's data profiling found controls alone
explain R^2 = 0.726 of weekly sales while all ten media channels alone explain only
0.575, and that any model omitting them lets media coefficients absorb the calendar.
The Bayesian model needs to be checked against the exact same baseline specification
the ridge model used, or a difference in results could come from a different baseline
rather than from the Bayesian vs. frequentist distinction this project exists to study.
"""
import numpy as np
import pandas as pd

PEAK_WEEK_COLS = [f"seas_week_{w}" for w in range(40, 49)]
MACRO_COLS = ["me_ics_all", "me_gas_dpg"]
MARKDOWN_COLS = ["mrkdn_valadd_edw", "mrkdn_pdm"]


def fourier_terms(week_of_year: np.ndarray, n_harmonics: int = 2, period: float = 52.0) -> pd.DataFrame:
    out = {}
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * week_of_year / period
        out[f"sin_{k}"] = np.sin(angle)
        out[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(out)


def build_controls(df: pd.DataFrame, n_harmonics: int = 2) -> pd.DataFrame:
    n = len(df)
    woy = df.index.isocalendar().week.values.astype(float)

    parts = [
        pd.DataFrame({"trend": np.arange(n) / n}, index=df.index),
        pd.DataFrame({"store_count": df["st_ct"].values}, index=df.index),
        df[MACRO_COLS].set_axis(["consumer_sentiment", "gas_price"], axis=1),
        df[MARKDOWN_COLS].set_axis(["markdown_valadd", "markdown_pdm"], axis=1),
        fourier_terms(woy, n_harmonics).set_index(df.index),
        df[PEAK_WEEK_COLS].set_axis([f"peak_wk{w}" for w in range(40, 49)], axis=1),
    ]
    controls = pd.concat(parts, axis=1)
    return controls.loc[:, controls.std() > 0]


def standardize_controls(controls: pd.DataFrame) -> pd.DataFrame:
    """Z-score every control column: mean 0, std 1.

    Same reasoning as the ridge project's fit_constrained_ridge, applied here for the
    Bayesian model instead of a ridge penalty: pymc-marketing's `Scaling` only covers
    `target` and `channel`, not `control_columns`, so without this, gamma_control's
    prior is mismatched in scale across columns like store_count (~600-700) vs.
    seasonality terms already in [-1, 1] -- see docs/CHALLENGES.md #2.

    Assumes zero-variance columns are already dropped (build_controls does this).
    """
    return (controls - controls.mean()) / controls.std()
