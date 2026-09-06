"""Holiday-week controls for the geo panel.

The national model gets its calendar from `features.build_controls`: two Fourier
harmonics plus explicit dummies for weeks 40-48. The geo panel has no such columns --
it carries a division, a week, six impression series and sales, nothing else -- so the
calendar has to be constructed from the week index here.

Why not just let the Fourier terms handle it. Fitting these two harmonics to the real
panel by OLS reproduces a seasonal curve spanning only -0.15 to +0.25 of max-scaled
sales, while the data's Q4 peak reaches 1.0. Two harmonics physically cannot make a
spike that sharp. Whatever they miss does not vanish -- it gets absorbed by whichever
regressor happens to correlate with Q4, and media spend in a retail business correlates
with Q4 strongly. `features.py`'s own docstring makes this argument for the national
model ("any model omitting them lets media coefficients absorb the calendar"); it
applies with more force here, where the spike is larger and the harmonics are the only
other calendar term.

Week range. Weeks 40-51 is the Oct-Dec retail holiday block, chosen as a calendar fact
rather than fitted -- the same reasoning behind the national model's 40-48. The data
agrees: mean max-scaled sales by ISO week run 0.19 at baseline against 0.30-0.82 across
40-51, peaking in week 48. The upper end is extended past the national model's 48
because this panel's peak clearly runs into weeks 49-51, which the national one does
not.
"""
import pandas as pd

PEAK_WEEKS = range(40, 52)
PEAK_WEEK_COLS = [f"peak_wk{w}" for w in PEAK_WEEKS]


def build_geo_controls(dates: pd.Series) -> pd.DataFrame:
    """One 0/1 column per holiday week, aligned to `dates`.

    Takes the date column of the long-format panel rather than the panel itself, so the
    same dummies apply to every division without any per-division bookkeeping -- a
    holiday week is a holiday week in all 26.
    """
    week_of_year = pd.DatetimeIndex(dates).isocalendar().week.to_numpy()
    controls = pd.DataFrame(
        {f"peak_wk{w}": (week_of_year == w).astype(float) for w in PEAK_WEEKS},
        index=pd.RangeIndex(len(dates)),
    )

    empty = [c for c in controls.columns if controls[c].sum() == 0]
    if empty:
        raise ValueError(
            f"Holiday-week controls {empty} never fire in this date range. A constant "
            "column has no identifiable coefficient and will not be shrunk toward "
            "anything useful -- drop it rather than model it."
        )
    return controls
