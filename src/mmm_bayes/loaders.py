"""National data loader.

Ported directly from mmm-marketing-project/src/mmm/loaders.py::load_national,
including its integrity checks -- this repo does not re-verify the data is real
(that forensic work, six checks against Federal Reserve series among them, already
happened in the ridge project; see docs/DATA.md). Re-running it here would be
redundant effort with no new evidence value.

The geo loader (Phase 4 onward) delegates to the ridge project's own
`mmm.loaders.load_geo` rather than reimplementing it. That loader carries the two
data defects documented in docs/DATA.md -- Division Z's duplicated 113-row block, and
the `Overall_Views` column that is the exact sum of two others and drives VIF to 1e5
-- and their fixes. A second implementation here would be a fork that silently drifts
from the one those defects were found against, the same argument
baseline_replication.py makes for importing `run_mmm` instead of copying it.
"""
import pandas as pd

from mmm_bayes.config import GEO_CSV, GEO_CSV_ENV_VAR, NATIONAL_CSV, geo_csv_path

PAID_CHANNELS = ["dm", "inst", "nsp", "auddig", "audtr", "vidtr", "viddig", "so", "on", "sem"]
ORGANIC_CHANNELS = ["em", "sms", "aff"]

SPEND_COLS = [f"mdsp_{c}" for c in PAID_CHANNELS]
IMPRESSION_COLS = [f"mdip_{c}" for c in PAID_CHANNELS]

# --- geo panel ---------------------------------------------------------------
# Re-exported from the ridge project so both repos cannot disagree about which
# columns are modelled. `Overall_Views` is deliberately absent: it is exactly
# Paid_Views + Organic_Views, and including it makes the design matrix rank
# deficient (docs/DATA.md defect #1).
GEO_CHANNELS = [
    "Paid_Views",
    "Organic_Views",
    "Google_Impressions",
    "Email_Impressions",
    "Facebook_Impressions",
    "Affiliate_Impressions",
]
GEO_DIM = "geo"
GEO_TARGET = "Sales"


def load_national() -> pd.DataFrame:
    """Weekly national data indexed by week-start date, sorted ascending."""
    df = pd.read_csv(NATIONAL_CSV, parse_dates=["wk_strt_dt"])
    df = df.sort_values("wk_strt_dt").set_index("wk_strt_dt")
    df.index.name = "week"

    _check(not df.index.duplicated().any(), "national: duplicate weeks")
    _check(df[SPEND_COLS + ["sales"]].notna().all().all(), "national: nulls in spend/sales")
    _check((df[SPEND_COLS] >= 0).all().all(), "national: negative spend")
    _check((df["sales"] > 0).all(), "national: non-positive sales")
    return df


def load_geo() -> pd.DataFrame:
    """Division x week panel: 26 divisions x 113 weeks, 6 impression/view channels.

    Delegates to the ridge project's `mmm.loaders.load_geo`, temporarily pointing its
    module-level `GEO_CSV` at whatever `config.geo_csv_path()` resolves to. That
    indirection exists because the file is not redistributable: this repo reads the
    ridge project's downloaded copy in place rather than keeping a second one that
    could drift out of sync.

    NOTE ON WHAT THIS DATA IS. It is NOT a geographic decomposition of the national
    dataset -- see docs/HIERARCHY.md. Different business, different channels
    (impressions and views, no dollar spend), and not one overlapping week. The
    hierarchical model built on it is a second model, not a geo-aware version of the
    national one.
    """
    from mmm import loaders as ridge_loaders

    path = geo_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No geo dataset found. Looked for ${GEO_CSV_ENV_VAR}, then {GEO_CSV}, "
            "then the ridge project's copy at ../mmm-marketing-project/data/raw/. "
            "It is a non-redistributable Kaggle dataset, so it is committed to neither "
            "repo. Fetch it with the ridge project's `python scripts/fetch_data.py`, "
            f"or point {GEO_CSV_ENV_VAR} at an existing copy. Only the geo model "
            "(Phase 4 onward) needs it; Phases 1-3 run without it."
        )

    # `mmm.loaders` does `from mmm.config import GEO_CSV`, so the name its load_geo
    # reads is the one on the loaders module -- patch and restore that one, not the
    # config module's, or the restore puts the value back in the wrong place.
    original = ridge_loaders.GEO_CSV
    ridge_loaders.GEO_CSV = path
    try:
        return ridge_loaders.load_geo()
    finally:
        ridge_loaders.GEO_CSV = original


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Data integrity check failed -- {message}. See docs/DATA.md.")


if __name__ == "__main__":
    nat = load_national()
    print(f"national: {len(nat)} weeks, {nat.index.min().date()} -> {nat.index.max().date()}")
    print(f"          {len(PAID_CHANNELS)} paid channels with spend, {len(ORGANIC_CHANNELS)} organic")
    print(f"          total spend ${nat[SPEND_COLS].sum().sum():,.0f} = "
          f"{nat[SPEND_COLS].sum().sum() / nat['sales'].sum():.1%} of sales")
    print("Integrity checks passed.")
