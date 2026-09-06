"""Project paths.

Resolved relative to this file rather than the working directory -- same pattern as
the ridge project (mmm-marketing-project/src/mmm/config.py), for the same
reason: `python -m mmm_bayes.baseline_replication` must behave identically no matter
where it's invoked from.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"

NATIONAL_CSV = RAW_DATA_DIR / "national_weekly.csv"

# Geo data is NOT copied into this repo: it's a non-redistributable Kaggle dataset
# (see docs/DATA.md, inherited from the ridge project). Phase 4 (hierarchical geo
# pooling) and Phase 6 (calibration against the geo-DiD estimator) both need it.
GEO_CSV = RAW_DATA_DIR / "geo_divisions.csv"

# Where the geo CSV is looked for, in order. The file is not redistributable, so this
# repo neither commits it nor copies it out of the ridge project -- it reads the ridge
# project's own copy in place. Two datasets on disk that must stay identical is a
# synchronisation problem nobody remembers they have; one copy, read from where it was
# downloaded, has no such failure mode.
GEO_CSV_ENV_VAR = "MMM_BAYES_GEO_CSV"


def geo_csv_path() -> Path:
    """Resolve the geo CSV: env var override, then this repo, then the ridge project.

    Returns the first candidate that exists. If none does, returns the local path so
    callers can raise an error naming the place a user would most expect it.
    """
    import os

    override = os.environ.get(GEO_CSV_ENV_VAR)
    candidates = [Path(override)] if override else []
    candidates.append(GEO_CSV)
    candidates.append(PROJECT_ROOT.parent / "mmm-marketing-project" / "data" / "raw" / "geo_divisions.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return GEO_CSV

# Phase 3 prior anchoring: the ridge project's fitted per-channel decays and
# half-points, cached so building the Bayesian model does not re-run a ~90s ridge fit
# every time. Regenerate with `python scripts/build_ridge_anchors.py`; the file records
# an MD5 of the national CSV so a stale cache is detected rather than silently used.
RIDGE_ANCHORS_JSON = DERIVED_DATA_DIR / "ridge_anchors.json"

# Phase 5 cached full fits. A 4-chain real-length geo fit is ~20 minutes on this
# machine; caching the InferenceData means every question asked of it afterwards --
# diagnostics, contribution intervals, Phase 9's comparison -- costs a read rather than
# another sampling run. Regenerate with `python scripts/run_full_fits.py`.
FULL_FIT_DIR = DERIVED_DATA_DIR / "fits"
NATIONAL_FIT_NC = FULL_FIT_DIR / "national_anchored.nc"
GEO_FIT_NC = FULL_FIT_DIR / "geo_pooled.nc"
