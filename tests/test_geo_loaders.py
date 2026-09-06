"""Tests for the geo loader and its file resolution.

The geo CSV is not redistributable, so it lives in exactly one place -- the ridge
project's download -- and this repo reads it there. That indirection is the fragile
part: it depends on a sibling checkout, an environment variable, and monkeypatching a
module-level constant inside another package. Each of those is tested, because a
resolution bug here silently changes which data the Phase 4 model is fit on.
"""
import pandas as pd
import pytest

from mmm_bayes import config
from mmm_bayes.loaders import GEO_CHANNELS, load_geo


def test_geo_csv_resolves_to_a_file_that_exists():
    path = config.geo_csv_path()
    assert path.exists(), (
        f"geo CSV not found at {path}. It is not committed to either repo -- fetch it "
        "with the ridge project's scripts/fetch_data.py, or set "
        f"{config.GEO_CSV_ENV_VAR}."
    )


def test_env_var_overrides_the_default_location(monkeypatch, tmp_path):
    """The escape hatch for anyone whose ridge checkout is somewhere else."""
    elsewhere = tmp_path / "geo_divisions.csv"
    elsewhere.write_text("Division,Calendar_Week\n", encoding="utf-8")
    monkeypatch.setenv(config.GEO_CSV_ENV_VAR, str(elsewhere))
    assert config.geo_csv_path() == elsewhere


def test_missing_env_var_target_falls_through_rather_than_failing(monkeypatch, tmp_path):
    """An env var pointing at nothing should not shadow a copy that does exist -- the
    resolver takes the first candidate that is actually present."""
    monkeypatch.setenv(config.GEO_CSV_ENV_VAR, str(tmp_path / "absent.csv"))
    resolved = config.geo_csv_path()
    assert resolved.exists()
    assert resolved.name == "geo_divisions.csv"


def test_loader_restores_the_ridge_projects_path_afterwards():
    """`load_geo` monkeypatches `mmm.loaders.GEO_CSV` to read a file from outside the
    ridge project. Leaving it patched would silently change what the ridge project's
    own code reads for the rest of the process -- including Phase 1's replication."""
    from mmm import loaders as ridge_loaders

    before = ridge_loaders.GEO_CSV
    load_geo()
    assert ridge_loaders.GEO_CSV == before


def test_loader_restores_the_path_even_when_loading_fails(monkeypatch, tmp_path):
    from mmm import loaders as ridge_loaders

    broken = tmp_path / "geo_divisions.csv"
    broken.write_text("Division,Calendar_Week\nA,not-a-date\n", encoding="utf-8")
    monkeypatch.setenv(config.GEO_CSV_ENV_VAR, str(broken))

    before = ridge_loaders.GEO_CSV
    # A KeyError from the missing Overall_Views column, in practice. Which exception
    # gets us there is not the point -- the point is that the `finally` restores the
    # ridge project's path on any of them.
    with pytest.raises((KeyError, ValueError)):
        load_geo()
    assert ridge_loaders.GEO_CSV == before, (
        "a failed load left the ridge project's GEO_CSV pointing at the broken file"
    )


def test_absent_geo_data_raises_a_message_that_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setenv(config.GEO_CSV_ENV_VAR, str(tmp_path / "absent.csv"))
    monkeypatch.setattr(config, "GEO_CSV", tmp_path / "also-absent.csv")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    import mmm_bayes.loaders as loaders

    monkeypatch.setattr(loaders, "geo_csv_path", lambda: tmp_path / "nowhere.csv")
    monkeypatch.setattr(loaders, "GEO_CSV", tmp_path / "also-absent.csv")

    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load_geo()


def test_geo_integrity_checks_travel_with_the_data():
    """The ridge loader's own assertions -- 113 duplicated Division Z rows dropped,
    balanced panel, no nulls -- are the reason this delegates instead of reimplementing.
    Confirm they actually ran by checking their postconditions."""
    geo = load_geo()
    assert not geo.duplicated(subset=["Division", "week"]).any()
    assert geo.groupby("Division").size().nunique() == 1
    assert geo[GEO_CHANNELS + ["Sales"]].notna().all().all()
    assert pd.api.types.is_datetime64_any_dtype(geo["week"])
