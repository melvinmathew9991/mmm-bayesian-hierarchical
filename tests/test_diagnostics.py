"""Tests for the Phase 5 diagnostics module.

Built on synthetic InferenceData rather than real fits, deliberately. A diagnostic that
only ever sees healthy output is untested where it matters -- the question is whether it
*fails* when it should, and a real fit cannot be asked to diverge on demand. These
construct chains with known pathologies and check the report says so.
"""
import numpy as np
import pytest
import xarray as xr
from arviz_base import from_dict

from mmm_bayes.diagnostics import (
    MAX_RHAT,
    MIN_EBFMI,
    MIN_ESS_BULK,
    compare_posteriors,
    divergence_locations,
    energy_summary,
    parameter_vars,
    per_chain_ebfmi,
    summarise,
)


def make_idata(n_chains=4, n_draws=800, seed=0, offset=0.0, divergent_at=None,
               shift_on_divergent=0.0, extra_vars=True):
    """Synthetic fit: well-mixed chains unless asked for otherwise.

    `divergent_at` marks draws as divergent; `shift_on_divergent` pushes `alpha` on
    exactly those draws, which is the pattern `divergence_locations` exists to find.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(offset, 1.0, size=(n_chains, n_draws, 3))
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if divergent_at is not None:
        diverging[:, divergent_at] = True
        alpha[:, divergent_at, 0] += shift_on_divergent

    posterior = {"alpha": alpha, "beta": rng.normal(offset, 1.0, size=(n_chains, n_draws))}
    if extra_vars:
        posterior["channel_contribution"] = rng.normal(
            5.0, 0.1, size=(n_chains, n_draws, 4, 3)
        )

    return from_dict(
        {
            "posterior": posterior,
            "sample_stats": {
                "diverging": diverging,
                "energy": rng.normal(100.0, 5.0, size=(n_chains, n_draws)),
            },
        },
        dims={"alpha": ["channel"], "channel_contribution": ["date", "channel"]},
        coords={"channel": ["a", "b", "c"], "date": [0, 1, 2, 3]},
    )


# ------------------------------------------------------------------ the report


def test_healthy_fit_passes_every_threshold():
    report = summarise(make_idata(), "healthy")
    assert report.passed, report.failures
    assert report.divergences == 0
    assert report.max_rhat <= MAX_RHAT
    assert report.min_ess_bulk >= MIN_ESS_BULK
    assert report.min_ebfmi >= MIN_EBFMI


def test_divergences_alone_fail_the_report():
    """The case the national model actually hit: every other diagnostic clean, 48
    divergences. A report that averaged its criteria would have called that a pass."""
    report = summarise(make_idata(divergent_at=slice(0, 40)), "divergent")
    assert not report.passed
    assert len(report.failures) == 1
    assert "divergences" in report.failures[0]
    assert report.max_rhat <= MAX_RHAT


def test_stuck_chain_is_caught_by_rhat():
    """One chain sampling a different distribution is the thing R-hat exists for."""
    good = make_idata(n_chains=3, seed=1, extra_vars=False)
    bad = make_idata(n_chains=1, seed=2, offset=8.0, extra_vars=False)
    merged = xr.concat([good.posterior.dataset, bad.posterior.dataset], dim="chain")

    combined = from_dict(
        {
            "posterior": {k: merged[k].values for k in merged.data_vars},
            "sample_stats": {
                "diverging": np.zeros((4, merged.sizes["draw"]), dtype=bool),
                "energy": np.random.default_rng(3).normal(100, 5, (4, merged.sizes["draw"])),
            },
        },
        dims={"alpha": ["channel"]},
        coords={"channel": ["a", "b", "c"]},
    )
    report = summarise(combined, "stuck chain")
    assert not report.passed
    assert report.max_rhat > MAX_RHAT
    assert any("R-hat" in f for f in report.failures)


def test_report_str_names_the_verdict_and_every_failure():
    text = str(summarise(make_idata(divergent_at=slice(0, 10)), "labelled"))
    assert text.startswith("labelled: FAIL")
    assert "divergences" in text
    assert str(summarise(make_idata(), "ok")).startswith("ok: PASS")


# ------------------------------------------------------- what gets diagnosed


def test_per_observation_derived_arrays_are_excluded():
    """`channel_contribution` is a deterministic function of the parameters -- 17,628
    entries on the geo model, whose R-hat adds nothing and costs more than the fit."""
    idata = make_idata()
    assert "channel_contribution" in idata.posterior
    assert "channel_contribution" not in parameter_vars(idata)
    assert set(parameter_vars(idata)) == {"alpha", "beta"}


def test_summary_covers_every_entry_of_every_parameter():
    report = summarise(make_idata(), "counted")
    assert report.n_entries == 4  # alpha over 3 channels, plus scalar beta


# --------------------------------------------------------- divergence location


def test_divergence_locations_finds_the_shifted_parameter():
    """The diagnostic that turned '48 divergences' into 'vidtr's decay'. Without it,
    the only available response to a divergence count is to raise target_accept and
    hope."""
    idata = make_idata(divergent_at=slice(0, 60), shift_on_divergent=3.0)
    table = divergence_locations(idata)

    assert table.index[0] == "alpha[a]", f"expected alpha[a] first, got {list(table.index)}"
    assert table.iloc[0]["z_shift"] > 1.0
    assert abs(table.iloc[1]["z_shift"]) < 1.0, "unshifted parameters should rank far below"


def test_divergence_locations_is_empty_when_nothing_diverged():
    assert divergence_locations(make_idata()).empty


# --------------------------------------------------------------- energy / BFMI


def test_per_chain_ebfmi_returns_one_value_per_chain():
    """arviz 1.3's `az.bfmi` hands back a DataTree for an InferenceData, and
    `np.asarray` on that raises rather than converting -- which broke the first full
    fit after it had already sampled."""
    ebfmi = per_chain_ebfmi(make_idata(n_chains=4))
    assert ebfmi.shape == (4,)
    assert np.isfinite(ebfmi).all()


def test_energy_summary_is_per_chain():
    table = energy_summary(make_idata(n_chains=4))
    assert list(table.index) == [0, 1, 2, 3]
    assert {"e_bfmi", "energy_mean", "energy_sd"} <= set(table.columns)


# ------------------------------------------------------------------ comparison


def test_compare_posteriors_measures_shift_in_reference_sds():
    """The unit is the point. A raw difference cannot distinguish noise from a
    conclusion changing; 2.6 sd on vidtr's decay is what made that check decisive."""
    reference = make_idata(seed=10, extra_vars=False)
    variant = make_idata(seed=10, offset=2.0, extra_vars=False)

    table = compare_posteriors(reference, variant, ["alpha"], "ref", "var")
    assert list(table.columns) == ["ref", "var", "reference_sd", "shift_in_sd"]
    assert table["shift_in_sd"].abs().min() > 1.5
    # sorted by absolute shift, largest first
    assert table["shift_in_sd"].abs().is_monotonic_decreasing


def test_compare_posteriors_reports_no_shift_for_an_identical_fit():
    idata = make_idata(seed=11, extra_vars=False)
    table = compare_posteriors(idata, idata, ["alpha"], "a", "b")
    np.testing.assert_allclose(table["shift_in_sd"].to_numpy(), 0.0, atol=1e-12)


def test_entry_labels_name_the_coordinate_not_the_position():
    """`adstock_alpha[vidtr]` rather than `adstock_alpha[5]` -- a positional label is
    one reordering away from pointing at the wrong channel."""
    table = compare_posteriors(make_idata(seed=12, extra_vars=False),
                               make_idata(seed=13, extra_vars=False), ["alpha"])
    assert set(table.index) == {"alpha[a]", "alpha[b]", "alpha[c]"}


def test_thresholds_are_the_conventional_ones():
    """Pinned so a future edit cannot quietly loosen a threshold to make a fit pass."""
    from mmm_bayes import diagnostics

    assert diagnostics.MAX_RHAT == 1.01
    assert diagnostics.MIN_ESS_BULK == 400
    assert diagnostics.MIN_ESS_TAIL == 400
    assert diagnostics.MIN_EBFMI == 0.3
    assert (diagnostics.FULL_CHAINS, diagnostics.FULL_DRAWS, diagnostics.FULL_TUNE) == (
        4, 1000, 1000,
    )


def test_load_idata_says_how_to_produce_a_missing_fit(tmp_path):
    from mmm_bayes.diagnostics import load_idata

    with pytest.raises(FileNotFoundError, match="run_full_fits.py"):
        load_idata(tmp_path / "absent.nc")


def test_save_and_load_round_trip(tmp_path):
    from mmm_bayes.diagnostics import load_idata, save_idata

    original = make_idata(n_chains=2, n_draws=50, extra_vars=False)
    path = save_idata(original, tmp_path / "fits" / "round_trip.nc")
    assert path.exists()

    restored = load_idata(path)
    np.testing.assert_allclose(
        np.asarray(restored.posterior["alpha"]), np.asarray(original.posterior["alpha"])
    )


def test_save_can_overwrite_the_file_it_was_loaded_from(tmp_path):
    """netCDF reads are lazy, so an idata loaded from a path still holds it open.
    Writing straight back raised "unable to truncate a file which is already open" --
    hit while re-saving the geo fit without its deterministics. The write goes via a
    temp file and a replace, which also means a crash mid-write cannot destroy the
    previous good fit."""
    from mmm_bayes.diagnostics import load_idata, save_idata

    path = tmp_path / "round_trip.nc"
    save_idata(make_idata(n_chains=2, n_draws=40), path)

    reloaded = load_idata(path)
    save_idata(reloaded, path, drop_derived=True)

    final = load_idata(path)
    assert "channel_contribution" not in final.posterior
    assert "alpha" in final.posterior
    assert not list(path.parent.glob("*.tmp")), "temp file left behind"
