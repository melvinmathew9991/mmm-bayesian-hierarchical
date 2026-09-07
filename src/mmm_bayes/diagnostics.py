"""Phase 5 -- convergence diagnostics at real chain length.

Phases 2-4 all sampled short chains and said, each time, that inference quality was
Phase 5's job. This is that job: run both models long enough for the diagnostics to mean
something, then say plainly whether they pass.

WHAT COUNTS AS PASSING, DECIDED BEFORE LOOKING
----------------------------------------------
Thresholds are set here rather than after seeing the numbers, because a threshold chosen
once the answer is visible is not a threshold. They are the conventional ones:

* **divergences == 0.** Any divergence means the sampler could not follow the geometry
  somewhere, and the posterior is biased in a direction nothing else reports.
* **R-hat <= 1.01** on every parameter (Vehtari et al. 2021, which is also why 4 chains
  is the minimum -- R-hat compares between- and within-chain variance, and two chains
  give a very noisy estimate of the former).
* **bulk ESS >= 400** per parameter, the usual floor for a stable posterior mean.
* **tail ESS >= 400**, which is what credible intervals actually depend on. Reported
  separately because a model can have healthy bulk ESS and unusable tails.
* **E-BFMI >= 0.3** per chain. Low E-BFMI means the momentum resampling is not
  exploring the energy distribution -- a failure mode that produces no divergences and
  no R-hat warning, and so goes unnoticed unless it is checked for.

Contribution numbers do NOT live here. `contribution_intervals` used to, returning
per-channel totals in max-scaled target units -- numbers that look like percentages and
are 64x off. It was orphaned and it was a trap, so it is gone; `mmm_bayes.attribution`
does the same job on the sales scale, and reconciles against observed sales.

DERIVED QUANTITIES ARE EXCLUDED, DELIBERATELY
---------------------------------------------
`channel_contribution` is a date x geo x channel array -- 17,628 entries on the geo
model. Its R-hat is a deterministic function of the parameters', so it adds no
information, and summarising it costs more wall time than the fit. `summarise` therefore
reports on the model's free parameters and the small derived scalars, and
`DERIVED_VARS` names exactly what is left out so the omission is inspectable rather than
implicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# Pass/fail thresholds, fixed in advance. See the module docstring.
MAX_RHAT = 1.01
MIN_ESS_BULK = 400
MIN_ESS_TAIL = 400
MIN_EBFMI = 0.3

# Real chain length. 4 chains because R-hat needs between-chain variance to be
# estimable; 1000/1000 because that is where ESS clears 400 for a model this size and
# is pymc's own default, not a number tuned until the diagnostics passed.
FULL_CHAINS = 4
FULL_DRAWS = 1000
FULL_TUNE = 1000

# Per-observation derived quantities: excluded from the summary (see module docstring).
DERIVED_VARS = (
    "channel_contribution",
    "fourier_contribution",
    # Added after the fact, and it moved a documented number. This list was written
    # against the national model's variable names; the multidimensional model emits the
    # summed-over-modes seasonality under its own name, which is (date x geo) -- 2,938
    # per-observation entries that were being counted as model parameters. Every "of
    # 4,849 parameter entries" figure in the docs was 61% derived array as a result, and
    # the true denominator is 1,911. Found while building geo_attribution, which needs
    # this term as a contribution component and so had to establish what it was.
    "yearly_seasonality_contribution",
    "control_contribution",
    "intercept_contribution",
    "total_media_contribution_original_scale",
    "mu",
    "y",
)


@dataclass
class DiagnosticReport:
    """Everything Phase 5 needs to say pass or fail, plus the evidence behind it."""

    label: str
    n_chains: int
    n_draws: int
    divergences: int
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    min_ebfmi: float
    n_entries: int
    n_rhat_over: int
    n_ess_bulk_under: int
    n_ess_tail_under: int
    worst: pd.DataFrame

    @property
    def failures(self) -> list[str]:
        """Which thresholds are breached, named. Empty means the fit passes."""
        out = []
        if self.divergences:
            out.append(f"{self.divergences} divergences (require 0)")
        if self.max_rhat > MAX_RHAT:
            out.append(
                f"max R-hat {self.max_rhat:.4f} > {MAX_RHAT} "
                f"({self.n_rhat_over} of {self.n_entries} entries)"
            )
        if self.min_ess_bulk < MIN_ESS_BULK:
            out.append(
                f"min bulk ESS {self.min_ess_bulk:.0f} < {MIN_ESS_BULK} "
                f"({self.n_ess_bulk_under} entries)"
            )
        if self.min_ess_tail < MIN_ESS_TAIL:
            out.append(
                f"min tail ESS {self.min_ess_tail:.0f} < {MIN_ESS_TAIL} "
                f"({self.n_ess_tail_under} entries)"
            )
        if self.min_ebfmi < MIN_EBFMI:
            out.append(f"min E-BFMI {self.min_ebfmi:.3f} < {MIN_EBFMI}")
        return out

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"{self.label}: {verdict}  ({self.n_chains} chains x {self.n_draws} draws, "
            f"{self.n_entries} parameter entries)",
            f"  divergences   {self.divergences}",
            f"  max R-hat     {self.max_rhat:.4f}   ({self.n_rhat_over} over {MAX_RHAT})",
            f"  min ESS bulk  {self.min_ess_bulk:7.0f} ({self.n_ess_bulk_under} under {MIN_ESS_BULK})",
            f"  min ESS tail  {self.min_ess_tail:7.0f} ({self.n_ess_tail_under} under {MIN_ESS_TAIL})",
            f"  min E-BFMI    {self.min_ebfmi:.3f}",
        ]
        for failure in self.failures:
            lines.append(f"  FAILS: {failure}")
        return "\n".join(lines)


def parameter_vars(idata) -> list[str]:
    """Posterior variables worth diagnosing: everything but the per-observation
    derived arrays named in DERIVED_VARS."""
    return [v for v in idata.posterior.data_vars if v not in DERIVED_VARS]


def per_chain_ebfmi(idata) -> np.ndarray:
    """E-BFMI as a plain per-chain array.

    `az.bfmi` in arviz 1.3 returns a DataTree when handed an InferenceData, and
    `np.asarray` on a DataTree raises rather than converting. Passing the energy
    DataArray directly sidesteps that and is what the function documents anyway.
    """
    return np.atleast_1d(np.asarray(az.bfmi(idata.sample_stats["energy"])))


def summarise(idata, label: str, worst_n: int = 8) -> DiagnosticReport:
    """Run every threshold in this module against a fitted model."""
    summary = az.summary(idata, var_names=parameter_vars(idata))
    ebfmi = per_chain_ebfmi(idata)

    return DiagnosticReport(
        label=label,
        n_chains=int(idata.posterior.sizes["chain"]),
        n_draws=int(idata.posterior.sizes["draw"]),
        divergences=int(idata.sample_stats["diverging"].sum()),
        max_rhat=float(summary["r_hat"].max()),
        min_ess_bulk=float(summary["ess_bulk"].min()),
        min_ess_tail=float(summary["ess_tail"].min()),
        min_ebfmi=float(ebfmi.min()),
        n_entries=len(summary),
        n_rhat_over=int((summary["r_hat"] > MAX_RHAT).sum()),
        n_ess_bulk_under=int((summary["ess_bulk"] < MIN_ESS_BULK).sum()),
        n_ess_tail_under=int((summary["ess_tail"] < MIN_ESS_TAIL).sum()),
        worst=summary.sort_values("r_hat", ascending=False)
        .head(worst_n)[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]],
    )


def divergence_locations(idata, var_names: list[str] | None = None,
                         top_n: int = 5) -> pd.DataFrame:
    """Where in parameter space the divergences fell, if any.

    A divergence count says something is wrong; this says where, by comparing the mean
    of each parameter on divergent draws against its mean on the rest. A parameter whose
    divergent draws sit far into one tail is the one whose geometry the sampler could
    not follow -- which is the difference between "raise target_accept and hope" and
    knowing which prior to reparametrise.
    """
    diverging = np.asarray(idata.sample_stats["diverging"]).ravel()
    if not diverging.any():
        return pd.DataFrame(columns=["divergent_mean", "other_mean", "sd", "z_shift"])

    rows = {}
    for name in var_names or parameter_vars(idata):
        values = idata.posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...)
        flat = np.asarray(values).reshape(len(diverging), -1)
        labels = _entry_labels(idata, name, flat.shape[1])
        for column, label in enumerate(labels):
            series = flat[:, column]
            sd = float(series.std())
            if sd == 0:
                continue
            div_mean = float(series[diverging].mean())
            other_mean = float(series[~diverging].mean())
            rows[label] = {
                "divergent_mean": div_mean,
                "other_mean": other_mean,
                "sd": sd,
                "z_shift": (div_mean - other_mean) / sd,
            }

    table = pd.DataFrame(rows).T
    return table.reindex(table["z_shift"].abs().sort_values(ascending=False).index).head(top_n)


def _entry_labels(idata, name: str, n_entries: int) -> list[str]:
    """`saturation_beta[A, Paid_Views]`-style labels for a flattened parameter."""
    if n_entries == 1:
        return [name]
    dims = [d for d in idata.posterior[name].dims if d not in ("chain", "draw")]
    coords = [list(idata.posterior.coords[d].values) for d in dims]
    index = pd.MultiIndex.from_product(coords) if len(coords) > 1 else pd.Index(coords[0])
    return [f"{name}[{', '.join(map(str, k))}]" if isinstance(k, tuple) else f"{name}[{k}]"
            for k in index]


def save_idata(idata, path: Path, drop_derived: bool = False) -> Path:
    """Persist a fit so the analysis below it can be re-run without re-sampling.

    A full geo fit is ~20 minutes on this machine; without caching, every question asked
    of it costs another 20. netCDF via h5netcdf rather than pickle because pickled
    InferenceData does not survive a library upgrade, and these fits are meant to be
    re-readable when Phase 9 compares against them.

    `drop_derived` discards the per-observation deterministics in `DERIVED_VARS` before
    writing. They dominate the file -- the geo fit's `channel_contribution` and
    `control_contribution` are (draws x 113 dates x 26 geos x channels), which is 1.1GB
    of the 1.1GB.

    The justification used to be "nothing reads them back, because `attribution` refuses
    panel fits". That was circular -- the refusal was the reason for the drop, and the
    drop is what would have made lifting the refusal impossible. `mmm_bayes.geo_attribution`
    now does read them, and the drop survives on a better argument: it recomputes them from
    the free RVs with `pm.compute_deterministics`, exactly (verified to 5.6e-17 against the
    national fit, which keeps its copies) and at ~2ms per draw-chain. Cheap to rebuild,
    expensive to store, so they are not stored.

    Left off by default so the national fit, whose contributions ARE the deliverable and
    which is small enough to keep them, does exactly that.
    """
    if drop_derived:
        for name in DERIVED_VARS:
            if name in idata.posterior:
                del idata.posterior[name]

    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling temp file and replace. Two reasons, one of which is a bug this
    # hit: netCDF reads are lazy, so an idata that came from `load_idata(path)` still
    # holds an open handle on `path`, and writing straight back raises "unable to
    # truncate a file which is already open". The other is the usual one -- a crash
    # mid-write leaves the previous good fit intact rather than a truncated file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    idata.to_netcdf(str(tmp), engine="h5netcdf")
    tmp.replace(path)
    return path


def load_idata(path: Path):
    """Read a cached fit, eagerly.

    `az.from_netcdf` is lazy by default, which leaves the file open behind the returned
    object. On Windows that makes the file unwritable: re-saving to the same path fails
    with "unable to truncate a file which is already open", and even an atomic
    temp-file-then-replace fails with `PermissionError`, because the target is still
    held. Both were hit while re-saving the geo fit without its deterministics.

    `.load()` pulls the arrays into memory and `.close()` releases the handle -- both
    are needed, because xarray keeps opened files in a global cache that `.load()` alone
    does not clear. It costs the fit's full size in RAM, which every caller here pays
    anyway since `summarise` touches every parameter, and buys the ability to rewrite a
    cache in place.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No cached fit at {path}. Produce it with:\n"
            "    python scripts/run_full_fits.py"
        )
    idata = az.from_netcdf(str(path)).load()
    idata.close()
    return idata


def energy_summary(idata) -> pd.DataFrame:
    """Per-chain E-BFMI, the diagnostic that fails silently.

    Neither a divergence count nor R-hat notices when NUTS's momentum resampling stops
    exploring the energy distribution; the chains just move too slowly through the tails
    and every other number looks fine. Reported per chain because a single bad chain is
    the usual presentation.
    """
    ebfmi = per_chain_ebfmi(idata)
    energy = idata.sample_stats["energy"]
    return pd.DataFrame(
        {
            "e_bfmi": ebfmi,
            "energy_mean": [float(energy.isel(chain=c).mean()) for c in range(len(ebfmi))],
            "energy_sd": [float(energy.isel(chain=c).std()) for c in range(len(ebfmi))],
        },
        index=pd.Index(range(len(ebfmi)), name="chain"),
    )


def compare_posteriors(reference, variant, var_names: list[str],
                       reference_label: str = "reference",
                       variant_label: str = "variant") -> pd.DataFrame:
    """Posterior means and sds side by side, with the shift in reference-sd units.

    The unit matters. An absolute difference in a parameter's mean says nothing without
    knowing how wide the posterior is; a shift of 0.2 sd is noise, a shift of 2 sd is
    the conclusion changing. Every sensitivity check in Phase 5 is read through this.
    """
    rows = {}
    for name in var_names:
        ref, var = reference.posterior[name], variant.posterior[name]
        ref_mean = ref.mean(("chain", "draw"))
        ref_sd = ref.std(("chain", "draw"))
        var_mean = var.mean(("chain", "draw"))

        labels = _entry_labels(reference, name, int(np.prod(ref_mean.shape)) or 1)
        for label, rm, rs, vm in zip(
            labels,
            np.asarray(ref_mean).ravel(),
            np.asarray(ref_sd).ravel(),
            np.asarray(var_mean).ravel(),
            strict=True,
        ):
            rows[label] = {
                reference_label: float(rm),
                variant_label: float(vm),
                "reference_sd": float(rs),
                "shift_in_sd": float((vm - rm) / rs) if rs else np.nan,
            }
    table = pd.DataFrame(rows).T
    return table.reindex(table["shift_in_sd"].abs().sort_values(ascending=False).index)
