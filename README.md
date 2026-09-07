# Bayesian Hierarchical MMM (pymc-marketing)

**Status: Phases 1-5 complete. Phase 6 opened; 7-11 not started.** Phase 5 found the
national model's headline quantity is not identified, and the first Phase 6 result is
that the geo model -- the intended calibration target -- is not identified either. The
second Phase 6 result is that the geo model's *per-division* decomposition, which is the
quantity a geo-DiD actually compares against, was anti-correlated with the raw data until
the channel scaling was changed. Neither model's aggregate media share should be quoted
without reading `docs/DIAGNOSTICS.md` first.
See `docs/BUILD_PLAN.md`-equivalent context: this repo is the Bayesian sequel to
[`mmm-marketing-project`](../mmm-marketing-project) -- the ridge-regression MMM
project, checked out at `D:\mmm-marketing-project` on this machine. It was built to
close that project's own stated gap: *"No posterior. Non-negative ridge stands in for
a hierarchical Bayesian MMM, so there are no credible intervals on any contribution."*

Repo: <https://github.com/melvinmathew9991/mmm-bayesian-hierarchical>
Stack: Python 3.13, `pymc` 6.2.0, `pymc-marketing` 1.1.0, `arviz`, `pandas`/`numpy`/`scipy`, pytest

---

## What's actually done vs. planned

| Phase | Status |
|---|---|
| 1. Baseline replication | ✅ Done. Refits the ridge project's exact model inside this repo; reproduces the documented headline numbers (12.4% media-attributed, 2.2% spend share) as a regression test. |
| 2. Bayesian core skeleton | ✅ Done. Compiles and samples end-to-end (verified with both a real ~500-draw run and a fast smoke-test in CI). Still reachable as `build_model(anchored=False)`. |
| 3. Prior specification (anchored) | ✅ Done. Adstock decay and saturation half-point are anchored per channel to the ridge project's fitted values, with the unit conversion pinned by tests. Media coefficient deliberately left unanchored. See `docs/PRIORS.md`. |
| 4. Hierarchical structure (geo pooling) | ✅ Done, **but not what the plan assumed** -- the geo dataset is a different business from the national one, so this is a second model, not a geo-aware version of Phases 1-3. Partial pooling across 26 divisions, with the pooling written explicitly because `dims=("geo",)` does not pool. See `docs/HIERARCHY.md`. |
| 5. Full fit + diagnostics (R-hat, ESS, divergences at real chain length) | ✅ Done, and it is the most important phase so far. Both models sampled at 4 chains. **The national model's media/baseline split is not identified** (posterior correlation -0.997), so its media share is set by the priors, not the data. Sampler settings settled on measured evidence. See `docs/DIAGNOSTICS.md`. |
| 6. Calibration against the ridge project's geo-DiD estimator | 🟡 Opened, and re-scoped twice. The panel cannot calibrate the national model's media share: **both models have the same identification pathology** (national corr −0.997, geo −0.981), so calibrating one against the other transfers a prior. The narrow per-division spread was then found to be caused by the **channel scaling**, not the pooling — and it was inverted, not just flattened. Refitting under `channel_scaling="target-relative"` leaves the aggregate split exactly as unidentified (corr −0.9826) but turns the per-division decomposition from anti-correlated with raw media intensity (−0.558) into tracking it (**+0.815**). That per-division quantity, not the panel-wide share, is what a DiD compares against. See `docs/DIAGNOSTICS.md`. |
| 7. External calibration (stretch) | ⬜ Not started. |
| 8. ROAS + budget optimization under uncertainty | ⬜ Not started. Must run on the **national** model -- the geo panel has impressions, not dollars, so no ROAS exists there. |
| 9. Ridge vs. Bayesian comparison | ⬜ Not started. |
| 10. Self-audit | 🟡 A mid-flight end-to-end audit of Phases 1-5 is done -- `docs/AUDIT.md`. All 24 documented figures reproduce, nine defects and gaps found, all nine fixed. Phase 6 then found the limit of that check: the geo entry count of 4,849 *reproduced exactly* and was still wrong, because the function computing it had a name gap (finding 14 below). **Recomputing a figure verifies the figure, not the definition behind it.** The full-project self-audit this phase names is still to come. |
| 11. Dashboard + docs | ⬜ Not started. |

## Real findings so far (see `docs/CHALLENGES.md` for full detail)

1. `model.fit(cores=None)` raises `ZeroDivisionError` on a single-core machine
   (`os.cpu_count() == 1`) -- a real bug hit during this build, not anticipated in
   advance. Fixed by passing `cores=1` explicitly.
2. The skeleton fit completes with zero divergences but emits a `RuntimeWarning:
   overflow encountered in dot`. **Two fix attempts, both tested and both failed:**
   target/channel scaling (`MMM`'s `scaling=` parameter) and control standardization
   (z-scoring, on top of the first fix) -- neither eliminated the warning, confirmed
   each time by re-running with the warning promoted to a hard error. Divergences
   have stayed at 0 across every attempt, which is the evidence for the working
   conclusion that this is a benign transient during NUTS's initial `jitter+
   adapt_diag` phase rather than a real posterior-geometry problem -- pinned down as
   a regression test (`test_skeleton_fit_has_zero_divergences_despite_overflow_warning`)
   rather than left as an assumption. Still an open item for Phase 5 to either
   properly resolve or formally confirm as benign. It fires under the anchored
   Phase 3 model too, so it is not prior-related.
3. Anchoring the priors moved divergences off zero: 2 at `target_accept=0.9`, 3 at
   0.95, 0 at 0.99, on an otherwise healthy run (max R-hat 1.0065, min bulk ESS 360).
   Left at the current default rather than tuned away, with the evidence recorded, so
   Phase 5's diagnostics pass decides it rather than a knob quietly bumped at Phase 3.
4. `python -m mmm_bayes.model` -- the command in this README -- died with
   `UnicodeEncodeError: '\u2009'` on a stock Windows cp1252 console, *after* sampling
   finished, throwing away the completed fit. PyMC's `rich` progress bar emits a thin
   space the console cannot encode. Fixed by detecting stdout's actual encoding rather
   than by disabling the bar on Windows.
5. **The build plan's Phase 4 is not buildable as written.** The geo dataset shares no
   channel, no dollar spend and not one week with the national dataset -- it is a
   different business. Phase 4 is therefore a second model, and Phase 8 can only run on
   the national one. Recorded as a plan defect rather than quietly re-scoped.
6. `MMM(dims=("geo",))` does **not** pool, despite pymc-marketing's own docstring saying
   panel dims "share hierarchical priors so that information is partially pooled". The
   defaults give 936 independent parameters over 2,938 observations. A model built on
   that trust would look hierarchical and borrow nothing. Pooling is now written
   explicitly, and the library's real default behaviour is pinned by a test.
7. Pooling paid off in an interpretable way: the fitted across-division scale for
   seasonality is 0.0020 against 0.1815 for media response. These divisions share a
   calendar almost exactly but not a media response -- a distinction a uniformly pooled
   or uniformly unpooled model would have erased.
8. The 845-parameter hierarchical model samples with **0 divergences**, where the
   50-parameter national model produced 1 on its best configuration. Non-centred
   parametrisation, chosen in advance for exactly that reason, is what did it.
9. **The national model's media/baseline split is not identified.** Posterior
   correlation between total media contribution and baseline is -0.997, and their sum is
   determined an order of magnitude more precisely than either part. The data pins down
   total sales -- trivially -- and says almost nothing about how much of it media caused.
   The model attributes 108% of sales to media against ridge's 12.4%, and no prior
   setting tested comes near closing that. This is a specification problem: it needs
   information from outside the time series, which is what Phase 6 is for.
10. **Divergences scale with chain length.** 2 at 2 chains x 500 draws, 48 at 4 x 1000,
   same model. A short run does not estimate a divergence rate imprecisely -- it makes a
   real problem look like rounding error.
11. **Two sampler improvements, each measured in isolation, are a disaster combined.**
   `init="adapt_diag"` alone: 48 divergences -> 12. `target_accept=0.99` alone: 48 -> 1.
   Both together: 280, with bulk ESS of 25. Sampler settings are not additive, and this
   was caught only because the combination was re-run before being believed.
12. The overflow `RuntimeWarning` open since Phase 2 **is** the jitter phase, confirmed
   by removing the jitter and watching it vanish. The jitter is kept anyway, because
   finding #11 shows it earns its place -- so the warning stays as a documented choice
   rather than an unsolved problem.
13. **The geo model is not identified either, and that undermines Phase 6's premise.**
   Asked of the panel for the first time -- it could not be asked before the panel had a
   decomposition -- media comes to 54.6% of sales [49.4%, 59.3%] with a media/baseline
   correlation of -0.981 and a sum determined 9.3x more sharply than either part. The
   plan was to calibrate the national model's unidentified split against this panel;
   both splits are set by their priors, so that calibration would transfer a prior, not
   evidence. What survives is the DiD estimator itself, which is placebo-validated and
   has a real MDE.
14. **A documented figure was 61% derived array.** Every "of 4,849 parameter entries"
   count in this project was counting `yearly_seasonality_contribution` -- 2,938
   per-observation values -- as model parameters, because `diagnostics.DERIVED_VARS` was
   written against the national model's variable names and the multidimensional model
   emits that term under a different one. True count 1,911. The conclusion holds, the
   rate does not: 12 of 1,911 over R-hat 1.01 is 0.63%, not 0.27%. It also broke a
   second claim -- the failures were described as "all non-centred offsets" and one is
   `y_sigma`. Found sideways, by an unrelated module needing to be precise about the
   same variable.

15. **The cross-division variation Phase 6 needs was removed by the channel scaling,
   not by the pooling — and it was inverted, not just flattened.** The narrow 48.7-59.0%
   media share across divisions was first blamed on the pooling prior. Measured, the
   pooling is innocent: `saturation_beta`'s across-geo scale sits at prior-CDF 0.07-0.46,
   inside a prior that left room the data declined to use. The cause is
   `DataDerivedScaling(method="max")` on the channels, which divides out each division's
   media-to-sales ratio — the exact contrast a DiD exploits. Raw intensity spans 11.17x
   across divisions and is near-orthogonal to division size (corr 0.155); the model sees
   1.79x, correlated **−0.457** with the truth. Divisions C and N run ~50k Google
   impressions against B's 459M, and per-channel scaling inflates all three to a scaled
   maximum of exactly 1.0. Fixed by
   `build_geo_model(channel_scaling="target-relative")`, which preserves the contrast
   (span 7.00x, corr +0.865) while leaving the prior calibration unchanged. **Refitted,
   it does not identify the media/baseline split** -- corr goes -0.9807 to -0.9826, and it
   was never going to, since that degeneracy is within-series and shared by all 26
   divisions. What it fixes is the per-division decomposition, the quantity a DiD actually
   compares against: correlation with raw media intensity goes from **-0.558 to +0.815**.
   The baseline model had given divisions C and N -- which run ~50k Google impressions
   against B's 459M -- the two *highest* media shares of all 26, at 58.2% and 59.0%; they
   are now 10.9% and 42.5%.

16. **A fix was documented, verified, and not actually connected.** `run_full_fits.run_one`
   unpacks `drop_derived` from its `FITS` table and never passes it to `save_idata`, so
   every cache the script regenerates keeps the per-observation deterministics that
   `docs/AUDIT.md` #8 records as removed. The regenerated geo fit came back at 1,113 MB --
   the exact pre-fix figure. Ruff does not flag an unused tuple-unpacking target and no
   test regenerates a cache, so nothing caught it. The audit had verified the 92.9%
   saving on the artefact rather than by running the script that is supposed to produce
   it, which is the same lesson finding #14 records: checking a figure is not checking the
   path that produces it. Fixed; the re-saved cache is 1,113.5 MB -> 31.9 MB with the
   diagnostic report identical.

## Setup

```bash
# 0. Create the environment (Python 3.13; a 3.13 venv at .venv is what this was
#    last verified against)
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip

# 1. Install the ridge project first (sibling dependency, not on PyPI).
#    Adjust the path to wherever mmm-marketing-project is checked out.
pip install -e ../mmm-marketing-project

# 2. Install this project, pinned versions first
pip install -r requirements.txt
pip install -e . --no-deps

# 3. Run Phase 1 (baseline replication)
python -m mmm_bayes.baseline_replication

# 4. Generate the Phase 3 prior anchors (runs the ridge fit once, ~90s, and caches
#    its decays/half-points to data/derived/ridge_anchors.json)
python scripts/build_ridge_anchors.py

# 5. Inspect the anchored priors: ridge input, unit conversion, prior parameters
python -m mmm_bayes.priors

# 6. Run the national Bayesian model (~500 draws, a few minutes on 1 core)
python -m mmm_bayes.model

# 7. Inspect the hierarchical geo model: pooled vs. unpooled parameter counts and the
#    prior predictive check. Add --fit to sample it (~5 minutes).
python -m mmm_bayes.geo_model

# 8. Fit both models at real chain length and cache them (~5 min + ~21 min)
python scripts/run_full_fits.py national
python scripts/run_full_fits.py geo --draws 500

# 9. The deliverable: per-channel contributions with credible intervals
python scripts/report_contributions.py

# 9b. The same for the geo panel -- per-division, and with the identification report
#     that shows why the panel-wide number should not be read on its own
python -m mmm_bayes.geo_attribution

# 10. Any of the Phase 5 open-item investigations
python scripts/sensitivity_checks.py intercept   # also: beta-scale, vidtr, init,
                                                 # accept, trend, decay-pooling

# 11. Run the fast test suite (excludes MCMC and ridge-refit tests by default)
pytest

# 12. Run the slow tests too (tiny MCMC chains plus a ridge refit)
pytest -m slow
```

## Data

`data/raw/national_weekly.csv` is copied directly from the ridge project's own
forensically-verified real dataset (see `docs/DATA.md`, inherited unchanged). This repo
does not re-verify the data is real -- that work already happened once and re-doing it
here would be redundant.

**The geo dataset is a different business, not a geographic breakdown of the national
one.** No shared channel, no dollar spend, no overlapping week. `docs/HIERARCHY.md` has
the table; this is the single most important thing to know before reading Phase 4.

The geo dataset (`data/raw/geo_divisions.csv`) is **not present in this repo** -- it's a
non-redistributable Kaggle dataset, same restriction as in the ridge project, so it is
deliberately not copied across. It's required starting at Phase 4 (hierarchical geo
pooling) and Phase 6 (calibration).

It is, however, already downloaded locally in the ridge project, and
`config.geo_csv_path()` reads it there rather than keeping a second copy: it tries
`$MMM_BAYES_GEO_CSV`, then `data/raw/geo_divisions.csv` here, then
`../mmm-marketing-project/data/raw/geo_divisions.csv`. One copy, read from where it was
downloaded, has no way to drift out of sync with itself. On a machine without it, fetch
it via the ridge project's own `scripts/fetch_data.py` first.

## Honest gaps, right now

- Decay and half-point priors are anchored; the intercept and likelihood-noise priors
  are still untouched library defaults, described in `docs/PRIORS.md` as "not yet
  reviewed" rather than as chosen.
- The Bayesian model's saturation is logistic, the ridge model's is Michaelis-Menten.
  The anchoring makes them agree at the half-point by construction, not away from it.
  Phase 9 must not read a ridge/Bayes contribution gap as an inference-method effect
  without accounting for that.
- Decay posteriors barely move off their anchors at short chain length. That is the
  weak identification the anchoring exists to handle -- it is not the data confirming
  the ridge estimates, and `docs/PRIORS.md` says so explicitly.
- The numerical-stability warning (finding #2 above) survived two independent fix
  attempts and is left open, with reasoning for treating it as likely benign, not a
  silent success claim.
- The geo model's Gaussian identity link puts 4.9% of its prior predictive on negative
  sales. That is structural, not a tuning failure; `link="log"` is the fix and is a
  Phase 5 decision because it changes the model's functional form.
- The geo model's `Google_Impressions` decay of 0.979 was suspected of absorbing an
  unmodelled trend. Phase 5 tested that and it is not: adding a trend control pushed the
  decay *up*. It is the same near-collinearity that afflicts the national model.
- **The national model's contribution numbers are not usable, and the repo says so where
  they are printed.** `report_contributions.py` prints a health warning above them rather
  than presenting a broken decomposition quietly. The intervals are real; the location of
  the media/baseline split inside them is not.
- The geo model clears every threshold except R-hat, and does so marginally (12 of 1,911
  entries, worst 1.0166, eleven non-centred offsets and one `y_sigma`) at 4 chains x 500
  draws. It has not been run long enough to clear R-hat on a machine that can hold a
  1000-draw fit.
- The likelihood-noise prior is still an untouched library default. The intercept prior
  was the other one, and reviewing it in Phase 5 found a real defect.
- Version control starts at the Phase 5 audit. Everything before it exists only as prose
  in `docs/CHALLENGES.md` -- no commit can be bisected or dated earlier than that.
- The geo model's contribution numbers carry the same warning the national model's do:
  media at 54.6% of panel sales is a symptom, not a finding. `geo_attribution` reports it
  with the identification report next to it, the same way `report_contributions.py` does.
- Phase 6 has a prerequisite built and a premise in doubt. Phases 7 onward are unbuilt.
  This is a working core with five real, verified phases -- not a finished project.
