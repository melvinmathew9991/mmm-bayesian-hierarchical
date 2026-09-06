# End-to-end audit, Phases 1–5

An independent pass over everything built so far: claims re-derived from the artefacts
rather than read off earlier summaries, source reviewed for defects, tests checked for
whether they can actually fail, and documents checked against the fits they describe.

This is not Phase 10's self-audit — that one is scoped to the finished project. This is a
mid-flight check before Phase 6 starts building on top of five phases of assumptions.

**Verdict: the documented numbers hold up, and the code has one real defect plus three
gaps.** Every diagnostic figure in `DIAGNOSTICS.md` reproduces exactly. One function
returns confidently wrong answers on an input nothing currently gives it. Four
dependencies were relied on without being declared. Two load-bearing invariants had no
test.

---

## What was verified, and held

| claim | how it was checked | result |
|---|---|---|
| 18 diagnostic figures in `DIAGNOSTICS.md` (both models: divergences, R-hat, ESS, E-BFMI, entry counts) | recomputed from the cached netCDF fits | **all 18 exact** |
| 6 attribution figures (media share, HDI bounds, intercept share, correlation, CV) | recomputed | **all 6 exact** |
| Phase 1: "media spend is 2.2% of revenue" | summed spend / summed sales | 2.174% ✓ |
| Phase 1: "12.4% media-attributed" | ridge contributions from the anchor cache / total sales | 12.446% ✓ |
| `docs/DATA.md`'s MD5 for `national_weekly.csv` | hashed the file on disk | matches, and matches the anchor cache's recorded hash ✓ |
| `PRIORS.md`: "every `lam` anchor lands inside [0.05, 0.56]" | recomputed the anchor table | 0.0511–0.5560 ✓ |
| `features.build_controls` is the ridge project's, unchanged | ran both on the same data, compared frames exactly | **identical**, 209×19 ✓ |
| `divergence_locations` attributes divergences to the right parameter | synthetic fit where only chain 2 diverges and only one parameter is shifted there | correctly ranked first, z 2.10 vs 0.05 for the rest ✓ |
| `channel_shares` (the headline deliverable) | recomputed from first principles; checked it sums to the independently-computed aggregate | matches to 1e-12 ✓ |
| `CHALLENGES.md` numbering | cross-referenced every `#N` citation in code and docs against the entries defined | 1–9 defined, no dangling references ✓ |

The ordering check on `divergence_locations` mattered most. It is the diagnostic that
turned "48 divergences" into "`adstock_alpha[vidtr]`", and it relies on two arrays —
`sample_stats["diverging"]` flattened, and the posterior stacked over `(chain, draw)` —
being in the same order. Had they disagreed, the `vidtr` finding would have been an
artefact. They agree.

---

## Defects found

### 1. HIGH — `attribution` returned confidently wrong numbers on panel fits *(fixed)*

Run `decompose` against the hierarchical geo fit and it reported the **intercept at
3051% of sales**, with the components failing to reconcile against observed sales by a
factor of **34**. No error, no warning.

Two errors compound:

* `n_periods = len(sales)` counts division-weeks (2,938) on a long-format panel, where
  the intercept applies once per **date** (113).
* `scale = sales.max()` is one global maximum, but pymc-marketing scales the target
  **per geo** — `target_scale` has 26 entries differing by more than 10×.

The bug was latent: nothing calls these functions with a panel fit, and the national path
is verified correct. What made it worth fixing rather than noting is that
`_component_totals` *explicitly handled panel dims* (`extra = [d for d in intercept.dims
...]`), so the code looked like it supported panels deliberately. Silent support that is
wrong is worse than no support.

**Fixed** by refusing panel fits with a `NotImplementedError` that names both errors, on
all three entry points, plus a test. Supporting panels properly means threading a
per-geo scale vector and a true period count through the module — a feature, not an audit
fix, and not attempted here.

This is the exact failure mode the project keeps writing about: `CHALLENGES` #9 is a
sensitivity check that would have silently compared a model against itself. That one was
caught by a dry run before it produced a number. This one produced numbers for anyone who
asked, and only an audit found it.

### 2. MEDIUM — four direct dependencies were undeclared *(fixed)*

`pyproject.toml` declared `numpy, pandas, scipy, pymc, pymc-marketing, arviz`. The code
also imports directly:

* `pymc_extras.prior.Prior` — in `priors.py`, `geo_model.py`, `sensitivity_checks.py`
* `xarray` — in `priors.py`, `diagnostics.py`
* `h5netcdf` + `h5py` — required by `save_idata` / `load_idata`

The first two arrived only as transitive dependencies of pymc-marketing; a release that
dropped either would break this project with an `ImportError` and no explanation. The
last two were in `requirements.txt` but not `pyproject.toml`, so `pip install -e .` on
its own produced a project whose Phase 5 caching fails at runtime rather than at install.

**Fixed** — all four declared, with comments recording why.

### 3. MEDIUM — the `features.py` port had no equality test *(fixed)*

`baseline_replication.verify_data_matches` asserts this repo's copy of the *data* is
identical to the ridge project's, and argues at length why that matters. Nothing made the
same assertion about the *controls*, even though every claim of the form "the same
baseline specification, so any difference traces to Bayesian-vs-frequentist" rests on it.

Verified identical during this audit (209×19, exact). **Test added**, so a drift in
either project would fail rather than quietly turn Phase 9 into a comparison of two
different models.

### 4. MEDIUM — the headline deliverable was untested *(fixed)*

`attribution.channel_shares` produces the per-channel contribution table with credible
intervals — the artefact this project exists to produce — and had no test at all. It is
correct: it matches a first-principles recompute to 1e-12 and its shares sum exactly to
the independently-computed aggregate media share. **Test added** covering both
properties. `features.fourier_terms` / `build_controls` / `standardize_controls` were also
untested; the latter two are now covered.

---

## Findings not fixed

### 5. HIGH — this is not a git repository

Five phases, roughly 5,100 lines of source, tests and documentation, and **no version
history**. There is no way to bisect a regression, revert a bad decision, or see when a
number changed. Several times in Phase 5 a default was changed, measured, and reverted;
none of that is recoverable from the working tree.

Two specific risks this creates:

* The sampler-settings reversal in `CHALLENGES` #2 (jitter dropped, then restored once
  the interaction was found) survives only because it was written into a document. Had it
  not been, the working tree would show the final state with no trace of the wrong turn.
* `data/derived/ridge_anchors.json` is a committed-by-intent artefact with a data hash.
  Nothing enforces that it and the code that reads it move together.

Not fixed because initialising a repository and choosing what the first commit contains
is the owner's decision, not an auditor's. It is the single largest risk to the project.

### 6. MEDIUM — 1.3 GB of cached fits

`data/derived/fits/geo_pooled.nc` is **1.11 GB**; `national_anchored.nc` is 196 MB.
`.gitignore` excludes the directory, so this is a disk-footprint issue rather than a
repository one, but it is worth knowing before a third model is cached.

Almost all of the geo file is the per-observation deterministics —
`channel_contribution` and `control_contribution` at (2,000 draws × 113 dates × 26 geos ×
channels). The diagnostics that justify caching the fit use none of it. Saving a slimmed
copy for diagnostics and the full one only when contributions are needed would cut this
by an order of magnitude. Not done because it changes what downstream phases can read
back, which is a design decision.

### 7. LOW — eight tests skip silently on a fresh clone

`tests/test_attribution.py` is guarded by `skipif` on the cached national fit existing.
Without it, eight tests — including the reconciliation invariant and the identification
finding — do not run. pytest does report the skip count, so this is visible to a reader
of the output rather than hidden, but a fresh clone's green `pytest` covers materially
less than it appears to.

### 8. LOW — `scipy` is declared as a runtime dependency and used only in tests

Harmless (it arrives transitively regardless) but the declaration is inaccurate. It
belongs in the `dev` extra.

---

## Judgements checked, and left alone

Not everything questionable is wrong. Three things looked like defects and are not:

* **`test_media_share_is_flagged_when_it_exceeds_what_is_possible` asserts the model is
  broken.** It is a deliberate known-bad-state test: it fails the day the model is fixed,
  forcing a rewrite rather than letting a stale expectation pass. Its docstring says so.
* **Four `assert ... is not None` smoke assertions.** Thin, but they catch a crash in a
  sampling path where a crash is the realistic failure. Legitimate.
* **Both models are recorded as FAIL.** The national model fails on a single divergence
  in 4,000 draws and the geo model on 13 marginal R-hat entries out of 4,849. Both
  thresholds were set before any fit ran and neither has been moved. Reporting a
  near-miss as a failure is the right call and the documents do it.

---

## Open before Phase 6

1. Initialise version control. (finding 5)
2. Decide whether `attribution` should support panel fits, or whether the geo model gets
   its own decomposition module. Phase 6 calibrates the geo model, so this will come up.
3. The `link="log"` item from `HIERARCHY.md` remains untouched — the geo model's Gaussian
   identity link puts 4.9% of its prior predictive on impossible negative sales.
4. The likelihood-noise prior on the national model is still an untouched library
   default; `PRIORS.md` describes it as "not yet reviewed", which is accurate.

## Reproducing this audit

Every check above is a short script over the cached fits and the two repos; the
non-obvious ones are now tests:

```bash
pytest                                   # 96 fast, includes the new guards
pytest -m slow                           # 12, includes the ridge-refit equality check
python scripts/report_contributions.py   # regenerates the verified figures
```
