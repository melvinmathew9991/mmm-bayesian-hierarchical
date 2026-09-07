# End-to-end audit, Phases 1–5

An independent pass over everything built so far: claims re-derived from the artefacts
rather than read off earlier summaries, source reviewed for defects, tests checked for
whether they can actually fail, and documents checked against the fits they describe.

This is not Phase 10's self-audit — that one is scoped to the finished project. This is a
mid-flight check before Phase 6 starts building on top of five phases of assumptions.

**Verdict: the documented numbers hold up. The code had nine defects and gaps; all nine
are fixed.** Every diagnostic figure in `DIAGNOSTICS.md` reproduces exactly, as do Phase
1's headline claims and the data-integrity chain. Against that, two functions returned
confidently wrong numbers on inputs nothing currently gave them, four dependencies were
relied on without being declared, two load-bearing invariants had no test, a cached fit
could not be rewritten in place, and the project had no version history.

Nothing found invalidates a conclusion in `DIAGNOSTICS.md`, `HIERARCHY.md` or
`PRIORS.md`. The defects were in code paths the findings did not travel through — which
is its own lesson about where bugs accumulate.

---

## What was verified, and held

| claim | how it was checked | result |
|---|---|---|
| 18 diagnostic figures in `DIAGNOSTICS.md` (both models: divergences, R-hat, ESS, E-BFMI, entry counts) | recomputed from the cached netCDF fits | **all 18 exact** — but see the note below on the entry counts |
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

### 5. HIGH — `diagnostics.contribution_intervals` was an orphaned trap *(fixed: removed)*

Never called by anything but its own tests, and it returned per-channel totals in
**max-scaled target units** — the exact error the sibling module's docstring warns about.
For `dm` it returned `11.094` where the true share of sales is `17.27%`: off by 64.2x,
and shaped like a percentage. Two tests exercised it as though it were a deliverable.

Removed rather than fixed, because `attribution.channel_shares` already does the job on
the sales scale and reconciles against observed sales. A second, wrong way to compute the
project's headline number is worse than no second way.

### 6. MEDIUM — rewriting a cached fit in place was impossible *(fixed)*

`save_idata(load_idata(p), p)` failed. `az.from_netcdf` is lazy, so the returned object
holds the file open; the write raised *"unable to truncate a file which is already
open"*, and an atomic temp-file-then-replace then failed with `PermissionError` because
Windows will not replace a held file either. Found while trying to slim the geo cache.

Fixed on both sides: `load_idata` now calls `.load()` **and** `.close()` — both are
needed, since xarray keeps opened files in a global cache that `.load()` alone does not
clear — and `save_idata` writes via a temp file and replaces, so a crash mid-write cannot
destroy the previous good fit.

### 7. LOW — `run_geo_fit` mislabelled caller-supplied models *(fixed)*

The result dict reported `pooled=True` regardless of whether the caller passed its own
model. The decay-pooling sensitivity check passes one that pools adstock *differently*,
so the label was attaching this function's description to someone else's model. Now
reports `None` when a model is supplied, with a test.

### 8. MEDIUM — the geo cache was 1.11 GB *(fixed: 79 MB)*

Almost all of it was per-observation deterministics that nothing reads back: the
diagnostics use parameters only, and `attribution` refuses panel fits outright.
`save_idata` gained a `drop_derived` flag, and `run_full_fits` sets it for the geo model
and not for the national one, whose contributions are the deliverable. **1,113 MB → 79 MB,
92.9% smaller, with the diagnostic report bit-identical before and after.** Total cache
footprint is now 275 MB rather than 1.3 GB.

### 9. HIGH — this was not a git repository *(fixed)*

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

**Fixed.** Initialised, with a `.gitattributes` pinning LF endings so the tree does not
show as wholly modified on a machine with different `autocrlf` settings, and one initial
commit of 41 files / 0.37 MB. Verified before committing that no cached fit, no virtual
environment and no non-redistributable geo data is tracked.

The history before this point is unrecoverable and exists only as prose in
`docs/CHALLENGES.md`. That is the cost of the finding, and it is already paid.

## Findings accepted, not fixed

### 10. LOW — eight tests skip silently on a fresh clone

`tests/test_attribution.py` is guarded by `skipif` on the cached national fit existing.
Without it, eight tests — including the reconciliation invariant and the identification
finding — do not run. pytest does report the skip count, so this is visible to a reader
of the output rather than hidden, but a fresh clone's green `pytest` covers materially
less than it appears to.

### 11. LOW — `scipy` is declared as a runtime dependency and used only in tests

`src/` never imports it; the tests do. Left declared: pymc requires scipy regardless, so
the declaration is redundant rather than wrong, and moving it to the `dev` extra would
make a fresh `pip install -e .` depend on a transitive edge staying put — the exact
fragility finding 2 was about.

---

## The limit of "recomputed, and it matched"

Phase 6 found one, and it is worth stating because it applies to every row above.

The geo model's entry count of **4,849** was recomputed from the cached fit and matched
exactly. It was still wrong. `diagnostics.parameter_vars` excludes the per-observation
arrays named in `DERIVED_VARS`, that list was written against the national model's
variable names, and the multidimensional model emits its seasonality term under a name
the list did not have — so 2,938 per-observation values were counted as parameters. The
true figure is 1,911, and the R-hat failure rate the documents reported was 6x too
optimistic (see `CHALLENGES.md` #12).

**Recomputing a figure verifies the figure against the code, not the code against
reality.** Every check in the table above ran the project's own functions over the
project's own artefacts, so any of them would reproduce a definitional error just as
faithfully. The checks that did *not* have this weakness are the ones with an independent
reference — Phase 1's 2.2% against summed raw columns, the CSV's MD5 against the file on
disk, `features.build_controls` against the ridge project's own implementation. That is
the pattern to prefer in the full-project audit Phase 10 still owes.

---

## Judgements checked, and left alone

Not everything questionable is wrong. Three things looked like defects and are not:

* **`test_media_share_is_flagged_when_it_exceeds_what_is_possible` asserts the model is
  broken.** It is a deliberate known-bad-state test: it fails the day the model is fixed,
  forcing a rewrite rather than letting a stale expectation pass. Its docstring says so.
* **Four `assert ... is not None` smoke assertions.** Thin, but they catch a crash in a
  sampling path where a crash is the realistic failure. Legitimate.
* **Both models are recorded as FAIL.** The national model fails on a single divergence
  in 4,000 draws and the geo model on 12 marginal R-hat entries out of 1,911. Both
  thresholds were set before any fit ran and neither has been moved. Reporting a
  near-miss as a failure is the right call and the documents do it.

---

## Open before Phase 6

1. ~~Decide whether `attribution` should support panel fits, or whether the geo model
   gets its own decomposition module.~~ **Decided and built: `mmm_bayes.geo_attribution`.**

   Neither of the two options as posed. The split is by responsibility rather than by
   model: *obtaining* per-draw component totals is what genuinely differs between the two
   fits, and everything done to those totals afterwards — shares, intervals, the
   identification report — does not differ at all, so `attribution._shares_table`,
   `._interval` and `._identification_table` are shared and the panel module supplies
   only the totals. `attribution` still refuses panel fits, now with a pointer instead of
   just a refusal.

   The question had a hidden prerequisite that only appeared on inspection: **the geo
   fit does not contain the contributions to decompose.** `save_idata(drop_derived=True)`
   discards them, and its stated reason was that nothing reads them back *because
   attribution refuses panels* — circular, and it would have made any answer to this item
   unimplementable against the cached fit. They are recomputed instead, which is both
   cheaper and exact; see `docs/CHALLENGES.md` #11.
2. The `link="log"` item from `HIERARCHY.md` remains untouched — the geo model's Gaussian
   identity link puts 4.9% of its prior predictive on impossible negative sales.
3. The likelihood-noise prior on the national model is still an untouched library
   default; `PRIORS.md` describes it as "not yet reviewed", which is accurate.

## Reproducing this audit

Every check above is a short script over the cached fits and the two repos; the
non-obvious ones are now tests:

```bash
pytest                                   # 106 fast (95 at audit time, + 11 for the
                                         # panel decomposition built for Phase 6)
pytest -m slow                           # 13, includes the ridge-refit equality check
python scripts/report_contributions.py   # regenerates the verified figures
```
