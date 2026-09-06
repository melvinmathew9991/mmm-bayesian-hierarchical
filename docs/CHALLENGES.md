# Challenges faced -- and how each was resolved

Consistent with the rest of the portfolio: documented as found, not cleaned up after the
fact. Every entry below came out of actually running Phases 1-5, not anticipated in
advance.

---

#: 1
**Challenge:** `model.fit(..., cores=None)` (the default) raised
`ZeroDivisionError: integer division or modulo by zero` inside PyMC's
`setup_cores_blas_cores`, before any sampling started.

**Root cause:** the build environment has `os.cpu_count() == 1`. PyMC's internal core-count
resolution logic computes `blas_cores // cores` where `cores` resolves to `0` on a
single-core machine under PyMC 6.2.0's current logic, producing a division by zero.

**Resolution:** pass `cores=1` explicitly to `model.fit(...)` rather than relying on the
default. Documented inline in `mmm_bayes/model.py::run_skeleton_fit`. Worth re-testing
this once the project moves to a multi-core machine for the full Phase 5 fit -- the
explicit `cores=1` should be replaced with an explicit, deliberate core count rather than
left as a leftover workaround.

---

#: 2
**Challenge:** sampling completed with zero divergences on both chains, but emitted
`RuntimeWarning: overflow encountered in dot` from PyMC's mass-matrix computation
(`quadpotential.py`) during every step.

**Root cause, not yet fully confirmed:** the skeleton model was built without setting
`MMM`'s `scaling` parameter (defaults to `None`, i.e. no scaling applied). Media spend in
this dataset ranges up to ~$158M (channel `dm`, see Phase 1 output) while other predictors
(Fourier terms, standardized controls) sit near unit scale -- the same five-order-of-
magnitude spread the ridge project's own `transforms.py` docstring explicitly calls out as
the reason it standardizes features before penalization: *"Raw media spend and store
counts differ by five orders of magnitude; without scaling, a single alpha penalises them
wildly unequally."* The same problem likely applies here to the sampler's mass matrix, not
just to a ridge penalty.

**Resolution attempt #1 (target/channel scaling via `MMM`'s `scaling=` parameter):
tried, did NOT fix it.** Added `Scaling(target=DataDerivedScaling(method="max"),
channel=DataDerivedScaling(method="max"))` to `build_model` -- this scales sales and
each channel's spend by their own max before any prior touches them. Re-ran with the
warning promoted to a hard error (`-W error::RuntimeWarning`) to confirm definitively:
the warning still fires, from the same line, during NUTS's `velocity_energy`
computation. **The scaling fix did not resolve the actual problem** -- recorded here
rather than left implied by moving on, since claiming a fix worked without checking is
exactly the failure mode this project's audit discipline exists to catch.

**Revised hypothesis:** `MMM`'s built-in `scaling=` only covers `target` and
`channel` (confirmed from the `Scaling` docstring) -- it does not touch
`control_columns`. The controls are still unscaled: `store_count` ranges ~599-720,
`consumer_sentiment` ~82-97, `gas_price` ~1.8-4.3, next to seasonality/peak terms
already in [-1, 1] or {0, 1}. That's a much smaller spread than the original
media-spend problem, but `gamma_control`'s prior is still likely mismatched in scale
across those columns, which could be driving the mass-matrix overflow during early
NUTS step-size adaptation.

**Next step, not yet done:** standardize control columns (z-score, mean 0 / std 1)
before passing them to `MMM`, the same way the ridge project's own
`transforms.py::fit_constrained_ridge` already standardizes controls for its ridge
penalty. This is a different fix from attempt #1, targeting a different part of the
design matrix -- worth testing on its own rather than assumed to follow automatically
from the channel/target fix.

**Resolution attempt #2 (standardize control columns, z-score): tried, ALSO did NOT
fix it.** Added `features.standardize_controls` (mean 0, std 1 per column) and wired
it into `build_dataframe`, on top of attempt #1's target/channel scaling. Re-ran with
`-W error::RuntimeWarning` again: the warning still fires, same line, same traceback
shape.

**Where this leaves it, honestly:** two different, reasoned fixes have not
eliminated the warning. What has NOT changed across all three attempts (unscaled,
then + target/channel scaling, then + control standardization) is that **divergences
stay at 0 and the model completes successfully every time** -- only the warning
persists. That pattern points toward a different explanation than data scale: the
warning fires during NUTS's initial `jitter+adapt_diag` exploratory phase, before the
mass matrix has adapted, which is a known point where PyMC's momentum computation can
transiently overflow on a model with this many parameters (30 controls/channels plus
adstock/saturation hyperparameters) without indicating a real posterior-geometry
problem -- the adaptation phase exists specifically to correct for exactly this kind
of early instability.

**RESOLVED IN PHASE 5. The jitter hypothesis was right.** `init="adapt_diag"` removes
the jitter from NUTS's initialisation and changes nothing else. Run against
`jitter+adapt_diag` at 4 chains x 1000 draws, same seed, same data:

| `init` | overflow warnings | divergences | max R-hat | min bulk ESS |
|---|---|---|---|---|
| `jitter+adapt_diag` (default) | **2** | 48 | 1.0054 | 1688 |
| `adapt_diag` | **0** | **12** | 1.0058 | 1409 |

The warning disappears completely. The posteriors do not move -- the largest shift
across every adstock and saturation parameter is 0.06 posterior standard deviations,
which is noise. So the working conclusion recorded below was correct: the warning was a
transient of the jitter phase, not a data-scale problem, which is exactly why two
reasoned scaling fixes failed to touch it. Scaling was never the cause.

**Dropping the jitter also looked like an unbudgeted win -- 48 divergences down to 12,
at no cost to R-hat, ESS or the posterior. It was made the default, and that was a
mistake.** Entry #4 was independently raising `target_accept` from 0.9 to 0.99, also
measured, also an improvement. Both were verified in isolation; neither was verified in
combination. Together they are far worse than either:

| `init` | `target_accept` | divergences | max R-hat | min bulk ESS |
|---|---|---|---|---|
| `jitter+adapt_diag` | 0.90 | 48 | 1.005 | 1688 |
| `jitter+adapt_diag` | **0.99** | **1** | **1.003** | **1301** |
| `adapt_diag` | 0.90 | 12 | 1.006 | 1409 |
| `adapt_diag` | 0.99 | **280** | **1.105** | **25** |

**Why they interact.** The jitter is what gives four chains different starting points.
This posterior has a near-collinear ridge between media and baseline -- correlation
-0.997, see docs/DIAGNOSTICS.md -- and on a long narrow ridge, initialisation diversity
is what lets chains land in different places and mix. Remove the jitter and shrink the
step size at the same time, and all four chains crawl along the same ridge from the same
point, never decorrelating. Bulk ESS of 25 out of 4,000 draws is chains that have barely
moved.

**Settled: keep `jitter+adapt_diag` (pymc's default, now stated explicitly) and take
`target_accept=0.99`.** One divergence out of 4,000 draws is the best measured
configuration. The overflow warning therefore stays -- but as a choice with a confirmed
cause, not an unsolved problem. Twelve, or one, is still not zero; the rest of that story
is the identification problem in docs/DIAGNOSTICS.md, which no sampler setting can fix.

**The lesson, which is the real content of this entry.** Two changes, each measured
against a proper baseline at full chain length, each a clear improvement, and their
combination is 280x worse than the better of them. Sampler settings are not additive.
Anything that touches geometry and anything that touches step size have to be evaluated
on the grid, not one at a time -- and the only reason this was caught is that the
combined configuration was re-run before being believed.

**What this cost, and the lesson.** Two fix attempts across Phase 2, both reasoned from
a plausible root cause (five orders of magnitude of spread in the design matrix) that
happened to be the wrong one. The test that settled it -- toggle the one flag named in
the hypothesis and see whether the symptom follows -- was available the whole time and
takes five minutes. The scaling attempts were tests of a *cause*; this was a test of the
*hypothesis actually written down*, and only the second kind can close the question.

**Original Phase 2 decision, kept for the record: stop chasing this blindly, defer to
Phase 5 with reasoning stated.**
Continuing to guess at fixes without a way to falsify each guess (each run costs
~100s+ and the warning gives no parameter-level detail) isn't a good use of effort at
the skeleton stage, especially given the divergence count says the actual inference
isn't failing. Phase 5's proper diagnostic pass (more chains, more draws, explicit
R-hat/ESS review, and testing `init="adapt_diag"` without jitter to isolate whether
the jitter phase specifically is the cause) is the right place to either resolve this
properly or confirm it's benign. Recorded as an open item, not a solved one --
readers should not assume later commits silently fixed this just because the code
around it changed.

---

#: 3
**Observation, not yet a resolved issue:** at the skeleton's short chain length
(500 draws, 500 tune, 2 chains), PyMC reported *"The rhat statistic is larger than
1.01 for some parameters."* Divergences remained 0 across all three scaling attempts
above. This is expected at this chain length -- pymc-marketing's own sampler output
recommends at least 4 chains for reliable R-hat -- and is explicitly deferred to
Phase 5 ("full fit + diagnostics") per the build plan, not treated as a new open item
here. Noted so it isn't silently lost between this log and Phase 5's own diagnostics
writeup.

---

#: 4
**Challenge:** switching on Phase 3's anchored priors moved the divergence count off
zero. The Phase 2 skeleton sampled 2 x 500 with 0 divergences every time; the anchored
model reports 2.

**Evidence gathered (2 chains, 500 draws, 500 tune, seed 42, anchored priors):**

| `target_accept` | divergences |
|---|---|
| 0.90 (current default) | 2 |
| 0.95 | 3 |
| 0.99 | 0 |

**Reading of it.** 2-3 divergences in 1000 post-tuning draws, non-monotonic between
0.90 and 0.95 and cleared entirely at 0.99, is the signature of a step size marginally
too large for the tightest part of the posterior -- not of a pathological geometry.
Supporting that: at the same run, max R-hat across all parameters is 1.0065 and
minimum bulk ESS is 360, both of which would look far worse if the sampler were
genuinely failing to explore a region. The LogNormal half-point priors have a long
right tail, which is the most likely place a fixed step size struggles.

**RESOLVED IN PHASE 5: `target_accept=0.99` is now the default, on evidence.** Re-run
at 4 chains x 1000 draws, where the counts mean something:

| `target_accept` | divergences | max R-hat | min bulk ESS | mean step size |
|---|---|---|---|---|
| 0.90 | 48 | 1.0054 | 1688 | 0.0316 |
| 0.95 | 11 | 1.0040 | 1297 | 0.0275 |
| 0.99 | **1** | 1.0033 | 1301 | 0.0155 |

Two things had to hold before raising it, and both do. The count falls **monotonically**
-- the non-monotone 2 / 3 / 0 seen at 2 x 500 was sampling noise, which is its own
lesson about short runs. And the posterior does not move: the largest shift between 0.9
and 0.99 across every adstock, saturation and coefficient parameter is **0.07 posterior
standard deviations**. A smaller step that changed the answer would mean the step size
had been hiding a geometry problem; one that does not is simply resolving the geometry.

Combined with the `init="adapt_diag"` change from entry #2, both defaults are now set by
measurement rather than inherited.

**Original Phase 3 decision, kept for the record: not fixed here, deliberately.** Raising
the default `target_accept` to 0.99 would make the number go away, and Phase 5 ("full fit
+ diagnostics") is the phase whose job is to decide between that, a reparametrisation,
and more tuning draws -- on a real chain length rather than on a 500-draw smoke run.
Bumping a sampler knob at Phase 3 to clear a warning, before the diagnostics phase has
looked at it, is how a real geometry problem gets buried.

**Note on the Phase 2 regression test.** `test_skeleton_fit_has_zero_divergences_
despite_overflow_warning` still asserts 0 divergences and still passes: it runs with
`anchored=False`, which is the specification that claim was established against. Entry
#2's conclusion is unchanged, and the overflow `RuntimeWarning` from that entry still
fires under the anchored model too -- so it is not prior-related either.

---

#: 5
**Challenge:** `python -m mmm_bayes.model` -- the exact command the README gives --
died with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2009' in
position 0` after sampling had already finished. The traceback ran through
`rich/_win32_console.py` and `rich/live.py` with no mention of encodings, and the
completed fit was lost.

**Root cause:** PyMC 6.2.0 renders its sampling progress with `rich`, whose progress
display uses a thin space (U+2009). A stock Windows console gives Python a cp1252
stdout, which has no code point for it. The write happens when the progress manager
exits, i.e. *after* the last draw, so the run burns its full sampling time and then
throws it away at the last moment.

**Resolution:** `model.console_supports_progressbar()` tries to encode U+2009 with
whatever `sys.stdout.encoding` actually is, and `run_skeleton_fit(progressbar=None)`
-- the default -- turns the bar on only if that succeeds. Keyed off the real stdout
encoding rather than off `platform == "win32"`, because `PYTHONIOENCODING=utf-8` makes
a Windows console perfectly capable and the bar is worth having when it works. Pinned
by `test_progressbar_is_disabled_on_a_console_that_cannot_encode_it`, which swaps in a
cp1252 stdout rather than trusting the current terminal to be the interesting case.

Not a modelling problem at all -- but it made the project's headline command fail on
the platform it was being built on, which is worth more than a footnote.

---

#: 6
**Challenge:** `MMM(dims=("geo",))` produces a model that looks hierarchical and is
not. pymc-marketing's own `MMM` docstring says that with panel dims, parameters "share
hierarchical priors so that information is partially pooled across geographies". In
1.1.0 the defaults do nothing of the kind.

**What the defaults actually are.** Every parameter simply gains a geo axis carrying an
independent prior per division -- `adstock_alpha` and `saturation_beta` at
`Beta(1, 3)` / `HalfNormal(2)` broadcast to `(geo, channel)`, with no shared
hyperparameter anywhere in `default_model_config`. On the geo panel that is 936 free
parameters over 2,938 observations with nothing tying the divisions together.

**Why it matters more than a documentation nit.** The model builds, samples, and
reports per-division estimates either way. Nothing about the output announces that no
pooling happened. A reader who trusted the docstring would ship an unpooled model
believing it was hierarchical, and the estimates for the smallest divisions -- 0.8% of
sales, 113 weeks -- would be almost pure prior with no borrowing from anywhere.

**Resolution:** pooling is written explicitly into `model_config` in
`geo_model.hierarchical_config`, per parameter rather than globally, with the reasoning
for each choice in docs/HIERARCHY.md. The actual default behaviour is pinned by
`tests/test_geo_model.py::test_unpooled_model_has_no_shared_hyperparameters`, so the
docstring's claim cannot be taken on trust by a future reader of this repo, and
`test_pooled_model_has_a_hyperparameter_for_every_pooled_block` fails if a later edit
drops a hyperprior.

---

#: 7
**Challenge:** Phase 4 as written in the build plan is not buildable. "Hierarchical
structure (geo pooling)" reads as adding a geography dimension to the Phases 1-3
national model. The geo dataset cannot serve that purpose -- it is a different
business.

**Evidence:** 6 impression/view channels against the national model's 10 dollar-spend
channels, with no name in common; no dollar spend at all; 2018-01-06 to 2020-02-29
against 2014-08-03 to 2018-07-29, with **zero** overlapping week-start dates; ~$3.7M of
weekly sales across 26 divisions against ~$99.6M nationally.

**Consequences, none avoidable:** the Phase 3 anchors have nothing to anchor to; the geo
model cannot produce a ROAS, so Phase 8 has to run on the national model; and this phase
does not put credible intervals on the national contributions -- Phases 2-3 already did
that, and it was always the project's actual headline gap.

**Resolution:** build the phase for what it can honestly be -- a hierarchical MMM with
partial pooling across the 26 divisions, on the geo panel, as a second model. That also
makes Phase 6 coherent for the first time: the ridge project's geo-DiD estimator runs on
this same panel, so calibrating against it is now like-for-like rather than
cross-dataset. Written up in docs/HIERARCHY.md and pinned by
`tests/test_geo_model.py::test_geo_is_a_different_dataset_from_national`, which fails if
a future data refresh ever makes the two panels overlap.

The alternative -- quietly building something geo-shaped and leaving the README to imply
it extended the national model -- is the failure mode this project's audit discipline
exists to catch, so it is recorded as a plan defect rather than silently re-scoped.

---

#: 8
**Challenge:** entry #1's `cores=1` workaround asked to be re-tested on a multi-core
machine. Re-tested at the start of Phase 5, and neither half of it holds here.

**What is different.** `os.cpu_count()` on this machine is **4**, not the 1 that entry
#1 records, and `cores=None` does not raise `ZeroDivisionError`. It raises something
else entirely:

    RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

**Root cause:** Windows has no `fork`. pymc's parallel sampling spawns fresh
interpreters that re-import the calling module, so a `model.fit(...)` sitting at module
level is re-executed by every child. The fix is the standard multiprocessing idiom -- an
`if __name__ == "__main__":` guard around the call -- and with one in place, parallel
sampling works: 4 chains in 2 jobs, 145 seconds for the national model at
4 x 1000 draws.

**Where that leaves entry #1.** The `ZeroDivisionError` was real, but it was specific to
a single-core environment; it is not a property of pymc 6.2.0 in general. `cores=1`
stays the default in `run_skeleton_fit` and `run_geo_fit` because those are called from
pytest, where the `__main__` guard does not exist and a spawn would re-import the test
module. The scripts that do have a guard (`run_full_fits.py`,
`sensitivity_checks.py`) pass `cores=2`, which is what made a 4-chain phase affordable
at all. `cores` remains explicit either way -- the point of entry #1 -- but it is now an
explicit choice per call site rather than a workaround.

**Measured, so the choice is not folklore:** national model, 2 chains x 400 iterations,
`cores=1` 56.1s against `cores=2` 45.4s. Only 1.2x, because on Windows spawn the
interpreter startup and PyTensor compilation are paid per process and dominate a short
run. The gain is larger at real chain length, where sampling dominates instead.

---

#: 9
**Challenge:** a sensitivity check that would have silently compared a model against
itself. Caught before it ran, by luck rather than by process.

**What happened.** The Phase 5 check for "does adstock decay actually vary by division?"
built the partially pooled arm by constructing the model and then assigning
`model.model_config["adstock_alpha"] = <hierarchical prior>`. That is the same pattern
used a few lines earlier for the intercept, where it works. For adstock it does not.

**Root cause:** `model_config` seeds the adstock and saturation TRANSFORMATION objects'
priors at construction time. After construction, the transformation holds its own copy;
mutating the dict changes nothing the model will read. The build succeeds, the fit runs,
and the "partially pooled" arm is the completely pooled model with a different label.

**How it surfaced.** A dry build of the variant printed its free random variables before
committing 20 minutes of sampling to it. The pooled arm showed a bare `adstock_alpha`
where a non-centred hierarchy must show `adstock_alpha_raw_offset`,
`adstock_alpha_raw_mu` and `adstock_alpha_raw_sigma`. Setting the prior on
`model.adstock.function_priors` instead produces all three.

**Resolution:** set transformation priors on the transformation
(`model.adstock.function_priors`, `model.saturation.function_priors`); `model_config`
after construction is only reliable for parameters the MMM owns directly, such as the
intercept and the likelihood. The two overrides Phase 5 actually depends on were then
re-verified by reading the compiled graph -- `saturation_beta`'s sigma and the
intercept's mu both check out -- rather than assumed from the fact that the fit ran.

**The general point, which is why this is written down rather than just fixed.** A
sensitivity check that fails loudly costs an afternoon. One that silently compares a
model against itself produces a confident null result -- "decay does not vary by
division, we tested it" -- that is worse than never having run it. Any check whose two
arms are supposed to differ should be asked to prove they differ, structurally, before
it is asked what the difference means.

---

#: 10
**Challenge:** `mmm_bayes.attribution` returned confidently wrong numbers on the
hierarchical geo fit -- intercept at **3051% of sales**, components failing to reconcile
against observed sales by a factor of **34**. No error, no warning. Found by the
end-to-end audit (docs/AUDIT.md), not by use.

**Root cause, two errors compounding:**

* `n_periods = len(sales)` counts division-weeks (2,938) on a long-format panel, where
  the intercept applies once per **date** (113).
* `scale = sales.max()` is a single global maximum, but pymc-marketing scales the target
  **per geo** -- `target_scale` holds 26 entries differing by more than 10x.

**Why it was worth fixing rather than noting.** The bug was latent: nothing calls these
functions with a panel fit, and the national path is verified correct against a
first-principles recompute. But `_component_totals` *explicitly handled panel dims*
(`extra = [d for d in intercept.dims if d not in ("chain", "draw")]`), so the code read
as though panels were supported on purpose. Code that silently mis-supports a case is
worse than code that does not support it, because the next person has no reason to check.

**Resolution:** all three public entry points now refuse a panel fit with a
`NotImplementedError` naming both errors, pinned by
`tests/test_attribution.py::test_panel_fits_are_refused_rather_than_mis_answered`.
Supporting panels properly means threading a per-geo scale vector and a true period count
through the module -- a feature, deliberately not attempted under an audit.

**How this relates to #9.** That entry is a sensitivity check that would have silently
compared a model against itself, caught by dry-running the variant before spending a fit
on it. This is the same class of failure and it was not caught that way: it produced
numbers for anyone who asked. The difference is that #9's arms were *supposed* to differ,
so there was something to check; here there was no signal at all until someone pointed
the function at an input it had never seen. Latent wrong answers do not surface from
use -- only from going looking.
