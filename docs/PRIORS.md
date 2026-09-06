# Prior specification

Every prior choice here, justified in one line, per the build plan's instruction that
this is the single most inspectable part of a Bayesian project to a reviewer.

**Scope: this file is about the NATIONAL model (Phases 1-3).** The hierarchical geo
model added in Phase 4 is a different model on a different dataset, with its own priors
and its own reasoning -- see `docs/HIERARCHY.md`. Nothing anchored here transfers there:
the two datasets share no channel, so there is nothing to anchor to.

**Status: Phase 3 done. Adstock decay and saturation half-point are anchored per
channel to the ridge project's own fitted values.** Everything else remains at
pymc-marketing's library defaults, and the table below says which is which and why.
The code is `src/mmm_bayes/priors.py`; run `python -m mmm_bayes.priors` to print the
per-channel anchor table.

`build_model(anchored=False)` still reproduces the Phase 2 specification exactly
(library defaults throughout, plain `LogisticSaturation`). It is kept because the
docs/CHALLENGES.md #2 regression tests were established against that model.

## The priors

| Parameter | Prior | Source | Justification |
|---|---|---|---|
| Adstock decay (`alpha`, per channel) | `Beta(mu * 6, (1 - mu) * 6)`, `mu` = the ridge decay clipped to [0.05, 0.95] | **Anchored** -- `data/derived/ridge_anchors.json` | Centred on evidence instead of on the library default's 0.25, with concentration 6 giving sd ~0.15 at mid-range means against 0.19 for the default `Beta(1, 3)` -- so it relocates the prior without materially tightening it. |
| Saturation half-point (`lam`, per channel) | `LogNormal(log(lam_anchor), 0.7)` | **Anchored** -- same file, converted (see below) | `lam` here is literally the half-saturation point, so the prior is stated in the ridge estimate's own units. sigma 0.7 is ~2x per sd: the data can move a half-point 4x within two sd. |
| Media coefficient (`saturation_beta`) | `HalfNormal(2)` (pymc-marketing default) | Library default, **deliberately not anchored** | Non-negative support mirrors the ridge project's floor-at-zero on media coefficients -- spending money cannot reduce sales. Left unanchored on purpose: this coefficient is the quantity the project exists to estimate independently, and anchoring it to the ridge estimate would prejudge Phase 9's comparison. |
| Target scaling | `DataDerivedScaling(method="max")` | `model.py::build_model` | Sales range ~$45M-$352M and needed scaling before any prior touches it (docs/CHALLENGES.md #2). |
| Channel scaling | `DataDerivedScaling(method="max")`, per channel | `model.py::build_model` | Spend ranges from near-zero to ~$3.1M across the 10 channels. This is also the divisor the half-point conversion below depends on. |
| Control coefficients | Library default (unconstrained), inputs z-scored | Library default + `features.standardize_controls` | Matches the ridge project's choice to leave controls free to take either sign. Z-scoring was resolution attempt #2 for docs/CHALLENGES.md #2; it did not fix the warning but is kept as good practice regardless. |
| Intercept | Library default `Normal(0, 2)` | Library default, **reviewed in Phase 5 and found wrong** | That default assumes a centred target. Here the target is max-scaled sales -- strictly positive, mean 0.31 -- so a prior centred on zero pushes the baseline toward nothing and leaves the non-negative media coefficients to carry it. Re-centring on the observed mean moves media-attributed sales from 104% to 84%. See `docs/DIAGNOSTICS.md`. |
| Likelihood noise | Library default | Library default | Not yet reviewed. |

## How the ridge half-points were converted

The ridge half-points cannot be used as-is: they live in different units at three
points in the pipeline. The full derivation is in `priors.py`'s module docstring; the
result is

    lam_anchor = ridge_half_point * (1 - ridge_decay) / max_t(spend_t)

because (1) the model divides each channel by its own max spend, (2) pymc-marketing's
`GeometricAdstock` normalises its weights to sum to 1 while the ridge IIR's sum to
`1 / (1 - d)`, and (3) the finite-window terms from `l_max = 8` cancel out.

That conversion is the highest-risk part of Phase 3 -- get it wrong and the model
still samples happily while being quietly biased -- so it is pinned by tests rather
than argued for on paper alone. `tests/test_priors.py` checks the channel divisor
against the model's own `channel_scale` variable, checks per channel on the real
series that pymc's normalised adstock really is `(1 - d)` times the ridge adstock
(median ratio within 0.03 of 1.0 for all ten), and checks that the saturation returns
exactly 0.5 at each anchored `lam`.

### One modelling change came with it

Phase 2 used `LogisticSaturation`, whose `lam` is an efficiency rate whose
half-saturation point sits at `ln(3)/lam`. Phase 3 uses
`InverseScaledLogisticSaturation`, which is **the same logistic curve reparametrised**
so that `lam` *is* the half-saturation point. This is a reparametrisation, not a
respecification -- the likelihood is unchanged -- and it means the prior is written in
the units the ridge estimate already has, instead of through a conversion factor
buried inside the prior.

**Still different from ridge, and worth knowing:** ridge saturates with
Michaelis-Menten, `x / (x + h)`, which approaches its asymptote more slowly than a
logistic. The two curves agree at the half-point by construction but not away from it.
`MichaelisMentenSaturation` exists in pymc-marketing if a later phase wants exact shape
parity with the ridge model; that is a modelling change rather than a prior change, so
it is out of Phase 3's scope and recorded here rather than assumed away. Phase 9 should
not attribute a ridge/Bayes contribution gap to inference method without accounting for
this.

## What the anchors are

Regenerate with `python scripts/build_ridge_anchors.py`. The cache records an MD5 of
`national_weekly.csv` and `load_ridge_anchors` refuses to use it if the CSV has
changed, so anchors fit on stale data raise rather than quietly biasing a posterior.

| channel | ridge decay | ridge half-point | max spend | `lam` anchor |
|---|---|---|---|---|
| dm | 0.80 | 3,710,973 | 2,409,896 | 0.3080 |
| inst | 0.20 | 163,656 | 590,148 | 0.2219 |
| nsp | 0.40 | 498,572 | 2,198,467 | 0.1361 |
| auddig | 0.80 | 33,625 | 13,065 | 0.5147 |
| audtr | 0.40 | 397,772 | 435,615 | 0.5479 |
| vidtr | 0.00 | 56,177 | 1,100,083 | 0.0511 |
| viddig | 0.20 | 8,909 | 104,352 | 0.0683 |
| so | 0.80 | 231,499 | 573,356 | 0.0808 |
| on | 0.80 | 1,934,121 | 695,750 | 0.5560 |
| sem | 0.80 | 5,516,488 | 3,134,565 | 0.3520 |

Every `lam` anchor lands inside [0.05, 0.56], comfortably within the [0, 1] range the
saturation input actually occupies. That is itself a check: a dropped or doubled scale
factor would have thrown these orders of magnitude off, which is why
`test_lam_anchors_land_inside_the_observed_input_range` asserts it.

## The two boundary cases, which are not symmetric

The ridge search ran a grid over decay in [0.0, 0.8] and 6 of 10 channels landed on a
boundary. Those two boundaries mean different things, and the priors treat them
differently on purpose:

* **Upper edge (0.8, four channels).** This is censoring -- the grid stopped, the truth
  may be higher. `Beta(4.8, 1.2)` has a 90% interval of [0.50, 0.98], so the posterior
  can move well above 0.8 if the data asks.
* **Lower edge (0.0, `vidtr`).** Zero is a hard edge of the parameter space, not a grid
  artefact: the search found no carryover. The mean is clipped to 0.05 to keep the
  prior proper (`Beta(0, k)` is not a distribution), giving `Beta(0.3, 5.7)` with a 90%
  interval of [0.00, 0.23].

`vidtr`'s prior is by some margin the tightest of the ten -- a Beta with mean 0.05
cannot be wide. That is accepted rather than tuned around, but it makes `vidtr` the one
channel whose posterior most deserves a prior-sensitivity check in Phase 5.

## What the first anchored fit did

From a short run (2 chains, 500 draws, 500 tune) -- not a real fit, Phase 5 is where
inference quality gets established:

* Decay posteriors sit close to their anchors (dm 0.85 vs 0.80, nsp 0.43 vs 0.40, sem
  0.77 vs 0.80) with sd 0.13-0.20. Read honestly, that means the data is **not** moving
  decay much -- which is the weak identification the anchoring exists to handle, not
  independent confirmation of the ridge estimates.
* Half-point posteriors did move: `auddig` 0.51 -> 1.07, `audtr` 0.55 -> 1.10, `dm`
  0.31 -> 0.55, while the small-`lam` channels barely shifted. Some of that gap is the
  LogNormal's own mean/median offset (its mean is the anchor x 1.28, not the anchor),
  so compare against 1.28x the anchor before calling a shift large.
* Max R-hat 1.0065, min bulk ESS 360, and 2 divergences -- see docs/CHALLENGES.md #4
  for the divergences, which are new in the anchored model.

## Reviewed in Phase 5

The intercept prior was the "not yet reviewed" row above, and reviewing it found a real
defect -- see `docs/DIAGNOSTICS.md`. The short version: the library default is centred
on zero, the target is not, and media ends up carrying the entire baseline.

The likelihood-noise prior remains an untouched library default. "Not yet reviewed" is
still the accurate description of it, not "chosen".

## The geo model's priors are elsewhere

`geo_model.hierarchical_config` sets the Phase 4 priors, and `docs/HIERARCHY.md`
justifies each one. Two differences from this file are worth stating explicitly:

* **Nothing there is anchored.** The ridge project never fit an MMM on the geo panel --
  it used that data only for placebo-validated DiD -- so there are no fitted decays or
  half-points to anchor to, and no channel in common with the national data either way.
  Those priors are weakly informative, with their scales set by prior predictive check
  against the geo panel.
* **The structure is the point there, not the location.** This file is about where the
  national priors are centred. `HIERARCHY.md` is about which parameters share a scale
  across 26 divisions and which do not -- a question the national model, with one
  geography, never had to answer.

The one thing carried across deliberately is the saturation family:
`InverseScaledLogisticSaturation` in both, so `lam` means the same thing (a
half-saturation point) in either model.
