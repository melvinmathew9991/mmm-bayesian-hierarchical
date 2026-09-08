# Hierarchical structure (Phase 4)

## The finding that reshaped this phase

The build plan reads as though Phase 4 adds a geography dimension to the model built in
Phases 1-3. **It cannot.** The geo dataset is not a geographic decomposition of the
national one — it is a different business:

| | national (Phases 1-3) | geo (Phase 4) |
|---|---|---|
| channels | 10, with dollar spend | 6, impressions and views only |
| spend data | yes | **none** |
| period | 2014-08-03 → 2018-07-29 | 2018-01-06 → 2020-02-29 |
| overlapping week-start dates | — | **0** |
| sales | ~$99.6M/week | ~$3.7M/week total across 26 divisions |
| rows | 209 weeks | 26 divisions × 113 weeks = 2,938 |

Three consequences follow, and none of them are optional:

1. **The Phase 3 anchors do not transfer.** There is no channel in common, so there is
   nothing to anchor to. The geo model's priors are weakly informative, set by prior
   predictive check against the geo panel itself.
2. **This model cannot produce a ROAS.** The panel has impressions, not dollars. Phase 8
   (budget optimisation under uncertainty) therefore has to run on the national model,
   or not at all.
3. **This model does not put credible intervals on the national contributions.** The
   project's headline gap — "no posterior on the ridge project's contributions" — is
   closed by Phases 2-3, not by this phase.

What Phase 4 *does* deliver is the thing it is named for: a hierarchical MMM with
partial pooling across 26 divisions. And it delivers it on the same data as the ridge
project's geo-DiD estimator, which is what makes Phase 6's calibration a like-for-like
comparison rather than a cross-dataset one.

This is documented rather than worked around because the alternative — quietly building
*something* geo-shaped and letting the README keep implying it extends the national
model — is exactly the failure this project's audit discipline exists to catch.
`tests/test_geo_model.py::test_geo_is_a_different_dataset_from_national` fails if the
two datasets ever start overlapping, so the claim cannot go stale unnoticed.

## `dims=("geo",)` does not pool

pymc-marketing's `MMM` docstring says that with panel dims, parameters "share
hierarchical priors so that information is partially pooled across geographies". In
1.1.0 the defaults do no such thing. `MMM(dims=("geo",))` gives every parameter a geo
axis with an **independent** prior per division:

```
adstock_alpha      (geo, channel)  156
saturation_lam     (geo, channel)  156
saturation_beta    (geo, channel)  156
gamma_control      (geo, control)  312
gamma_fourier      (geo, fourier)  104
intercept          (geo,)           26
y_sigma            (geo,)           26
                                   ---
                                   936 free parameters over 2,938 observations
```

Nothing is shared between divisions. The pooling in this phase is therefore written
explicitly into `model_config`. See `docs/CHALLENGES.md` #6.

## What is pooled, and why each choice

| Parameter | Treatment | Reasoning |
|---|---|---|
| `adstock_alpha` (decay) | **Complete pooling** — one per channel, no geo axis | Carryover is a property of the medium, not the division: an email keeps working for as long as it keeps working whether it lands in division A or division T. Phase 3 also showed decay barely moves off its prior with 209 national weeks; 156 per-geo decays on 113 weeks each would be noise with a geo label. **156 → 6.** |
| `saturation_beta` (media effect) | **Partial pooling**, non-centred lognormal, per-channel location and scale | This is where genuine division-level differences plausibly live, and the quantity the model exists to estimate. |
| `saturation_lam` (half-point) | **Partial pooling**, same form | Channels are max-scaled per geo, so half-points are comparable across divisions on the scaled axis — the condition under which pooling is meaningful. |
| `intercept` | **Partial pooling**, and constrained positive | Baseline sales cannot be negative. See the prior predictive section. |
| `gamma_fourier` (seasonality) | **Partial pooling** | Measured directly: an OLS fit of the four Fourier modes on the raw panel gives coefficients no larger than 0.13 with an across-division sd of ~0.01. The divisions share a seasonal shape almost exactly, so the pooling should shrink this nearly to a single curve. |
| `gamma_control` (holiday weeks) | **Partial pooling** | Same argument, wider prior — these carry the Q4 spike. |
| `y_sigma` | **Partial pooling** | Divisions differ in size and so in noise. |

Everything hierarchical is **non-centred** (`centered=False` with `transform="exp"` for
the positive parameters). Twenty-six groups with a weakly identified scale is the
textbook funnel, and Phase 3 already saw this sampler produce divergences on far simpler
geometry (`docs/CHALLENGES.md` #4).

### Counting parameters is the wrong way to judge pooling

Pooled: 845 free parameters. Unpooled: 936. Almost all of that 91-parameter difference
is the adstock change. Non-centred pooling swaps an independent prior per group for an
offset per group plus two hyperparameters, so the raw count barely moves.

What pooling buys is a smaller *effective* number of parameters — group offsets shrunk
toward zero by a scale the data estimates. That is why the tests check fitted
across-geo scales rather than parameter counts, and why
`test_pooling_is_not_a_parameter_count_saving` exists to stop a future edit from
optimising the wrong number.

## Holiday weeks are not optional here

The geo panel carries no controls at all — a division, a week, six impression series and
sales. Seasonality has to be built from the week index.

Two Fourier harmonics cannot do it alone. Fitted to the real panel by OLS, they
reproduce a seasonal curve spanning only −0.15 to +0.25 of max-scaled sales, while the
observed Q4 peak reaches 1.0. Whatever the harmonics miss does not vanish: it is
absorbed by whichever regressor correlates with Q4, and in a retail business media
correlates with Q4 strongly. The national model's `features.py` makes this argument for
itself ("any model omitting them lets media coefficients absorb the calendar"); it
applies with more force here.

So the model adds 0/1 dummies for **ISO weeks 40-51**, the Oct-Dec retail block. The
range is a calendar fact rather than a fit, though the data agrees: mean max-scaled
sales run 0.19 at baseline against 0.30-0.82 across weeks 40-51, peaking in week 48. The
upper end extends past the national model's week 48 because this panel's peak clearly
runs into weeks 49-51.

## The prior predictive check, and what it caught

Every prior scale in `geo_model.hierarchical_config` was set by checking what it implies
about *sales*, not by inspecting the prior in isolation. That is what surfaced two
problems reading the priors did not:

1. **The intercept could go negative.** An unconstrained hierarchical Normal put 12.5%
   of the prior predictive below zero. Baseline sales cannot be negative; the intercept
   is now exponentiated.
2. **The seasonality prior, not the noise prior, was responsible.** Decomposing the
   prior predictive: the noise term's own sd had a posterior median of 0.020, while the
   Fourier contribution alone spanned −0.475 to +0.469. Tightening the noise prior — the
   obvious first move — changed almost nothing.

Final state (max-scaled sales, 400 prior draws):

| | prior predictive | observed |
|---|---|---|
| q01 | −0.194 | 0.113 |
| q25 | 0.267 | 0.160 |
| q50 | 0.447 | 0.190 |
| q95 | 0.938 | 0.543 |
| q99 | 1.198 | 0.981 |
| P(y < 0) | 4.9% | 0% |

The prior brackets the data at both tails, which is the property that matters. It also
sits about 2.4× above the observed median, which is a deliberate trade: raising the
intercept lowers the impossible mass and raises the predictive level, and those move
together. 4.9% on negative sales was judged the better side of that trade than a
better-centred prior with 8%+.

**The residual 4.9% is structural, not a tuning failure.** An additive Gaussian model
with an identity link and symmetric seasonality priors will always put some mass below
zero. `MMM(link="log")` is the structural fix and would make it exactly zero. It is not
adopted *here* because it changes the model's functional form — a Phase 5 decision, on a
real chain length, not a Phase 4 prior adjustment.

**Built in Phase 6 as `build_geo_model(link="log")`.** It does exactly what this section
predicted: prior predictive `P(y < 0) = 0.0000`, by construction rather than by tuning.
It also removes the trade this section had to make. The identity-link intercept was
centred at 0.30 rather than the observed 0.246 specifically to halve the impossible mass,
at the documented cost of *"a predictive median 2.4x the observed one"*. Under the log
link no such purchase is needed — predictive median 0.213 against an observed 0.190, 12%
high rather than 135%.

The change is larger than a flag, which is why it is `log_hierarchical_config` and not an
argument to `hierarchical_config`. Under `link="log"` pymc-marketing uses a `LogNormal`
likelihood with `median(y) = exp(mu) · target_scale`, so the whole linear predictor moves
to log space and every prior above changes meaning. Two break outright:

| prior | identity link | what log space needs |
|---|---|---|
| intercept | `transform="exp"`, strictly positive, centred at `log(0.30)` | **−1.705** (OLS on `log` max-scaled sales), across-geo sd 0.071, **and it must be able to go negative** |
| holiday weeks | `Normal(0, 0.18)` | coefficients reach **1.704** (`peak_wk48` averages 1.432) |
| seasonality | `Normal(0, 0.15)`, across-geo `HalfNormal(0.05)` | \|max\| 0.161, across-geo sd up to 0.060 |
| likelihood scale | `HalfNormal(HalfNormal(0.04))` on max-scaled sales | residual log sd **0.197** |

The intercept is the one that could not have been left alone. A strictly positive
log-space intercept has `exp(0) = 1.0` as its *smallest* expressible baseline — every
week at or above that division's own peak — so it cannot reach `−1.7` at all. And the
reason it was exp-transformed in the first place was to keep baseline sales positive,
which under a LogNormal likelihood is guaranteed. The structural fix makes the workaround
obsolete rather than relocating it; that is the clearest single argument for the log link
in this model.

`saturation_beta` keeps its location, by arithmetic rather than by inertia: producing a
media share `s` needs a total log-lift of `log(1/(1−s))` across six channels, so `s`
between 3% and 61% wants per-channel values of roughly 0.006–0.156 — the range
`exp(Normal(−3.5, 1.0))` already spans. `saturation_lam` and `adstock_alpha` are
untouched because both live in *channel* units, which the link does not touch.

**Two cautions.** pymc-marketing 1.1.0 warns that `link="log"` is *"experimental and
under active development… API and behavior may change in future releases without
deprecation warnings"*. And the decomposition stops being additive: media is reported as
the counterfactual `exp(mu) − exp(mu − mu_media)`, so `geo_attribution`'s additive sum is
meaningless under this link and now refuses it explicitly.

## What the first fit showed

From a short run (2 chains, 500 draws, 500 tune) — not a real fit; Phase 5 owns
inference quality:

| | |
|---|---|
| divergences | **0** |
| max R-hat (4,849 entries — see note) | 1.0239 |
| entries with R-hat > 1.05 | 0 |
| entries with R-hat > 1.01 | 64 |
| min bulk ESS | 245 |

The entry counts in this table are on the pre-correction denominator: 2,938 of those
4,849 are `yearly_seasonality_contribution`, a per-observation deterministic that
`diagnostics.DERIVED_VARS` failed to exclude until Phase 6's opening work. The true
parameter count is 1,911. These particular numbers come from a short run that was never
cached, so they are annotated rather than recomputed — `docs/DIAGNOSTICS.md` carries the
corrected figures for the real fit. See `docs/CHALLENGES.md` #12.

**Zero divergences**, across three independent 2×500 runs. Worth noting against Phase 3,
where the *national* model — one geography, ~30 parameters — produced 2 at the same
settings (`docs/CHALLENGES.md` #4). An 845-parameter hierarchical model sampling more
cleanly than a small flat one is the non-centred parametrisation doing exactly what it
was chosen for.

The 64 entries above R-hat 1.01 are almost all non-centred *offsets* — `N(0, 1)`
nuisance variables whose mixing matters far less than the parameters they compose into
— plus one division's `y_sigma`. Nothing exceeds 1.05, and the worst is 1.024. That is a
chain-length question at 2 chains and 500 draws, not a geometry one, and it is Phase 5's
to settle.

(R-hat here is computed over the model's own parameters. Summarising the full posterior
also covers `channel_contribution`, a date × geo × channel derived quantity with ~17,600
entries whose diagnostics add nothing the parameters do not already say — and take
longer to compute than the fit itself.)

The `RuntimeWarning: overflow encountered in dot` from `docs/CHALLENGES.md` #2 fires
here too. It has now appeared on the unanchored national model, the anchored national
model, and this hierarchical geo model — three different specifications on two
different datasets. That is further evidence for the "benign, sampler-internal"
reading and against anything data- or prior-specific.

**The pooling works, and it is informative.** Posterior means of the across-geo scales:

| scale | posterior mean | reading |
|---|---|---|
| `gamma_fourier_sigma` | 0.0020 | Seasonality is effectively identical across all 26 divisions — shrunk almost to complete pooling, exactly as the OLS measurement predicted. |
| `gamma_control_sigma` | 0.0116 | Holiday response nearly identical too. |
| `intercept_contribution_raw_sigma` | 0.1702 | Baselines genuinely differ. |
| `saturation_beta_raw_sigma` | 0.1815 | Media response genuinely differs by division. |
| `saturation_lam_raw_sigma` | 0.1919 | So do half-points. |

That two-orders-of-magnitude gap between the seasonal and the media scales is the
substantive result of the phase: **these divisions share a calendar but not a media
response.** A model that pooled everything equally would have imposed the first and
destroyed the second. `tests/test_geo_fit.py::test_seasonality_shrinks_further_than_media_response`
pins it.

**Pooled decay estimates** (one per channel, shared across all divisions):

| channel | mean | sd |
|---|---|---|
| Google_Impressions | 0.979 | 0.012 |
| Organic_Views | 0.730 | 0.126 |
| Affiliate_Impressions | 0.328 | 0.261 |
| Paid_Views | 0.299 | 0.211 |
| Facebook_Impressions | 0.043 | 0.027 |
| Email_Impressions | 0.010 | 0.011 |

`Google_Impressions` at 0.979 with sd 0.012 deserves suspicion rather than
interpretation. A decay that close to 1 makes the adstocked series behave like a slow
level rather than a carryover, which is what a channel absorbing an unmodelled trend
looks like. The 8-week window bounds the damage, but this is the first thing Phase 5
should test — starting with whether it survives a trend term.

`Affiliate_Impressions` and `Paid_Views` have sds nearly as large as their means: those
two decays are essentially unidentified, and their posteriors are close to their priors.

## Open items handed to Phase 5 — and what came back

- ~~`Google_Impressions` decay pinned near 1 — test against an explicit trend term.~~
  **Done, hypothesis refuted.** There is a real declining trend (coefficient −0.146, sd
  0.015), but adding it pushed the decay *up* to 0.989 and moved `Organic_Views` a full
  1.4 sd the same way. A decay near 1 is competition among near-collinear level-like
  terms, not one missing control. See `docs/DIAGNOSTICS.md`.
- ~~Whether complete pooling of decay is defensible against a partially pooled
  alternative.~~ **Done, inconclusive.** Every fitted across-division scale sits within
  0.40 prior sd of its own prior mean — 113 weeks per division cannot say. Complete
  pooling costs nothing in the central estimates (max difference 0.046) and stays the
  default; the argument for it is still a priori, now known to be untestable here rather
  than merely untested.
- ~~64 of 4,849 parameter entries above R-hat 1.01 at 2×500.~~ **Done.** At 4 chains ×
  500 draws with `target_accept=0.99`: **0 divergences**, 12 of 1,911 entries over 1.01
  (max 1.0166, eleven non-centred offsets and one `y_sigma`), bulk ESS 422. Marginal,
  and limited by the draw count this machine's memory allows rather than by geometry.
  The denominator is the corrected one; see the note above the first-fit table.
- ~~The Gaussian identity link puts 4.9% of the prior predictive on negative sales;
  `link="log"` is the structural fix.~~ **Done, in Phase 6.**
  `build_geo_model(link="log")` gives `P(y < 0) = 0.0000` and a better-calibrated prior
  predictive besides (median 12% above observed, against the identity link's 135%). It
  needed a re-specified prior set rather than a flag — see the section above — and it
  makes the exp-transformed intercept, which existed only to keep baselines positive,
  unnecessary. The library flags the link itself as experimental.
  **The fit it produces does not converge** — 486 of 1,885 entries over R-hat 1.01
  against the identity link's 12, for reasons that are not the shorter chain. See
  "What the log-link fit showed" below and `docs/CHALLENGES.md` #15. The structural
  fix to the prior and a sampleable posterior turned out to trade against each other.

## What the log-link fit showed

4 chains × 300 draws (1,000 tune), `target_accept=0.99`, `cores=1` — 28.2 minutes.
Draws are 300 rather than 500 because the log link registers one more per-observation
deterministic (`y_original_scale`) and 500 exceeded this machine's memory *after*
sampling finished; see `scripts/run_full_fits.py`.

**It does not pass.** 486 of 1,885 parameter entries above R-hat 1.01, 75 above 1.05,
max 1.1189, min bulk ESS 36 — against the identity link's 12, 0, 1.0166 and 421. Zero
divergences and min E-BFMI 0.888 in both. The gap is not the shorter chain: median bulk
ESS falls further than 1.67x fewer draws accounts for, and the worst parameter is 85x
worse per draw. The mechanism — a non-identification that was benign under the identity
link and is not under this one — is `docs/CHALLENGES.md` #15.

**The across-geo scales are the part of this fit that can be read**, and they reproduce
the Phase 4 result under a different functional form:

| scale | identity | log | |
|---|---|---|---|
| `gamma_fourier_sigma` | 0.0020 | 0.0147 | seasonality still near-completely pooled |
| `gamma_control_sigma` | 0.0116 | 0.0239 | holiday response too |
| `saturation_beta_raw_sigma` | 0.1815 | 0.1711 | **media response still genuinely differs** |
| `saturation_lam_raw_sigma` | 0.1919 | 0.1896 | so do half-points |
| `intercept_contribution_sigma` | 0.1702 | 0.0779 | not comparable — the log link drops the exp transform the identity version is scaled by |

The two media scales move by less than 6% across a change of functional form, and the
order-of-magnitude gap between seasonal and media pooling survives. **These divisions
share a calendar but not a media response** was the substantive finding of Phase 4, and
it is now known not to be an artefact of the additive link. That is the one inference
this fit adds, and it happens to be the one that does not depend on the parameters that
mixed badly.

Nothing resting on the channel-level parameters themselves should be read off this fit.
