# Full fit and diagnostics (Phase 5)

Phases 2, 3 and 4 each sampled short chains and each said, in writing, that inference
quality was Phase 5's problem. This is Phase 5.

**Headline: the national model samples well and cannot answer the question it was built
to answer.** Every convergence threshold passes except divergences. And the media/baseline
split is not identified — posterior correlation between total media contribution and
baseline is **−0.997**, and their sum is determined an order of magnitude more precisely than
either part. The data pins down total sales, which is trivial, and says almost nothing
about how much of it media caused.

That is a specification problem, not a sampling problem, and no convergence diagnostic
can see it.

## What counts as passing, decided before looking

Set in `diagnostics.py` before any full fit was run. A threshold chosen after seeing the
answer is not a threshold.

| criterion | threshold | why |
|---|---|---|
| divergences | 0 | a divergence means the sampler could not follow the geometry somewhere, and biases the posterior in a direction nothing else reports |
| R-hat | ≤ 1.01 | Vehtari et al. 2021 — also why 4 chains is the minimum, since R-hat needs between-chain variance to be estimable |
| bulk ESS | ≥ 400 | the usual floor for a stable posterior mean |
| tail ESS | ≥ 400 | what credible intervals actually depend on; reported separately because bulk can be healthy while tails are not |
| E-BFMI | ≥ 0.3 | catches momentum resampling failing to explore the energy distribution — a failure that produces no divergences and no R-hat warning |

Real chain length is **4 chains × 1000 draws × 1000 tune**, pymc's own default rather
than a number raised until things passed.

### Short chains were hiding the divergences

At 2 chains × 500 draws the national model showed **2** divergences. At 4 × 1000 it shows
**48**. Divergences are counted per draw, so a short run does not merely estimate the
rate imprecisely — it makes a real problem look like rounding error. Every "looks fine at
this chain length" note in Phases 2–4 has to be read with that in mind.

## National model: FAIL, by one divergence

Final configuration — 4 chains × 1000 draws, `init="jitter+adapt_diag"`,
`target_accept=0.99`, anchored priors, 50 parameter entries, 5.4 minutes.

| | value | verdict |
|---|---|---|
| divergences | **1** | ✗ |
| max R-hat | 1.0033 (0 over 1.01) | ✓ |
| min bulk ESS | 1301 | ✓ |
| min tail ESS | 1329 | ✓ |
| min E-BFMI | 0.748 | ✓ |

The threshold was set at zero before any fit was run, so one divergence is a fail and is
reported as one. In practical terms it is a single draw in four thousand, on the best
configuration found: everything else is comfortably clear, ESS is three times the floor,
and the remaining divergence could not be removed without making other diagnostics
worse (see the interaction table below).

### Getting there: the settings, and an interaction that nearly went unnoticed

The starting configuration — pymc's defaults, `target_accept=0.9` — gave **48**
divergences. Two changes were then measured separately, each against a proper baseline
at full chain length, and each was an improvement. Setting both produced a model far
worse than either:

| `init` | `target_accept` | divergences | max R-hat | min bulk ESS |
|---|---|---|---|---|
| `jitter+adapt_diag` | 0.90 | 48 | 1.005 | 1688 |
| `jitter+adapt_diag` | **0.99** | **1** | **1.003** | **1301** |
| `adapt_diag` | 0.90 | 12 | 1.006 | 1409 |
| `adapt_diag` | 0.99 | **280** | **1.105** | **25** |

Bulk ESS of 25 out of 4,000 draws is four chains that have barely moved. The cause is
the ridge described later in this document: jitter is what gives chains different
starting points, and on a long narrow ridge that diversity is what lets them mix.
Remove the jitter *and* shrink the step size and all four crawl along the same ridge
from the same place.

**Sampler settings are not additive.** Anything touching initialisation and anything
touching step size have to be evaluated on the grid rather than one at a time. This was
caught only because the combined configuration was re-run before being believed. See
`docs/CHALLENGES.md` #2.

### The overflow warning: cause confirmed, jitter kept anyway

`CHALLENGES.md` #2 had been open since Phase 2, where two reasoned scaling fixes failed
to shift a `RuntimeWarning: overflow encountered in dot`, and the working conclusion was
that it came from NUTS's jitter phase. Phase 5 tested that directly — `init="adapt_diag"`
removes the jitter and nothing else:

| `init` | overflow warnings | divergences |
|---|---|---|
| `jitter+adapt_diag` | 2 | 48 |
| `adapt_diag` | **0** | 12 |

posterior shift between them: at most 0.06 sd, across every parameter.

The warning is the jitter phase, confirmed. Scaling was never the cause, which is why
two scaling fixes did nothing. The jitter is kept regardless, because the table above
shows it earns its place — so the warning remains, now as a documented choice rather
than an open question.

### Where the divergences are

With one divergence left there is nothing to localise, so this is read off the
**48-divergence** run at `target_accept=0.9` — the configuration that had enough of them
to say where they were. `diagnostics.divergence_locations` compares each parameter's mean
on divergent draws against its mean on the rest, in posterior-sd units, which turns a
count into a name:

| parameter | divergent mean | other mean | shift (sd) |
|---|---|---|---|
| `adstock_alpha[vidtr]` | 0.139 | 0.054 | **+0.96** |
| `saturation_beta[audtr]` | 0.177 | 0.091 | +0.70 |
| `adstock_alpha[dm]` | 0.764 | 0.847 | −0.66 |
| `saturation_lam[sem]` | 0.266 | 0.580 | −0.64 |
| `saturation_beta[vidtr]` | 0.054 | 0.037 | +0.54 |

`vidtr` is at the top — and `vidtr` is the channel `docs/PRIORS.md` singled out as
carrying the tightest, most boundary-pinned prior of the ten, `Beta(0.3, 5.7)`, whose
density is unbounded at zero. A prior with a density spike against a parameter boundary
is a textbook divergence source. Two open items handed to Phase 5 from two different
documents turn out to be the same problem.

### `vidtr`'s decay is entirely prior-driven

`PRIORS.md` asked for a sensitivity check on this specific channel. Refitting with
`vidtr` alone moved to the library default `Beta(1, 3)`, everything else unchanged:

| channel | anchored | relaxed | shift (sd) |
|---|---|---|---|
| **vidtr** | **0.054** | **0.286** | **+2.64** |
| so | 0.828 | 0.816 | −0.10 |
| on | 0.791 | 0.780 | −0.07 |
| auddig | 0.851 | 0.843 | −0.06 |
| inst | 0.163 | 0.155 | −0.06 |
| audtr | 0.456 | 0.444 | −0.06 |
| nsp | 0.432 | 0.440 | +0.04 |
| dm | 0.848 | 0.845 | −0.03 |
| sem | 0.771 | 0.767 | −0.02 |
| viddig | 0.204 | 0.203 | −0.01 |

Every other channel moves less than a tenth of a standard deviation. `vidtr` moves 2.64.
Its posterior was the prior, not the data — the concern `PRIORS.md` raised, confirmed.
Relaxing it also made sampling *worse* (295 divergences), so "just widen the prior" is
not the fix. The honest reading is that the data carries almost no information about
`vidtr`'s carryover.

## The finding that matters: the media/baseline split is not identified

The national model passes R-hat, ESS and E-BFMI. It also says this:

| component | posterior mean | share of sales |
|---|---|---|
| media | 24,304,315,648 | **107.6%** |
| controls | −0 | −0.0% |
| intercept | −1,710,005,663 | −7.6% |
| **sum** | **22,594,309,985** | **100.0%** |
| observed sales | 22,583,339,975 | 100.0% |

Media-attributed share: **107.6%, 94% HDI [62.7%, 167.9%]**, against the ridge project's
**12.4%** on the same data and the same channels.

**This is not a units bug.** The components reconcile to observed total sales to within
0.01%, which was the first thing ruled out —
`tests/test_attribution.py::test_components_sum_to_observed_sales` keeps it ruled out.

### The diagnosis, in three steps

The three steps below were run at `target_accept=0.9`, before #4 settled on 0.99. That
affects the divergence counts they report, not the shares: the final fit at 0.99 gives
107.6% against 105.1% at 0.9, so the conclusion is unchanged and the arithmetic below is
quoted as it was measured.

**Step 1 — the intercept prior is genuinely wrong, and explains about a fifth of it.**
pymc-marketing's default is `Normal(0, 2)`, the row `PRIORS.md` had been calling "not yet
reviewed" since Phase 3. That default assumes a centred target; here the target is
max-scaled sales, strictly positive, averaging 0.31. A prior centred on zero pushes the
baseline toward nothing. Re-centring it on the observed mean:

| | media share | intercept share |
|---|---|---|
| `Normal(0, 2)` (default) | 104.0% | −4.0% |
| `Normal(0.31, 0.15)` | 83.9% | +16.3% |

Real, and not enough.

**Step 2 — the media-coefficient prior moves it further, and still not enough.** With the
intercept centred, varying `saturation_beta ~ HalfNormal(σ)` twentyfold:

| σ | sum of betas | media share | intercept share | divergences |
|---|---|---|---|---|
| 2.0 (default) | 1.081 | 83.9% | 16.3% | 104 |
| 0.5 | 0.934 | 81.7% | 18.4% | 47 |
| 0.1 | 0.533 | 66.5% | 33.6% | 81 |

Ridge's answer needs the ten betas to sum to about 0.1. Even a prior with a mean of 0.08
per channel yields 0.53 — the likelihood is pulling them up, not the prior holding them
there. **My working hypothesis going in was that the prior scale was the cause. The data
says otherwise, and this table is why.**

**Step 3 — the two components are nearly collinear.** The test: if two components are
separately identified, their sum should be no better determined than either part.

| component | mean | sd | CV |
|---|---|---|---|
| media | 2.430e10 | 6.581e9 | 0.271 |
| baseline | −1.710e9 | 6.611e9 | 3.866 |
| **media + baseline** | **2.259e10** | **4.803e8** | **0.021** |

posterior correlation, media vs baseline: **−0.9974**

The sum is determined an order of magnitude more sharply than either part. Saturated,
adstocked, always-positive media is nearly a constant plus noise on 209 weekly
observations, which makes it nearly collinear with a scalar intercept. The likelihood
fixes the total and is close to indifferent about the split; whatever breaks the tie is
the prior.

Ridge gets 12.4% because its L2 penalty breaks the same tie in the other direction. That
is not ridge being right — it is a different arbitrary tiebreak, and it should be read
that way in Phase 9.

`attribution.identification_report` computes this, and
`scripts/report_contributions.py` refuses to present the intervals without it.

### What would actually fix it

Information from outside the time series. Nothing internal to these 209 weeks can
separate media from baseline, so no prior, reparametrisation or sampler setting will do
it honestly — they can only choose the answer.

That makes **Phase 6 (calibration against the geo-DiD estimator) load-bearing rather
than a nice-to-have**, and it is the argument for pymc-marketing's
`add_lift_test_measurements`, which exists for exactly this. The catch is already known:
the ridge project established that *no experiment exists in the geo data* — no division
ever switches a channel off — so what Phase 6 can supply is a validated estimator and an
MDE, not a measured lift. Phase 6 should open by confronting that.

**Convergence diagnostics cannot catch any of this.** A misspecified model can be sampled
perfectly. That is why `tests/test_attribution.py` is a separate suite from
`tests/test_diagnostics.py`: one asks whether the chains explored the posterior, the
other whether the posterior describes the business.

## Per-channel intervals

Produced, with the health warning attached, by `scripts/report_contributions.py`. Shares
of total sales, 94% credible intervals, ridge point estimates alongside:

| channel | mean | q03 | q97 | ridge |
|---|---|---|---|---|
| dm | 17.27% | 0.87% | 51.43% | 0.77% |
| sem | 16.06% | 0.81% | 46.92% | 0.51% |
| so | 14.45% | 0.58% | 42.43% | 0.68% |
| viddig | 12.30% | 0.71% | 34.23% | 3.90% |
| inst | 11.63% | 2.09% | 22.53% | 1.74% |
| on | 10.00% | 0.38% | 28.72% | 0.39% |
| vidtr | 9.82% | 0.35% | 33.79% | 4.11% |
| auddig | 8.26% | 0.39% | 24.59% | 0.34% |
| audtr | 4.23% | 0.14% | 14.39% | 0.00% |
| nsp | 3.61% | 0.10% | 12.36% | 0.00% |

The project's stated purpose was to put credible intervals on these numbers, and it has.
The intervals are wide — most span two orders of magnitude — and that width is the
honest output of a model whose media total is not identified. A narrower interval here
would be a worse result, not a better one.

## Geo model: FAIL, marginally, on R-hat

4 chains × **500** draws (1000 tune), `target_accept=0.99`, 4,849 parameter entries,
21 minutes.

| | value | verdict |
|---|---|---|
| divergences | **0** | ✓ |
| max R-hat | 1.0166 (13 of 4,849 over 1.01) | ✗ |
| min bulk ESS | 422 | ✓ |
| min tail ESS | 496 | ✓ |
| min E-BFMI | 0.868 | ✓ |

**Zero divergences on an 845-parameter hierarchical model, where the 50-parameter flat
one has one.** That is the non-centred parametrisation earning its place, exactly as
Phase 4 predicted when it chose it.

The R-hat failure is marginal and concentrated: 13 entries out of 4,849, worst 1.0166,
and every one of them a non-centred *offset* — `gamma_fourier_offset[V, cos_2]`,
`saturation_beta_raw_offset[U, Organic_Views]`, and similar. Those are the `N(0, 1)`
nuisance variables the hierarchy is built from, not quantities anyone reports. Bulk ESS
sits at 422 against a floor of 400, which says the same thing from the other side: this
is a draw-count limit, not a geometry problem.

**Why 500 draws and not 1000.** A machine constraint, stated as one rather than dressed
up. The geo model's `channel_contribution` and `control_contribution` are
(draws × 113 dates × 26 geos × channels) — roughly 1.7GB of deterministics at 1000
draws, and the run was OOM-killed twice. Restricting `var_names` does not help:
pymc-marketing already samples only the free RVs and then recomputes deterministics in
one vectorised pass, so the peak is structural. Four chains at 500 draws clears the ESS
floor, barely, and the honest reading is that this model has not been run long enough to
clear R-hat on a machine that can hold it.

## The two Phase 4 hand-offs: one refuted, one inconclusive

Both ran at 2 chains × 500 draws, `target_accept=0.9` — paired comparisons, same seed
both arms, asking whether a posterior *moves* rather than whether either arm converged.

### `Google_Impressions`' 0.979 decay is not an unmodelled trend

`HIERARCHY.md` flagged that decay as suspicious: a geometric alpha that close to 1 makes
the adstocked series behave like a slow level, which is what a channel standing in for a
missing trend term looks like. The geo panel has no trend control, so adding one is the
direct test.

| channel | no trend | with trend | shift (sd) |
|---|---|---|---|
| Organic_Views | 0.728 | 0.922 | **+1.42** |
| Google_Impressions | 0.979 | 0.989 | +0.81 |
| Facebook_Impressions | 0.043 | 0.062 | +0.62 |
| Affiliate_Impressions | 0.330 | 0.300 | −0.12 |
| Email_Impressions | 0.010 | 0.010 | +0.04 |
| Paid_Views | 0.314 | 0.309 | −0.02 |

fitted trend coefficient: **−0.146** (sd 0.015)

**The hypothesis is refuted, and refuted in the informative direction.** There is a real
declining trend in the panel — the coefficient is far from zero and tightly estimated —
but modelling it explicitly did not free `Google_Impressions` from its level-like role.
Its decay went *up*, and `Organic_Views` moved a full 1.4 sd the same way.

That is the national model's problem again, in a different dataset: adding one more
slowly-varying regressor to a model that already has several does not resolve which of
them owns the level. It redistributes it. A decay pinned near 1 is a symptom of
competition among near-collinear level-like terms, not of one specific missing control.

### Whether decay varies by division: the data cannot say

`HIERARCHY.md` pooled decay completely across divisions on an a priori argument —
carryover is a property of the medium, not the geography — and flagged that the argument
had never been tested. Testing it means letting decay vary under a hierarchical prior and
reading the fitted across-division scale.

| channel | across-division scale | distance from its own prior mean |
|---|---|---|
| Paid_Views | 0.418 | +0.06 prior sd |
| Affiliate_Impressions | 0.409 | +0.03 prior sd |
| Organic_Views | 0.343 | −0.19 prior sd |
| Email_Impressions | 0.322 | −0.25 prior sd |
| Google_Impressions | 0.302 | −0.32 prior sd |
| Facebook_Impressions | 0.278 | −0.40 prior sd |

The scales look substantial until they are compared against the prior that generated
them. `HalfNormal(0.5)` has a mean of 0.399 and an sd of 0.301; **every posterior scale
sits within 0.40 prior sd of that mean.** The data moved none of them. 113 weeks per
division cannot say whether decay varies by division, so this test does not support
complete pooling and does not challenge it either.

What it does establish is that complete pooling costs nothing in the central estimates:
the completely pooled decays and the means of the partially pooled ones differ by at most
0.046 across all six channels. Complete pooling remains the right default for the
parameter count it saves; the argument for it is still a priori, and now known to be
untestable on this data rather than merely untested.

Reading a posterior scale without checking where its prior sat would have turned "the
data is silent" into "decay varies by division, we measured it". The same check is what
made the `vidtr` result decisive in the other direction.

## Open items carried in from earlier phases

| item | source | status |
|---|---|---|
| `vidtr` prior sensitivity | PRIORS.md | **done** — prior-driven, 2.64 sd |
| intercept prior "not yet reviewed" | PRIORS.md | **done** — wrong, and worth ~20 points of the gap |
| `cores=1` workaround on a multi-core machine | CHALLENGES #1 | **done** — see CHALLENGES #8 |
| overflow `RuntimeWarning` — resolve or confirm benign | CHALLENGES #2 | **done** — it is the jitter phase; jitter kept anyway |
| `target_accept` decision | CHALLENGES #4 | **done** — 0.99, monotone and posterior-neutral |
| `Google_Impressions` decay pinned at 0.979 | HIERARCHY.md | **done** — not a trend; a level-competition symptom |
| complete pooling of decay, argued not tested | HIERARCHY.md | **done** — inconclusive; the data is silent |
| geo model at real chain length | this phase | **done** — 0 divergences, marginal R-hat |
| Gaussian identity link, 4.9% negative prior mass | HIERARCHY.md | **not addressed** — see below |

## What Phase 5 did not do

The `link="log"` question from `HIERARCHY.md` — the geo model's Gaussian identity link
puts 4.9% of its prior predictive on impossible negative sales — is still open. It was
in scope and did not get done, and it is a model-form change rather than a diagnostic, so
it belongs with whichever phase next revisits the geo specification. Recorded here as
skipped rather than quietly dropped from the list.

## Reproducing this

```bash
python scripts/run_full_fits.py                 # both models, cached to data/derived/fits/
python scripts/report_contributions.py          # the intervals, with the health warning
python scripts/sensitivity_checks.py intercept  # and: beta-scale, vidtr, init, accept, trend, decay-pooling
```

Fits are cached as netCDF because every question asked of a fit afterwards should cost a
read rather than another sampling run. `scripts/run_full_fits.py` carries an
`if __name__ == "__main__":` guard, without which parallel sampling on Windows fails —
see `docs/CHALLENGES.md` #8. And see #9 for why any sensitivity check should be made to
prove its two arms actually differ before it is asked what the difference means.
