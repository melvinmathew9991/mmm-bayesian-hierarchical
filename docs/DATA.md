# Datasets

This project models **real** marketing data. Two datasets are used, because no single
public dataset supports every component of the pipeline. Each is used only for what it
can actually support.

| | `raw/national_weekly.csv` | `raw/geo_divisions.csv` |
|---|---|---|
| Grain | 1 market x weekly | 26 divisions x weekly |
| Span | 2014-08-03 - 2018-07-29 (209 wks) | 2018-01-06 - 2020-02-29 (113 wks) |
| Rows | 209 | 3,051 |
| **Dollar spend** | **yes** (10 channels) | **no** (impressions/views only) |
| Impressions | yes (13 channels) | yes (6 channels) |
| Controls | store count, macro, 22 holidays, 19 seasonality, markdown | none |
| **Geo dimension** | **no** | **yes** (26 divisions) |
| Used for | naive model, MMM fit, budget optimizer | geo-holdout / DiD validation |

The split is forced: the national set is the only one with dollar spend (so it is the
only one that can drive a budget optimizer), and the division set is the only one with
a geo dimension (so it is the only one that can support a holdout experiment).

## Sources

**`raw/national_weekly.csv`**
- From <https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv>
- Retrieved 2026-08-08. MD5 `fc003779f05e12cbd4b633451167c61d`.
- Repo code is MIT licensed. **The repo does not document the data's provenance** and
  never states whether it is real, anonymised, or simulated. See caveat below.

**`raw/geo_divisions.csv`** -- NOT committed to this repository. Kaggle publishes it
under "Data files (c) Original Authors", which grants no redistribution rights. Fetch it
with the ridge project's `python scripts/fetch_data.py` (this repo has no fetch script of
its own). On this machine it is already downloaded there, at
`../mmm-marketing-project/data/raw/geo_divisions.csv`.
- From <https://www.kaggle.com/datasets/yugagrawal95/sample-media-spends-data>
  (served unauthenticated via `https://www.kaggle.com/api/v1/datasets/download/yugagrawal95/sample-media-spends-data`)
- Retrieved 2026-08-08. MD5 `5254f9360154e3ea5e176879bacc673f`.
- Described by its publisher as weekly division-level media data from a large company.

### Licensing and redistribution

Neither dataset is covered by this repository's terms; both are third-party.

- **`raw/national_weekly.csv`** is redistributed here from
  <https://github.com/sibylhe/mmm_stan>, which is MIT licensed. The MIT notice and
  copyright of that project travel with this file; it is included for reproducibility and
  no ownership is claimed over it.
- **`raw/geo_divisions.csv`** is **not** redistributed. Kaggle publishes it under
  *"Data files © Original Authors"*, which grants no redistribution rights, so this
  project links to the source and fetches it on demand instead.

Check the terms at both sources before reusing either dataset, and particularly before
using either commercially.

### Provenance — what was actually verified

Neither publisher states whether their data is real, anonymised or simulated. Rather than
leaving that as an open worry, the datasets were tested directly. The conclusion is that
**both are anchored in real-world data, and neither shows a fabrication signature** — with
one specific limit stated at the end.

**1. Macro columns match real published series.** The national set carries a consumer
sentiment and a gas price column. Both were compared against the authoritative series on
FRED for the exact dates in the file:

| Dataset column | Reference series | Correlation | Mean abs. difference |
|---|---|---|---|
| `me_gas_dpg` | FRED `GASREGW` (US regular retail gasoline) | **0.9967** | $0.10/gal |
| `me_ics_all` | FRED `UMCSENT` (U. Michigan sentiment) | **0.9417** | 1.01 index points |

The gas series reproduces the 2014-15 oil collapse ($3.60 -> $2.31) and the early-2016
trough ($1.87) week by week. The small constant gas offset is consistent with a regional
weighting rather than the national average. `UMCSENT` is monthly, so weekly interpolation
accounts for most of the sentiment residual.

**2. Moving holidays land on the real calendar.** A fixed seasonal bump is trivial to
fake; Easter is not, since it moves by up to three weeks a year. All four Easter flags are
exact — 2015-04-05, 2016-03-27, 2017-04-16, 2018-04-01 — as are all four Black Friday
weeks.

**3. Benford's Law.** National media spend spans 6.3 orders of magnitude, so the test
applies: chi-square 11.4 against a 15.5 critical value at 5%. Consistent.

**4. Digit fingerprints.** 98.8% of spend values carry non-zero cents, and terminal
digits are uniform (chi-square p = 0.62). Fabricated financial figures cluster on round
numbers.

**5. Operational messiness.** Channels go dark in ragged patterns (`so` 7 weeks, `nsp` 6,
`vidtr` 3). The duplicated Division Z block (defect #2 below) is itself weak evidence
*for* real provenance — generators do not ship stacked duplicates.

**6. Panel noise structure (geo set).** Smaller divisions are proportionally noisier
(r = -0.33 between size and residual sd), residuals are decidedly non-normal
(Shapiro p = 3e-27), and residual cross-correlation sits 2.6 standard deviations *above*
an independent null, meaning divisions retain shared shocks the way real regions do. None
of that is what a shares-times-one-curve generator produces.

Two tests were tried and **discarded as inapplicable**, recorded here so they are not
repeated: Benford on the geo set (its 2.4 orders of magnitude is below the ~3 the test
needs, and it returns misleading rejections), and any reading of the residual correlation
sign that does not first subtract the -1/(N-1) artifact introduced by removing the
cross-sectional mean.

**The limit that remains.** These tests establish real-world anchoring. They do *not*
establish that the sales and spend figures are an unmodified extract from a specific
company. Real macro columns and a real calendar can be attached to rescaled or otherwise
anonymised business figures, and anonymisation is standard practice that would preserve
essentially every signature above — multiplying all sales by a constant leaves Benford,
digit uniformity and every correlation untouched. So: **the data is real in origin and
plausibly real in magnitude, but the levels may be rescaled.** Ratios, elasticities and
relative findings are safe; absolute dollar figures should not be quoted as a specific
company's revenue.

Remaining oddity: `Email_Impressions` in the geo set is non-integer on 100% of rows while
every other impression column is a clean integer, suggesting that column was derived or
rescaled rather than measured directly.

## Column reference — national

- `mdip_*` — media **impressions**, 13 channels
- `mdsp_*` — media **spend** in dollars, 10 channels
  (`dm` direct mail, `inst` insert, `nsp` newspaper, `auddig` digital audio,
  `audtr` radio, `vidtr` TV, `viddig` digital video, `so` social, `on` online display,
  `sem` search)
- `em`, `sms`, `aff` have impressions but **no spend** — treat as organic channels, not
  as paid channels in a budget optimiser
- `sales` — weekly revenue
- `st_ct` — store count; drops 16% across the window. Worth controlling for, but it
  explains far less than its size suggests: alone with a trend it reaches only R² 0.016,
  against 0.660 once seasonality is added. Seasonality is the dominant baseline driver.
- `me_ics_all`, `me_gas_dpg` — consumer sentiment, gas price
- `mrkdn_*`, `va_pub_*` — markdown / promotional depth
- `hldy_*` (22), `seas_*` (19) — holiday and seasonality indicators

## Column reference — geo

`Division` (A-Z), `Calendar_Week`, `Paid_Views`, `Organic_Views`, `Google_Impressions`,
`Email_Impressions`, `Facebook_Impressions`, `Affiliate_Impressions`, `Overall_Views`,
`Sales`.

## Known defects (handled in `mmm/loaders.py`, do not rediscover these)

1. **`Overall_Views` is a sum identity.** It equals `Paid_Views + Organic_Views` to
   within a rounding residual. Including all three makes the design matrix rank
   deficient and inflates VIF to absurd values: 10,965 pooled across all divisions, and
   up to 112,882 when a single division is fitted on its own. Dropping `Overall_Views`
   brings VIF down to 3.1 pooled (2.3-2.7 per division). It is dropped at load time. Any
   "severe multicollinearity" finding that survives only because this column is
   present is an artifact, not a result.

2. **Division `Z` is duplicated.** It has 226 rows — exactly two stacked copies of the
   same 113 weeks. Deduplicated at load time. Every other division has 113 rows.

3. **No dollar spend in the geo set.** The columns are impressions and views despite
   the dataset's title. Budget optimisation in dollars is not possible here; do not
   invent CPMs to manufacture spend.

## Finding that changes the project's thesis

The original synthetic generator produced pairwise channel-spend correlations of
0.90-0.94 and VIF 8-13, and the project's narrative was built on that "textbook
multicollinearity failure".

**Neither real dataset reproduces this.** After removing the sum-identity column (all
figures pooled across the full sample, so the three rows are comparable):

| | corr (max) | VIF (max) | VIF > 10 |
|---|---|---|---|
| Synthetic (old) | 0.94 | 13.0 | 4 of 5 |
| National (real) | 0.58 | 3.9 | 0 of 10 |
| Geo (real) | 0.77 | 3.1 | 0 of 6 |

Naive OLS still fails on real data — on the national set R^2 is 0.575 with 2 of 10
channels wrong-signed and 6 of 10 insignificant — but the cause is **omitted variables**
(no trend, no store count, no seasonality, no adstock/saturation), not channels
competing for credit. The project's headline claim has to be rewritten around that.
