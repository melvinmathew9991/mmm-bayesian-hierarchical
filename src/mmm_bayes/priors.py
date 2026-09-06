"""Phase 3 -- prior specification, anchored to the ridge project's fitted values.

Phase 2 ran on pymc-marketing's library defaults. This module replaces the two priors
the build plan calls out -- adstock decay and saturation half-point -- with soft,
per-channel priors centred on what the ridge project's own coordinate-ascent search
actually found, and wide enough for 209 weeks of data to move them.

Why only those two. Decay and half-point are transform hyperparameters: notoriously
weakly identified from 209 weekly observations across 10 correlated channels, and
exactly where an uninformative prior leaves the sampler wandering. The media
coefficient is deliberately NOT anchored, even though the ridge fit hands us one per
channel. That coefficient IS the quantity this project exists to estimate
independently -- anchoring it to the ridge estimate would prejudge the comparison
Phase 9 is supposed to make. Its prior stays at the library default (docs/PRIORS.md).

THE UNITS PROBLEM, which is most of the work here
-------------------------------------------------
The ridge half-points cannot be dropped into pymc-marketing as-is. They live in a
different space at three separate points in the pipeline:

1. Saturation shape. Ridge saturates with Michaelis-Menten, ``x / (x + h)``, where
   ``h`` is literally the half-saturation point. Phase 2 used ``LogisticSaturation``,
   whose ``lam`` is an efficiency rate, not a half-point (its half-saturation sits at
   ``ln(3)/lam``). Rather than bury a conversion factor inside a prior, this module
   switches to ``InverseScaledLogisticSaturation`` -- the SAME logistic curve,
   reparametrised so ``lam`` is directly the half-saturation point. That is a
   reparametrisation, not a respecification: the likelihood is unchanged, and the
   prior can now be stated in the units the ridge estimate is already in.

   (The curve SHAPE still differs from ridge's Michaelis-Menten -- logistic
   approaches its asymptote faster. ``MichaelisMentenSaturation`` exists in
   pymc-marketing if a later phase wants exact shape parity; that is a modelling
   change, not a prior change, so it is out of Phase 3's scope. Flagged in
   docs/PRIORS.md rather than silently assumed away.)

2. Adstock normalisation. Ridge adstock is the unnormalised IIR
   ``a_t = s_t + d * a_{t-1}``, whose weights sum to ``1 / (1 - d)``.
   pymc-marketing's ``GeometricAdstock`` defaults to ``normalize=True``, whose weights
   sum to 1. The series the saturation sees is therefore smaller by ``(1 - d)``.

3. Channel scaling. ``build_model`` divides each channel by its own max spend before
   anything else (verified against the model's own ``channel_scale`` variable in
   tests/test_priors.py, not assumed).

Composing the three, with ``L = ADSTOCK_L_MAX``:

    z_half = h * (1 - d) / max_t(spend_t)

Derivation of the cancellation: pymc's normalised L-lag adstock of scaled spend is
``A_L(s) / (scale * S_L)`` with ``S_L = (1 - d**L) / (1 - d)``, while ridge's infinite
adstock relates to the finite one by ``A_L ~= A_inf * (1 - d**L)``. The ``(1 - d**L)``
terms cancel and only ``(1 - d)`` survives. Checked numerically per channel in
tests/test_priors.py: the median ratio of pymc's actual normalised adstock to
``(1 - d) * ridge_adstock / scale`` is within 0.03 of 1.0 for all ten channels. The
residual spread (about +/-20% at the 5th/95th percentiles, and only on the four
channels sitting at d = 0.8) is the L-lag truncation, an order of magnitude tighter
than the prior width below -- so it is absorbed by the prior, not ignored.

Prior widths
------------
Both priors are deliberately soft. The point of anchoring is to put the sampler in
the right neighbourhood, not to pin it there.

* alpha (decay): Beta reparametrised by mean and concentration, mean at the ridge
  decay, concentration 6 -- an sd of about 0.15 at mid-range means. pymc-marketing's
  default Beta(1, 3) has sd 0.19, so this is barely tighter than the library default
  while being centred on evidence rather than on 0.25.
* lam (half-point): LogNormal on the log of the converted half-point, sigma 0.7 --
  roughly a factor of 2 per standard deviation, so the data can move a half-point by
  4x within two sd.

Boundary decays. The ridge search ran a grid over [0.0, 0.8] and 6 of 10 channels
landed on a boundary, which means "at least this extreme", not "exactly this". The
Beta mean is clipped into [0.05, 0.95] to keep the prior proper (a mean of exactly 0
gives Beta(0, k), which is not a distribution), and the width above lets a
boundary-pinned channel move off the boundary in either direction.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, InverseScaledLogisticSaturation

from mmm_bayes.config import NATIONAL_CSV, RIDGE_ANCHORS_JSON
from mmm_bayes.loaders import PAID_CHANNELS, SPEND_COLS, load_national

# Beta pseudo-count for the decay prior. 6 gives sd ~= 0.15 at mid-range means,
# against 0.19 for pymc-marketing's own Beta(1, 3) default -- anchored but not tight.
DECAY_CONCENTRATION = 6.0

# Keeps Beta(mu*k, (1-mu)*k) proper when the ridge grid returned exactly 0.0 or 0.8.
DECAY_MEAN_CLIP = (0.05, 0.95)

# Log-scale sd for the half-point prior: ~2x per sd, ~4x within two.
HALF_POINT_LOG_SIGMA = 0.7


def national_csv_md5() -> str:
    """MD5 of the raw national CSV, used to detect an anchor cache fit on other data."""
    return hashlib.md5(NATIONAL_CSV.read_bytes()).hexdigest()


def load_ridge_anchors(path: Path = RIDGE_ANCHORS_JSON) -> dict:
    """Read the cached ridge anchors, verifying they were fit on the current data.

    Raises rather than falling back to library defaults: a silent fallback would
    produce a model that the docs describe as anchored and that is not, which is
    exactly the kind of gap this project exists to surface rather than paper over.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No ridge anchors at {path}. Generate them with:\n"
            "    python scripts/build_ridge_anchors.py\n"
            "That runs the Phase 1 ridge fit (~90s) and caches its decays/half-points."
        )
    anchors = json.loads(path.read_text(encoding="utf-8"))

    cached_md5 = anchors.get("national_csv_md5")
    current_md5 = national_csv_md5()
    if cached_md5 != current_md5:
        raise ValueError(
            f"Stale ridge anchors at {path}: cached against national_weekly.csv MD5 "
            f"{cached_md5}, but the CSV on disk is {current_md5}. The anchors describe "
            "a different dataset than the model would be fit on. Regenerate with "
            "`python scripts/build_ridge_anchors.py`."
        )

    missing = [c for c in PAID_CHANNELS if c not in anchors["channels"]]
    if missing:
        raise ValueError(f"Ridge anchors are missing channels: {missing}")
    return anchors


def channel_scales(df: pd.DataFrame | None = None) -> pd.Series:
    """Per-channel max spend -- the divisor ``DataDerivedScaling(method="max")`` uses.

    Recomputed here rather than read off a built model, so priors can be constructed
    before the model exists. tests/test_priors.py asserts these match the model's own
    ``channel_scale`` variable element-wise, so the duplication is pinned, not trusted.
    """
    if df is None:
        df = load_national()
    scales = df[SPEND_COLS].max()
    scales.index = PAID_CHANNELS
    return scales


def half_point_to_lam(half_point: float, decay: float, scale: float) -> float:
    """Convert a ridge half-point (unnormalised-adstocked dollars) into the model's
    saturation input units (max-scaled spend, normalised adstock).

    See the module docstring for the derivation: the finite-window terms cancel and
    only ``(1 - decay)`` and the channel scale survive.
    """
    if scale <= 0:
        raise ValueError(
            f"Channel scale must be positive, got {scale}. A zero max spend means the "
            "channel never spent anything in this window -- drop it from PAID_CHANNELS "
            "rather than modelling it."
        )
    return float(half_point) * (1.0 - float(decay)) / float(scale)


def anchor_table(anchors: dict | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-channel table of every quantity feeding a prior, in one inspectable place.

    This is the artefact a reviewer should read first: ridge input, unit conversion and
    resulting prior parameters side by side, so a units error is visible rather than
    buried inside a Prior repr.
    """
    anchors = load_ridge_anchors() if anchors is None else anchors
    scales = channel_scales(df)

    rows = []
    for ch in PAID_CHANNELS:
        a = anchors["channels"][ch]
        decay, half = float(a["decay"]), float(a["half_point"])
        scale = float(scales[ch])
        lam = half_point_to_lam(half, decay, scale)
        mu = float(np.clip(decay, *DECAY_MEAN_CLIP))
        rows.append({
            "channel": ch,
            "ridge_decay": decay,
            "ridge_half_point": half,
            "max_spend": scale,
            "lam_anchor": lam,
            "beta_a": mu * DECAY_CONCENTRATION,
            "beta_b": (1.0 - mu) * DECAY_CONCENTRATION,
            "lognormal_mu": float(np.log(lam)),
        })
    return pd.DataFrame(rows).set_index("channel")


def _by_channel(values: np.ndarray) -> xr.DataArray:
    """Label a per-channel parameter vector with the channel coordinate.

    Passing a bare array works but makes the prior positional: it would line up with
    whatever order the model happens to put `channel` in. Labelling it means xarray
    aligns by channel name, so a reordering of PAID_CHANNELS can no longer silently
    hand `dm`'s half-point to `sem`. (It also drops pymc-extras' "implicit conversion
    of array-like parameter" warning, which is that same hazard being pointed out.)
    """
    return xr.DataArray(values, dims="channel", coords={"channel": list(PAID_CHANNELS)})


def adstock_alpha_prior(table: pd.DataFrame) -> Prior:
    """Per-channel Beta prior on the geometric decay, centred on the ridge decay."""
    return Prior(
        "Beta",
        alpha=_by_channel(table["beta_a"].to_numpy()),
        beta=_by_channel(table["beta_b"].to_numpy()),
        dims="channel",
    )


def saturation_lam_prior(table: pd.DataFrame) -> Prior:
    """Per-channel LogNormal prior on the half-saturation point.

    ``InverseScaledLogisticSaturation`` defines ``lam`` AS the half-saturation point,
    so this prior is stated directly in the converted ridge units, with no further
    conversion factor hidden inside it.
    """
    return Prior(
        "LogNormal",
        mu=_by_channel(table["lognormal_mu"].to_numpy()),
        sigma=HALF_POINT_LOG_SIGMA,
        dims="channel",
    )


def anchored_transforms(
    l_max: int, table: pd.DataFrame | None = None
) -> tuple[GeometricAdstock, InverseScaledLogisticSaturation]:
    """The (adstock, saturation) pair carrying the anchored priors.

    ``beta`` (the media coefficient) is left at InverseScaledLogisticSaturation's own
    default HalfNormal -- deliberately unanchored, see the module docstring.
    """
    table = anchor_table() if table is None else table
    return (
        GeometricAdstock(l_max=l_max, priors={"alpha": adstock_alpha_prior(table)}),
        InverseScaledLogisticSaturation(priors={"lam": saturation_lam_prior(table)}),
    )


if __name__ == "__main__":
    pd.set_option("display.width", 150)
    _anchors = load_ridge_anchors()
    _table = anchor_table(_anchors)
    print("=" * 96)
    print("PHASE 3 -- ANCHORED PRIORS (ridge estimates translated into model units)")
    print("=" * 96)
    print(f"Anchors generated {_anchors['generated']}, "
          f"national CSV MD5 {_anchors['national_csv_md5'][:12]}...")
    print()
    print(_table.to_string(float_format=lambda v: f"{v:,.4f}"))
    print()
    # The Prior objects hold labelled DataArrays, whose repr is several screens long.
    # The columns above already carry every number in them, so print the shape of the
    # priors rather than their contents.
    print(f"alpha (decay)     : Beta(beta_a, beta_b) per channel, concentration "
          f"{DECAY_CONCENTRATION:g}, mean clipped to {DECAY_MEAN_CLIP}")
    print(f"lam (half-point)  : LogNormal(lognormal_mu, {HALF_POINT_LOG_SIGMA}) per "
          f"channel -- 90% band is the anchor x [{np.exp(-1.645 * HALF_POINT_LOG_SIGMA):.2f}, "
          f"{np.exp(1.645 * HALF_POINT_LOG_SIGMA):.2f}]")
    print("beta (media coef) : library default, deliberately NOT anchored -- it is the "
          "quantity Phase 9 compares.")

    # Constructed anyway, so a broken prior spec fails here rather than at model build.
    adstock_alpha_prior(_table)
    saturation_lam_prior(_table)
