"""Generate data/derived/ridge_anchors.json -- the Phase 3 prior anchors.

Runs the Phase 1 ridge fit and caches the two per-channel quantities Phase 3 anchors
priors to: the fitted geometric decay and the fitted saturation half-point.

Why cache at all. The ridge fit is a coordinate-ascent search over decay and
half-point grids and takes ~90s. Re-running it every time `mmm_bayes.priors` builds a
Prior would make every model build, every test and every notebook import pay for it.

Why cache to a file in the repo rather than hardcode the numbers in priors.py.
Hardcoded constants lose their provenance the moment they are pasted: nothing then
records which ridge run or which dataset produced them. The JSON keeps the fit's own
metadata alongside the values, including an MD5 of the national CSV, and
`priors.load_ridge_anchors` refuses to use the cache if that MD5 no longer matches --
so anchors fit on stale data raise rather than quietly biasing a Bayesian posterior.

The ridge coefficients and contributions are cached too. Phase 3 does NOT use them
(anchoring the media coefficient would prejudge the very comparison Phase 9 makes --
see priors.py), but Phase 9 needs exactly these numbers from exactly this fit, and
capturing them here means that comparison runs against a recorded fit rather than a
re-run that might land elsewhere.

Usage:
    python scripts/build_ridge_anchors.py
"""
import json
from datetime import datetime, timezone

from mmm_bayes.baseline_replication import replicate_ridge_baseline, summarize
from mmm_bayes.config import DERIVED_DATA_DIR, RIDGE_ANCHORS_JSON
from mmm_bayes.loaders import PAID_CHANNELS, load_national
from mmm_bayes.priors import national_csv_md5


def build() -> dict:
    df = load_national()
    result = replicate_ridge_baseline(verbose=False)
    table = summarize(result)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": (
            "mmm_bayes.baseline_replication.replicate_ridge_baseline -- the ridge "
            "project's own run_mmm, refit in this repo on this repo's copy of the data"
        ),
        "national_csv_md5": national_csv_md5(),
        "n_weeks": len(df),
        "week_range": [str(df.index.min().date()), str(df.index.max().date())],
        "alpha": float(result["alpha"]),
        "r2_in_sample": float(result["r2_in_sample"]),
        "channels": {
            ch: {
                "decay": float(table.loc[ch, "decay"]),
                "half_point": float(table.loc[ch, "half_point"]),
                # Recorded for Phase 9, NOT used for anchoring -- see module docstring.
                "ridge_coef": float(table.loc[ch, "ridge_coef"]),
                "ridge_contribution": float(table.loc[ch, "ridge_contribution"]),
            }
            for ch in PAID_CHANNELS
        },
    }


if __name__ == "__main__":
    print("Running the Phase 1 ridge fit to extract prior anchors (~90s)...")
    anchors = build()

    DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RIDGE_ANCHORS_JSON.write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {RIDGE_ANCHORS_JSON}")
    print(f"  national CSV MD5 {anchors['national_csv_md5']}")
    print(f"  {anchors['n_weeks']} weeks, {anchors['week_range'][0]} -> {anchors['week_range'][1]}")
    print(f"  ridge alpha {anchors['alpha']:.4g}, in-sample R^2 {anchors['r2_in_sample']:.3f}")
    print("\n  channel    decay       half_point")
    for ch, v in anchors["channels"].items():
        print(f"  {ch:9s} {v['decay']:5.2f} {v['half_point']:16,.2f}")
