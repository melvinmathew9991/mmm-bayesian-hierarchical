import pandas as pd

from mmm_bayes.loaders import PAID_CHANNELS, SPEND_COLS, load_national


def test_load_national_shape():
    df = load_national()
    assert len(df) == 209
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_national_matches_ridge_project():
    """The whole point of this repo copying the CSV is that it stays identical to
    the ridge project's own load. If this ever fails, the copy is stale."""
    from mmm.loaders import load_national as load_national_upstream

    local = load_national()
    upstream = load_national_upstream()
    pd.testing.assert_frame_equal(local, upstream, check_exact=True)


def test_spend_columns_present_and_nonnegative():
    df = load_national()
    assert set(SPEND_COLS).issubset(df.columns)
    assert (df[SPEND_COLS] >= 0).all().all()


def test_sales_positive():
    df = load_national()
    assert (df["sales"] > 0).all()


def test_paid_channels_count():
    assert len(PAID_CHANNELS) == 10


def test_control_features_match_the_ridge_projects_own():
    """The port in features.py must stay identical to `mmm.features.build_controls`.

    Phase 1 asserts the DATA matches the ridge project's; nothing asserted that the
    CONTROLS did, even though every claim about "the same baseline specification" rests
    on it. Verified by audit and pinned here: a drift in either project's control set
    would silently make the Bayesian/ridge comparison a comparison of two different
    models.
    """
    import pandas as pd
    from mmm.features import build_controls as ridge_build_controls

    from mmm_bayes.features import build_controls
    from mmm_bayes.loaders import load_national

    df = load_national()
    pd.testing.assert_frame_equal(
        build_controls(df), ridge_build_controls(df), check_exact=True
    )


def test_standardised_controls_are_centred_and_unit_scale():
    from mmm_bayes.features import build_controls, standardize_controls
    from mmm_bayes.loaders import load_national

    controls = standardize_controls(build_controls(load_national()))
    assert abs(controls.mean()).max() < 1e-12
    assert abs(controls.std() - 1.0).max() < 1e-12
    assert not controls.isna().any().any(), "a zero-variance column would divide by zero"
