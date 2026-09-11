import pandas as pd
import pytest

from src.fraudshield.features.pipeline import FraudFeatureTransformer, PCA_COLS


def make_df(n=5):
    data = {c: [float(i) for i in range(n)] for c in PCA_COLS}
    data["Time"] = [0.0, 3600.0, 7200.0, 90000.0, 172000.0]
    data["Amount"] = [10.0, 0.0, 500.5, 25.99, 3.0]
    return pd.DataFrame(data)


def test_fit_transform_shape():
    df = make_df()
    feat = FraudFeatureTransformer().fit(df)
    out = feat.transform(df)
    assert len(out) == len(df)
    assert set(feat.feature_names_) == set(out.columns)


def test_amount_log1p_nonnegative_and_monotonic():
    df = make_df()
    feat = FraudFeatureTransformer().fit(df)
    out = feat.transform(df)
    assert (out["amount_log1p"] >= 0).all()
    # amount_log1p must be monotonic with raw Amount
    assert out["amount_log1p"].is_monotonic_increasing == (df["Amount"].is_monotonic_increasing)


def test_hour_of_day_range():
    df = make_df()
    feat = FraudFeatureTransformer().fit(df)
    out = feat.transform(df)
    assert out["hour_of_day"].between(0, 23).all()


def test_zscore_fit_on_train_only_not_leaked():
    """Scaler statistics must come from fit() data, not transform() data."""
    train = make_df()
    feat = FraudFeatureTransformer().fit(train)
    mean_, scale_ = feat.amount_scaler.mean_[0], feat.amount_scaler.scale_[0]

    other = make_df()
    other["Amount"] = [999.0, 999.0, 999.0, 999.0, 999.0]
    feat.transform(other)  # should not refit
    assert feat.amount_scaler.mean_[0] == mean_
    assert feat.amount_scaler.scale_[0] == scale_


def test_missing_required_column_raises():
    df = make_df().drop(columns=["V1"])
    feat = FraudFeatureTransformer()
    with pytest.raises(KeyError):
        feat.fit(df).transform(df)
