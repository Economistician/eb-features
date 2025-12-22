from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_features.panel.calendar import add_calendar_features


def _make_df(
    *,
    n: int = 48,
    start: str = "2025-01-01 00:00:00",
    freq: str = "h",
) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=n, freq=freq)
    return pd.DataFrame({"timestamp": ts})


def test_add_calendar_features_adds_expected_base_columns_and_names() -> None:
    df = _make_df(n=10)

    df_out, feature_cols, calendar_cols = add_calendar_features(
        df,
        timestamp_col="timestamp",
        calendar_features=["hour", "dow", "dom", "month", "is_weekend"],
        use_cyclical_time=False,
    )

    expected_calendar_cols = ["hour", "dayofweek", "dayofmonth", "month", "is_weekend"]
    assert calendar_cols == expected_calendar_cols
    assert feature_cols == expected_calendar_cols

    # Ensure columns exist
    for c in expected_calendar_cols:
        assert c in df_out.columns

    # Ensure base columns are integer-ish dtypes
    assert pd.api.types.is_integer_dtype(df_out["hour"])
    assert pd.api.types.is_integer_dtype(df_out["dayofweek"])
    assert pd.api.types.is_integer_dtype(df_out["dayofmonth"])
    assert pd.api.types.is_integer_dtype(df_out["month"])
    assert pd.api.types.is_integer_dtype(df_out["is_weekend"])


def test_add_calendar_features_with_cyclical_time_adds_sin_cos_when_base_present() -> None:
    # Two days to ensure dow changes, many hours to ensure hour changes
    df = _make_df(n=48)

    df_out, feature_cols, calendar_cols = add_calendar_features(
        df,
        timestamp_col="timestamp",
        calendar_features=["hour", "dow"],
        use_cyclical_time=True,
    )

    assert calendar_cols == ["hour", "dayofweek"]

    # feature_cols should include base + cyclical in deterministic order
    expected = ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    assert feature_cols == expected

    for c in expected:
        assert c in df_out.columns

    # Range sanity
    assert df_out["hour"].between(0, 23).all()
    assert df_out["dayofweek"].between(0, 6).all()

    # Cyclical outputs are finite and within [-1, 1] (allow tiny float jitter)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert np.isfinite(df_out[c].to_numpy()).all()
        assert (df_out[c].abs() <= 1.0000001).all()


def test_add_calendar_features_does_not_add_cyclical_if_base_missing() -> None:
    df = _make_df(n=24)

    # Request only month; cyclical features should not be added
    df_out, feature_cols, calendar_cols = add_calendar_features(
        df,
        timestamp_col="timestamp",
        calendar_features=["month"],
        use_cyclical_time=True,
    )

    assert calendar_cols == ["month"]
    assert feature_cols == ["month"]
    assert "hour_sin" not in df_out.columns
    assert "dow_sin" not in df_out.columns


def test_add_calendar_features_is_weekend_correctness_known_dates() -> None:
    # 2025-01-04 is Saturday, 2025-01-05 is Sunday, 2025-01-06 is Monday
    ts = pd.to_datetime(
        ["2025-01-04 12:00:00", "2025-01-05 12:00:00", "2025-01-06 12:00:00"]
    )
    df = pd.DataFrame({"timestamp": ts})

    df_out, feature_cols, calendar_cols = add_calendar_features(
        df,
        timestamp_col="timestamp",
        calendar_features=["dow", "is_weekend"],
        use_cyclical_time=False,
    )

    assert calendar_cols == ["dayofweek", "is_weekend"]
    assert feature_cols == ["dayofweek", "is_weekend"]

    # pandas dayofweek: Mon=0 ... Sun=6
    assert df_out["dayofweek"].tolist() == [5, 6, 0]
    assert df_out["is_weekend"].tolist() == [1, 1, 0]


def test_add_calendar_features_raises_on_missing_timestamp_col() -> None:
    df = pd.DataFrame({"not_timestamp": [pd.Timestamp("2025-01-01")]})

    with pytest.raises(KeyError, match="Timestamp column .* not found"):
        add_calendar_features(
            df,
            timestamp_col="timestamp",
            calendar_features=["hour"],
            use_cyclical_time=False,
        )


def test_add_calendar_features_raises_on_invalid_feature_key() -> None:
    df = _make_df(n=5)

    with pytest.raises(ValueError, match="Unsupported calendar feature"):
        add_calendar_features(
            df,
            timestamp_col="timestamp",
            calendar_features=["hour", "bad_key"],
            use_cyclical_time=False,
        )


def test_add_calendar_features_accepts_timezone_aware_timestamps() -> None:
    # Timezone-aware timestamps should work; values reflect localized representation in the series.
    ts = pd.date_range("2025-01-01 00:00:00", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})

    df_out, feature_cols, calendar_cols = add_calendar_features(
        df,
        timestamp_col="timestamp",
        calendar_features=["hour", "dow"],
        use_cyclical_time=True,
    )

    assert calendar_cols == ["hour", "dayofweek"]
    for c in ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert c in df_out.columns
    assert df_out["hour"].between(0, 23).all()
    assert df_out["dayofweek"].between(0, 6).all()
    assert np.isfinite(df_out[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].to_numpy()).all()