from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_features.panel.engineering import FeatureConfig, FeatureEngineer


def _make_panel_df(n_entities: int = 2, n_steps: int = 10) -> pd.DataFrame:
    """
    Create a small, deterministic panel DataFrame for unit tests.

    The DataFrame includes:
    - entity_id: integer entity key
    - timestamp: hourly timestamps
    - target: non-negative numeric target
    - reg_num: numeric regressor passthrough
    - static_cat: non-numeric static metadata passthrough (will be encoded)
    """
    rows = []
    base_ts = pd.Timestamp("2025-01-01 00:00:00")
    for e in range(n_entities):
        for t in range(n_steps):
            rows.append(
                {
                    "entity_id": e,
                    "timestamp": base_ts + pd.Timedelta(hours=t),
                    "target": float(10 * e + t),  # deterministic, non-negative
                    "reg_num": float(100 + 10 * e + t),
                    "static_cat": "A" if e % 2 == 0 else "B",
                }
            )
    return pd.DataFrame(rows)


def test_engineer_transform_shapes_and_feature_names() -> None:
    df = _make_panel_df(n_entities=2, n_steps=12)

    config = FeatureConfig(
        lag_steps=[1, 2],
        rolling_windows=[3],
        rolling_stats=["mean"],
        calendar_features=["hour", "dow", "is_weekend"],
        use_cyclical_time=True,
        regressor_cols=["reg_num"],
        static_cols=["static_cat"],
        dropna=True,
        leakage_safe_rolling=True,
    )

    eng = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")
    X, y, names = eng.transform(df, config)

    # Expected feature names (order matters: lags -> rolling -> calendar (incl cyclical) -> passthrough)
    expected = [
        "lag_1",
        "lag_2",
        "roll_3_mean",
        "hour",
        "dayofweek",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "static_cat",
        "reg_num",
    ]
    assert names == expected

    # With dropna=True, lag_2 needs t>=2; leakage-safe roll(3) uses shift(1) and needs 3 prior points,
    # so first fully-defined roll value occurs at t>=3. Combined => earliest kept is t>=3 per entity.
    expected_rows = 2 * (12 - 3)
    assert X.shape == (expected_rows, len(expected))
    assert y.shape == (expected_rows,)

    # Basic sanity: finite values only
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_engineer_requires_monotonic_timestamps_within_entity() -> None:
    df = _make_panel_df(n_entities=1, n_steps=6)

    # Break monotonicity (swap two timestamps)
    df.loc[2, "timestamp"], df.loc[3, "timestamp"] = (
        df.loc[3, "timestamp"],
        df.loc[2, "timestamp"],
    )

    config = FeatureConfig(lag_steps=[1], rolling_windows=None, calendar_features=["hour"])
    eng = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")

    with pytest.raises(ValueError, match="strictly increasing|monotonic|Timestamps"):
        eng.transform(df, config)


def test_engineer_raises_on_negative_target() -> None:
    df = _make_panel_df(n_entities=1, n_steps=6)
    df.loc[4, "target"] = -1.0

    config = FeatureConfig(lag_steps=[1], rolling_windows=None, calendar_features=["hour"])
    eng = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")

    with pytest.raises(ValueError, match="Negative values found in target"):
        eng.transform(df, config)


def test_engineer_dropna_false_drops_rows_missing_engineered_history() -> None:
    """
    When dropna=False, FeatureEngineer should still require engineered features (lags/rolling/calendar)
    to be present, but it may allow passthrough columns to contain NaNs for upstream imputation.

    This test verifies that rows without engineered history are removed (so X remains finite),
    and that we keep the remaining rows.
    """
    df = _make_panel_df(n_entities=1, n_steps=6)

    config = FeatureConfig(
        lag_steps=[2],
        rolling_windows=[3],
        rolling_stats=["mean"],
        calendar_features=["hour"],
        use_cyclical_time=False,
        regressor_cols=["reg_num"],
        static_cols=["static_cat"],
        dropna=False,
        leakage_safe_rolling=True,
    )

    eng = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")
    X, y, names = eng.transform(df, config)

    # Earliest fully-defined engineered features:
    # - lag_2 defined at t>=2
    # - leakage-safe roll_3_mean defined at t>=3
    # Combined => keep t>=3 => 6 - 3 = 3 rows for this single entity
    assert X.shape == (3, len(names))
    assert y.shape == (3,)

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert names[0] == "lag_2"
    assert "roll_3_mean" in names
    assert "hour" in names


def test_engineer_regressor_autodetect_excludes_core_and_static() -> None:
    df = _make_panel_df(n_entities=1, n_steps=8)

    # Add another numeric column that should be auto-detected
    df["extra_num"] = np.arange(len(df), dtype=float)

    config = FeatureConfig(
        lag_steps=[1],
        rolling_windows=None,
        calendar_features=["hour"],
        regressor_cols=None,  # auto-detect
        static_cols=["static_cat"],  # exclude from regressors
        dropna=True,
    )

    eng = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")
    X, y, names = eng.transform(df, config)

    # Must include reg_num and extra_num, exclude entity/timestamp/target as features
    assert "reg_num" in names
    assert "extra_num" in names

    # static_cat is explicitly requested, so it should be present
    assert "static_cat" in names

    assert "entity_id" not in names
    assert "timestamp" not in names
    assert "target" not in names

    # Shape sanity
    assert X.shape[1] == len(names)
    assert X.shape[0] == y.shape[0]