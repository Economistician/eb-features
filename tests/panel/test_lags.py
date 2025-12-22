from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_features.panel.lags import add_lag_features


def _make_panel_df(n_entities: int = 2, n_steps: int = 6) -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2025-01-01 00:00:00")
    for e in range(n_entities):
        for t in range(n_steps):
            rows.append(
                {
                    "entity_id": e,
                    "timestamp": base_ts + pd.Timedelta(hours=t),
                    "target": float(100 * e + t),
                }
            )
    return pd.DataFrame(rows)


def test_add_lag_features_adds_expected_columns_and_values() -> None:
    df = _make_panel_df(n_entities=2, n_steps=6)

    df_out, cols = add_lag_features(
        df,
        entity_col="entity_id",
        target_col="target",
        lag_steps=[1, 2],
    )

    assert cols == ["lag_1", "lag_2"]
    assert all(c in df_out.columns for c in cols)

    # Validate values within each entity
    for e in df["entity_id"].unique():
        sub = df_out[df_out["entity_id"] == e].sort_values("timestamp")
        y = sub["target"].to_numpy()

        lag1 = sub["lag_1"].to_numpy()
        lag2 = sub["lag_2"].to_numpy()

        # First row has no history
        assert np.isnan(lag1[0])
        assert np.isnan(lag2[0])

        # Second row has lag_1 but not lag_2
        assert lag1[1] == y[0]
        assert np.isnan(lag2[1])

        # Thereafter both defined
        assert lag1[2] == y[1]
        assert lag2[2] == y[0]
        assert lag1[5] == y[4]
        assert lag2[5] == y[3]


def test_add_lag_features_empty_or_none_returns_copy_no_cols() -> None:
    df = _make_panel_df(n_entities=1, n_steps=4)

    out1, cols1 = add_lag_features(
        df, entity_col="entity_id", target_col="target", lag_steps=None
    )
    assert cols1 == []
    assert out1 is not df
    assert set(out1.columns) == set(df.columns)

    out2, cols2 = add_lag_features(
        df, entity_col="entity_id", target_col="target", lag_steps=[]
    )
    assert cols2 == []
    assert out2 is not df
    assert set(out2.columns) == set(df.columns)


def test_add_lag_features_requires_columns() -> None:
    df = pd.DataFrame({"entity_id": [0, 0, 0], "target": [1.0, 2.0, 3.0]})

    with pytest.raises(KeyError, match="Entity column|entity"):
        add_lag_features(df, entity_col="missing", target_col="target", lag_steps=[1])

    with pytest.raises(KeyError, match="Target column|target"):
        add_lag_features(df, entity_col="entity_id", target_col="missing", lag_steps=[1])


def test_add_lag_features_raises_on_non_positive_lag() -> None:
    df = _make_panel_df(n_entities=1, n_steps=4)

    with pytest.raises(ValueError, match="positive|Lag"):
        add_lag_features(df, entity_col="entity_id", target_col="target", lag_steps=[0])

    with pytest.raises(ValueError, match="positive|Lag"):
        add_lag_features(df, entity_col="entity_id", target_col="target", lag_steps=[-1])


def test_add_lag_features_does_not_cross_entity_boundaries() -> None:
    # Construct data where targets would be obviously wrong if leakage across entity occurred.
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0, 1, 1, 1],
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "target": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
        }
    )

    df_out, cols = add_lag_features(
        df, entity_col="entity_id", target_col="target", lag_steps=[1]
    )
    assert cols == ["lag_1"]

    sub0 = df_out[df_out["entity_id"] == 0].sort_values("timestamp")
    sub1 = df_out[df_out["entity_id"] == 1].sort_values("timestamp")

    assert np.isnan(sub0["lag_1"].iloc[0])
    assert sub0["lag_1"].iloc[1] == 0.0
    assert sub0["lag_1"].iloc[2] == 1.0

    assert np.isnan(sub1["lag_1"].iloc[0])
    assert sub1["lag_1"].iloc[1] == 100.0
    assert sub1["lag_1"].iloc[2] == 101.0