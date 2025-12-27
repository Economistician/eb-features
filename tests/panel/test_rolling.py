from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_features.panel.rolling import add_rolling_features


def _make_panel_df(n_entities: int = 2, n_steps: int = 8) -> pd.DataFrame:
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


def test_add_rolling_features_adds_expected_columns_and_values_leakage_safe() -> None:
    df = _make_panel_df(n_entities=1, n_steps=8)

    df_out, cols = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[3],
        rolling_stats=["mean", "sum", "min", "max", "median"],
        min_periods=None,  # defaults to w
        leakage_safe=True,
    )

    expected_cols = [
        "roll_3_mean",
        "roll_3_sum",
        "roll_3_min",
        "roll_3_max",
        "roll_3_median",
    ]
    assert cols == expected_cols
    assert all(c in df_out.columns for c in expected_cols)

    # For leakage_safe=True:
    # rolling computed on target.shift(1), window=3, min_periods=3.
    # So first defined at t=3 (0-indexed): mean(target[0],target[1],target[2])
    sub = df_out.sort_values("timestamp").reset_index(drop=True)
    y = sub["target"].to_numpy()

    for i in range(3):
        assert np.isnan(sub.loc[i, "roll_3_mean"])

    t = 3
    window_vals = np.array([y[0], y[1], y[2]], dtype=float)
    assert sub.loc[t, "roll_3_mean"] == pytest.approx(window_vals.mean())
    assert sub.loc[t, "roll_3_sum"] == pytest.approx(window_vals.sum())
    assert sub.loc[t, "roll_3_min"] == pytest.approx(window_vals.min())
    assert sub.loc[t, "roll_3_max"] == pytest.approx(window_vals.max())
    assert sub.loc[t, "roll_3_median"] == pytest.approx(np.median(window_vals))

    # Next point t=4 uses [y1,y2,y3]
    t = 4
    window_vals = np.array([y[1], y[2], y[3]], dtype=float)
    assert sub.loc[t, "roll_3_mean"] == pytest.approx(window_vals.mean())


def test_add_rolling_features_leakage_unsafe_includes_current_value() -> None:
    df = _make_panel_df(n_entities=1, n_steps=6)

    df_out_safe, _ = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[3],
        rolling_stats=["mean"],
        leakage_safe=True,
    )
    df_out_unsafe, _ = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[3],
        rolling_stats=["mean"],
        leakage_safe=False,
    )

    safe = df_out_safe.sort_values("timestamp").reset_index(drop=True)
    unsafe = df_out_unsafe.sort_values("timestamp").reset_index(drop=True)
    y = df["target"].to_numpy()

    # At t=2:
    # unsafe roll_3_mean exists (window includes y0,y1,y2), safe should still be NaN
    assert np.isnan(safe.loc[2, "roll_3_mean"])
    assert unsafe.loc[2, "roll_3_mean"] == pytest.approx(np.mean([y[0], y[1], y[2]]))

    # At t=3:
    # safe uses y0,y1,y2; unsafe uses y1,y2,y3
    assert safe.loc[3, "roll_3_mean"] == pytest.approx(np.mean([y[0], y[1], y[2]]))
    assert unsafe.loc[3, "roll_3_mean"] == pytest.approx(np.mean([y[1], y[2], y[3]]))


def test_add_rolling_features_min_periods_allows_earlier_values() -> None:
    df = _make_panel_df(n_entities=1, n_steps=5)

    df_out, cols = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[3],
        rolling_stats=["mean"],
        min_periods=1,
        leakage_safe=True,
    )
    assert cols == ["roll_3_mean"]

    sub = df_out.sort_values("timestamp").reset_index(drop=True)
    y = sub["target"].to_numpy()

    # leakage_safe uses shift(1)
    # t=0 -> shifted is NaN => roll is NaN even with min_periods=1
    assert np.isnan(sub.loc[0, "roll_3_mean"])

    # t=1 -> window contains [y0] => mean=y0
    assert sub.loc[1, "roll_3_mean"] == pytest.approx(y[0])

    # t=2 -> window contains [y0,y1] => mean
    assert sub.loc[2, "roll_3_mean"] == pytest.approx(np.mean([y[0], y[1]]))

    # t=3 -> window contains [y0,y1,y2] => mean
    assert sub.loc[3, "roll_3_mean"] == pytest.approx(np.mean([y[0], y[1], y[2]]))


def test_add_rolling_features_empty_or_none_returns_copy_no_cols() -> None:
    df = _make_panel_df(n_entities=1, n_steps=4)

    out1, cols1 = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=None,
        rolling_stats=["mean"],
    )
    assert cols1 == []
    assert out1 is not df
    assert set(out1.columns) == set(df.columns)

    out2, cols2 = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[],
        rolling_stats=["mean"],
    )
    assert cols2 == []
    assert out2 is not df
    assert set(out2.columns) == set(df.columns)


def test_add_rolling_features_requires_columns() -> None:
    df = pd.DataFrame({"entity_id": [0, 0, 0], "target": [1.0, 2.0, 3.0]})

    with pytest.raises(KeyError, match=r"Entity column|entity"):
        add_rolling_features(
            df,
            entity_col="missing",
            target_col="target",
            rolling_windows=[3],
            rolling_stats=["mean"],
        )

    with pytest.raises(KeyError, match=r"Target column|target"):
        add_rolling_features(
            df,
            entity_col="entity_id",
            target_col="missing",
            rolling_windows=[3],
            rolling_stats=["mean"],
        )


def test_add_rolling_features_raises_on_non_positive_window() -> None:
    df = _make_panel_df(n_entities=1, n_steps=5)

    with pytest.raises(ValueError, match=r"Rolling window must be positive|positive"):
        add_rolling_features(
            df,
            entity_col="entity_id",
            target_col="target",
            rolling_windows=[0],
            rolling_stats=["mean"],
        )

    with pytest.raises(ValueError, match=r"Rolling window must be positive|positive"):
        add_rolling_features(
            df,
            entity_col="entity_id",
            target_col="target",
            rolling_windows=[-2],
            rolling_stats=["mean"],
        )


def test_add_rolling_features_raises_on_unsupported_stat() -> None:
    df = _make_panel_df(n_entities=1, n_steps=5)

    with pytest.raises(ValueError, match="Unsupported rolling stat"):
        add_rolling_features(
            df,
            entity_col="entity_id",
            target_col="target",
            rolling_windows=[3],
            rolling_stats=["mean", "bad_stat"],
        )


def test_add_rolling_features_does_not_cross_entity_boundaries() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0, 1, 1, 1],
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "target": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
        }
    )

    df_out, cols = add_rolling_features(
        df,
        entity_col="entity_id",
        target_col="target",
        rolling_windows=[2],
        rolling_stats=["mean"],
        leakage_safe=True,
    )
    assert cols == ["roll_2_mean"]

    sub0 = (
        df_out[df_out["entity_id"] == 0].sort_values("timestamp").reset_index(drop=True)
    )
    sub1 = (
        df_out[df_out["entity_id"] == 1].sort_values("timestamp").reset_index(drop=True)
    )

    # leakage_safe=True uses shift(1), window=2, min_periods=2
    # entity 0: first defined at t=2 -> mean(y0,y1) = 0.5
    assert np.isnan(sub0.loc[0, "roll_2_mean"])
    assert np.isnan(sub0.loc[1, "roll_2_mean"])
    assert sub0.loc[2, "roll_2_mean"] == pytest.approx(np.mean([0.0, 1.0]))

    # entity 1: first defined at its t=2 -> mean(100,101)=100.5
    assert np.isnan(sub1.loc[0, "roll_2_mean"])
    assert np.isnan(sub1.loc[1, "roll_2_mean"])
    assert sub1.loc[2, "roll_2_mean"] == pytest.approx(np.mean([100.0, 101.0]))
