r"""
Panel feature engineering orchestrator.

This module defines a lightweight, frequency-agnostic feature engineering utility for
panel time-series data (entity-by-timestamp). The implementation is intentionally
*stateless*: each call constructs features from the provided input DataFrame and
configuration.

The output is designed for classical supervised learning pipelines that expect a
fixed-width design matrix ``X`` and target vector ``y``.

Features
--------
Given an entity identifier column and a target series ``y_t`` (per entity), the feature
pipeline can construct:

1) Lag features:

$$
\mathrm{lag}_k(t) = y_{t-k}
$$

2) Rolling window statistics over the last ``w`` observations (leakage-safe by default):

$$
\mathrm{roll\_mean}_w(t) = \frac{1}{w}\sum_{j=1}^{w} y_{t-j}
$$

3) Calendar features derived from timestamp: hour, day-of-week, day-of-month, month,
and weekend indicator.

4) Optional cyclical encodings for periodic calendar features:

$$
\sin\left(2\pi \frac{\mathrm{hour}}{24}\right), \quad
\cos\left(2\pi \frac{\mathrm{hour}}{24}\right)
$$

and similarly for day-of-week with period 7.

5) Optional passthrough features:
numeric regressors and static metadata columns.

Notes
-----
- Lags and rolling windows are expressed in **index steps** (rows) at the input frequency.
- All time-dependent features are computed strictly within each entity.
- Passthrough non-numeric columns are encoded using stable integer category codes for the
  values present in the provided DataFrame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from eb_features.panel.calendar import add_calendar_features
from eb_features.panel.encoders import encode_non_numeric_as_category_codes
from eb_features.panel.lags import add_lag_features
from eb_features.panel.rolling import add_rolling_features
from eb_features.panel.validation import (
    ensure_columns_present,
    validate_monotonic_timestamps,
    validate_required_columns,
)

# Use internal variables to avoid Final reassignment errors from Pyright
try:
    from eb_features.panel.constants import (
        DEFAULT_CALENDAR_FEATURES,
        DEFAULT_LAG_STEPS,
        DEFAULT_ROLLING_STATS,
        DEFAULT_ROLLING_WINDOWS,
    )

    _LAG_INIT = DEFAULT_LAG_STEPS
    _ROLL_WIN_INIT = DEFAULT_ROLLING_WINDOWS
    _ROLL_STAT_INIT = DEFAULT_ROLLING_STATS
    _CAL_INIT = DEFAULT_CALENDAR_FEATURES
except Exception:  # pragma: no cover
    _LAG_INIT = (1, 2, 24)
    _ROLL_WIN_INIT = (3, 24)
    _ROLL_STAT_INIT = ("mean", "std", "min", "max", "sum")
    _CAL_INIT = ("hour", "dow", "month", "is_weekend")


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for panel time-series feature engineering.
    """

    lag_steps: Sequence[int] | None = field(default_factory=lambda: list(_LAG_INIT))
    rolling_windows: Sequence[int] | None = field(default_factory=lambda: list(_ROLL_WIN_INIT))
    rolling_stats: Sequence[str] = field(default_factory=lambda: list(_ROLL_STAT_INIT))
    calendar_features: Sequence[str] = field(default_factory=lambda: list(_CAL_INIT))
    use_cyclical_time: bool = True

    regressor_cols: Sequence[str] | None = None
    static_cols: Sequence[str] | None = None

    dropna: bool = True
    leakage_safe_rolling: bool = True


class FeatureEngineer:
    """
    Transform panel time-series data into a model-ready ``(X, y, feature_names)`` triple.
    """

    def __init__(
        self,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        target_col: str = "target",
    ) -> None:
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col

    def transform(
        self,
        df: pd.DataFrame,
        config: FeatureConfig,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Transform a panel DataFrame into ``(X, y, feature_names)``.
        """
        validate_required_columns(
            df,
            required_cols=(self.entity_col, self.timestamp_col, self.target_col),
        )

        validate_monotonic_timestamps(
            df, entity_col=self.entity_col, timestamp_col=self.timestamp_col
        )

        df_work = df.copy()
        df_work = df_work.sort_values([self.entity_col, self.timestamp_col], kind="mergesort")

        # ------------------------------------------------------------------
        # Identify passthrough columns
        # ------------------------------------------------------------------
        static_cols = list(config.static_cols or [])
        ensure_columns_present(df_work, columns=static_cols, label="Static")

        if config.regressor_cols is not None:
            regressor_cols = list(config.regressor_cols)
        else:
            exclude = {
                self.entity_col,
                self.timestamp_col,
                self.target_col,
                *static_cols,
            }
            numeric_cols = df_work.select_dtypes(include=["number"]).columns
            regressor_cols = [c for c in numeric_cols if c not in exclude]

        ensure_columns_present(df_work, columns=regressor_cols, label="Regressor")

        # ------------------------------------------------------------------
        # Build engineered features
        # ------------------------------------------------------------------
        feature_cols: list[str] = []
        engineered_cols: list[str] = []

        df_work, lag_cols = add_lag_features(
            df_work,
            entity_col=self.entity_col,
            target_col=self.target_col,
            lag_steps=config.lag_steps,
        )
        feature_cols.extend(lag_cols)
        engineered_cols.extend(lag_cols)

        df_work, roll_cols = add_rolling_features(
            df_work,
            entity_col=self.entity_col,
            target_col=self.target_col,
            rolling_windows=config.rolling_windows,
            rolling_stats=config.rolling_stats,
            leakage_safe=config.leakage_safe_rolling,
        )
        feature_cols.extend(roll_cols)
        engineered_cols.extend(roll_cols)

        df_work, cal_feature_cols, _cal_base_cols = add_calendar_features(
            df_work,
            timestamp_col=self.timestamp_col,
            calendar_features=config.calendar_features,
            use_cyclical_time=config.use_cyclical_time,
        )
        feature_cols.extend(cal_feature_cols)
        engineered_cols.extend(cal_feature_cols)

        feature_cols.extend(static_cols)
        feature_cols.extend(regressor_cols)

        # ------------------------------------------------------------------
        # Final cleaning & extraction
        # ------------------------------------------------------------------
        df_work = df_work[~df_work[self.target_col].isna()]

        if (df_work[self.target_col] < 0).any():
            raise ValueError("Negative values found in target column; expected >= 0.")

        # Type narrowing for dropna overloads
        assert isinstance(df_work, pd.DataFrame)

        if feature_cols:
            if config.dropna:
                df_work = df_work.dropna(axis=0, subset=feature_cols)
            elif engineered_cols:
                df_work = df_work.dropna(axis=0, subset=engineered_cols)

        # Ensure we are working with a DataFrame for the final matrix creation
        final_df = df_work[feature_cols].copy()
        if not isinstance(final_df, pd.DataFrame):
            final_df = pd.DataFrame(final_df)

        # Encode any remaining non-numeric feature columns.
        if any(not is_numeric_dtype(final_df[c]) for c in final_df.columns):
            final_df = encode_non_numeric_as_category_codes(final_df)

        # Final cast to satisfy to_numpy access
        feature_frame = cast(pd.DataFrame, final_df)

        X_values = feature_frame.to_numpy(dtype=float)
        y_values = df_work[self.target_col].to_numpy(dtype=float)

        if not np.isfinite(X_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in feature matrix X.")
        if not np.isfinite(y_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in target vector y.")

        return X_values, y_values, feature_cols
