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

try:
    # Prefer a single source of truth for allowed feature keys and defaults.
    from eb_features.panel.constants import (
        DEFAULT_CALENDAR_FEATURES,
        DEFAULT_LAG_STEPS,
        DEFAULT_ROLLING_STATS,
        DEFAULT_ROLLING_WINDOWS,
    )
except Exception:  # pragma: no cover
    DEFAULT_LAG_STEPS = (1, 2, 24)
    DEFAULT_ROLLING_WINDOWS = (3, 24)
    DEFAULT_ROLLING_STATS = ("mean", "std", "min", "max", "sum")
    DEFAULT_CALENDAR_FEATURES = ("hour", "dow", "month", "is_weekend")


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for panel time-series feature engineering.

    Notes
    -----
    - All lag steps and rolling window lengths are expressed in **index steps** (rows).
    - Lags and rolling windows are computed **within each entity**.
    - If ``dropna=True``, rows that lack full lag/rolling history are dropped after feature
      construction.

    Attributes
    ----------
    lag_steps : Sequence[int] | None
        Positive lag offsets (in steps) applied to the target. For each ``k`` in ``lag_steps``,
        the feature ``lag_{k}`` is added.
    rolling_windows : Sequence[int] | None
        Positive rolling window lengths (in steps) applied to the target. For each ``w`` in
        ``rolling_windows`` and each stat in ``rolling_stats``, the feature ``roll_{w}_{stat}``
        is added.
    rolling_stats : Sequence[str]
        Rolling statistics to compute. Allowed values are:
        ``{"mean", "std", "min", "max", "sum", "median"}``.
    calendar_features : Sequence[str]
        Calendar features derived from ``timestamp_col``. Allowed values:
        ``{"hour", "dow", "dom", "month", "is_weekend"}``.
    use_cyclical_time : bool
        If True, add sine/cosine encodings for hour and day-of-week when those base columns
        are present.
    regressor_cols : Sequence[str] | None
        Numeric external regressors to pass through. If None, numeric columns are auto-detected
        excluding entity/timestamp/target and ``static_cols``.
    static_cols : Sequence[str] | None
        Entity-level metadata columns already present on the input DataFrame. These are passed
        through directly as features.
    dropna : bool
        If True, drop rows with NaNs in any feature columns.
        If False, drop rows with NaNs in *engineered* feature columns only (lags/rolling/calendar),
        allowing passthrough regressors/static columns to contain NaNs for upstream imputation.
    leakage_safe_rolling : bool
        If True, rolling window features exclude the current target value ``y_t`` by computing
        statistics over ``y_{t-1}, \\ldots, y_{t-w}``.
    """

    lag_steps: Sequence[int] | None = field(
        default_factory=lambda: list(DEFAULT_LAG_STEPS)
    )
    rolling_windows: Sequence[int] | None = field(
        default_factory=lambda: list(DEFAULT_ROLLING_WINDOWS)
    )
    rolling_stats: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_ROLLING_STATS)
    )
    calendar_features: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_CALENDAR_FEATURES)
    )
    use_cyclical_time: bool = True

    regressor_cols: Sequence[str] | None = None
    static_cols: Sequence[str] | None = None

    dropna: bool = True
    leakage_safe_rolling: bool = True


class FeatureEngineer:
    """
    Transform panel time-series data into a model-ready ``(X, y, feature_names)`` triple.

    The input is expected to be a long-form DataFrame with at least:

    - ``entity_col``: series identifier (e.g., store_id, sku_id)
    - ``timestamp_col``: datetime-like timestamp
    - ``target_col``: numeric target to be predicted

    This transformer computes features strictly within entity, requires timestamps to be
    strictly increasing within each entity, and can optionally drop rows lacking sufficient
    history.

    Notes
    -----
    - This class does not store fitted state. Feature generation is deterministic given the
      inputs.
    - Non-numeric feature columns are encoded using integer category codes for the values
      present in the provided DataFrame.
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

        Validation policy
        -----------------
        - We validate monotonicity on the **input row order** first (to catch bad upstream ordering).
        - We then sort by (entity, timestamp) for deterministic feature computation.

        Raises
        ------
        KeyError
            If required columns or configured passthrough columns are missing.
        ValueError
            If timestamps are not strictly increasing within each entity (in input row order),
            if the target contains negative values, or if non-finite values are present in
            the resulting ``X``/``y``.
        """
        validate_required_columns(
            df,
            required_cols=(self.entity_col, self.timestamp_col, self.target_col),
        )

        # IMPORTANT: validate in the provided row order (do NOT sort first),
        # so we can catch upstream ordering issues.
        validate_monotonic_timestamps(
            df, entity_col=self.entity_col, timestamp_col=self.timestamp_col
        )

        # Work on a copy, sorted by entity / timestamp for deterministic feature building
        df_work = df.copy()
        df_work = df_work.sort_values(
            [self.entity_col, self.timestamp_col], kind="mergesort"
        )

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

        # Passthrough: static + regressors
        feature_cols.extend(static_cols)
        feature_cols.extend(regressor_cols)

        # ------------------------------------------------------------------
        # Final cleaning & extraction
        # ------------------------------------------------------------------
        df_work = df_work[~df_work[self.target_col].isna()]

        # For demand-like series, enforce non-negative target.
        if (df_work[self.target_col] < 0).any():
            raise ValueError("Negative values found in target column; expected >= 0.")

        # NaN handling:
        # - dropna=True: strict, drop rows missing any feature (engineered or passthrough)
        # - dropna=False: permissive, still require engineered features to be present
        if feature_cols:
            if config.dropna:
                df_work = df_work.dropna(subset=feature_cols)
            else:
                if engineered_cols:
                    df_work = df_work.dropna(subset=engineered_cols)

        feature_frame = df_work[feature_cols].copy()

        # Encode any remaining non-numeric feature columns.
        if any(not is_numeric_dtype(feature_frame[c]) for c in feature_frame.columns):
            feature_frame = encode_non_numeric_as_category_codes(feature_frame)

        X_values = feature_frame.to_numpy(dtype=float)
        y_values = df_work[self.target_col].to_numpy(dtype=float)

        if not np.isfinite(X_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in feature matrix X.")
        if not np.isfinite(y_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in target vector y.")

        return X_values, y_values, feature_cols
