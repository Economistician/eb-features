"""
Configuration objects for panel time-series feature engineering.

This module defines declarative configuration used by the panel feature engineering
pipeline. The configuration is designed to be:

- **frequency-agnostic** (lags/windows are expressed in index steps, not wall-clock time)
- **stateless** (config describes what to compute; no fitted state is stored)
- **explicit** (feature families are enabled/disabled via lists/flags)

Notes
-----
This configuration is consumed by
[`FeatureEngineer`][eb_features.panel.engineering.FeatureEngineer] and related helper
modules (lags, rolling, calendar, encoders).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# -------------------------------------------------------------------------
# Public constants (useful for validation, docs, and IDE discoverability)
# -------------------------------------------------------------------------

#: Allowed rolling statistics for rolling window features.
ALLOWED_ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "sum", "median")

#: Allowed calendar feature keys derived from timestamps.
ALLOWED_CALENDAR_FEATURES: tuple[str, ...] = (
    "hour",
    "dow",
    "dom",
    "month",
    "is_weekend",
)


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for panel time-series feature engineering.

    The feature engineering pipeline assumes a long-form panel DataFrame with an entity
    identifier column, a timestamp column, and a numeric target column. This configuration
    describes which feature families to generate and which passthrough columns to include.

    Notes
    -----
    - Lag steps and rolling windows are expressed in **index steps** (rows) at the input
      frequency, not in wall-clock units.
    - Lags and rolling windows are computed **within each entity**.
    - If ``dropna=True``, rows lacking sufficient lag/rolling history are dropped after
      feature construction.

    Attributes
    ----------
    lag_steps : Sequence[int] | None
        Positive lag offsets (in steps) applied to the target. For each ``k`` in ``lag_steps``,
        the feature ``lag_{k}`` is added.
    rolling_windows : Sequence[int] | None
        Positive rolling window lengths (in steps) applied to the target. For each ``w`` in
        ``rolling_windows`` and each stat in ``rolling_stats``, the feature
        ``roll_{w}_{stat}`` is added.
    rolling_stats : Sequence[str]
        Rolling statistics to compute. Allowed values are:
        ``{"mean", "std", "min", "max", "sum", "median"}``.
    calendar_features : Sequence[str]
        Calendar features derived from the timestamp column. Allowed values are:
        ``{"hour", "dow", "dom", "month", "is_weekend"}``.
    use_cyclical_time : bool
        If True, add sine/cosine encodings for hour and day-of-week when those base
        calendar columns are present.
    regressor_cols : Sequence[str] | None
        Numeric external regressors to pass through. If None, numeric columns may be
        auto-detected by the calling pipeline (excluding entity/timestamp/target and
        ``static_cols``).
    static_cols : Sequence[str] | None
        Entity-level metadata columns already present on the input DataFrame. These are
        passed through directly as features.
    dropna : bool
        If True, drop rows with NaNs in any engineered feature columns (typically caused
        by lags and rolling windows).
    """

    lag_steps: Sequence[int] | None = field(default_factory=lambda: (1, 2, 24))

    rolling_windows: Sequence[int] | None = field(default_factory=lambda: (3, 24))
    rolling_stats: Sequence[str] = field(
        default_factory=lambda: ("mean", "std", "min", "max", "sum")
    )

    calendar_features: Sequence[str] = field(
        default_factory=lambda: ("hour", "dow", "month", "is_weekend")
    )
    use_cyclical_time: bool = True

    regressor_cols: Sequence[str] | None = None
    static_cols: Sequence[str] | None = None

    dropna: bool = True
