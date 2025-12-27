r"""
Shared constants for panel feature engineering.

This module centralizes small, stable configuration values used across the
``eb_features.panel`` subpackage. Keeping these definitions in one place prevents
validation drift between modules and provides a single reference point for both
implementation and documentation.

Notes
-----
- These constants are intentionally minimal and low-churn.
- They define *allowed values*, *default configurations*, and *calendar parameters*
  used consistently across feature builders.
"""

from __future__ import annotations

from typing import Final

# -----------------------------------------------------------------------------
# Allowed feature keys
# -----------------------------------------------------------------------------
ALLOWED_ROLLING_STATS: frozenset[str] = frozenset(
    {"mean", "std", "min", "max", "sum", "median"}
)
r"""
Allowed rolling-window summary statistics.

Each statistic corresponds to a feature name of the form:

$$
\mathrm{roll\_{w}\_{stat}}(t)
$$

where ``w`` is the window length (in index steps) and ``stat`` is one of the allowed values.
"""


ALLOWED_CALENDAR_FEATURES: frozenset[str] = frozenset(
    {"hour", "dow", "dom", "month", "is_weekend"}
)
r"""
Allowed calendar features derived from the timestamp column.

Calendar features are added as integer-valued columns and may optionally be accompanied by
cyclical encodings (sine/cosine) for periodic components.
"""

# -----------------------------------------------------------------------------
# Default configuration values (used by FeatureConfig)
# -----------------------------------------------------------------------------
DEFAULT_LAG_STEPS: Final[tuple[int, ...]] = (1, 2, 24)

DEFAULT_ROLLING_WINDOWS: Final[tuple[int, ...]] = (3, 24)
DEFAULT_ROLLING_STATS: Final[tuple[str, ...]] = ("mean", "std", "min", "max", "sum")

DEFAULT_CALENDAR_FEATURES: Final[tuple[str, ...]] = (
    "hour",
    "dow",
    "month",
    "is_weekend",
)

# -----------------------------------------------------------------------------
# Calendar / cyclical encoding parameters
# -----------------------------------------------------------------------------
HOUR_PERIOD: Final[int] = 24
"""Period used for cyclical hour-of-day encodings."""

DOW_PERIOD: Final[int] = 7
"""Period used for cyclical day-of-week encodings."""

# pandas dt.dayofweek convention: Monday=0 ... Sunday=6
WEEKEND_DAYS: Final[tuple[int, int]] = (5, 6)
"""Day-of-week values corresponding to Saturday and Sunday."""
