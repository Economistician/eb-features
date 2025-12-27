r"""
Calendar and time-derived features for panel time series.

This module provides utilities to derive calendar/time features from a timestamp column.
It is designed for *panel* time-series data (entity-by-timestamp) and is typically used
as part of a broader feature engineering pipeline.

Supported base calendar features
--------------------------------
Given a timestamp column ``timestamp_col``:

- ``"hour"``: Hour of day in ``[0, 23]``
- ``"dow"``: Day of week in ``[0, 6]`` where Monday=0 (pandas convention)
- ``"dom"``: Day of month in ``[1, 31]``
- ``"month"``: Month in ``[1, 12]``
- ``"is_weekend"``: Weekend indicator (Saturday/Sunday) as ``0/1``

Optional cyclical encodings
---------------------------
Certain calendar attributes are periodic and can be represented with sine/cosine pairs:

- hour (period 24):

$$
\sin\left(2\pi \frac{\mathrm{hour}}{24}\right),\quad
\cos\left(2\pi \frac{\mathrm{hour}}{24}\right)
$$

- day of week (period 7):

$$
\sin\left(2\pi \frac{\mathrm{dow}}{7}\right),\quad
\cos\left(2\pi \frac{\mathrm{dow}}{7}\right)
$$

These encodings are added only when:
- the corresponding base feature (hour or dow) is present, and
- ``use_cyclical_time=True``.

Notes
-----
- Feature construction is *stateless*: functions operate only on the provided DataFrame.
- Time features are derived from ``pandas.to_datetime`` conversion of ``timestamp_col``.
  Timezone-aware timestamps are supported; feature values reflect the timestamp's local
  representation as stored in the column.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from eb_features.panel.constants import (
    ALLOWED_CALENDAR_FEATURES,
    DOW_PERIOD,
    HOUR_PERIOD,
    WEEKEND_DAYS,
)


def add_calendar_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    calendar_features: Sequence[str],
    use_cyclical_time: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    r"""
    Add calendar/time-derived features to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a timestamp column.
    timestamp_col : str
        Name of the timestamp column.
    calendar_features : Sequence[str]
        Base calendar features to derive. Allowed values are:
        ``{"hour", "dow", "dom", "month", "is_weekend"}``.
    use_cyclical_time : bool, default True
        If True, add sine/cosine encodings for hour and/or day-of-week when those
        base features are included.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of ``df`` with the requested calendar features added as columns.
    feature_cols : list[str]
        Names of all features added by this call, including cyclical encodings (if any).
    calendar_cols : list[str]
        Names of base calendar feature columns added (excludes cyclical encodings).

    Raises
    ------
    KeyError
        If ``timestamp_col`` is not present in ``df``.
    ValueError
        If an unsupported calendar feature is requested.

    Notes
    -----
    The derived columns are currently named:

    - ``hour``
    - ``dayofweek`` (for requested ``"dow"``)
    - ``dayofmonth`` (for requested ``"dom"``)
    - ``month``
    - ``is_weekend``

    This naming is stable and intended to be referenced by downstream steps.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column {timestamp_col!r} not found in DataFrame.")

    _validate_calendar_features(calendar_features)

    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    calendar_cols: list[str] = []
    feature_cols: list[str] = []

    for name in calendar_features:
        if name == "hour":
            col = "hour"
            df_out[col] = ts.dt.hour.astype("int16")
            calendar_cols.append(col)
        elif name == "dow":
            col = "dayofweek"
            df_out[col] = ts.dt.dayofweek.astype("int16")
            calendar_cols.append(col)
        elif name == "dom":
            col = "dayofmonth"
            df_out[col] = ts.dt.day.astype("int16")
            calendar_cols.append(col)
        elif name == "month":
            col = "month"
            df_out[col] = ts.dt.month.astype("int16")
            calendar_cols.append(col)
        elif name == "is_weekend":
            col = "is_weekend"
            df_out[col] = ts.dt.dayofweek.isin(WEEKEND_DAYS).astype("int8")
            calendar_cols.append(col)
        else:  # pragma: no cover
            # Guarded by _validate_calendar_features, but keep a defensive fallback.
            raise ValueError(
                f"Unsupported calendar feature {name!r}. Allowed: {sorted(ALLOWED_CALENDAR_FEATURES)}."
            )

    feature_cols.extend(calendar_cols)

    # ---------------------------------------------------------------------
    # Optional cyclical encodings
    # ---------------------------------------------------------------------
    if use_cyclical_time:
        if "hour" in calendar_cols:
            hour = df_out["hour"].astype(float)
            df_out["hour_sin"] = np.sin(2.0 * np.pi * hour / float(HOUR_PERIOD))
            df_out["hour_cos"] = np.cos(2.0 * np.pi * hour / float(HOUR_PERIOD))
            feature_cols.extend(["hour_sin", "hour_cos"])

        if "dayofweek" in calendar_cols:
            dow = df_out["dayofweek"].astype(float)
            df_out["dow_sin"] = np.sin(2.0 * np.pi * dow / float(DOW_PERIOD))
            df_out["dow_cos"] = np.cos(2.0 * np.pi * dow / float(DOW_PERIOD))
            feature_cols.extend(["dow_sin", "dow_cos"])

    return df_out, feature_cols, calendar_cols


def _validate_calendar_features(calendar_features: Iterable[str]) -> None:
    r"""
    Validate that requested calendar features are supported.

    Parameters
    ----------
    calendar_features : Iterable[str]
        Candidate calendar feature keys.

    Raises
    ------
    ValueError
        If any feature key is unsupported.
    """
    requested = list(calendar_features)
    invalid = [f for f in requested if f not in ALLOWED_CALENDAR_FEATURES]
    if invalid:
        raise ValueError(
            "Unsupported calendar feature(s) requested: "
            f"{invalid}. Allowed: {sorted(ALLOWED_CALENDAR_FEATURES)}."
        )
