r"""
Rolling window feature construction for panel time series.

This module provides stateless utilities to compute rolling window statistics of a
target series within each entity of a panel (entity-by-timestamp) dataset.

Rolling windows are expressed in **index steps** (rows) at the input frequency rather
than wall-clock units. This keeps the transformation frequency-agnostic.

Definitions
-----------
For a target series ``y_t`` and a window length ``w``, the rolling mean feature is:

$$
\mathrm{roll\_mean}_w(t) = \frac{1}{w}\sum_{j=1}^{w} y_{t-j}
$$

This formulation explicitly uses *past* values only (``y_{t-1}`` through ``y_{t-w}``),
which avoids target leakage when predicting at time ``t``.

The resulting feature columns are named ``roll_{w}_{stat}``, e.g., ``roll_24_mean``.

Notes
-----
- Rolling features are computed strictly within each entity.
- By default this module computes rolling statistics on ``target.shift(1)`` within each
  entity, so that the current target value ``y_t`` is not included in the window.
- The calling pipeline is responsible for sorting data by ``(entity, timestamp)`` and for
  deciding how to handle NaNs introduced by rolling windows (e.g., dropping rows).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from eb_features.panel.constants import ALLOWED_ROLLING_STATS


def add_rolling_features(
    df: pd.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    rolling_windows: Sequence[int] | None,
    rolling_stats: Sequence[str],
    min_periods: int | None = None,
    leakage_safe: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    r"""
    Add rolling window statistics features to a panel DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least ``entity_col`` and ``target_col``.
    entity_col : str
        Name of the entity identifier column.
    target_col : str
        Name of the numeric target column used to compute rolling statistics.
    rolling_windows : Sequence[int] | None
        Positive rolling window lengths (in steps). For each ``w`` in ``rolling_windows`` and
        each stat in ``rolling_stats``, the feature ``roll_{w}_{stat}`` is added. If None or
        empty, no rolling features are added.
    rolling_stats : Sequence[str]
        Rolling statistics to compute. Allowed values are:
        ``{"mean", "std", "min", "max", "sum", "median"}``.
    min_periods : int | None, default None
        Minimum number of observations in the window required to produce a value. If None,
        defaults to ``w`` (full-window requirement).
    leakage_safe : bool, default True
        If True, compute rolling statistics on the *lagged* target (``target.shift(1)`` within
        each entity), so that the current ``y_t`` is excluded from the window.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of ``df`` with rolling feature columns added.
    feature_cols : list[str]
        Names of the rolling feature columns added.

    Raises
    ------
    KeyError
        If ``entity_col`` or ``target_col`` is missing from ``df``.
    ValueError
        If any rolling window is non-positive, if ``min_periods`` is invalid, or if an
        unsupported stat is requested.

    Notes
    -----
    Rolling features introduce missing values at the beginning of each entity's series.
    These are typically removed downstream when ``dropna=True`` or handled via imputation.
    """
    if entity_col not in df.columns:
        raise KeyError(f"Entity column {entity_col!r} not found in DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in DataFrame.")

    if not rolling_windows:
        return df.copy(), []

    _validate_rolling_stats(rolling_stats)

    if min_periods is not None and int(min_periods) <= 0:
        raise ValueError(
            f"min_periods must be a positive integer or None; got {min_periods!r}."
        )

    df_out = df.copy()
    feature_cols: list[str] = []

    # Compute rolling features within each entity.
    grp = df_out.groupby(entity_col)[target_col]
    series_for_roll = grp.shift(1) if leakage_safe else df_out[target_col]

    for w in rolling_windows:
        if w <= 0:
            raise ValueError(f"Rolling window must be positive; got {w}.")

        mp = w if min_periods is None else int(min_periods)
        if mp > w:
            raise ValueError(
                f"min_periods must be <= window length w; got min_periods={mp}, w={w}."
            )

        # rolling() produces a MultiIndex keyed by entity; we drop the entity level to align to df rows.
        roll = series_for_roll.groupby(df_out[entity_col]).rolling(
            window=w, min_periods=mp
        )

        for stat in rolling_stats:
            col_name = f"roll_{w}_{stat}"

            if stat == "mean":
                values = roll.mean()
            elif stat == "std":
                values = roll.std()
            elif stat == "min":
                values = roll.min()
            elif stat == "max":
                values = roll.max()
            elif stat == "sum":
                values = roll.sum()
            elif stat == "median":
                values = roll.median()
            else:  # pragma: no cover
                raise RuntimeError(f"Unexpected rolling stat {stat!r}.")

            df_out[col_name] = values.reset_index(level=0, drop=True)
            feature_cols.append(col_name)

    return df_out, feature_cols


def _validate_rolling_stats(rolling_stats: Sequence[str]) -> None:
    r"""
    Validate that requested rolling stats are supported.

    Parameters
    ----------
    rolling_stats : Sequence[str]
        Candidate rolling statistic keys.

    Raises
    ------
    ValueError
        If any stat key is unsupported.
    """
    invalid = [s for s in rolling_stats if s not in ALLOWED_ROLLING_STATS]
    if invalid:
        raise ValueError(
            f"Unsupported rolling stat(s) {invalid}. Allowed: {sorted(ALLOWED_ROLLING_STATS)}."
        )
