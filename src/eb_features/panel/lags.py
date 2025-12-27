r"""
Lag feature construction for panel time series.

This module provides stateless utilities to construct lagged versions of a target
series within each entity of a panel (entity-by-timestamp) dataset.

Lag features are expressed in **index steps** (rows) at the input frequency rather than
wall-clock units. This makes the transformation frequency-agnostic (5-min, 30-min,
hourly, daily, etc.), assuming the panel is sorted by timestamp within each entity.

Definition
----------
For a target series ``y_t`` and lag step ``k``:

$$
\mathrm{lag}_k(t) = y_{t-k}
$$

The resulting feature column is named ``lag_{k}``.

Notes
-----
- Lag features are computed strictly within each entity using grouped shifts.
- The calling pipeline is responsible for handling missing values introduced by lagging
  (e.g., dropping rows or applying imputation).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    lag_steps: Sequence[int] | None,
) -> tuple[pd.DataFrame, list[str]]:
    r"""
    Add target lag features to a panel DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least ``entity_col`` and ``target_col``.
    entity_col : str
        Name of the entity identifier column.
    target_col : str
        Name of the numeric target column to be lagged.
    lag_steps : Sequence[int] | None
        Positive lag offsets (in steps). For each ``k`` in ``lag_steps``, the feature
        ``lag_{k}`` is added. If None or empty, no lag features are added.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of ``df`` with lag feature columns added.
    feature_cols : list[str]
        Names of the lag feature columns added.

    Raises
    ------
    KeyError
        If ``entity_col`` or ``target_col`` is missing from ``df``.
    ValueError
        If any lag step is non-positive.

    Notes
    -----
    Lagging introduces missing values for the first ``k`` observations of each entity.
    These are typically removed downstream when ``dropna=True`` or handled via
    imputation.
    """
    if entity_col not in df.columns:
        raise KeyError(f"Entity column {entity_col!r} not found in DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in DataFrame.")

    if not lag_steps:
        return df.copy(), []

    df_out = df.copy()
    feature_cols: list[str] = []

    for k in lag_steps:
        if k <= 0:
            raise ValueError(f"Lag steps must be positive; got {k}.")
        col_name = f"lag_{k}"
        df_out[col_name] = df_out.groupby(entity_col)[target_col].shift(k)
        feature_cols.append(col_name)

    return df_out, feature_cols
