"""
Validation utilities for panel feature engineering.

This module centralizes lightweight input validation helpers used throughout the
panel feature engineering subpackage.

Key invariants
--------------
- Required columns must exist.
- Within each entity, timestamps must be **strictly increasing in the given row order**.
  (No sorting is performed inside validation; callers may sort afterward for deterministic
  computation, but validation should catch out-of-order input.)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame, *, required_cols: Sequence[str]
) -> None:
    """
    Validate that required columns exist on a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    required_cols : Sequence[str]
        Columns that must be present.

    Raises
    ------
    KeyError
        If any required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}.")


def ensure_columns_present(
    df: pd.DataFrame, *, columns: Iterable[str], label: str
) -> None:
    """
    Ensure that a set of configured columns exists on a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : Iterable[str]
        Columns that must be present.
    label : str
        Human-readable label used in error messages (e.g., "Static", "Regressor").

    Raises
    ------
    KeyError
        If any specified columns are missing.
    """
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} column(s) not found in DataFrame: {missing}.")


def validate_monotonic_timestamps(
    df: pd.DataFrame,
    *,
    entity_col: str,
    timestamp_col: str,
) -> None:
    """
    Validate that timestamps are strictly increasing within each entity in row order.

    This function does **not** sort. It checks the DataFrame exactly as provided.

    Parameters
    ----------
    df : pandas.DataFrame
        Input panel DataFrame.
    entity_col : str
        Entity identifier column.
    timestamp_col : str
        Timestamp column.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If any entity has non-strictly increasing timestamps (ties or out-of-order),
        or if timestamps cannot be parsed.
    """
    validate_required_columns(df, required_cols=(entity_col, timestamp_col))

    # Convert once; preserve timezone awareness if present.
    try:
        ts = pd.to_datetime(df[timestamp_col])
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to convert {timestamp_col!r} to datetime.") from e

    # Fast path: if there is only one row, it's trivially valid.
    if len(df) <= 1:
        return

    # We must respect the current row order per entity. groupby(..., sort=False)
    # keeps the first-seen entity order and does not reorder rows within groups.
    tmp = pd.DataFrame({entity_col: df[entity_col], "_ts": ts})

    # Check strict increase: ts[i] > ts[i-1]
    for ent, g in tmp.groupby(entity_col, sort=False):
        g_ts = g["_ts"].to_numpy()

        if g_ts.size <= 1:
            continue

        is_increasing = g_ts[1:] > g_ts[:-1]
        if not is_increasing.all():
            # Provide a small diagnostic: first offending index position within entity.
            bad_pos = int((~is_increasing).nonzero()[0][0] + 1)
            raise ValueError(
                "Timestamps must be strictly increasing within each entity. "
                f"Found non-monotonic sequence for entity={ent!r} at position {bad_pos} "
                "(in the provided row order)."
            )
