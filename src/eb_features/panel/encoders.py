"""
Encoding utilities for panel feature engineering.

This module provides small, stateless helpers to make feature matrices numeric.

Current scope
-------------
The panel feature engineering pipeline produces a feature frame that may contain
a mixture of numeric and non-numeric columns (e.g., entity metadata strings).
Many downstream estimators expect purely numeric arrays. The helper in this module
encodes non-numeric columns using pandas categorical codes.

Important
---------
Categorical codes are stable **only within the provided DataFrame**. Because this
module is intentionally stateless (no fitted mapping is persisted), codes may differ
between training and inference if category sets or ordering differ.

For production modeling pipelines that require consistent encodings across datasets,
consider:
- pre-encoding categoricals upstream (one-hot, target encoding, etc.), or
- introducing a fitted encoder with persisted category mappings.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def encode_non_numeric_as_category_codes(
    feature_frame: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    dtype: str = "int32",
) -> pd.DataFrame:
    """
    Encode non-numeric feature columns as categorical codes.

    Parameters
    ----------
    feature_frame : pandas.DataFrame
        Feature DataFrame whose columns will be encoded as needed.
    columns : Iterable[str] | None, default None
        Columns to consider for encoding. If None, all columns are considered.
    dtype : str, default "int32"
        Output dtype for encoded columns. (Booleans are converted to 0/1 and cast
        to this dtype; categorical codes are integers cast to this dtype.)

    Returns
    -------
    pandas.DataFrame
        Copy of ``feature_frame`` where:
        - boolean columns are converted to {0, 1}
        - non-numeric, non-boolean columns are replaced by categorical integer codes

    Notes
    -----
    - Missing values in non-numeric columns are assigned the code ``-1`` by pandas.
    - Category ordering is made deterministic by sorting observed values by their
      string representation before assigning codes.
    """
    df = feature_frame.copy()
    selected = list(df.columns) if columns is None else list(columns)

    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in feature_frame: {missing}.")

    for col in selected:
        s = df[col]

        # Convert booleans to 0/1 (numeric), for a clean numeric feature matrix.
        if is_bool_dtype(s):
            df[col] = s.astype("int8").astype(dtype)
            continue

        # Leave numeric columns unchanged.
        if is_numeric_dtype(s):
            continue

        # Deterministic category ordering for a given input DataFrame.
        observed = s.dropna().unique().tolist()
        categories = sorted(observed, key=lambda x: str(x))

        cat = pd.Categorical(s, categories=categories, ordered=False)
        df[col] = cat.codes.astype(dtype)

    return df
