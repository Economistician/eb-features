from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_features.panel.encoders import encode_non_numeric_as_category_codes


def test_encode_non_numeric_as_category_codes_encodes_object_and_preserves_numeric() -> None:
    df = pd.DataFrame(
        {
            "num": [1.0, 2.5, 3.0],
            "cat": ["a", "b", "a"],
            "booly": [True, False, True],
        }
    )

    out = encode_non_numeric_as_category_codes(df)

    # Shape preserved
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)

    # Numeric column preserved exactly (dtype may stay numeric)
    assert np.allclose(out["num"].to_numpy(dtype=float), df["num"].to_numpy(dtype=float))

    # Non-numeric columns become numeric and finite
    assert np.issubdtype(out["cat"].dtype, np.number)
    assert np.issubdtype(out["booly"].dtype, np.number)
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_encode_non_numeric_as_category_codes_is_deterministic_for_same_input() -> None:
    df = pd.DataFrame({"x": ["b", "a", "b", "c"]})

    out1 = encode_non_numeric_as_category_codes(df)
    out2 = encode_non_numeric_as_category_codes(df)

    assert out1.equals(out2)


def test_encode_non_numeric_as_category_codes_handles_missing_values() -> None:
    df = pd.DataFrame({"x": ["a", None, "b", None]})

    out = encode_non_numeric_as_category_codes(df)

    # Category codes use -1 for NaN by pandas convention; ensure column is numeric
    assert np.issubdtype(out["x"].dtype, np.number)

    # Ensure the NaN rows map to -1 consistently
    codes = out["x"].to_numpy(dtype=float)
    assert (codes[[1, 3]] == -1).all()


def test_encode_non_numeric_as_category_codes_does_not_mutate_input() -> None:
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    df_copy = df.copy(deep=True)

    _ = encode_non_numeric_as_category_codes(df)

    assert df.equals(df_copy)


def test_encode_non_numeric_as_category_codes_raises_on_non_dataframe() -> None:
    with pytest.raises((TypeError, AttributeError)):
        # type: ignore[arg-type]
        encode_non_numeric_as_category_codes(["a", "b"])