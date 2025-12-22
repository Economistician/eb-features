from __future__ import annotations

import pandas as pd
import pytest

from eb_features.panel.validation import (
    ensure_columns_present,
    validate_monotonic_timestamps,
    validate_required_columns,
)


def test_validate_required_columns_passes_when_present() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 1],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
            "target": [1.0, 2.0, 3.0],
        }
    )

    # Should not raise
    validate_required_columns(df, required_cols=("entity_id", "timestamp", "target"))


def test_validate_required_columns_raises_when_missing() -> None:
    df = pd.DataFrame({"entity_id": [0], "timestamp": [pd.Timestamp("2025-01-01")]})

    with pytest.raises(KeyError, match="missing|required|Missing|Input DataFrame"):
        validate_required_columns(df, required_cols=("entity_id", "timestamp", "target"))


def test_ensure_columns_present_passes_on_empty_list() -> None:
    df = pd.DataFrame({"a": [1]})
    ensure_columns_present(df, columns=[], label="Anything")  # should not raise


def test_ensure_columns_present_raises_for_missing_with_label() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(KeyError, match="Static|missing|not found|columns"):
        ensure_columns_present(df, columns=["a", "c"], label="Static")


def test_validate_monotonic_timestamps_passes_when_strictly_increasing() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0, 1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                    "2025-01-05 00:00:00",
                    "2025-01-05 01:00:00",
                ]
            ),
            "target": [1, 2, 3, 10, 11],
        }
    )

    validate_monotonic_timestamps(df, entity_col="entity_id", timestamp_col="timestamp")


def test_validate_monotonic_timestamps_raises_when_equal_timestamps() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:00", "2025-01-01 01:00:00"]
            ),
            "target": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing|monotonic|Timestamps"):
        validate_monotonic_timestamps(df, entity_col="entity_id", timestamp_col="timestamp")


def test_validate_monotonic_timestamps_raises_when_decreasing() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 02:00:00", "2025-01-01 01:00:00"]
            ),
            "target": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing|monotonic|Timestamps"):
        validate_monotonic_timestamps(df, entity_col="entity_id", timestamp_col="timestamp")


def test_validate_monotonic_timestamps_works_with_timezone_aware() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [0, 0, 0],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 01:00:00", "2025-01-01 02:00:00"],
                utc=True,
            ),
            "target": [1, 2, 3],
        }
    )

    validate_monotonic_timestamps(df, entity_col="entity_id", timestamp_col="timestamp")