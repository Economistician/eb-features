from __future__ import annotations

import dataclasses

import pytest

from eb_features.panel.engineering import FeatureConfig


def test_feature_config_defaults_are_populated_and_non_empty() -> None:
    cfg = FeatureConfig()

    # Defaults should be present and non-empty (unless intentionally changed upstream)
    assert cfg.lag_steps is not None
    assert len(list(cfg.lag_steps)) > 0

    assert cfg.rolling_windows is not None
    assert len(list(cfg.rolling_windows)) > 0

    assert len(list(cfg.rolling_stats)) > 0
    assert len(list(cfg.calendar_features)) > 0

    # Basic boolean defaults
    assert isinstance(cfg.use_cyclical_time, bool)
    assert isinstance(cfg.dropna, bool)
    assert isinstance(cfg.leakage_safe_rolling, bool)


def test_feature_config_is_frozen() -> None:
    cfg = FeatureConfig()

    # dataclasses.FrozenInstanceError is raised on assignment
    with pytest.raises(dataclasses.FrozenInstanceError):
        # type: ignore[misc]
        cfg.dropna = False


def test_feature_config_default_factories_do_not_share_state() -> None:
    cfg1 = FeatureConfig()
    cfg2 = FeatureConfig()

    # They should be distinct list objects (no shared mutable state)
    assert cfg1.lag_steps is not cfg2.lag_steps
    assert cfg1.rolling_windows is not cfg2.rolling_windows
    assert cfg1.rolling_stats is not cfg2.rolling_stats
    assert cfg1.calendar_features is not cfg2.calendar_features

    # Mutating cfg1's lists should not affect cfg2 (even though cfg itself is frozen,
    # the contained lists can still be mutated, so we test for independence)
    cfg1.lag_steps.append(999)  # type: ignore[union-attr]
    assert 999 not in list(cfg2.lag_steps)


def test_feature_config_accepts_none_for_optional_sequences() -> None:
    cfg = FeatureConfig(lag_steps=None, rolling_windows=None)

    assert cfg.lag_steps is None
    assert cfg.rolling_windows is None
