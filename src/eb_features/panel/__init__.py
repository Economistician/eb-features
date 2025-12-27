"""
Panel time-series feature engineering.

This subpackage provides lightweight, frequency-agnostic feature engineering utilities
for *panel* time-series data, i.e., data indexed by:

- an **entity** identifier (store, SKU, customer, sensor, etc.), and
- a **timestamp** column

The primary public interface is:

- [`FeatureConfig`][eb_features.panel.engineering.FeatureConfig]: declarative feature configuration
- [`FeatureEngineer`][eb_features.panel.engineering.FeatureEngineer]: stateless transformer that
  produces a model-ready ``(X, y, feature_names)`` triple from a long-form DataFrame.

Design goals
------------
- **Stateless**: no fit/transform lifecycle; features are generated deterministically from inputs.
- **Frequency-agnostic**: lags and rolling windows are expressed in index steps, not wall-clock units.
- **Leakage-aware**: feature construction is intended to use only information available at or
  before prediction time (depending on configuration and implementation details).

Notes
-----
This subpackage is intentionally conservative in scope. It focuses on producing features suitable
for classical supervised learning pipelines (tree models, linear models, shallow neural nets) that
expect a fixed-width design matrix.

See Also
--------
- [`eb_features.panel.engineering.FeatureEngineer`][eb_features.panel.engineering.FeatureEngineer]
- [`eb_features.panel.engineering.FeatureConfig`][eb_features.panel.engineering.FeatureConfig]
"""

from __future__ import annotations

from eb_features.panel.engineering import FeatureConfig, FeatureEngineer

__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
]
