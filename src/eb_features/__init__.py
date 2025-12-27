"""
Feature engineering utilities for the Electric Barometer ecosystem.

This package provides modular, domain-agnostic feature engineering components
used across forecasting, evaluation, and operational modeling workflows.

Subpackages
-----------
panel
    Feature engineering utilities for panel (entity x timestamp) time-series data.

Notes
-----
This package intentionally exposes a small public API. Most functionality is
accessed through subpackages such as ``eb_features.panel``.
"""

from __future__ import annotations

__all__: list[str] = []
