# Electric Barometer · Features (`eb-features`)

[![CI](https://github.com/Economistician/eb-features/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-features/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-features)
![PyPI](https://img.shields.io/pypi/v/eb-features)

Feature engineering primitives for panel-based forecasting systems, designed to integrate seamlessly with the Electric Barometer ecosystem.

---

## Overview

`eb-features` is a modular feature engineering library for panel-based forecasting systems. It provides reusable, deterministic transformations for constructing time-aware features across entities observed over time.

Within the Electric Barometer ecosystem, `eb-features` serves as the upstream feature construction layer, producing standardized inputs for downstream evaluation and metric components. While designed to integrate seamlessly with Electric Barometer, the package remains framework-agnostic and can be used independently in other forecasting workflows.

---

## Role in the Electric Barometer Ecosystem

`eb-features` defines the feature engineering primitives used throughout the Electric Barometer ecosystem. It is responsible for constructing deterministic, panel-aware input features—such as lags, rolling aggregations, and calendar encodings—that serve as the foundational inputs to forecasting and evaluation workflows.

This package focuses exclusively on feature construction and validation. It does not perform model training, forecast generation, metric evaluation, or decision logic. Those responsibilities are handled by downstream layers in the ecosystem that consume engineered features for modeling, selection, and operational assessment.

By separating feature semantics from modeling and evaluation concerns, `eb-features` provides a stable, reusable foundation that ensures consistency and reproducibility across forecasting pipelines operating on heterogeneous panel data.

---

## Installation

`eb-features` is distributed as a standard Python package.

```bash
pip install eb-features
```

The package supports Python 3.10 and later.

---

## Core Concepts

- **Panel-aware feature construction** — Features are constructed with explicit awareness of entity boundaries and temporal ordering, ensuring correctness in multi-entity forecasting settings.
- **Deterministic transformations** — Feature generation is designed to be reproducible and free of stochastic behavior, supporting auditability and consistent downstream evaluation.
- **Temporal causality** — All features respect time directionality, preventing information leakage from future observations into historical feature sets.
- **Rolling and lag semantics** — Common forecasting features such as lags and rolling aggregates are treated as first-class primitives with clear, well-defined behavior.
- **Validation by construction** — Feature pipelines include explicit checks and constraints to ensure structural validity before model training or evaluation.

---

## Minimal Example

The example below shows how to construct lagged and rolling features for panel data while preserving entity boundaries and temporal ordering.

```python
import pandas as pd
from eb_features.panel.lags import add_lag_features
from eb_features.panel.rolling import add_rolling_features

# Example panel data
df = pd.DataFrame({
    "entity_id": ["A", "A", "A", "B", "B"],
    "date": pd.date_range("2024-01-01", periods=5, freq="D"),
    "y": [10, 12, 11, 7, 9],
})

# Add lagged features
df = add_lag_features(
    df,
    value_col="y",
    lags=[1, 2],
    entity_col="entity_id",
    time_col="date",
)

# Add rolling features
df = add_rolling_features(
    df,
    value_col="y",
    windows=[3],
    entity_col="entity_id",
    time_col="date",
)

print(df)
```

---

## License

BSD 3-Clause License.  
© 2025 Kyle Corrie.

<!-- Trusted publishing verification -->