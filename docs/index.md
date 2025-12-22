# eb-features

**eb-features** is the feature engineering library of the **Electric Barometer** ecosystem.

It provides a principled, leakage-safe set of utilities for constructing
**panel time-series features**—features derived from entity × timestamp data
that are suitable for classical machine learning and forecasting pipelines.

The focus of `eb-features` is not generic feature transformation, but
**operationally correct feature construction**: features that respect
time ordering, entity boundaries, and real-world deployment constraints.

---

## Naming convention

Electric Barometer packages follow a consistent naming convention:

- **Distribution names** (used with `pip install`) use hyphens  
  e.g. `pip install eb-features`
- **Python import paths** use underscores  
  e.g. `import eb_features`

This follows standard Python packaging practices and avoids ambiguity between
package names and module imports.

---

## What this package provides

### Panel-aware lag features
Lagged representations of the target series computed **strictly within each entity**.

- Arbitrary positive lag steps (expressed in index steps)
- No cross-entity leakage
- Deterministic naming (`lag_k`)

---

### Rolling window statistics
Rolling summary features over recent history, designed for forecasting use.

- Multiple statistics: mean, sum, min, max, median, std
- Configurable window lengths
- **Leakage-safe by default** (excludes the current target value)

---

### Calendar and time-derived features
Time-based features extracted from timestamp columns.

- Hour of day, day of week, day of month, month
- Weekend indicators
- Optional **cyclical encodings** (sine / cosine) for periodic components

---

### Feature orchestration
A unified feature engineering interface for panel data.

- Deterministic feature construction
- Explicit configuration via `FeatureConfig`
- Automatic passthrough of regressors and static metadata
- Safe handling of missing history

---

### Encoding and validation utilities
Supporting utilities to ensure model-ready feature matrices.

- Stateless encoding of non-numeric features using categorical codes
- Validation of required columns and monotonic timestamps
- Guardrails against invalid configurations and data leakage

---

## Documentation structure

- **API Reference**  
  All feature engineering utilities are documented automatically from
  NumPy-style docstrings in the source code using `mkdocstrings`.

Conceptual motivation, design rationale, and formal treatment of leakage,
readiness, and operational constraints are documented in the companion
research repository **eb-papers**.

---

## Intended audience

This documentation is intended for:

- data scientists and ML practitioners working with time-series data
- forecasting and demand-planning teams
- operations and service analytics leaders
- researchers studying leakage, readiness, and deployment-safe modeling

The emphasis throughout is on **operational correctness**, not convenience
or generic feature generation.

---

## Relationship to the Electric Barometer framework

`eb-features` provides the **feature layer** of the Electric Barometer ecosystem.
It is designed to be used alongside:

- **eb-metrics** — cost-aware and readiness-focused evaluation metrics
- **eb-evaluation** — structured forecast evaluation workflows
- **eb-adapters** — integrations with external forecasting systems
- **eb-papers** — formal definitions, theory, and technical notes

Together, these components support a unified approach to building,
evaluating, and selecting forecasts that are **deployment-ready**, not
just statistically accurate.