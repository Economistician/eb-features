# Electric barometer Feature Engineering (`eb-features`)

**eb-features** is the feature engineering layer of the **Electric Barometer** ecosystem.

It provides a structured, opinionated set of **panel-aware feature construction utilities**
for time-series modeling in operational environments—contexts where *temporal structure*,
*entity boundaries*, and *leakage safety* matter as much as model choice itself.

This package focuses on **deterministic, stateless feature generation** for classical
supervised learning pipelines, producing clean, model-ready design matrices from
long-form panel data.

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

### Panel-safe lag features
Lagged versions of the target series constructed **strictly within entity** boundaries.

- Configurable lag steps (index-based, frequency-agnostic)
- Deterministic naming (`lag_1`, `lag_24`, etc.)
- Explicit handling of missing history

---

### Leakage-aware rolling statistics
Rolling-window summaries designed for forecasting workflows.

- Mean, sum, min, max, std, median
- Configurable window sizes
- **Leakage-safe by default** (excludes current target value)
- Optional early availability via `min_periods`

---

### Calendar and time-derived features
Calendar attributes derived from timestamp columns.

- Hour, day-of-week, day-of-month, month
- Weekend indicators
- Optional cyclical encodings (sine/cosine) for periodic components
- Timezone-aware timestamp support

---

### Passthrough regressors and static features
Support for mixing engineered temporal features with:

- Numeric external regressors
- Static entity-level metadata
- Automatic regressor detection when not explicitly specified

Non-numeric passthrough columns are encoded using stable, dataset-local
categorical codes.

---

### Validation and guardrails
Built-in validation to catch common modeling errors early.

- Required-column checks
- Strict monotonic timestamp enforcement within entity
- Protection against cross-entity leakage
- Non-finite value detection before model handoff

---

## Design principles

`eb-features` is intentionally:

- **Stateless** — no fitted encoders or persisted mappings  
- **Deterministic** — same input + config → same output  
- **Frequency-agnostic** — works with hourly, daily, or irregular data  
- **Panel-aware** — entity boundaries are first-class constraints  

This makes it suitable for batch modeling, experimentation, and reproducible
forecast evaluation pipelines.

---

## Documentation structure

- **API Reference**  
  All feature builders and utilities are documented automatically from
  NumPy-style docstrings using `mkdocstrings`.

Conceptual motivation and modeling guidance for these features live in the
companion repositories:

- **eb-metrics** — operationally meaningful forecast metrics  
- **eb-evaluation** — structured forecast evaluation workflows  
- **eb-papers** — formal definitions and technical notes  

---

## Intended audience

This package is intended for:

- data scientists and applied ML practitioners
- forecasting and demand-planning teams
- operations and service analytics engineers
- researchers working with panel time-series data

The emphasis throughout is on **correct feature construction under operational
constraints**, not generic time-series convenience.

---

## Relationship to the Electric Barometer framework

`eb-features` provides the **feature engineering layer** of the Electric Barometer
ecosystem.

It is designed to work in concert with:

- **eb-metrics** — how forecasts are evaluated  
- **eb-evaluation** — how forecasts are compared and selected  
- **eb-adapters** — how forecasts integrate with external systems  

Together, these components support a disciplined, end-to-end approach to
*forecast readiness*—from raw data, to features, to evaluation.