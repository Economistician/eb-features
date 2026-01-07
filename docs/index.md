# eb-features

`eb-features` provides feature engineering utilities used to construct model-ready inputs from raw or intermediate Electric Barometer data.

This package focuses on **feature definition and transformation**, not model training, evaluation, or optimization policy logic.

## Scope

This package is responsible for:

- Defining reusable feature transformations
- Encoding temporal, contextual, and structural signals
- Producing standardized feature sets for downstream modeling
- Supporting consistent feature semantics across the EB ecosystem

It intentionally avoids evaluation logic, optimization policies, or runtime orchestration.

## Contents

- **Feature builders**  
  Utilities for constructing derived features from raw inputs

- **Feature transforms**  
  Reusable transformations applied across modeling workflows

## API reference

- [Feature APIs](api/)
