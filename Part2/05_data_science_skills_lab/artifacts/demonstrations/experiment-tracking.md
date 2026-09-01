# experiment-tracking demonstration

## Scenario

**Dataset:** House Prices: Advanced Regression Techniques  
**Kaggle:** `https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques`  
**Objective:** Log data hash, parameters, folds, metrics and artifacts across runs.

## Acquire

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

## Skill workflow demonstrated

1. Confirm the business decision, unit of analysis, target/metric definition, and data provenance.
2. Validate schema, grain, missingness, duplicates, temporal boundaries, and leakage risks.
3. Execute: Log data hash, parameters, folds, metrics and artifacts across runs.
4. Compare against a simple baseline or current-state definition where applicable.
5. Quantify uncertainty, sensitivity, limitations, and affected stakeholder groups.
6. Save code, parameters, data version, evidence, and an action-oriented handoff.

## Dataset contract

The demonstration uses the dataset for leakage-safe regression and feature engineering. Raw data remains immutable; derived data and transformations are versioned. Identifiers and sensitive attributes are minimized in outputs.

## Acceptance criteria

- The stated unit of analysis is preserved through joins and aggregation.
- Preprocessing is fit on training data only for predictive tasks.
- Temporal tasks use time-respecting validation; experiments report uncertainty.
- Results include at least one baseline, slice, or reconciliation check.
- Claims distinguish correlation, prediction, and causal evidence.
- Deliverables identify an owner, next action, limitation, and reproducible command.

## Evidence status

`COMPLETE — protocol, dataset mapping, command, checks, and handoff specified.`
