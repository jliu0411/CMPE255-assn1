# data-cleaning demonstration

## Scenario

**Dataset:** Titanic — Machine Learning from Disaster  
**Kaggle:** `https://www.kaggle.com/competitions/titanic`  
**Objective:** Build train-only age/fare imputation and categorical cleanup.

## Acquire

```bash
kaggle competitions download -c titanic
```

## Skill workflow demonstrated

1. Confirm the business decision, unit of analysis, target/metric definition, and data provenance.
2. Validate schema, grain, missingness, duplicates, temporal boundaries, and leakage risks.
3. Execute: Build train-only age/fare imputation and categorical cleanup.
4. Compare against a simple baseline or current-state definition where applicable.
5. Quantify uncertainty, sensitivity, limitations, and affected stakeholder groups.
6. Save code, parameters, data version, evidence, and an action-oriented handoff.

## Dataset contract

The demonstration uses the dataset for survival classification with mixed tabular data. Raw data remains immutable; derived data and transformations are versioned. Identifiers and sensitive attributes are minimized in outputs.

## Acceptance criteria

- The stated unit of analysis is preserved through joins and aggregation.
- Preprocessing is fit on training data only for predictive tasks.
- Temporal tasks use time-respecting validation; experiments report uncertainty.
- Results include at least one baseline, slice, or reconciliation check.
- Claims distinguish correlation, prediction, and causal evidence.
- Deliverables identify an owner, next action, limitation, and reproducible command.

## Evidence status

`COMPLETE — protocol, dataset mapping, command, checks, and handoff specified.`
