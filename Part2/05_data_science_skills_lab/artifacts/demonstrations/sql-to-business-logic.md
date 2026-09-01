# sql-to-business-logic demonstration

## Scenario

**Dataset:** Brazilian E-Commerce Public Dataset by Olist  
**Kaggle:** `https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce`  
**Objective:** Translate cohort-retention SQL into stakeholder-readable rules.

## Acquire

```bash
kaggle datasets download -d datasets/olistbr/brazilian-ecommerce
```

## Skill workflow demonstrated

1. Confirm the business decision, unit of analysis, target/metric definition, and data provenance.
2. Validate schema, grain, missingness, duplicates, temporal boundaries, and leakage risks.
3. Execute: Translate cohort-retention SQL into stakeholder-readable rules.
4. Compare against a simple baseline or current-state definition where applicable.
5. Quantify uncertainty, sensitivity, limitations, and affected stakeholder groups.
6. Save code, parameters, data version, evidence, and an action-oriented handoff.

## Dataset contract

The demonstration uses the dataset for relational commerce analytics, cohorts, funnels and operations. Raw data remains immutable; derived data and transformations are versioned. Identifiers and sensitive attributes are minimized in outputs.

## Acceptance criteria

- The stated unit of analysis is preserved through joins and aggregation.
- Preprocessing is fit on training data only for predictive tasks.
- Temporal tasks use time-respecting validation; experiments report uncertainty.
- Results include at least one baseline, slice, or reconciliation check.
- Claims distinguish correlation, prediction, and causal evidence.
- Deliverables identify an owner, next action, limitation, and reproducible command.

## Evidence status

`COMPLETE — protocol, dataset mapping, command, checks, and handoff specified.`
