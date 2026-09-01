# Data Card

- **Dataset:** Customer Personality Analysis
- **Source:** Kaggle, `imakash3011/customer-personality-analysis`
- **Unit:** One row per customer
- **Coverage:** Demographics, household, enrollment, recency, two-year product spend, purchase channels, campaign acceptance, complaints
- **Target:** None; this is unsupervised learning
- **Bundled data:** Deterministic synthetic schema-compatible sample for reproducibility
- **Sensitive attributes:** Income, household and relationship information require careful handling
- **Limitations:** Historical, small, unclear geographic context, self/operational reporting errors, no causal campaign outcomes
- **Governance:** Minimize access, do not expose individual records in telemetry, audit additions, and avoid consequential individual targeting
