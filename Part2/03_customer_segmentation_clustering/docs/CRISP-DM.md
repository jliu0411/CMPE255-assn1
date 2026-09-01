# CRISP-DM Report

## 1. Business understanding

The marketing team needs a manageable set of behavioral audiences for differentiated retention, cross-sell, and reactivation experiments. Success requires segments that are distinct, stable across random initializations, understandable to operators, sufficiently sized for experiments, and assignable to new customers. The segments must not be used for eligibility, pricing discrimination, or other consequential decisions.

## 2. Data understanding

The Customer Personality Analysis dataset contains about 2,240 customer records covering demographics, product spending over two years, channel purchases, promotion acceptance, recency, and complaints. Known issues include missing income, date parsing, unusual birth years, skewed monetary values, and constant administrative columns. The bundled deterministic sample mirrors its schema but is synthetic; real business conclusions require the Kaggle source.

## 3. Data preparation

The pipeline validates its schema, median-imputes income, removes duplicate identifiers and impossible records, engineers customer-level behavioral aggregates and channel shares, clips heavily skewed signals at the 1st/99th percentiles, and applies RobustScaler. Identifiers, categorical labels, and constant fields are excluded from distance calculations. PCA is fit only for visualization and not clustering.

## 4. Modeling

K-Means candidates with `k=2…8` use multiple initializations. Selection combines normalized silhouette (higher is better), Davies-Bouldin (lower is better), and cross-seed Adjusted Rand stability. Calinski-Harabasz and inertia remain visible for diagnostic context. Persona names are assigned after fitting from centroid profiles and are not training labels.

## 5. Evaluation

Internal metrics measure geometry, not commercial usefulness. Review cluster sizes, centroid differences, stability, PCA distortion, outliers, and profile coherence with stakeholders. Validate value through randomized campaign experiments with incremental conversion, retention, margin, unsubscribe rate, and fairness slices. Avoid claiming natural or permanent customer types.

## 6. Deployment

The artifact packages imputation-derived features, robust scaling, K-Means, PCA, metadata, and version. FastAPI validates incoming profiles and applies the same transformations before nearest-centroid assignment. Monitor feature missingness, distribution drift, cluster share changes, centroid distance, API errors, and business experiment outcomes. Retrain on a governed schedule or when drift thresholds trigger review.
