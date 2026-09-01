# Mosaic — Customer Segmentation Intelligence

An end-to-end CRISP-DM clustering project built around Kaggle's **Customer Personality Analysis** dataset. Mosaic turns customer records into defensible, interpretable personas and serves them through an analytics dashboard and assignment API.

## Quick start

```powershell
python scripts/generate_sample_data.py
python -m src.train
python -m uvicorn app.main:app --reload --port 8003
```

Open http://127.0.0.1:8003.

## Use Kaggle data

Install/configure the Kaggle CLI, then:

```powershell
python scripts/download_data.py
python -m src.train --data data/raw/marketing_campaign.csv
```

The source file is tab-delimited despite its `.csv` extension; ingestion detects the delimiter.

## What the pipeline does

- Audits missingness, duplicates, impossible ages, outlier income, and negative activity
- Engineers age, household size, children, total spend, total purchases, campaign acceptance, tenure, and channel shares
- Winsorizes skewed monetary/activity signals and applies robust scaling
- compares K-Means solutions from 2–8 clusters with silhouette, Calinski-Harabasz, Davies-Bouldin, inertia, and stability
- Selects a solution using a documented composite score
- Projects customers to two PCA dimensions for visualization only
- Produces stable persona labels, profiles, recommended actions, artifacts, and training telemetry
- Assigns new customers without retraining through a validated FastAPI endpoint

## Tests

```powershell
python -m pytest -q
```

See [CRISP-DM.md](docs/CRISP-DM.md) for objectives, risks, evaluation, and deployment guidance. Segments describe behavioral patterns in this dataset; they are not intrinsic identities and must not be used for discriminatory treatment.
