# RideCast NYC — End-to-End Taxi Trip Duration Prediction

An end-to-end CRISP-DM data science project for the Kaggle **NYC Taxi Trip Duration** challenge. It includes data acquisition, validation, feature engineering, model training and comparison, evaluation, artifact packaging, a FastAPI deployment, tests, and a polished responsive prediction experience.

## Quick start

```powershell
python scripts/generate_sample_data.py
python -m src.train
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. A trained demo artifact is included after running the first two commands.

## Use the real Kaggle data

Accept the competition rules and configure Kaggle credentials, then run:

```powershell
python scripts/download_data.py
python -m src.train --data data/raw/train.csv --max-rows 500000
```

Remove `--max-rows` for the complete training set. The downloader uses the official Kaggle CLI and never stores credentials in this repository.

## Project layout

```text
app/                 FastAPI service and web UI
artifacts/           Versioned model and metrics
data/raw/            Raw/sample data (large Kaggle files ignored)
docs/                CRISP-DM report and model card
notebooks/           Reproducible EDA script
scripts/             Data acquisition and sample generation
src/                 Features, validation, and training pipeline
tests/               Unit and API tests
```

## Quality commands

```powershell
python -m pytest -q
python notebooks/01_eda.py
```

The Kaggle competition evaluates RMSLE and expects `id,trip_duration` submission columns. Production predictions are constrained to positive durations and the UI clearly identifies the model as an estimate.
