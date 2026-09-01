# Affinity — Market Basket Intelligence

An end-to-end CRISP-DM association-pattern project for Kaggle's popular Online Retail dataset. Affinity cleans invoice lines, mines frequent itemsets and directional rules, validates them on a chronological holdout, and turns the results into a recommendation service and decision-ready admin dashboard.

## Quick start

```powershell
python scripts/generate_sample_data.py
python -m src.mine
python -m uvicorn app.main:app --reload --port 8004
```

Open http://127.0.0.1:8004.

## Use Kaggle data

Configure the Kaggle CLI, then run:

```powershell
python scripts/download_data.py
python -m src.mine --data "data/raw/Online Retail.csv" --min-support 0.015 --min-confidence 0.30 --min-lift 1.20
```

CSV and XLSX are supported (`openpyxl` is needed for XLSX). Large data is pruned to common items before pair/triple generation to bound memory.

## Mining and evaluation

- Removes cancellations, returns, zero-priced lines, missing descriptions, duplicates, and service/non-product codes
- Normalizes product descriptions and collapses invoice lines to unique basket items
- Uses an inspectable Apriori implementation for itemsets up to size three
- Computes support, confidence, lift, leverage, conviction, and rule coverage
- Uses the last 20% of invoices chronologically as a holdout
- Reports confidence drift and only labels sufficiently supported rules as stable
- Saves dashboard-ready JSON and a deployable rules artifact

Read [CRISP-DM.md](docs/CRISP-DM.md) before interpreting rules. Association is not causation: merchandising changes require controlled experiments.
