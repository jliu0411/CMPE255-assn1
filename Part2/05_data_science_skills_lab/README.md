# Atlas — Data Science Skills Lab

Atlas is an executable coverage lab for 46 installed agent skills: 15 from [`param087/agent-ml-skills`](https://github.com/param087/agent-ml-skills) and 31 from [`nimrodfisher/data-analytics-skills`](https://github.com/nimrodfisher/data-analytics-skills). Every skill is mapped to a popular Kaggle dataset, a concrete task, a reproducible command, an evidence report, and acceptance criteria.

## Installed globally

The skills were installed in `C:\Users\James\.codex\skills`. Restart Codex or begin a new turn before expecting automatic skill activation.

## Run the lab

```powershell
cd CMPE255-assn1/Part2/05_data_science_skills_lab
python scripts/run_lab.py
python -m uvicorn app.main:app --reload --port 8005
```

Open http://127.0.0.1:8005. The runner creates one auditable report per skill under `artifacts/demonstrations/` and a machine-readable `artifacts/results.json`.

## Demonstration design

The lab uses scenario-level demonstrations instead of copying full Kaggle datasets into git. Each report includes:

- dataset source and download command;
- business question and expected input contract;
- skill-specific workflow and deliverable;
- executable starter code or analysis protocol;
- leakage, privacy, validation, and interpretation checks;
- explicit completion criteria.

Several compatible skills share a dataset intentionally: this demonstrates realistic handoffs such as planning → EDA → cleaning → feature engineering → pipeline → evaluation → serving. Dataset downloads require Kaggle credentials and acceptance of each dataset's terms.

## Coverage gate

```powershell
python scripts/check_coverage.py
```

The gate fails unless all 46 unique skills have a dataset, scenario, category, evidence file, and completed status.
