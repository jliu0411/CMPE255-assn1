# One-command project launchers

Each Part 2 project has a self-contained PowerShell launcher. Open a terminal in the desired project directory and run:

```powershell
.\run.ps1
```

| Project | URL | Launcher behavior |
|---|---|---|
| `00_dynamic_todo_workspace` | http://localhost:4173 | Starts the Node todo application |
| `01_nyc_taxi_trip_prediction` | http://127.0.0.1:8000 | Installs missing dependencies, creates sample data/model when absent, starts FastAPI |
| `02_nano_llm_transformer` | http://127.0.0.1:8002 | Installs missing dependencies, prepares data, trains a 200-step checkpoint when absent, starts FastAPI |
| `03_customer_segmentation_clustering` | http://127.0.0.1:8003 | Creates sample data/segments when absent and starts FastAPI |
| `04_associative_pattern_mining` | http://127.0.0.1:8004 | Creates sample transactions/rules when absent and starts FastAPI |
| `05_data_science_skills_lab` | http://127.0.0.1:8005 | Regenerates evidence, checks 46-skill coverage, starts FastAPI |

From the `Part2` directory, you can launch a project without changing directories:

```powershell
& ".\00_dynamic_todo_workspace\run.ps1"
& ".\01_nyc_taxi_trip_prediction\run.ps1"
& ".\02_nano_llm_transformer\run.ps1"
& ".\03_customer_segmentation_clustering\run.ps1"
& ".\04_associative_pattern_mining\run.ps1"
& ".\05_data_science_skills_lab\run.ps1"
```

Stop a running server with `Ctrl+C`.

## Optional parameters

```powershell
.\run.ps1 -Port 9000
```

Projects 01 and 03 accept `-Retrain`; project 04 accepts `-Remine`. NanoLM accepts `-Retrain` and `-TrainSteps`:

```powershell
.\run.ps1 -Retrain -TrainSteps 500
```
