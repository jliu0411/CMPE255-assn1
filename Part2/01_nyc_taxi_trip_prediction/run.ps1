param([int]$Port = 8000, [switch]$Retrain)
$ErrorActionPreference = 'Stop'; Push-Location $PSScriptRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is required.' }
    $missing = python -c "import importlib.util; print(','.join(x for x in ['fastapi','uvicorn','pandas','numpy','sklearn','joblib'] if importlib.util.find_spec(x) is None))"
    if ($missing) { Write-Host "Installing missing dependencies: $missing" -ForegroundColor Yellow; python -m pip install -r requirements.txt; if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' } }
    if (-not (Test-Path 'data/raw/sample_train.csv')) { python scripts/generate_sample_data.py; if ($LASTEXITCODE -ne 0) { throw 'Sample-data generation failed.' } }
    $modelPath = 'artifacts/taxi_duration_model.joblib'
    $needsTraining = $Retrain -or -not (Test-Path $modelPath)
    if (-not $needsTraining) {
        # Joblib/scikit-learn artifacts are not guaranteed to load after a
        # library upgrade. Validate the artifact before starting the API.
        # Windows PowerShell can promote a native program's stderr to a
        # terminating error when ErrorActionPreference is Stop. A failed load
        # is expected here, so silence only this probe and inspect its exit code.
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        python -c "import joblib; artifact=joblib.load(r'$modelPath'); assert 'model' in artifact and 'metadata' in artifact" 1>$null 2>$null
        $artifactLoadExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorActionPreference
        if ($artifactLoadExitCode -ne 0) {
            Write-Host 'The saved model is incompatible with this Python environment. Retraining it now...' -ForegroundColor Yellow
            $needsTraining = $true
        }
    }
    if ($needsTraining) {
        python -m src.train
        if ($LASTEXITCODE -ne 0) { throw 'Model training failed.' }
    }
    Write-Host "Starting RideCast NYC at http://127.0.0.1:$Port" -ForegroundColor Green
    python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "RideCast exited with code $LASTEXITCODE." }
} finally { Pop-Location }
