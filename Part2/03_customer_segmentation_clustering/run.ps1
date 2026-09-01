param([int]$Port = 8003, [switch]$Retrain)
$ErrorActionPreference = 'Stop'; Push-Location $PSScriptRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is required.' }
    $missing = python -c "import importlib.util; print(','.join(x for x in ['fastapi','uvicorn','pandas','numpy','sklearn','joblib'] if importlib.util.find_spec(x) is None))"
    if ($missing) { Write-Host "Installing missing dependencies: $missing" -ForegroundColor Yellow; python -m pip install -r requirements.txt; if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' } }
    if (-not (Test-Path 'data/raw/sample_marketing_campaign.csv')) { python scripts/generate_sample_data.py; if ($LASTEXITCODE -ne 0) { throw 'Sample-data generation failed.' } }
    if ($Retrain -or -not (Test-Path 'artifacts/segments.joblib')) { python -m src.train; if ($LASTEXITCODE -ne 0) { throw 'Segmentation training failed.' } }
    Write-Host "Starting Mosaic at http://127.0.0.1:$Port" -ForegroundColor Green
    python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Mosaic exited with code $LASTEXITCODE." }
} finally { Pop-Location }
