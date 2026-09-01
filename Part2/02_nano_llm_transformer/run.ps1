param([int]$Port = 8002, [int]$TrainSteps = 200, [switch]$Retrain)
$ErrorActionPreference = 'Stop'; Push-Location $PSScriptRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is required.' }
    $missing = python -c "import importlib.util; print(','.join(x for x in ['torch','fastapi','uvicorn','numpy'] if importlib.util.find_spec(x) is None))"
    if ($missing) { Write-Host "Installing missing dependencies: $missing (first run may take several minutes)..." -ForegroundColor Yellow; python -m pip install -r requirements.txt; if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' } }
    if (-not (Test-Path 'data/processed/train.bin') -or -not (Test-Path 'data/processed/val.bin')) { python scripts/prepare_data.py; if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' } }
    if ($Retrain -or -not (Test-Path 'artifacts/model.pt')) {
        Write-Host "Training the laptop model for $TrainSteps steps..." -ForegroundColor Yellow
        python -m core.train --max-steps $TrainSteps --eval-interval 25
        if ($LASTEXITCODE -ne 0) { throw 'Model training failed.' }
    }
    Write-Host "Starting Lumen at http://127.0.0.1:$Port" -ForegroundColor Green
    python -m uvicorn server.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Lumen exited with code $LASTEXITCODE." }
} finally { Pop-Location }
