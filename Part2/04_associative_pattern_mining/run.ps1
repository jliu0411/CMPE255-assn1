param([int]$Port = 8004, [switch]$Remine)
$ErrorActionPreference = 'Stop'; Push-Location $PSScriptRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is required.' }
    $missing = python -c "import importlib.util; print(','.join(x for x in ['fastapi','uvicorn','pandas','numpy'] if importlib.util.find_spec(x) is None))"
    if ($missing) { Write-Host "Installing missing dependencies: $missing" -ForegroundColor Yellow; python -m pip install -r requirements.txt; if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' } }
    if (-not (Test-Path 'data/raw/sample_online_retail.csv')) { python scripts/generate_sample_data.py; if ($LASTEXITCODE -ne 0) { throw 'Sample-data generation failed.' } }
    if ($Remine -or -not (Test-Path 'artifacts/rules.json')) { python -m src.mine; if ($LASTEXITCODE -ne 0) { throw 'Association-rule mining failed.' } }
    Write-Host "Starting Affinity at http://127.0.0.1:$Port" -ForegroundColor Green
    python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Affinity exited with code $LASTEXITCODE." }
} finally { Pop-Location }
