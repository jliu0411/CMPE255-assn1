param([int]$Port = 4173)
$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'fullstack-test'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js 20+ is required. Install Node.js and retry.' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'npm is required. Install Node.js and retry.' }
Push-Location $project
try {
    $env:PORT = $Port
    Write-Host "Starting Momentum at http://localhost:$Port" -ForegroundColor Green
    npm start
    if ($LASTEXITCODE -ne 0) { throw "Momentum exited with code $LASTEXITCODE." }
} finally { Pop-Location }
