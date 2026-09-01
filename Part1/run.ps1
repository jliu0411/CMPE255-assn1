$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
Set-Location $scriptRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found on PATH. Install Python 3.10+ or activate your virtual environment."
    exit 1
}
$pythonExe = $pythonCommand.Source

# Ensure symbols render in Windows PowerShell and plots work without a GUI.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"

Write-Host "Checking and installing Python dependencies..." -ForegroundColor Cyan
& $pythonExe -m pip install --disable-pip-version-check --quiet -r (Join-Path $scriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed (exit code $LASTEXITCODE)."
    exit $LASTEXITCODE
}

& $pythonExe -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, joblib"
if ($LASTEXITCODE -ne 0) {
    Write-Error "A required Python package could not be imported after installation."
    exit 1
}

$chunkScripts = @(
    "01_CRISP_DM_Chunk1_Business_and_Data_Understanding.py",
    "02_CRISP_DM_Chunk2_EDA_Visualization.py",
    "03_CRISP_DM_Chunk3_Data_Preparation.py",
    "04_CRISP_DM_Chunk4_Baseline_and_Feature_Selection.py",
    "05_CRISP_DM_Chunk5_Multiple_Algorithms.py",
    "06_CRISP_DM_Chunk6_Final_Evaluation_and_Diagnostics.py",
    "07_CRISP_DM_Chunk7_Deployment_and_Recommendations.py"
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Running full Part 1 CRISP-DM workflow" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($script in $chunkScripts) {
    Write-Host "Executing $script ..." -ForegroundColor Yellow
    & $pythonExe $script

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Chunk failed: $script (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }

    Write-Host "Completed: $script" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Part 1 workflow completed successfully." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting NBA Salary Predictor web UI..." -ForegroundColor Cyan
Write-Host "Open: http://127.0.0.1:8006" -ForegroundColor Green

# Repeated runs replace only this project's server. An unrelated process on
# the same port is reported instead of being terminated.
$existingListener = Get-NetTCPConnection -LocalPort 8006 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($existingListener) {
    $existingServer = Get-CimInstance Win32_Process -Filter "ProcessId = $($existingListener.OwningProcess)"
    if ($existingServer.CommandLine -match "salary_ui_server\.py") {
        Write-Host "Restarting existing Part 1 UI..." -ForegroundColor Yellow
    }
    else {
        Write-Error "Port 8006 is already used by another application (PID $($existingListener.OwningProcess))."
        exit 1
    }
}

# A Windows virtual-environment launcher may leave a parent and child process.
# Remove every process for this uniquely named server before starting one copy.
$existingProjectServers = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "salary_ui_server\.py" -and $_.ProcessId -ne $PID }
)
foreach ($serverProcess in $existingProjectServers) {
    Stop-Process -Id $serverProcess.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($existingProjectServers.Count -gt 0) {
    Start-Sleep -Milliseconds 500
}

$uiProcess = Start-Process -FilePath $pythonExe -ArgumentList "salary_ui_server.py" -WorkingDirectory $scriptRoot -PassThru
Start-Sleep -Seconds 1
if ($uiProcess.HasExited) {
    Write-Error "The UI server failed to start (exit code $($uiProcess.ExitCode))."
    exit $uiProcess.ExitCode
}

Write-Host "UI process started with PID: $($uiProcess.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "Generated outputs are in the data/ and visualizations/ folders." -ForegroundColor Cyan
