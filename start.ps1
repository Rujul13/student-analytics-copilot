$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$apiPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $apiPython)) {
    throw "Python environment not found. Follow the setup steps in README.md first."
}

Start-Process -FilePath $apiPython `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory (Join-Path $projectRoot "backend") `
    -WindowStyle Hidden

Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev") `
    -WorkingDirectory (Join-Path $projectRoot "frontend") `
    -WindowStyle Hidden

Write-Output "Student Analytics Copilot is starting at http://127.0.0.1:5173"

