# Launches ensemble_gui.py using the project's virtual environment.
#
# The venv is kept OUTSIDE OneDrive (e.g. %USERPROFILE%\venvs\histopathology) to
# avoid Windows long-path errors and to keep package files from syncing.
#
# The venv location is resolved in this priority order:
#   1. $env:HISTO_VENV    - path to the venv root, or directly to python.exe
#   2. .venv              - a virtual environment inside this project folder
#   3. default            - %USERPROFILE%\venvs\histopathology
$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot

# Quiet TensorFlow's C++ startup logging (oneDNN/absl) before Python imports it.
# The GUI redirects Python stdout/stderr to its log pane separately, so this
# only suppresses the pre-import native noise, not the GUI's own output.
$env:TF_CPP_MIN_LOG_LEVEL = "1"

function Resolve-VenvPython([string]$root) {
    if ([string]::IsNullOrWhiteSpace($root)) { return $null }
    if ($root.ToLower().EndsWith("python.exe")) { return $root }
    return (Join-Path $root "Scripts\python.exe")
}

$defaultVenv = Join-Path $env:USERPROFILE "venvs\histopathology"
$candidates = @(
    (Resolve-VenvPython $env:HISTO_VENV),
    (Join-Path $scriptDir ".venv\Scripts\python.exe"),
    (Join-Path $defaultVenv "Scripts\python.exe")
) | Where-Object { $_ }

$venvPython = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $venvPython) {
    Write-Host "Virtual environment not found." -ForegroundColor Red
    Write-Host "Looked in: `$env:HISTO_VENV, .\.venv, and $defaultVenv" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Create one with:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv `"$defaultVenv`""
    Write-Host "  `"$defaultVenv\Scripts\python.exe`" -m pip install -r `"$scriptDir\requirements.txt`""
    Write-Host ""
    Write-Host "Or point HISTO_VENV at an existing venv:" -ForegroundColor Yellow
    Write-Host "  `$env:HISTO_VENV = 'C:\path\to\your\venv'"
    exit 1
}

& $venvPython "$scriptDir\ensemble_gui.py" @args
