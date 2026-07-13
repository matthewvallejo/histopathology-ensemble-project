@echo off
setlocal
REM Double-click this file to launch the Ensemble GUI using the project venv.
REM
REM The venv location is resolved in this priority order:
REM   1. HISTO_VENV   - path to the venv root, or directly to pythonw.exe
REM   2. .venv        - a virtual environment inside this project folder
REM   3. default      - %USERPROFILE%\venvs\histopathology
REM
REM Quiet TensorFlow's C++ startup logging (oneDNN/absl) before Python imports
REM it; the GUI captures Python output separately, so this is purely cosmetic.
set "TF_CPP_MIN_LOG_LEVEL=1"

set "VENV_PYTHON="
if defined HISTO_VENV (
    if exist "%HISTO_VENV%\Scripts\pythonw.exe" (
        set "VENV_PYTHON=%HISTO_VENV%\Scripts\pythonw.exe"
    ) else (
        set "VENV_PYTHON=%HISTO_VENV%"
    )
)
if not defined VENV_PYTHON if exist "%~dp0.venv\Scripts\pythonw.exe" set "VENV_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not defined VENV_PYTHON if exist "%USERPROFILE%\venvs\histopathology\Scripts\pythonw.exe" set "VENV_PYTHON=%USERPROFILE%\venvs\histopathology\Scripts\pythonw.exe"

if not defined VENV_PYTHON (
    echo Virtual environment not found.
    echo Looked for HISTO_VENV, .\.venv, and %%USERPROFILE%%\venvs\histopathology
    echo Set HISTO_VENV to your venv folder, create a .venv here, or see run_gui.ps1.
    pause
    exit /b 1
)
if not exist "%VENV_PYTHON%" (
    echo Python not found at %VENV_PYTHON%
    pause
    exit /b 1
)
start "" "%VENV_PYTHON%" "%~dp0ensemble_gui.py"
