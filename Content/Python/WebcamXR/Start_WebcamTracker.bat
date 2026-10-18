@echo off
rem Starts the webcam tracker. On first run it creates a Python virtual
rem environment in %LOCALAPPDATA%\WebcamXR\venv and installs the packages
rem (needs internet once). The environment is kept outside the Unreal project
rem so it is not copied, packaged or scanned with the project content.
rem Extra arguments are passed to the tracker, e.g. --camera 1 or --osc-ip 192.168.1.20
setlocal
cd /d "%~dp0"

set "VENV=%LOCALAPPDATA%\WebcamXR\venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed"

if exist "%VENV_PY%" goto install

echo Creating Python environment in %VENV% ...
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv "%VENV%"
) else (
    python -m venv "%VENV%"
)
if not exist "%VENV_PY%" (
    echo.
    echo ERROR: Could not create the Python environment.
    echo Install Python 3 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
    echo then run this file again.
    pause
    exit /b 1
)

:install
rem Reinstall packages only when requirements.txt changed since the last install.
if exist "%STAMP%" (
    fc /b requirements.txt "%STAMP%" >nul 2>nul
    if not errorlevel 1 goto run
)
echo Installing packages (first run only, this can take a few minutes) ...
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)
copy /y requirements.txt "%STAMP%" >nul

:run
echo Starting webcam tracker. Press q in the camera window to quit.
rem -B: do not write __pycache__ into the project's Content folder.
"%VENV_PY%" -B combined_tracking.py %*
if errorlevel 1 pause
