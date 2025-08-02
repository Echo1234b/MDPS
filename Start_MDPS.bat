@echo off
echo Starting MDPS...
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python run_mdps.py
if errorlevel 1 (
    echo Error starting MDPS
    pause
)
