@echo off
title SentinAI - Hackathon Live Demonstration
cd /d "%~dp0"

echo ========================================================
echo   SentinAI - Hackathon Live Demonstration Launcher
echo ========================================================
echo.

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" demo.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo.py
) else (
    python demo.py
)

pause

