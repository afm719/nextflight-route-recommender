@echo off
setlocal
title Next Trip - AI Recommender System

echo =========================================================
echo       NEXT TRIP: AI ROUTE RECOMMENDER SYSTEM
echo =========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed on this system.
    echo Please install Python from https://www.python.org/
    pause
    exit /b
)

if not exist "info" (
    echo [1/3] Creating virtual environment (info)...
    python -m venv info
)

echo [2/3] Activating environment and installing dependencies...
echo       (This might take a while the first time)
call info\Scripts\activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo [3/3] Starting the AI server...
echo.
echo =========================================================
echo  THE PROJECT WILL OPEN IN YOUR BROWSER SHORTLY...
echo  (If it doesn't, go to: http://127.0.0.1:8000)
echo =========================================================
echo.

start "" "http://127.0.0.1:8000"

uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info

pause