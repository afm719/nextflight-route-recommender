#!/bin/bash

echo "========================================================="
echo "      NEXT TRIP: AI ROUTE RECOMMENDER SYSTEM"
echo "========================================================="
echo ""

if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 is not installed on this system."
    echo "Please install Python from https://www.python.org/"
    exit 1
fi

if [ ! -d "info" ]; then
    echo "[1/3] Creating virtual environment (info)..."
    python3 -m venv info
fi

echo "[2/3] Activating environment and installing dependencies..."
source info/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/3] Starting the AI server..."
echo ""
echo "========================================================="
echo " THE PROJECT WILL OPEN IN YOUR BROWSER SHORTLY..."
echo " (If it doesn't, go to: http://127.0.0.1:8000)"
echo "========================================================="
echo ""

sleep 2
if which xdg-open > /dev/null; then
  xdg-open http://127.0.0.1:8000 &
elif which open > /dev/null; then
  open http://127.0.0.1:8000 &
fi

uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info