@echo off
echo 🐍 Starting SentimentIQ Python Backend...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Navigate to backend directory
cd backend

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

echo.
echo 🚀 Starting the API server...
echo 📍 Backend will be available at: http://localhost:5000
echo 🌐 Frontend should be running at: http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py