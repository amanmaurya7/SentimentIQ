#!/bin/bash

echo "🐍 Starting SentimentIQ Python Backend..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip and try again."
    exit 1
fi

# Navigate to backend directory
cd backend

echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "🚀 Starting the API server..."
echo "📍 Backend will be available at: http://localhost:5000"
echo "🌐 Frontend should be running at: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py