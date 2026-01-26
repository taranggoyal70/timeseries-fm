#!/bin/bash

echo "========================================="
echo "Starting Chronos-2 Backend Server"
echo "========================================="

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Start the server
echo ""
echo "========================================="
echo "Backend server starting on port 8000"
echo "API Docs: http://localhost:8000/docs"
echo "========================================="
echo ""

python app.py
