#!/bin/bash

echo "========================================="
echo "Starting Chronos-2 Frontend Server"
echo "========================================="

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies (this may take a few minutes)..."
    npm install
else
    echo "Dependencies already installed"
fi

# Start the development server
echo ""
echo "========================================="
echo "Frontend server starting on port 3000"
echo "Open: http://localhost:3000"
echo "========================================="
echo ""

npm run dev
