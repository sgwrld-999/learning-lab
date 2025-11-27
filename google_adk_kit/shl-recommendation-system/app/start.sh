#!/bin/bash

# SHL Recommendation System - Startup Script

echo "🚀 Starting SHL Assessment Recommendation System"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY"
    echo "   Then run this script again."
    exit 1
fi

# Check if data file exists
DATA_FILE="../data/individual-assessment.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "   Please ensure the data file exists."
    exit 1
fi

echo "✓ Environment ready"
echo "✓ Data file found ($(wc -l < $DATA_FILE) lines)"
echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python main.py
