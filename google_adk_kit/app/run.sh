#!/bin/bash

# Activate virtual environment and run the SHL Recommendation System

echo "🚀 Starting SHL Assessment Recommendation System"
echo "================================================"

# Activate virtual environment
source /Users/siddhantgond/Desktop/Github_Modules/google_adk_kit/vir_env/bin/activate

# Navigate to app directory
cd /Users/siddhantgond/Desktop/Github_Modules/google_adk_kit/shl-recommendation-system/app

echo "✓ Virtual environment activated"
echo "✓ Starting server on http://localhost:8000"
echo ""

# Run the application
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
