#!/bin/bash

# Federated Learning Template - Quick Setup and Test Script

echo "🚀 Setting up Federated Learning Template with FLWR + PyTorch"
echo "=============================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Check installations
echo "🔍 Checking installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import flwr; print(f'Flower: {flwr.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

echo ""
echo "✅ All dependencies installed successfully!"
echo ""

# Run system information
echo "📊 System Information:"
python3 config.py

# Run tests
echo "🧪 Running system tests..."
python3 test_system.py

echo ""
echo "🎯 Quick Start Options:"
echo "======================="
echo ""
echo "1. Run basic example:"
echo "   python3 main.py --create-sample --num-rounds 3"
echo ""
echo "2. Run with custom configuration:"
echo "   python3 main.py --create-sample --model-type advanced --num-rounds 5 --batch-size 64"
echo ""
echo "3. Run all examples:"
echo "   python3 examples.py --all"
echo ""
echo "4. Run specific example:"
echo "   python3 examples.py --example 3"
echo ""
echo "5. Use your own datasets:"
echo "   python3 main.py --dataset-paths data1.csv data2.csv data3.csv"
echo ""
echo "6. Use datasets from directory:"
echo "   python3 main.py --data-dir /path/to/csv/files"
echo ""

# Test basic functionality
echo "🚀 Testing basic functionality..."
python3 main.py --create-sample --num-rounds 1

echo ""
echo "🎉 Setup complete! The federated learning template is ready to use."
echo ""
echo "📚 For more information, check README.md"
echo "🔧 For configuration options, see config.py"
echo "📋 For examples, run: python3 examples.py --all"
