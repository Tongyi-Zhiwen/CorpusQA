#!/bin/bash

# CorpusQA Setup Script
# This script helps set up the CorpusQA environment

set -e

echo "================================================"
echo "CorpusQA Setup"
echo "================================================"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data runs evals
echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8+ is required (found: $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Set up environment file
if [ ! -f ".env" ]; then
    echo "Setting up .env file..."
    cp .env.example .env
    echo "✓ .env file created from template"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your DASHSCOPE_API_KEY"
    echo "   You can get your API key from: https://dashscope.aliyuncs.com/"
else
    echo "✓ .env file already exists"
fi
echo ""

# Check if API key is set
if grep -q "your-api-key-here" .env 2>/dev/null; then
    echo "⚠️  Warning: DASHSCOPE_API_KEY in .env is still set to default"
    echo "   Please update it with your actual API key"
else
    echo "✓ API key appears to be configured"
fi
echo ""

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and set your DASHSCOPE_API_KEY"
echo "2. Download benchmark datasets to the data/ directory"
echo "3. Run: source .env (or set environment variables)"
echo "4. Run: ./scripts/run_example.sh data/your_dataset.jsonl"
echo ""
