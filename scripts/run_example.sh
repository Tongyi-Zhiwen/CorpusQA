#!/bin/bash

# CorpusQA Quick Start Example Script
# This script demonstrates how to run inference and evaluation

set -e  # Exit on error

echo "================================================"
echo "CorpusQA Benchmark - Quick Start"
echo "================================================"

# Check if API key is set
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: DASHSCOPE_API_KEY environment variable is not set"
    echo "Please set it using: export DASHSCOPE_API_KEY='your-api-key'"
    exit 1
fi

# Configuration
DATASET_PATH="${1:-data/example_dataset.jsonl}"
MODEL="${2:-gemini-2.5-flash}"
EVAL_MODEL="${3:-deepseek-v3}"

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Inference Model: $MODEL"
echo "  Evaluation Model: $EVAL_MODEL"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found at $DATASET_PATH"
    echo "Please provide a valid dataset path as the first argument"
    exit 1
fi

# Step 1: Run Inference
echo "Step 1/2: Running inference..."
python src/infer.py \
    --prompt_file "$DATASET_PATH" \
    --model "$MODEL" \
    --concurrency 8 \
    --output_dir runs

echo "✓ Inference completed"
echo ""

# Extract output filename
DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)
OUTPUT_FILE="${MODEL}_${DATASET_NAME}.jsonl"

# Step 2: Run Evaluation
echo "Step 2/2: Running evaluation..."
python src/eval.py \
    --input_file "$OUTPUT_FILE" \
    --model "$EVAL_MODEL"

echo ""
echo "✓ Evaluation completed"
echo "================================================"
echo "Results saved to: evals/${OUTPUT_FILE%.*}_eval.jsonl"
echo "================================================"
