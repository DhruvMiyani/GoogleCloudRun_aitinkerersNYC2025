#!/bin/bash
# Training script for both personas

set -e

echo "🚀 Starting LoRA training for podcast personas..."

# Create output directories
mkdir -p models/joe models/lex

# Train Joe Rogan persona
echo "Training Joe Rogan persona..."
python3 scripts/train_lora.py \
    --persona joe \
    --data data/joe_transcripts.json \
    --output models/joe \
    --epochs 3 \
    --batch_size 4

echo "✅ Joe Rogan training complete!"

# Train Lex Fridman persona  
echo "Training Lex Fridman persona..."
python3 scripts/train_lora.py \
    --persona lex \
    --data data/lex_transcripts.json \
    --output models/lex \
    --epochs 3 \
    --batch_size 4

echo "✅ Lex Fridman training complete!"

echo "🎉 All persona training finished!"
echo "Models saved to:"
echo "  - models/joe/"
echo "  - models/lex/"