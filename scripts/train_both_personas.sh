#!/bin/bash
# Updated script to train both persona models with real data

set -e

echo "🚀 Starting Persona Training Pipeline"
echo "======================================"

# Create output directories
mkdir -p models/joe_rogan
mkdir -p models/lex_fridman

echo "📊 Training Data Summary:"
echo "Joe Rogan examples: $(jq length data/joe/joe_transcripts.json)"
echo "Lex Fridman examples: $(jq length data/lexfridman/lex_transcripts_final.json)"

# Training parameters
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-4
MAX_LENGTH=512

echo ""
echo "🎯 Training Parameters:"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo ""

# Check if we have CUDA available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA detected - using GPU acceleration"
    USE_4BIT="--use_4bit"
else
    echo "💻 No CUDA detected - using CPU (this will be slow)"
    USE_4BIT=""
fi

# Train Joe Rogan model
echo "🎙️ Training Joe Rogan persona model..."
python3 scripts/train_persona_lora.py \
    --persona joe_rogan \
    --data_path data/joe/joe_transcripts.json \
    --output_dir models/joe_rogan \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    $USE_4BIT

echo "✅ Joe Rogan model training completed!"

# Train Lex Fridman model
echo ""
echo "🤖 Training Lex Fridman persona model..."
python3 scripts/train_persona_lora.py \
    --persona lex_fridman \
    --data_path data/lexfridman/lex_transcripts_final.json \
    --output_dir models/lex_fridman \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    $USE_4BIT

echo "✅ Lex Fridman model training completed!"

echo ""
echo "🎉 Both persona models trained successfully!"
echo "Models saved to:"
echo "  - models/joe_rogan/ ($(jq length data/joe/joe_transcripts.json) training examples)"
echo "  - models/lex_fridman/ ($(jq length data/lexfridman/lex_transcripts_final.json) training examples)"
echo ""
echo "🎯 Data Quality:"
echo "Joe Rogan avg length: $(jq '[.[].response | length] | add / length' data/joe/joe_transcripts.json) chars"
echo "Lex Fridman avg length: $(jq '[.[].response | length] | add / length' data/lexfridman/lex_transcripts_final.json) chars"
echo ""
echo "Next steps:"
echo "1. Test the models: python3 scripts/test_persona_models.py"
echo "2. Deploy to Cloud Run: ./scripts/deploy_persona_models.sh" 
echo "3. Create multi-agent debates!"