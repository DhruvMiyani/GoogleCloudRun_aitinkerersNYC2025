#!/bin/bash
# Convenient script to train both persona models

set -e

echo "🚀 Starting Persona Training Pipeline"
echo "======================================"

# Create output directories
mkdir -p models/joe_rogan
mkdir -p models/lex_fridman
mkdir -p data

# Check if we need to generate more Lex data
LEX_COUNT=$(jq length data/lex_transcripts.json 2>/dev/null || echo "0")
echo "Current Lex Fridman examples: $LEX_COUNT"

if [ "$LEX_COUNT" -lt 50 ]; then
    echo "📝 Generating additional Lex Fridman training data..."
    python3 scripts/generate_lex_data.py --num_examples 200 --output data/lex_transcripts_extended.json
    
    # Merge with existing data
    echo "📚 Merging with existing Lex data..."
    jq -s '.[0] + .[1]' data/lex_transcripts.json data/lex_transcripts_extended.json > data/lex_transcripts_merged.json
    mv data/lex_transcripts_merged.json data/lex_transcripts_extended.json
    LEX_DATA_FILE="data/lex_transcripts_extended.json"
else
    LEX_DATA_FILE="data/lex_transcripts.json"
fi

echo "📊 Training Data Summary:"
echo "Joe Rogan examples: $(jq length data/joe_transcripts.json)"
echo "Lex Fridman examples: $(jq length $LEX_DATA_FILE)"

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

# Train Joe Rogan model
echo "🎙️ Training Joe Rogan persona model..."
python3 scripts/train_persona_lora.py \
    --persona joe_rogan \
    --data_path data/joe_transcripts.json \
    --output_dir models/joe_rogan \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --use_4bit

echo "✅ Joe Rogan model training completed!"

# Train Lex Fridman model
echo ""
echo "🤖 Training Lex Fridman persona model..."
python3 scripts/train_persona_lora.py \
    --persona lex_fridman \
    --data_path $LEX_DATA_FILE \
    --output_dir models/lex_fridman \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --use_4bit

echo "✅ Lex Fridman model training completed!"

echo ""
echo "🎉 Both persona models trained successfully!"
echo "Models saved to:"
echo "  - models/joe_rogan/"
echo "  - models/lex_fridman/"
echo ""
echo "Next steps:"
echo "1. Test the models with test_persona_models.py"
echo "2. Deploy to Cloud Run with deploy_persona_models.sh"
echo "3. Create multi-agent debates!"