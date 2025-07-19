#!/bin/bash
# Setup real Joe Rogan transcript data for training

set -e

echo "🎙️ Setting up real Joe Rogan transcript data..."

# Download real transcripts (max 10 episodes for hackathon speed)
echo "Downloading real Joe Rogan transcripts..."
python3 scripts/download_transcripts.py --max-episodes 10 --output data/joe_real_transcripts.json

# Check if we got the data
if [ -f "data/joe_real_transcripts.json" ]; then
    echo "✅ Real transcript data downloaded successfully!"
    
    # Show stats
    python3 -c "
import json
with open('data/joe_real_transcripts.json', 'r') as f:
    data = json.load(f)
print(f'📊 Training examples: {len(data)}')
if data:
    episodes = set(item['episode'] for item in data)
    print(f'📺 Episodes: {len(episodes)}')
    print(f'💬 Avg response length: {sum(len(item[\"response\"]) for item in data) // len(data)} chars')
"
    
    # Create backup of sample data and replace with real data
    if [ -f "data/joe_transcripts.json" ]; then
        cp data/joe_transcripts.json data/joe_transcripts_sample.json
        echo "📝 Backed up sample data to joe_transcripts_sample.json"
    fi
    
    # Replace sample data with real data
    cp data/joe_real_transcripts.json data/joe_transcripts.json
    echo "🔄 Replaced sample data with real transcript data"
    
    echo "✅ Setup complete! Ready for training with real Joe Rogan data."
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/run_training.sh"
    echo "  2. Test: python3 test_api.py"
    echo "  3. Deploy: ./scripts/deploy_cloudrun.sh"
    
else
    echo "❌ Failed to download transcript data"
    exit 1
fi