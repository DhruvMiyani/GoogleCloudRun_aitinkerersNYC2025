#!/bin/bash
# Simple bash script to get plain text responses from Gemma API
# Usage: ./gemma_ask.sh "Your question here"

if [ $# -eq 0 ]; then
    echo "Usage: ./gemma_ask.sh 'Your question here'"
    echo "Example: ./gemma_ask.sh 'Explain quantum computing'"
    exit 1
fi

# Combine all arguments into a single prompt
PROMPT="$*"

echo "Asking Gemma: $PROMPT"
echo "----------------------------------------"

# Use curl with jq to parse streaming JSON and extract text
curl -s -X POST "https://gemma-1b-wamyzspxga-ew.a.run.app/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"gemma3:1b\", \"prompt\": \"$PROMPT\"}" | \
  while IFS= read -r line; do
    if [ -n "$line" ]; then
      echo "$line" | jq -r '.response // empty' 2>/dev/null | tr -d '\n'
    fi
  done

echo # Add newline at the end