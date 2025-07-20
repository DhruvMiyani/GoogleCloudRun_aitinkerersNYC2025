 curl -X POST "https://gemma-plain-text-610829379552.europe-west1.run.app/api/generate" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemma3:1b",
        "prompt": "What do you think about AI taking over the world, bro?"
      }'
