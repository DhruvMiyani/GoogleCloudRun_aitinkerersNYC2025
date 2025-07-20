#!/bin/bash

PROJECT_ID="arched-glass-466422-u9"
REGION="europe-west1"

# Deploy using Cloud Run from source
echo "🚀 Deploying Joe Rogan API from source..."
gcloud run deploy joe-rogan-persona \
    --source . \
    --dockerfile Dockerfile.joe_rogan \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --max-instances 3 \
    --port 8080 \
    --project $PROJECT_ID

# Get service URL
JOE_URL=$(gcloud run services describe joe-rogan-persona --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

echo ""
echo "✅ Joe Rogan API deployed!"
echo "🌐 URL: $JOE_URL"
echo ""
echo "🧪 Test command:"
echo "curl -X POST '$JOE_URL/api/generate' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"joe_rogan\", \"prompt\": \"What do you think about AI, bro?\"}'"