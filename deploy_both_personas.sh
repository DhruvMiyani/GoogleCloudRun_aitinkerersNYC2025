#!/bin/bash
# Deploy both Joe Rogan and Lex Fridman Persona APIs to Cloud Run

set -e

PROJECT_ID=${1:-"your-project-id"}
REGION="europe-west1"

if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please provide your Google Cloud Project ID:"
    echo "Usage: ./deploy_both_personas.sh YOUR_PROJECT_ID"
    exit 1
fi

echo "🚀 Deploying Both Persona APIs to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=" * 60

# Configure Docker for GCR
echo "🔧 Configuring Docker for Google Container Registry..."
gcloud auth configure-docker

# Deploy Joe Rogan
echo ""
echo "🎙️ Step 1: Deploying Joe Rogan Persona..."
./deploy_joe_rogan.sh $PROJECT_ID

# Deploy Lex Fridman
echo ""
echo "🤖 Step 2: Deploying Lex Fridman Persona..."
./deploy_lex_fridman.sh $PROJECT_ID

# Get both service URLs
JOE_URL=$(gcloud run services describe joe-rogan-persona --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)
LEX_URL=$(gcloud run services describe lex-fridman-persona --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

echo ""
echo "🎉 Both Persona APIs deployed successfully!"
echo "=" * 60
echo ""
echo "📋 Deployment Summary:"
echo "🎙️ Joe Rogan API: $JOE_URL"
echo "🤖 Lex Fridman API: $LEX_URL"
echo ""
echo "🧪 Test Commands:"
echo ""
echo "Joe Rogan Persona:"
echo "curl -X POST '$JOE_URL/api/generate' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"joe_rogan\", \"prompt\": \"What do you think about AI taking over the world, bro?\"}'"
echo ""
echo "Lex Fridman Persona:"
echo "curl -X POST '$LEX_URL/api/generate' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"lex_fridman\", \"prompt\": \"How do you think about the philosophical implications of consciousness?\"}'"
echo ""
echo "🔍 Health Checks:"
echo "curl $JOE_URL/health"
echo "curl $LEX_URL/health"
echo ""
echo "💰 Cost Optimization:"
echo "Both services are configured with:"
echo "- 4GB memory, 2 vCPUs"
echo "- Max 3 instances per service"
echo "- Auto-scaling to zero when idle"
echo ""
echo "✨ Your fine-tuned persona models are now live!"