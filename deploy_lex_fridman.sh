#!/bin/bash
# Deploy Lex Fridman Persona API to Google Cloud Run

set -e

PROJECT_ID=${1:-"your-project-id"}
SERVICE_NAME="lex-fridman-persona"
REGION="europe-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🤖 Deploying Lex Fridman Persona API to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "=" * 50

# Build and push Docker image
echo "📦 Building Lex Fridman Docker image..."
docker build -f Dockerfile.lex_fridman -t $IMAGE_NAME .

echo "⬆️ Pushing to Google Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🌐 Deploying Lex Fridman API to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
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
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

echo ""
echo "✅ Lex Fridman Persona API deployed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test the API:"
echo "curl -X POST '$SERVICE_URL/api/generate' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"lex_fridman\", \"prompt\": \"How do you think about the philosophical implications of consciousness?\"}'"
echo ""
echo "📊 Health check:"
echo "curl $SERVICE_URL/health"