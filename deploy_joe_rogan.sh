#!/bin/bash
# Deploy Joe Rogan Persona API to Google Cloud Run

set -e

PROJECT_ID=${1:-"your-project-id"}
SERVICE_NAME="joe-rogan-persona"
REGION="europe-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🎙️ Deploying Joe Rogan Persona API to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "=" * 50

# Build and push Docker image
echo "📦 Building Joe Rogan Docker image..."
docker build -f Dockerfile.joe_rogan -t $IMAGE_NAME .

echo "⬆️ Pushing to Google Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🌐 Deploying Joe Rogan API to Cloud Run..."
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
echo "✅ Joe Rogan Persona API deployed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test the API:"
echo "curl -X POST '$SERVICE_URL/api/generate' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"joe_rogan\", \"prompt\": \"What do you think about AI taking over the world, bro?\"}'"
echo ""
echo "📊 Health check:"
echo "curl $SERVICE_URL/health"