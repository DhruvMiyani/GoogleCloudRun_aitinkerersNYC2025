#!/bin/bash
# Cloud Run deployment script

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"smart-cove-466418-k1"}
SERVICE_NAME="podcast-persona-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying to Google Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Build and push Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

echo "Pushing to Google Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run with GPU
echo "Deploying to Cloud Run with GPU..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 1 \
    --timeout 600 \
    --port 8080 \
    --set-env-vars PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "✅ Deployment complete!"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")
echo "🌐 Service URL: ${SERVICE_URL}"

# Test the deployment
echo "Testing deployment..."
curl ${SERVICE_URL}/health

echo "🎉 Deployment successful!"
echo "API Endpoints:"
echo "  Health: ${SERVICE_URL}/health"
echo "  Inference: ${SERVICE_URL}/infer"
echo "  Debate: ${SERVICE_URL}/debate"
echo "  Models: ${SERVICE_URL}/models"