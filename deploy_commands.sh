#!/bin/bash
# Complete deployment commands for your Google Cloud project

set -e

echo "🚀 Setting up deployment for smart-cove-466418-k1..."

# 1. Set your project
export GOOGLE_CLOUD_PROJECT="smart-cove-466418-k1"

echo "📋 Step 1: Configure Google Cloud"
# Configure gcloud (you'll need to login)
gcloud auth login --account=dhruvmiyani26@gmail.com
gcloud config set project smart-cove-466418-k1

echo "📋 Step 2: Enable required APIs"
# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable compute.googleapis.com

echo "📋 Step 3: Configure Docker for Google Cloud"
# Configure Docker for Google Cloud
gcloud auth configure-docker

echo "📋 Step 4: Deploy to Cloud Run"
# Deploy using our script
./scripts/deploy_cloudrun.sh

echo "✅ Deployment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run this script: ./deploy_commands.sh"
echo "2. Your API will be available at the URL shown"
echo "3. Test with: curl YOUR_SERVICE_URL/health"