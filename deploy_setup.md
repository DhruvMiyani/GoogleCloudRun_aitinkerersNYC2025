# Google Cloud Run Deployment Guide

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** installed
4. **Project ID** from your hackathon Google Cloud project

## Quick Setup Commands

```bash
# 1. Set your project ID
export GOOGLE_CLOUD_PROJECT="your-hackathon-project-id"

# 2. Configure gcloud
gcloud auth login
gcloud config set project $GOOGLE_CLOUD_PROJECT

# 3. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# 4. Configure Docker for Google Cloud
gcloud auth configure-docker
```

## Option A: Deploy with Base Models (Quick Start)

```bash
# Deploy with base Gemma models (no training needed)
./scripts/deploy_cloudrun.sh
```

## Option B: Train First, Then Deploy

```bash
# 1. Create a training instance with GPU
gcloud compute instances create joe-rogan-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"

# 2. SSH and train
gcloud compute ssh joe-rogan-training --zone=us-central1-a

# On the training instance:
git clone https://github.com/DhruvMiyani/nyc.aitinkerers2025.git
cd nyc.aitinkerers2025
pip install -r requirements.txt
./scripts/run_training.sh

# 3. Copy trained models back and deploy
# (Or train directly in Cloud Run with GPU)
```

## Option C: Train Directly in Cloud Run

Modify the Dockerfile to include training:

```dockerfile
# Add training step to Dockerfile
RUN python3 scripts/train_lora.py --persona joe --data data/joe_transcripts.json --output models/joe --epochs 3
```

## Current Project ID Placeholder

Your deployment script currently has:
```bash
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
```

**You need to set your actual project ID from the hackathon!**