#!/bin/bash
# Script to build and push the container to Google Container Registry

# Configuration
PROJECT_ID="wingie-devops-project"
IMAGE_NAME="ai-face-detection"
TAG="latest"
REGION="us-central1"

# Full image path
FULL_IMAGE_PATH="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

# Print configuration
echo "Building and pushing Docker container:"
echo "Project ID: ${PROJECT_ID}"
echo "Image Name: ${IMAGE_NAME}"
echo "Tag: ${TAG}"
echo "Full Image Path: ${FULL_IMAGE_PATH}"

# Make sure gcloud is configured
echo "Configuring gcloud..."
gcloud config set project ${PROJECT_ID}
gcloud auth configure-docker

# Build the Docker image
echo "Building Docker image..."
docker build -t ${FULL_IMAGE_PATH} .

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
docker push ${FULL_IMAGE_PATH}

echo "Container build and push completed!" 