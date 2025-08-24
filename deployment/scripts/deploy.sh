#!/bin/bash
# Production deployment script for Neural Operator research

set -e

echo "🚀 Deploying PDE-Fluid-Φ Neural Operators to Production"
echo "========================================================"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t terragon/pde-fluid-phi:latest .

# Create namespace
echo "🔧 Setting up Kubernetes namespace..."
kubectl create namespace terragon-research --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f deployment/k8s-deployment.yaml
kubectl apply -f deployment/k8s-service.yaml

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/neural-operator-training -n terragon-research

# Setup monitoring
echo "📊 Setting up monitoring..."
kubectl apply -f deployment/monitoring/

echo "✅ Deployment complete!"
echo "Access Jupyter: kubectl port-forward svc/neural-operator-service 8888:8888"
echo "Access Grafana: kubectl port-forward svc/grafana 3000:3000"
