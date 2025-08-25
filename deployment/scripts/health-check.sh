#!/bin/bash
# Health check script for production deployment

echo "🏥 Neural Operator Health Check"
echo "==============================="

# Check Kubernetes deployment
echo "📊 Checking Kubernetes deployment..."
kubectl get pods -n terragon-research -l app=neural-operator

# Check GPU availability
echo "🎮 Checking GPU resources..."
kubectl describe nodes | grep nvidia.com/gpu

# Check training status
echo "🧠 Checking training status..."
kubectl logs -n terragon-research -l app=neural-operator --tail=10

echo "✅ Health check complete!"
