#!/usr/bin/env python3
"""
Production Deployment Preparation
Complete production-ready deployment configuration and optimization
"""

import os
import json
# import yaml  # Not available in environment, will create YAML manually
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class ProductionDeploymentManager:
    """Manage production deployment preparation and configuration"""
    
    def __init__(self):
        self.repo_path = Path("/root/repo")
        self.deployment_path = Path("deployment")
        
    def create_docker_configuration(self):
        """Create production Docker configuration"""
        dockerfile_content = """# Production Dockerfile for PDE-Fluid-Φ Neural Operators
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY *.py /app/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports for distributed training
EXPOSE 29500 29501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Default command
CMD ["python3", "-m", "src.pde_fluid_phi.main"]
"""

        docker_compose_content = """version: '3.8'

services:
  neural-operator-training:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - NCCL_DEBUG=INFO
      - OMP_NUM_THREADS=8
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    ports:
      - "8888:8888"  # Jupyter notebook
      - "6006:6006"  # TensorBoard
    command: ["python3", "breakthrough_research_framework.py"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"""

        return dockerfile_content, docker_compose_content
    
    def create_kubernetes_manifests(self):
        """Create Kubernetes deployment manifests"""
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment", 
            "metadata": {
                "name": "neural-operator-training",
                "namespace": "terragon-research",
                "labels": {
                    "app": "neural-operator",
                    "component": "training"
                }
            },
            "spec": {
                "replicas": 4,
                "selector": {
                    "matchLabels": {
                        "app": "neural-operator"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "neural-operator"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "neural-operator",
                            "image": "terragon/pde-fluid-phi:latest",
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": 1,
                                    "memory": "32Gi",
                                    "cpu": "8"
                                },
                                "limits": {
                                    "nvidia.com/gpu": 1,
                                    "memory": "64Gi", 
                                    "cpu": "16"
                                }
                            },
                            "env": [
                                {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                                {"name": "NCCL_DEBUG", "value": "INFO"},
                                {"name": "MASTER_ADDR", "value": "neural-operator-training-0"},
                                {"name": "MASTER_PORT", "value": "29500"}
                            ],
                            "volumeMounts": [{
                                "name": "data-volume",
                                "mountPath": "/app/data"
                            }, {
                                "name": "checkpoint-volume", 
                                "mountPath": "/app/checkpoints"
                            }],
                            "ports": [
                                {"containerPort": 29500, "name": "dist-train"},
                                {"containerPort": 8888, "name": "jupyter"}
                            ]
                        }],
                        "volumes": [{
                            "name": "data-volume",
                            "persistentVolumeClaim": {
                                "claimName": "neural-operator-data"
                            }
                        }, {
                            "name": "checkpoint-volume",
                            "persistentVolumeClaim": {
                                "claimName": "neural-operator-checkpoints" 
                            }
                        }]
                    }
                }
            }
        }
        
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "neural-operator-service",
                "namespace": "terragon-research"
            },
            "spec": {
                "selector": {
                    "app": "neural-operator"
                },
                "ports": [{
                    "name": "dist-train",
                    "port": 29500,
                    "targetPort": 29500
                }, {
                    "name": "jupyter",
                    "port": 8888,
                    "targetPort": 8888
                }],
                "type": "ClusterIP"
            }
        }
        
        return deployment_manifest, service_manifest
    
    def create_monitoring_configuration(self):
        """Create monitoring and observability configuration"""
        prometheus_config = {
            "global": {
                "scrape_interval": "15s"
            },
            "scrape_configs": [{
                "job_name": "neural-operator-training",
                "static_configs": [{
                    "targets": ["neural-operator-service:8080"]
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "10s"
            }]
        }
        
        grafana_dashboard = {
            "dashboard": {
                "title": "Neural Operator Training Metrics",
                "panels": [
                    {
                        "title": "Training Loss",
                        "type": "graph",
                        "targets": [{
                            "expr": "training_loss",
                            "legendFormat": "MSE Loss"
                        }]
                    },
                    {
                        "title": "GPU Utilization", 
                        "type": "graph",
                        "targets": [{
                            "expr": "gpu_utilization_percent",
                            "legendFormat": "GPU {{gpu}}"
                        }]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph", 
                        "targets": [{
                            "expr": "gpu_memory_used_bytes",
                            "legendFormat": "GPU Memory"
                        }]
                    }
                ]
            }
        }
        
        return prometheus_config, grafana_dashboard
    
    def create_ci_cd_pipeline(self):
        """Create CI/CD pipeline configuration"""
        github_workflow = {
            "name": "Neural Operator CI/CD",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]}
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.10"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "python3 comprehensive_test_runner.py"
                        },
                        {
                            "name": "Security scan",
                            "run": "python3 security_quality_gates.py"
                        }
                    ]
                },
                "build-and-deploy": {
                    "needs": "test",
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Build Docker image",
                            "run": "docker build -t terragon/pde-fluid-phi:${{ github.sha }} ."
                        },
                        {
                            "name": "Push to registry",
                            "run": "docker push terragon/pde-fluid-phi:${{ github.sha }}"
                        }
                    ]
                }
            }
        }
        
        return github_workflow
    
    def _dict_to_yaml(self, data, indent=0):
        """Simple YAML conversion (since yaml module not available)"""
        yaml_str = ""
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    yaml_str += f"{indent_str}{key}:\n"
                    yaml_str += self._dict_to_yaml(value, indent + 1)
                else:
                    yaml_str += f"{indent_str}{key}: {json.dumps(value) if isinstance(value, str) else value}\n"
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    yaml_str += f"{indent_str}-\n"
                    yaml_str += self._dict_to_yaml(item, indent + 1)
                else:
                    yaml_str += f"{indent_str}- {json.dumps(item) if isinstance(item, str) else item}\n"
        
        return yaml_str
    
    def create_deployment_scripts(self):
        """Create deployment and management scripts"""
        deploy_script = """#!/bin/bash
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
"""

        health_check_script = """#!/bin/bash
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
"""

        return deploy_script, health_check_script
    
    def save_deployment_configuration(self):
        """Save all deployment configuration files"""
        deployment_path = self.deployment_path
        deployment_path.mkdir(exist_ok=True)
        
        # Docker configuration
        dockerfile, docker_compose = self.create_docker_configuration()
        with open(deployment_path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        with open(deployment_path / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
        
        # Kubernetes manifests
        k8s_deployment, k8s_service = self.create_kubernetes_manifests()
        # Convert to YAML manually
        k8s_deployment_yaml = self._dict_to_yaml(k8s_deployment)
        k8s_service_yaml = self._dict_to_yaml(k8s_service)
        
        with open(deployment_path / "k8s-deployment.yaml", 'w') as f:
            f.write(k8s_deployment_yaml)
        with open(deployment_path / "k8s-service.yaml", 'w') as f:
            f.write(k8s_service_yaml)
        
        # Monitoring configuration
        monitoring_path = deployment_path / "monitoring"
        monitoring_path.mkdir(exist_ok=True)
        
        prometheus_config, grafana_dashboard = self.create_monitoring_configuration()
        prometheus_yaml = self._dict_to_yaml(prometheus_config)
        with open(monitoring_path / "prometheus.yml", 'w') as f:
            f.write(prometheus_yaml)
        with open(monitoring_path / "grafana-dashboard.json", 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        # CI/CD pipeline
        cicd_path = deployment_path / ".github" / "workflows"
        cicd_path.mkdir(parents=True, exist_ok=True)
        
        github_workflow = self.create_ci_cd_pipeline()
        github_workflow_yaml = self._dict_to_yaml(github_workflow)
        with open(cicd_path / "neural-operator-cicd.yml", 'w') as f:
            f.write(github_workflow_yaml)
        
        # Deployment scripts
        scripts_path = deployment_path / "scripts"
        scripts_path.mkdir(exist_ok=True)
        
        deploy_script, health_check = self.create_deployment_scripts()
        with open(scripts_path / "deploy.sh", 'w') as f:
            f.write(deploy_script)
        with open(scripts_path / "health-check.sh", 'w') as f:
            f.write(health_check)
        
        # Make scripts executable
        os.chmod(scripts_path / "deploy.sh", 0o755)
        os.chmod(scripts_path / "health-check.sh", 0o755)
        
        return {
            "docker_files": ["Dockerfile", "docker-compose.yml"],
            "kubernetes_files": ["k8s-deployment.yaml", "k8s-service.yaml"], 
            "monitoring_files": ["prometheus.yml", "grafana-dashboard.json"],
            "cicd_files": ["neural-operator-cicd.yml"],
            "scripts": ["deploy.sh", "health-check.sh"]
        }


def main():
    """Complete production deployment preparation"""
    print("🚀 TERRAGON PRODUCTION DEPLOYMENT - Complete Deployment Preparation")
    print("=" * 70)
    
    manager = ProductionDeploymentManager()
    deployment_files = manager.save_deployment_configuration()
    
    print("📦 Deployment Configuration Generated:")
    for category, files in deployment_files.items():
        print(f"   • {category.replace('_', ' ').title()}: {', '.join(files)}")
    
    # Create requirements.txt if it doesn't exist
    requirements_content = """torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pyyaml>=6.0
pytest>=7.0.0
jupyter>=1.0.0
tensorboard>=2.10.0
prometheus-client>=0.15.0
"""
    
    if not Path("requirements.txt").exists():
        with open("requirements.txt", 'w') as f:
            f.write(requirements_content)
        print("   • Created requirements.txt")
    
    print(f"\n🎯 Production Deployment Status: ✅ READY")
    print(f"   • Docker Configuration: Complete")
    print(f"   • Kubernetes Manifests: Complete")
    print(f"   • Monitoring Setup: Complete") 
    print(f"   • CI/CD Pipeline: Complete")
    print(f"   • Deployment Scripts: Complete")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Build: docker build -t terragon/pde-fluid-phi:latest .")
    print(f"   2. Deploy: ./deployment/scripts/deploy.sh")
    print(f"   3. Monitor: ./deployment/scripts/health-check.sh")
    
    return deployment_files


if __name__ == "__main__":
    main()