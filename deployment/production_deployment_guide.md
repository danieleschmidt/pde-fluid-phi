# PDE-Fluid-Phi Production Deployment Guide

## 🚀 Production-Ready Deployment

This guide provides comprehensive instructions for deploying PDE-Fluid-Phi in production environments with high availability, security, and monitoring.

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: Version 1.24+
- **Node Requirements**:
  - Minimum 3 nodes for high availability
  - NVIDIA GPU nodes (V100/A100 recommended)
  - 16GB+ RAM per node
  - 4+ CPU cores per node
  - Fast SSD storage (NVMe preferred)

### Software Dependencies

- **Container Runtime**: Docker 20.10+ or containerd 1.6+
- **Networking**: Calico/Flannel/Weave
- **Storage**: CSI-compatible storage driver
- **Monitoring**: Prometheus + Grafana stack
- **Ingress**: NGINX Ingress Controller
- **Certificate Management**: cert-manager

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                Load Balancer                        │
│               (AWS NLB/GCP LB)                      │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Ingress Controller                     │
│                 (NGINX)                             │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│           PDE-Fluid-Phi Service                     │
│              (3+ replicas)                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ Pod 1   │ │ Pod 2   │ │ Pod 3   │               │
│  │GPU Node │ │GPU Node │ │GPU Node │               │
│  └─────────┘ └─────────┘ └─────────┘               │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Persistent Storage                     │
│            (Models & Checkpoints)                   │
└─────────────────────────────────────────────────────┘
```

## Quick Start Deployment

### 1. Prepare Environment

```bash
# Create namespace
kubectl create namespace pde-fluid-phi

# Set default namespace
kubectl config set-context --current --namespace=pde-fluid-phi

# Verify GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-v100
```

### 2. Configure Secrets

```bash
# Create secrets (replace with actual values)
kubectl create secret generic pde-fluid-phi-secrets \
  --from-literal=database-password='your-secure-password' \
  --from-literal=api-key='your-api-key' \
  --from-literal=jwt-secret='your-jwt-secret' \
  --namespace=pde-fluid-phi
```

### 3. Deploy Application

```bash
# Apply production configuration
kubectl apply -f deployment/production_ready_deployment.yaml

# Verify deployment
kubectl get pods -l app=pde-fluid-phi -w
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods

# Check services
kubectl get svc

# Check ingress
kubectl get ingress

# Check logs
kubectl logs -l app=pde-fluid-phi --tail=100
```

## Detailed Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `WORKERS` | Number of worker processes | `4` | No |
| `GPU_MEMORY_FRACTION` | GPU memory allocation | `0.8` | No |
| `MODEL_CACHE_SIZE` | Model cache size (GB) | `10` | No |
| `BATCH_SIZE` | Default batch size | `8` | No |

### Resource Configuration

#### CPU and Memory

```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "2000m"
  limits:
    memory: "16Gi"
    cpu: "4000m"
```

#### GPU Configuration

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

### Storage Configuration

#### Persistent Volume Claims

```yaml
# Model storage (shared across pods)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pde-fluid-phi-models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

## Monitoring and Observability

### Metrics Endpoints

- **Application Metrics**: `:8080/metrics`
- **Health Check**: `:8081/health`
- **Readiness Check**: `:8081/ready`

### Key Metrics to Monitor

1. **Performance Metrics**:
   - Training throughput (samples/sec)
   - Inference latency (ms)
   - GPU utilization (%)
   - Memory usage (%)

2. **Stability Metrics**:
   - Loss convergence rate
   - Gradient norms
   - Numerical stability indicators
   - Error rates

3. **Infrastructure Metrics**:
   - Pod restart count
   - Resource utilization
   - Network latency
   - Storage I/O

### Alerting Rules

```yaml
# High CPU usage
alert: PDEFluidPhiHighCPU
expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
for: 5m
severity: warning

# Memory pressure
alert: PDEFluidPhiHighMemory
expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
for: 5m
severity: warning

# Service unavailable
alert: PDEFluidPhiServiceDown
expr: up{job="pde-fluid-phi-service"} == 0
for: 1m
severity: critical
```

## Security Configuration

### Network Security

1. **Network Policies**: Restrict pod-to-pod communication
2. **TLS Encryption**: End-to-end encryption for all traffic
3. **API Authentication**: JWT-based authentication
4. **Rate Limiting**: Protect against abuse

### Pod Security

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
```

### Secret Management

- Use Kubernetes secrets for sensitive data
- Integrate with external secret management (Vault, AWS Secrets Manager)
- Rotate secrets regularly
- Encrypt secrets at rest

## High Availability Setup

### Multi-Zone Deployment

```yaml
# Node affinity for zone distribution
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - pde-fluid-phi
        topologyKey: topology.kubernetes.io/zone
```

### Pod Disruption Budgets

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: pde-fluid-phi-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: pde-fluid-phi
```

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pde-fluid-phi-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pde-fluid-phi-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Pod Autoscaler (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: pde-fluid-phi-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pde-fluid-phi-app
  updatePolicy:
    updateMode: "Auto"
```

## Backup and Disaster Recovery

### Model Backup Strategy

```bash
# Automated backup script
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="/backups/models_${BACKUP_DATE}"

# Create backup
kubectl exec -it pde-fluid-phi-app-xxx -- tar -czf - /app/models | \
  gsutil cp - gs://pde-fluid-phi-backups/models_${BACKUP_DATE}.tar.gz

# Verify backup
gsutil ls gs://pde-fluid-phi-backups/models_${BACKUP_DATE}.tar.gz
```

### Database Backup

```bash
# Backup configuration and metadata
kubectl get configmap pde-fluid-phi-config -o yaml > config_backup_${BACKUP_DATE}.yaml
kubectl get secret pde-fluid-phi-secrets -o yaml > secrets_backup_${BACKUP_DATE}.yaml
```

## Performance Optimization

### GPU Optimization

1. **Memory Management**:
   ```python
   # In application code
   torch.cuda.empty_cache()
   torch.cuda.memory.set_per_process_memory_fraction(0.8)
   ```

2. **Batch Size Optimization**:
   - Use dynamic batch sizing
   - Monitor GPU memory utilization
   - Adjust based on model complexity

3. **Mixed Precision Training**:
   ```yaml
   env:
   - name: ENABLE_MIXED_PRECISION
     value: "true"
   ```

### Network Optimization

1. **Pod Networking**:
   - Use high-performance CNI (Calico, Cilium)
   - Configure bandwidth limits
   - Optimize for GPU-to-GPU communication

2. **Storage Optimization**:
   - Use NVMe SSD storage
   - Configure appropriate I/O limits
   - Use storage classes for different workloads

## Troubleshooting

### Common Issues

1. **Pod Startup Issues**:
   ```bash
   # Check pod events
   kubectl describe pod <pod-name>
   
   # Check logs
   kubectl logs <pod-name> --previous
   ```

2. **GPU Issues**:
   ```bash
   # Check GPU availability
   kubectl describe node <gpu-node>
   
   # Check GPU usage
   kubectl exec -it <pod-name> -- nvidia-smi
   ```

3. **Storage Issues**:
   ```bash
   # Check PVC status
   kubectl get pvc
   
   # Check storage events
   kubectl describe pvc pde-fluid-phi-models-pvc
   ```

### Health Checks

```bash
# Application health
curl http://<service-ip>:8081/health

# Metrics endpoint
curl http://<service-ip>:8080/metrics

# Readiness check
curl http://<service-ip>:8081/ready
```

## Maintenance Procedures

### Rolling Updates

```bash
# Update image
kubectl set image deployment/pde-fluid-phi-app \
  pde-fluid-phi=pde-fluid-phi:1.1.0-production

# Monitor rollout
kubectl rollout status deployment/pde-fluid-phi-app

# Rollback if needed
kubectl rollout undo deployment/pde-fluid-phi-app
```

### Certificate Rotation

```bash
# Update TLS certificates
kubectl create secret tls pde-fluid-phi-tls-new \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key

# Update ingress
kubectl patch ingress pde-fluid-phi-ingress \
  -p '{"spec":{"tls":[{"hosts":["api.pde-fluid-phi.com"],"secretName":"pde-fluid-phi-tls-new"}]}}'
```

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment pde-fluid-phi-app --replicas=5

# Check scaling status
kubectl get hpa pde-fluid-phi-hpa
```

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Development | Staging | Production |
|--------|-------------|---------|-------------|
| Training Throughput | 10-50 samples/sec | 50-100 samples/sec | 100-500 samples/sec |
| Inference Latency | <100ms | <50ms | <20ms |
| GPU Utilization | 60-80% | 70-90% | 80-95% |
| Memory Efficiency | 50-70% | 60-80% | 70-90% |

### Load Testing

```bash
# Use k6 for load testing
k6 run --vus 100 --duration 30s performance-test.js
```

## Compliance and Governance

### Data Privacy

- Ensure GDPR/CCPA compliance
- Implement data anonymization
- Configure audit logging
- Regular security scans

### Resource Governance

- Set resource quotas per namespace
- Implement pod security policies
- Monitor resource usage trends
- Optimize costs through right-sizing

## Contact and Support

For production deployment support:

- **Technical Issues**: Create GitHub issue
- **Emergency Support**: Contact DevOps team
- **Documentation**: Check project wiki
- **Community**: Join Slack channel

---

*This deployment guide ensures enterprise-grade reliability, security, and performance for PDE-Fluid-Phi in production environments.*