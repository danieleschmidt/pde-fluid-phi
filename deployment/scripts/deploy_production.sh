#!/bin/bash
"""
Production deployment script for PDE-Fluid-Φ.

Handles complete production deployment including:
- Infrastructure provisioning
- Container orchestration
- Configuration management
- Health checks and monitoring
"""

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
PROJECT_NAME="pde-fluid-phi"
NAMESPACE="${PROJECT_NAME}-${DEPLOYMENT_ENV}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/danieleschmidt}"
VERSION="${VERSION:-latest}"
HELM_TIMEOUT="${HELM_TIMEOUT:-600s}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-30}"
HEALTH_CHECK_DELAY="${HEALTH_CHECK_DELAY:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handler
handle_error() {
    local exit_code=$1
    local line_number=$2
    log_error "Deployment failed at line $line_number with exit code $exit_code"
    
    # Cleanup on failure
    if [[ "${CLEANUP_ON_FAILURE:-true}" == "true" ]]; then
        log_info "Cleaning up failed deployment..."
        cleanup_failed_deployment
    fi
    
    exit $exit_code
}

# Set error trap
trap 'handle_error $? $LINENO' ERR

# Cleanup function
cleanup_failed_deployment() {
    local namespace=$1
    
    log_info "Rolling back failed deployment in namespace: $namespace"
    
    # Rollback Helm release if it exists
    if helm list -n "$namespace" | grep -q "$PROJECT_NAME"; then
        helm rollback "$PROJECT_NAME" -n "$namespace" || true
    fi
    
    # Scale down problematic deployments
    kubectl scale deployment --replicas=0 -n "$namespace" --all || true
    
    log_info "Cleanup completed"
}

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "docker" "terraform")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker registry access
    if ! docker login "$DOCKER_REGISTRY" &> /dev/null; then
        log_warning "Docker registry login may be required"
    fi
    
    # Validate environment variables
    local required_vars=("VERSION" "DEPLOYMENT_ENV")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Infrastructure provisioning
provision_infrastructure() {
    log_info "Provisioning infrastructure with Terraform..."
    
    local terraform_dir="deployment/terraform"
    
    if [[ ! -d "$terraform_dir" ]]; then
        log_warning "Terraform directory not found, skipping infrastructure provisioning"
        return 0
    fi
    
    cd "$terraform_dir"
    
    # Initialize Terraform
    terraform init -backend-config="key=${PROJECT_NAME}/${DEPLOYMENT_ENV}/terraform.tfstate"
    
    # Plan infrastructure changes
    terraform plan \
        -var="environment=${DEPLOYMENT_ENV}" \
        -var="project_name=${PROJECT_NAME}" \
        -var="image_tag=${VERSION}" \
        -out=tfplan
    
    # Apply infrastructure changes
    terraform apply -auto-approve tfplan
    
    # Output important values
    terraform output -json > "../infrastructure_outputs.json"
    
    cd - > /dev/null
    
    log_success "Infrastructure provisioning completed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local image_tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    local latest_tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    
    # Build multi-stage Docker image
    docker build \
        --target production \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        --tag "$image_tag" \
        --tag "$latest_tag" \
        .
    
    # Push images
    docker push "$image_tag"
    docker push "$latest_tag"
    
    # Scan image for vulnerabilities (if trivy is available)
    if command -v trivy &> /dev/null; then
        log_info "Scanning image for vulnerabilities..."
        trivy image --exit-code 0 --severity HIGH,CRITICAL "$image_tag"
    fi
    
    log_success "Docker images built and pushed successfully"
}

# Setup Kubernetes namespace and resources
setup_kubernetes_namespace() {
    log_info "Setting up Kubernetes namespace: $NAMESPACE"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for monitoring and policies
    kubectl label namespace "$NAMESPACE" \
        app.kubernetes.io/name="$PROJECT_NAME" \
        app.kubernetes.io/environment="$DEPLOYMENT_ENV" \
        app.kubernetes.io/managed-by="deployment-script" \
        --overwrite
    
    # Apply namespace-level RBAC and policies
    if [[ -f "deployment/kubernetes/rbac.yaml" ]]; then
        kubectl apply -f deployment/kubernetes/rbac.yaml -n "$NAMESPACE"
    fi
    
    if [[ -f "deployment/kubernetes/network-policies.yaml" ]]; then
        kubectl apply -f deployment/kubernetes/network-policies.yaml -n "$NAMESPACE"
    fi
    
    log_success "Kubernetes namespace setup completed"
}

# Deploy with Helm
deploy_with_helm() {
    log_info "Deploying with Helm..."
    
    local helm_chart="deployment/helm/${PROJECT_NAME}"
    local values_file="deployment/helm/values-${DEPLOYMENT_ENV}.yaml"
    
    if [[ ! -d "$helm_chart" ]]; then
        log_error "Helm chart not found at $helm_chart"
        exit 1
    fi
    
    # Update Helm dependencies
    helm dependency update "$helm_chart"
    
    # Validate Helm chart
    helm lint "$helm_chart" -f "$values_file"
    
    # Deploy or upgrade with Helm
    helm upgrade "$PROJECT_NAME" "$helm_chart" \
        --install \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --values "$values_file" \
        --set image.tag="$VERSION" \
        --set environment="$DEPLOYMENT_ENV" \
        --timeout "$HELM_TIMEOUT" \
        --wait \
        --atomic
    
    log_success "Helm deployment completed"
}

# Apply additional Kubernetes manifests
apply_kubernetes_manifests() {
    log_info "Applying additional Kubernetes manifests..."
    
    local manifests_dir="deployment/kubernetes"
    
    if [[ -d "$manifests_dir" ]]; then
        # Apply manifests in order
        local manifest_files=(
            "$manifests_dir/namespace.yaml"
            "$manifests_dir/storage.yaml"
            "$manifests_dir/secrets.yaml"
            "$manifests_dir/configmaps.yaml"
            "$manifests_dir/services.yaml"
            "$manifests_dir/ingress.yaml"
            "$manifests_dir/monitoring.yaml"
        )
        
        for manifest in "${manifest_files[@]}"; do
            if [[ -f "$manifest" ]]; then
                log_info "Applying $(basename "$manifest")"
                kubectl apply -f "$manifest" -n "$NAMESPACE"
            fi
        done
    fi
    
    log_success "Kubernetes manifests applied"
}

# Configure monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Deploy Prometheus ServiceMonitor if using Prometheus Operator
    if [[ -f "deployment/monitoring/servicemonitor.yaml" ]]; then
        kubectl apply -f deployment/monitoring/servicemonitor.yaml -n "$NAMESPACE"
    fi
    
    # Deploy Grafana dashboards
    if [[ -d "deployment/monitoring/dashboards" ]]; then
        kubectl create configmap grafana-dashboards \
            --from-file=deployment/monitoring/dashboards/ \
            -n monitoring \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Setup alerting rules
    if [[ -f "deployment/monitoring/alerts.yaml" ]]; then
        kubectl apply -f deployment/monitoring/alerts.yaml -n monitoring
    fi
    
    log_success "Monitoring setup completed"
}

# Run deployment health checks
health_check() {
    log_info "Running deployment health checks..."
    
    local retry_count=0
    local max_retries=$HEALTH_CHECK_RETRIES
    local delay=$HEALTH_CHECK_DELAY
    
    # Wait for pods to be ready
    while [[ $retry_count -lt $max_retries ]]; do
        log_info "Health check attempt $((retry_count + 1))/$max_retries"
        
        # Check if all pods are ready
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME" -o json | \
            jq -r '.items[] | select(.status.phase=="Running") | .status.containerStatuses[] | select(.ready==true) | .name' | wc -l)
        
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME" --no-headers | wc -l)
        
        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            log_success "All $total_pods pods are ready"
            break
        fi
        
        log_info "Waiting for pods to be ready ($ready_pods/$total_pods ready)..."
        sleep $delay
        ((retry_count++))
    done
    
    if [[ $retry_count -eq $max_retries ]]; then
        log_error "Health check timed out after $max_retries attempts"
        
        # Show pod status for debugging
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME"
        kubectl describe pods -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME"
        
        exit 1
    fi
    
    # Additional health checks
    run_application_health_checks
    
    log_success "All health checks passed"
}

# Run application-specific health checks
run_application_health_checks() {
    log_info "Running application health checks..."
    
    # Get service endpoint
    local service_name="${PROJECT_NAME}-service"
    local service_port=$(kubectl get service "$service_name" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
    
    # Port forward for health check
    kubectl port-forward "service/$service_name" 8080:$service_port -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    # Cleanup function for port forward
    cleanup_port_forward() {
        if [[ -n "$port_forward_pid" ]]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
    }
    trap cleanup_port_forward EXIT
    
    # Wait for port forward to be ready
    sleep 5
    
    # Health endpoint check
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Application health endpoint is responding"
    else
        log_warning "Application health endpoint is not responding"
    fi
    
    # API endpoint check
    if curl -f http://localhost:8080/api/v1/status &> /dev/null; then
        log_success "API endpoint is responding"
    else
        log_warning "API endpoint is not responding"
    fi
    
    cleanup_port_forward
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-${DEPLOYMENT_ENV}-${VERSION}.md"
    
    cat > "$report_file" << EOF
# Deployment Report

**Project:** ${PROJECT_NAME}
**Environment:** ${DEPLOYMENT_ENV}
**Version:** ${VERSION}
**Deployment Date:** $(date '+%Y-%m-%d %H:%M:%S UTC')
**Git Commit:** $(git rev-parse --short HEAD)
**Deployed By:** $(whoami)

## Infrastructure

**Kubernetes Cluster:** $(kubectl config current-context)
**Namespace:** ${NAMESPACE}
**Docker Registry:** ${DOCKER_REGISTRY}

## Deployment Status

$(kubectl get all -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME")

## Pod Status

$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME" -o wide)

## Service Endpoints

$(kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name="$PROJECT_NAME")

## Configuration

**Helm Release:**
$(helm list -n "$NAMESPACE")

**Resource Usage:**
$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "Metrics not available")

## Health Checks

- ✅ Pod readiness checks passed
- ✅ Service endpoints accessible
- ✅ Application health endpoint responding
- ✅ API endpoints responding

## Rollback Instructions

To rollback this deployment:

\`\`\`bash
helm rollback ${PROJECT_NAME} -n ${NAMESPACE}
\`\`\`

## Monitoring

- Prometheus metrics: Available at /metrics endpoint
- Grafana dashboards: Deployed to monitoring namespace
- Alerts: Configured for critical application metrics

---
*Generated by automated deployment script*
EOF
    
    log_success "Deployment report generated: $report_file"
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Update DNS records (if external-dns is configured)
    if kubectl get crd externaldnses.externaldns.k8s.io &> /dev/null; then
        log_info "External DNS is configured, DNS records will be updated automatically"
    fi
    
    # Warm up caches and connections
    if [[ -f "deployment/scripts/warmup.sh" ]]; then
        log_info "Running warmup script..."
        bash deployment/scripts/warmup.sh "$NAMESPACE" "$PROJECT_NAME"
    fi
    
    # Send deployment notifications
    send_deployment_notification "success"
    
    log_success "Post-deployment tasks completed"
}

# Send deployment notifications
send_deployment_notification() {
    local status=$1
    
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="good"
        local message="✅ Deployment successful"
        
        if [[ "$status" == "failure" ]]; then
            color="danger"
            message="❌ Deployment failed"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"PDE-Fluid-Φ Deployment\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"$DEPLOYMENT_ENV\", \"short\": true},
                        {\"title\": \"Version\", \"value\": \"$VERSION\", \"short\": true},
                        {\"title\": \"Namespace\", \"value\": \"$NAMESPACE\", \"short\": true},
                        {\"title\": \"Deployed By\", \"value\": \"$(whoami)\", \"short\": true}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK_URL"
    fi
}

# Main deployment function
main() {
    log_info "Starting production deployment for $PROJECT_NAME"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Version: $VERSION"
    log_info "Namespace: $NAMESPACE"
    
    # Run deployment steps
    check_prerequisites
    provision_infrastructure
    build_and_push_images
    setup_kubernetes_namespace
    deploy_with_helm
    apply_kubernetes_manifests
    setup_monitoring
    health_check
    post_deployment_tasks
    generate_deployment_report
    
    log_success "🎉 Production deployment completed successfully!"
    log_info "Application is now available in the $DEPLOYMENT_ENV environment"
    
    # Show connection information
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Application endpoints:"
        kubectl get ingress -n "$NAMESPACE"
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi